import copy
import functools
import logging
import os
import pathlib
import pickle
import sys
import time

import pandas as pd
import ptflops
import torch
from torch import nn
from torch.utils import data
from torchvision import datasets, models, transforms

ALWAYS_DOWNLOAD_MODEL = False
SHUFFLE = False
BATCH_SIZE = 32
VAL_MAX_ITERS = 100
TIM_MAX_ITERS = 0

PRUNE_STEP_SIZE = 256
ADAPTIVE_THRESHOLD = 0.9

# ========================================================================================
# Projective pruning
# ========================================================================================


def self_normal_equations(
    W: torch.Tensor,
    lmbda: float,
    dtype: torch.dtype,
    device: torch.device,
):
    n = W.shape[0]
    G = W @ W.T
    I = torch.eye(n, dtype=dtype, device=device)
    Gi = torch.inverse(G + lmbda * I)
    Q = torch.zeros((n, n), dtype=dtype, device=device)
    for i in range(n):
        mask = torch.arange(n) != i
        a = Gi[mask][:, mask]
        c = Gi[i, mask]
        d = Gi[i, i]
        L = a - torch.outer(c / d, c)
        R = W[mask] @ W[i]
        Q[i, mask] = L @ R
    return Q


def proj_prune_step(
    W: torch.Tensor,
    M: torch.Tensor,
    Q: torch.Tensor,
    Ua: torch.Tensor,
    Ub: torch.Tensor,
    Ug: torch.Tensor,
    alpha: float,
    beta: float,
    gamma: float,
):
    indices = torch.nonzero(M, as_tuple=True)[0]
    dists = torch.norm(Q[M] @ W - W[M], dim=1)
    i = indices[torch.argmin(dists)]
    W[i] = 0
    M[i] = 0
    Q[:, i] = 0
    Q[M] /= 1 + alpha * Q[i]
    Ua *= 1 + alpha * Q[i]
    Ub *= 1 + beta * Q[i]
    Ug *= 1 + gamma * Q[i]
    return W, M, Q, Ua, Ub, Ug


def proj_prune(
    Ws: torch.Tensor,
    bs: torch.Tensor,
    Vs: torch.Tensor,
    alpha: float,
    beta: float,
    gamma: float,
    lmbda: float,
    amount: int,
    dtype: torch.dtype,
    device: torch.device,
):
    N = len(Ws)
    Ms, Qs, Uas, Ubs, Ugs = [], [], [], [], []
    for i in range(N):
        n = Ws[i].shape[0]
        Ms.append(torch.ones(n, dtype=torch.bool, device=device))
        Qs.append(torch.zeros((n, n), dtype=dtype, device=device))
        Uas.append(torch.ones(n, dtype=dtype, device=device))
        Ubs.append(torch.ones(n, dtype=dtype, device=device))
        Ugs.append(torch.ones(n, dtype=dtype, device=device))
    for i in range(N):
        Qs[i] = self_normal_equations(Ws[i], lmbda, dtype=dtype, device=device)

    scores = torch.ones(N)
    for a in range(amount):
        for i in range(N):
            s = torch.norm(Qs[i][Ms[i]] @ Ws[i] - Ws[i][Ms[i]], dim=1)
            s /= torch.sum(s)
            scores[i] = torch.min(s)
        li = torch.argmin(scores)
        # ri = torch.argmin(scores[li])

        r = 1 - a / amount
        Ws[li], Ms[li], Qs[li], Uas[li], Ubs[li], Ugs[li] = proj_prune_step(
            W=Ws[li],
            M=Ms[li],
            Q=Qs[li],
            Ua=Uas[li],
            Ub=Ubs[li],
            Ug=Ugs[li],
            alpha=alpha * r,
            beta=beta * r,
            gamma=gamma * r,
        )

    Ws_ = []
    bs_ = []
    Vs_ = []
    for i in range(N):
        M = Ms[i]
        Ws_.append(Ws[i][M] * Uas[i][M].unsqueeze(1))
        bs_.append(bs[i][M] * Ubs[i][M])
        Vs_.append(Vs[i][:, M] * Ubs[i][M].unsqueeze(0))

    return Ws_, bs_, Vs_


# ========================================================================================
# SVD pruning
# ========================================================================================


def svd3_reduction(A, B, k):
    # A @ B
    U0, S0, V0 = torch.linalg.svd(A, full_matrices=False)
    U1, S1, V1 = torch.linalg.svd(B, full_matrices=False)
    U2, S2, V2 = torch.linalg.svd(V0 @ U1, full_matrices=False)

    r = S2.shape[0] - k
    U2 = U2[:, :r]
    S2 = torch.diag(S2.diag()[:r])
    V2 = V2[:r]

    A = U0 @ S0 @ U2
    B = S2 @ V2 @ S1 @ V1
    return A, B


def svd3_prune(
    W: torch.Tensor,
    b: torch.Tensor,
    V: torch.Tensor,
    amount: int,
):
    W = torch.column_stack((b, W))
    V, W = svd3_reduction(V, W, amount)
    b = W[:, 0]
    W = W[:, 1:]
    return W, b, V


# ========================================================================================
# Utils
# ========================================================================================


def get_logger(filepath: pathlib.Path) -> logging.Logger:
    os.makedirs(filepath.parent, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(message)s")

    file_handler = logging.FileHandler(filepath, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.handlers.clear()
    # logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    return logger


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # if torch.backends.mps.is_available():
    #     return torch.device("mps")
    return torch.device("cpu")


# ========================================================================================
# Evaluation
# ========================================================================================


def top1_correct(output, target) -> int:
    pred = output.argmax(dim=1)
    correct = pred.eq(target).sum().item()
    return correct


def top5_correct(output, target) -> int:
    _, pred = output.topk(5, dim=1)
    correct = pred.eq(target.view(-1, 1).expand_as(pred)).sum().item()
    return correct


def evaluate(model, criterion, dataloader, device):
    if VAL_MAX_ITERS == 0:
        return 0, 0, 0

    model.eval()

    num_correct1 = 0
    num_correct5 = 0
    num_total = 0
    total_loss = 0

    for i, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        output = model(X)

        total_loss += criterion(output, y).item()
        num_correct1 += top1_correct(output, y)
        num_correct5 += top5_correct(output, y)
        num_total += X.shape[0]

        if i >= VAL_MAX_ITERS:
            break

    acc1 = num_correct1 / num_total
    acc5 = num_correct5 / num_total
    loss = total_loss

    return acc1, acc5, loss


def measure_size(model) -> tuple:
    param_size = 0
    param_num = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_num += param.nelement()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / (1024**2)
    return size_all_mb, param_num


def time_inference(model, dataloader, device) -> float:
    if TIM_MAX_ITERS == 0:
        return 0

    model.eval()

    num_total = 0
    total_time = 0

    for i, (X, _) in enumerate(dataloader):
        X = X.to(device)

        start_time = time.time()
        model(X)
        end_time = time.time()

        if i != 0:
            total_time += end_time - start_time
            num_total += X.shape[0]

        if i >= TIM_MAX_ITERS:
            break

    if num_total == 0:
        return 0

    time_avg_ms = total_time / num_total * 1000
    return time_avg_ms


def complexity(model, dataloader) -> int:
    first_batch = next(iter(dataloader))
    input_shape = tuple(first_batch[0][0].shape)
    macs, _ = ptflops.get_model_complexity_info(model, input_shape, as_strings=False, print_per_layer_stat=False, backend="aten")
    return macs


def gather_metrics(model, dataloader, criterion, device) -> dict:
    acc1, acc5, loss = evaluate(model, criterion, dataloader, device)
    model_size, param_num = measure_size(model)
    infer_time = time_inference(model, dataloader, device)
    macs = complexity(model, dataloader)

    return {
        "acc1": acc1,
        "acc5": acc5,
        "loss": loss,
        "macs": macs,
        "params": param_num,
        "model_size": model_size,
        "inference_time": infer_time,
    }


def acc1_metric_stop(orig, curr, threshold) -> bool:
    r = curr["acc1"] / orig["acc1"]
    return r < threshold


# ========================================================================================
# Models and Data
# ========================================================================================


def get_imagenet(
    path: pathlib.Path = pathlib.Path("data/imagenet"),
    train: bool = False,
    transform: transforms.functional = None,
) -> data.Dataset:
    return datasets.ImageNet(
        root=path,
        split=("train" if train else "val"),
        transform=transform,
    )


def get_imagenet_dataloader(
    batch_size: int,
    transform: transforms.functional,
    device: torch.device,
    train: bool = False,
) -> data.DataLoader:
    imagenet = get_imagenet(train=train, transform=transform)
    dataloader = data.DataLoader(
        imagenet,
        batch_size=batch_size,
        shuffle=SHUFFLE,
        generator=torch.Generator(device=device),
    )
    return dataloader


def load_model(
    path: pathlib.Path,
    model_init: models.VisionTransformer,
    weights: models.WeightsEnum,
    eval_mode: bool = True,
) -> nn.Module:
    if ALWAYS_DOWNLOAD_MODEL or not path.exists():
        model = model_init(weights=weights)
        os.makedirs(path.parent, exist_ok=True)
        torch.save(model.state_dict(), path)
    else:
        model = model_init(weights=None)
        model.load_state_dict(torch.load(path, weights_only=True))
    if eval_mode:
        model.eval()
    return model


def vit_fc_param_get(model):
    Ws = []
    bs = []
    Vs = []
    for layer in model.encoder.layers:
        Ws.append(layer.mlp[0].weight.data)
        bs.append(layer.mlp[0].bias.data)
        Vs.append(layer.mlp[3].weight.data)
    return Ws, bs, Vs


def vit_fc_param_update(model, Ws, bs, Vs):
    for li in range(len(model.encoder.layers)):
        model.encoder.layers[li].mlp[0].weight.data = Ws[li]
        model.encoder.layers[li].mlp[0].bias.data = bs[li]
        model.encoder.layers[li].mlp[3].weight.data = Vs[li]
    return model


# ========================================================================================
# Pruners
# ========================================================================================


def ln_prune_uniform(Ws, bs, Vs, amount, params):
    for li in range(len(Ws)):
        scores = torch.norm(Ws[li], p=params["norm"], dim=1)
        mask = torch.topk(scores, k=Ws[li].shape[0] - amount).indices
        Ws[li] = Ws[li][mask]
        bs[li] = bs[li][mask]
        Vs[li] = Vs[li][:, mask]
    return Ws, bs, Vs


def ln_prune_adaptive(Ws, bs, Vs, amounts, params):
    for li in range(len(Ws)):
        if amounts[li] == 0:
            continue
        scores = torch.norm(Ws[li], p=params["norm"], dim=1)
        mask = torch.topk(scores, k=Ws[li].shape[0] - amounts[li]).indices
        Ws[li] = Ws[li][mask]
        bs[li] = bs[li][mask]
        Vs[li] = Vs[li][:, mask]
    return Ws, bs, Vs


def proj_prune_uniform(Ws, bs, Vs, amount, params):
    for li in range(len(Ws)):
        Ws_, bs_, Vs_ = proj_prune(
            Ws=[Ws[li]],
            bs=[bs[li]],
            Vs=[Vs[li]],
            alpha=params["alpha"],
            beta=params["beta"],
            gamma=params["gamma"],
            lmbda=1e-3,
            amount=amount,
            dtype=Ws[li].dtype,
            device=Ws[li].device,
        )
        Ws[li], bs[li], Vs[li] = Ws_[0], bs_[0], Vs_[0]
    return Ws, bs, Vs


def proj_prune_adaptive(Ws, bs, Vs, amounts, params):
    for li in range(len(Ws)):
        if amounts[li] == 0:
            continue
        Ws_, bs_, Vs_ = proj_prune(
            Ws=[Ws[li]],
            bs=[bs[li]],
            Vs=[Vs[li]],
            alpha=params["alpha"],
            beta=params["beta"],
            gamma=params["gamma"],
            lmbda=1e-3,
            amount=amounts[li],
            dtype=Ws[li].dtype,
            device=Ws[li].device,
        )
        Ws[li], bs[li], Vs[li] = Ws_[0], bs_[0], Vs_[0]
    return Ws, bs, Vs


def proj_prune_global(Ws, bs, Vs, amount, params):
    Ws, bs, Vs = proj_prune(
        Ws=Ws,
        bs=bs,
        Vs=Vs,
        alpha=params["alpha"],
        beta=params["beta"],
        gamma=params["gamma"],
        lmbda=1e-3,
        amount=amount,
        dtype=Ws[0].dtype,
        device=Ws[0].device,
    )
    return Ws, bs, Vs


def svd3_prune_uniform(Ws, bs, Vs, amount, params):
    for i in range(len(Ws)):
        Ws[i], bs[i], Vs[i] = svd3_prune(Ws[i], bs[i], Vs[i], amount)
    return Ws, bs, Vs


def prune(get_fn, prune_fn, update_fn, model, amount, params):
    Ws, bs, Vs = get_fn(model)
    Ws, bs, Vs = prune_fn(Ws, bs, Vs, amount, params)
    model = update_fn(model, Ws, bs, Vs)
    return model


# ========================================================================================
# Preprocessing
# ========================================================================================

vit_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

vit_b_transform = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        vit_normalize,
    ]
)

vit_l_16_transform = transforms.Compose(
    [
        transforms.Resize(242, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        vit_normalize,
    ]
)

vit_l_32_transform = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        vit_normalize,
    ]
)

vit_h_transform = transforms.Compose(
    [
        transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(518),
        transforms.ToTensor(),
        vit_normalize,
    ]
)

# ========================================================================================
# Experiments
# ========================================================================================


def vit_experiment(
    output_path: pathlib.Path,
    model_path,
    model_init,
    model_weights,
    model_transform,
    get_fn,
    prune_fn,
    update_fn,
    amounts,
    **params,
):
    os.makedirs(output_path, exist_ok=True)
    logger = get_logger(output_path / "experiment.log")
    device = get_device()
    dataloader = get_imagenet_dataloader(BATCH_SIZE, model_transform, device, train=False)
    criterion = nn.CrossEntropyLoss()

    logger.info(f"Starting experiment {output_path}")

    metrics_seq = []
    for amount in amounts:
        logger.info(f"Pruning {amount}...")
        model = load_model(
            path=model_path,
            model_init=model_init,
            weights=model_weights,
        ).to(device)
        model = prune(
            get_fn=get_fn,
            prune_fn=prune_fn,
            update_fn=update_fn,
            model=model,
            amount=amount,
            params=params,
        )
        logger.info(f"Gathering metrics...")
        metrics = gather_metrics(
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            device=device,
        )
        logger.info(metrics)
        metrics_seq.append(metrics)

    df = pd.DataFrame(metrics_seq).reset_index()
    df.to_csv(output_path / "results.csv", index=False)

    logger.info("Experiment finished")


def vit_experiment_adaptive(
    output_path: pathlib.Path,
    model_path,
    model_init,
    model_weights,
    model_transform,
    get_fn,
    prune_fn,
    update_fn,
    num_layers,
    prune_step_size,
    metric_stop_fn,
    **params,
):
    os.makedirs(output_path, exist_ok=True)
    logger = get_logger(output_path / "experiment.log")
    device = get_device()
    dataloader = get_imagenet_dataloader(BATCH_SIZE, model_transform, device, train=False)
    criterion = nn.CrossEntropyLoss()
    metric_stop_fn = functools.partial(metric_stop_fn, threshold=ADAPTIVE_THRESHOLD ** (1 / num_layers))

    logger.info(f"Starting experiment {output_path}")

    logger.info(f"Loading the original model")
    model = load_model(
        path=model_path,
        model_init=model_init,
        weights=model_weights,
    ).to(device)
    logger.info(f"Gathering metrics of the original model")
    orig_metrics = gather_metrics(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        device=device,
    )
    logger.info(orig_metrics)

    amounts = [0 for _ in range(num_layers)]
    metrics_seq = [orig_metrics]

    for li in reversed(range(num_layers)):
        while True:
            prev_model = copy.deepcopy(model)

            amounts[li] += prune_step_size
            tmp_amounts = [0 for _ in range(num_layers)]
            tmp_amounts[li] = prune_step_size

            logger.info(f"Pruning {amounts[li]} in layer {li}")
            model = prune(
                get_fn=get_fn,
                prune_fn=prune_fn,
                update_fn=update_fn,
                model=model,
                amount=tmp_amounts,
                params=params,
            )
            logger.info(f"Gathering metrics")
            metrics = gather_metrics(
                model=model,
                dataloader=dataloader,
                criterion=criterion,
                device=device,
            )
            logger.info(metrics)

            if metric_stop_fn(orig_metrics, metrics) or model.encoder.layers[li].mlp[0].weight.data.shape[0] <= prune_step_size:
                model = prev_model
                amounts[li] -= prune_step_size
                orig_metrics = metrics_seq[-1]
                break

            metrics_seq.append(metrics)

    df = pd.DataFrame(metrics_seq).reset_index()
    df.to_csv(output_path / "results.csv", index=False)

    logger.info("Experiment finished")


def vit_norm_analysis(
    output_path: pathlib.Path,
    model_path,
    model_init,
    model_weights,
    num_layers,
    **params,
):
    os.makedirs(output_path, exist_ok=True)
    logger = get_logger(output_path / "analysis.log")
    device = get_device()

    logger.info(f"Starting analysis {output_path}")

    logger.info(f"Loading the original model")
    model = load_model(
        path=model_path,
        model_init=model_init,
        weights=model_weights,
    ).to(device)

    layer_data = []
    for li in range(num_layers):
        logger.info(f"Analysing layer {li}")
        W = model.encoder.layers[li].mlp[0].weight.data
        b = model.encoder.layers[li].mlp[0].bias.data
        V = model.encoder.layers[li].mlp[3].weight.data
        Q = self_normal_equations(W, 1e-3, dtype=W.dtype, device=device)
        proj_dists = torch.norm(Q @ W - W, dim=1)
        weights = {
            "W": W.norm(dim=1).to("cpu").tolist(),
            "b": b.to("cpu").tolist(),
            "V": V.norm(dim=0).to("cpu").tolist(),
            "proj_dists": proj_dists.to("cpu").tolist(),
        }
        logger.info(f"W: {W.shape}, b: {b.shape}, V: {V.shape}")
        layer_data.append(weights)

    with open(output_path / "weights.pkl", "wb") as f:
        pickle.dump(layer_data, f)

    logger.info("Experiment finished")


# ========================================================================================
# Entrypoint
# ========================================================================================


def main():
    dir = pathlib.Path("outputs")
    os.makedirs(dir, exist_ok=True)

    amounts_uniform = [round(i**1.1 * 50) for i in range(24)]
    adaptive_prune_step_size = PRUNE_STEP_SIZE
    adaptive_prune_metric_stop = functools.partial(acc1_metric_stop)

    dir = dir / sys.argv[1] / sys.argv[2] / sys.argv[3]
    match sys.argv[1]:
        case "uniform":
            experiment = functools.partial(
                vit_experiment,
                output_path=dir,
                get_fn=vit_fc_param_get,
                update_fn=vit_fc_param_update,
                amounts=amounts_uniform,
            )
            ln_pruner = ln_prune_uniform
            proj_pruner = proj_prune_uniform
        case "adaptive":
            experiment = functools.partial(
                vit_experiment_adaptive,
                output_path=dir,
                get_fn=vit_fc_param_get,
                update_fn=vit_fc_param_update,
                prune_step_size=adaptive_prune_step_size,
                metric_stop_fn=adaptive_prune_metric_stop,
            )
            ln_pruner = ln_prune_adaptive
            proj_pruner = proj_prune_adaptive
        case "norm_analysis":
            experiment = functools.partial(
                vit_norm_analysis,
                output_path=dir,
            )
            ln_pruner = None
            proj_pruner = None

    vit_b_16_experiment = functools.partial(
        experiment,
        model_path=pathlib.Path("checkpoints/vit_b_16_imagenet.pt"),
        model_init=models.vit_b_16,
        model_weights=models.ViT_B_16_Weights.IMAGENET1K_V1,
        model_transform=vit_b_transform,
        num_layers=12,
    )

    vit_b_32_experiment = functools.partial(
        experiment,
        model_path=pathlib.Path("checkpoints/vit_b_32_imagenet.pt"),
        model_init=models.vit_b_32,
        model_weights=models.ViT_B_32_Weights.IMAGENET1K_V1,
        model_transform=vit_b_transform,
        num_layers=12,
    )

    vit_l_16_experiment = functools.partial(
        experiment,
        model_path=pathlib.Path("checkpoints/vit_l_16_imagenet.pt"),
        model_init=models.vit_l_16,
        model_weights=models.ViT_L_16_Weights.IMAGENET1K_V1,
        model_transform=vit_l_16_transform,
        num_layers=24,
    )

    vit_l_32_experiment = functools.partial(
        experiment,
        model_path=pathlib.Path("checkpoints/vit_l_32_imagenet.pt"),
        model_init=models.vit_l_32,
        model_weights=models.ViT_L_32_Weights.IMAGENET1K_V1,
        model_transform=vit_l_32_transform,
        num_layers=24,
    )

    vit_h_14_experiment = functools.partial(
        experiment,
        model_path=pathlib.Path("checkpoints/vit_h_14_imagenet.pt"),
        model_init=functools.partial(models.vit_h_14, image_size=518),
        model_weights=models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1,
        model_transform=vit_h_transform,
        num_layers=32,
    )

    match sys.argv[2]:
        case "vit_b_16":
            experiment_fn = vit_b_16_experiment
        case "vit_b_32":
            experiment_fn = vit_b_32_experiment
        case "vit_l_16":
            experiment_fn = vit_l_16_experiment
        case "vit_l_32":
            experiment_fn = vit_l_32_experiment
        case "vit_h_14":
            experiment_fn = vit_h_14_experiment

    match sys.argv[3]:
        case "l1":
            experiment_fn(prune_fn=ln_pruner, norm=1)
        case "l2":
            experiment_fn(prune_fn=ln_pruner, norm=2)
        case "proj000":
            experiment_fn(prune_fn=proj_pruner, alpha=0, beta=0, gamma=0)
        case "proj550":
            experiment_fn(prune_fn=proj_pruner, alpha=0.5, beta=0.5, gamma=0)
        case "proj500":
            experiment_fn(prune_fn=proj_pruner, alpha=0.5, beta=0, gamma=0)
        case "proj5n50":
            experiment_fn(prune_fn=proj_pruner, alpha=0.5, beta=-0.5, gamma=0)


if __name__ == "__main__":
    with torch.no_grad():
        with get_device():
            main()
