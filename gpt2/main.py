import pathlib
import sys
import time
from itertools import chain

import pandas as pd
import torch
from datasets import load_dataset
from gpt2 import CONFIG_CLASSES, convert_openai_statedict
from gpt2 import GPT2LMHeadModel as GPT2
from loguru import logger
from sklearn.model_selection import ParameterGrid
from torch import nn
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel as GPT2OpenAI
from transformers import GPT2Tokenizer


def get_device():
    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    logger.info(f"Using device: {device}")
    return device


def self_normal_equations(
    W: torch.Tensor,
    lmbda: float,
    dtype: torch.dtype,
    device: torch.device,
):
    logger.trace(f"Solving self-normal-equations: {W.shape}")
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


def prune_none(Q, K, topk, device):
    return torch.arange(Q.shape[0])


def prune_random(Q, K, topk, device):
    return torch.randperm(Q.shape[0])[:topk]


def prune_norm(Q, K, topk, device):
    norms = Q.norm(dim=1, p=2) + K.norm(dim=1, p=2)
    indices = torch.topk(norms, topk).indices
    return indices


def prune_projective(Q, K, topk, device):
    A = self_normal_equations(Q, 1e-3, Q.dtype, device)
    B = self_normal_equations(K, 1e-3, K.dtype, device)
    norms = A.norm(dim=1, p=2) + B.norm(dim=1, p=2)
    indices = torch.topk(norms, topk).indices
    return indices


PRUNE_FN_MAP = {
    "none": prune_none,
    "random": prune_random,
    "norm": prune_norm,
    "projective": prune_projective,
}


def prune(
    model,
    attn_prune_fn,
    attn_prune_rate,
    mlp_prune_fn,
    mlp_prune_rate,
    device,
):
    model.to(device)

    logger.info(f"Pruning attention with rate {attn_prune_rate}")
    for block in model.transformer.h:
        logger.trace(f"Pruning attention in block: {block}")

        Q_0 = block.attn.c_attn.weight[0 : model.config.n_head * model.config.qk_dim]
        q_0 = block.attn.c_attn.bias[0 : model.config.n_head * model.config.qk_dim]
        K_0 = block.attn.c_attn.weight[model.config.n_head * model.config.qk_dim : 2 * model.config.n_head * model.config.qk_dim]
        k_0 = block.attn.c_attn.bias[model.config.n_head * model.config.qk_dim : 2 * model.config.n_head * model.config.qk_dim]
        V_0 = block.attn.c_attn.weight[2 * model.config.n_head * model.config.qk_dim :]
        v_0 = block.attn.c_attn.bias[2 * model.config.n_head * model.config.qk_dim :]

        Q_1 = []
        q_1 = []
        K_1 = []
        k_1 = []

        for head_idx in range(model.config.n_head):
            i = head_idx * model.config.qk_dim
            j = (head_idx + 1) * model.config.qk_dim

            Q = Q_0[i:j]
            q = q_0[i:j]
            K = K_0[i:j]
            k = k_0[i:j]

            topk_attn = int((1 - attn_prune_rate) * Q.shape[0])
            indices = attn_prune_fn(Q, K, topk_attn, device)
            Q_ = Q[indices]
            K_ = K[indices]
            q_ = q[indices]
            k_ = k[indices]

            Q_1.append(Q_)
            q_1.append(q_)
            K_1.append(K_)
            k_1.append(k_)

        Q_1 = torch.cat(Q_1)
        q_1 = torch.cat(q_1)
        K_1 = torch.cat(K_1)
        k_1 = torch.cat(k_1)

        QKV = torch.cat((Q_1, K_1, V_0))
        qkv = torch.cat((q_1, k_1, v_0))

        block.attn.c_attn.weight = nn.Parameter(QKV)
        block.attn.c_attn.bias = nn.Parameter(qkv)

    model.config.qk_dim = int((1 - attn_prune_rate) * model.config.qk_dim)

    logger.info(f"Pruning mlp with rate {mlp_prune_rate}")
    for block in model.transformer.h:
        logger.trace("Pruning mlp in block: {block}")
        A = block.mlp.c_fc.weight
        a = block.mlp.c_fc.bias
        B = block.mlp.c_proj.weight

        topk_mlp = int((1 - mlp_prune_rate) * Q.shape[0])
        indices = mlp_prune_fn(A, B.T, topk_mlp, device)
        A_ = A[indices]
        B_ = B[:, indices]
        a_ = a[indices]

        block.mlp.c_fc.weight = nn.Parameter(A_)
        block.mlp.c_fc.bias = nn.Parameter(a_)
        block.mlp.c_proj.weight = nn.Parameter(B_)

    model.config.intermediate_dim = block.mlp.c_fc.weight.shape[0]

    return model


def get_gpt2(model_name):
    logger.info(f"Loading model: {model_name}")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model_openai = GPT2OpenAI.from_pretrained(model_name)
    config = CONFIG_CLASSES[model_name]()
    model = GPT2(config)
    sd = convert_openai_statedict(model.state_dict(), model_openai.state_dict())
    model.load_state_dict(sd)
    return model, tokenizer


def get_dataloader(path, name, batch_size, context_size, tokenizer):
    logger.info(f"Loading dataset: {path}{f'/{name}' if name else ''}")
    test = load_dataset(path, name, split="test", trust_remote_code=True)
    test = "\n\n".join(test[test.column_names[0]])
    tokens = tokenizer.encode(test, return_tensors="pt")
    batches = tokens[0].split(context_size)
    dataloader = DataLoader(batches, batch_size=batch_size, drop_last=False)
    logger.info(f"Loaded {len(batches)} batches")
    return dataloader


def evaluate(model, dataloader, device):
    logger.info("Evaluating model")
    model.eval()
    nlls = []
    for batch_idx, inputs in enumerate(dataloader):
        logger.debug(f"Evaluating batch: [{batch_idx}/{len(dataloader)}]")
        inputs = inputs.to(device)
        targets = inputs.clone()
        targets[:, : -targets.shape[1] - 1] = -100
        _, loss = model(inputs, labels=targets)
        nlls.append(loss)
    ppl = torch.exp(torch.stack(nlls).mean()).item()
    logger.info(f"Perplexity: {ppl}")
    return ppl


PRUNE_GRID = {
    "model": ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
    "attn_prune_rate": [0.1, 0.2, 0.3],
    "attn_prune": ["random", "norm", "projective"],
    "mlp_prune_rate": [0],
    "mlp_prune": ["none"],
    "dataset": [
        ("ptb_text_only", "", "ptb"),
        ("wikitext", "wikitext-2-raw-v1", "wikitext2"),
    ],
}

NOPRUNE_GRID = {
    "model": PRUNE_GRID["model"],
    "attn_prune_rate": [0],
    "attn_prune": ["none"],
    "mlp_prune_rate": [0],
    "mlp_prune": ["none"],
    "dataset": PRUNE_GRID["dataset"],
}


def main():
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    torch.manual_seed(42)
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("medium")
    torch.backends.cuda.enable_flash_sdp(False)
    batch_size = 1
    context_size = 1024
    device = get_device()

    data = []
    param_chain = chain(ParameterGrid(NOPRUNE_GRID), ParameterGrid(PRUNE_GRID))
    for params in param_chain:
        # params = {
        #     "model": "gpt2",
        #     "attn_prune_rate": 0.3,
        #     "attn_prune": "random",
        #     "mlp_prune_rate": 0.3,
        #     "mlp_prune": "random",
        #     "dataset": ("wikitext", "wikitext-2-raw-v1", "wikitext2"),
        # }
        model, tokenizer = get_gpt2(params["model"])
        model = prune(
            model,
            PRUNE_FN_MAP[params["attn_prune"]],
            params["attn_prune_rate"],
            PRUNE_FN_MAP[params["mlp_prune"]],
            params["mlp_prune_rate"],
            device,
        )
        dataloader = get_dataloader(
            path=params["dataset"][0],
            name=params["dataset"][1],
            batch_size=batch_size,
            context_size=context_size,
            tokenizer=tokenizer,
        )

        logger.info("Warmup")
        x = next(iter(dataloader)).to(device)
        model(x, labels=x)
        torch.cuda.synchronize()

        t0 = time.time()
        ppl = evaluate(model, dataloader, device)
        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0
        tps = (len(dataloader) * batch_size * context_size) / dt
        logger.info(f"Tokens per sec: {tps} tok/s")
        logger.info(f"Total evaluation time: {dt} s")

        params["tps"] = tps
        params["ppl"] = ppl
        params["dataset"] = params["dataset"][2]
        data.append(params)
        logger.success(params)

    results_dir = pathlib.Path("results")
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / sys.argv[1]
    df = pd.DataFrame(data).reset_index()
    df.to_csv(results_file, index=False)
    logger.success(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
