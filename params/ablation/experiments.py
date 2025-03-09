import logging
import os
from collections import OrderedDict
from functools import partial
from pathlib import Path

import pandas as pd
import torch
from datasets import (
    FeatureDataset,
    download_imagenet_features_alexnet,
    download_imagenet_features_vgg,
    get_cifar10,
    get_cifar100,
)
from models import (
    get_cifar10_vgg16,
    get_cifar100_vgg16,
    get_imagenet_alexnet,
    get_imagenet_vgg16,
)
from pruners import ln_prune, proj_prune, proj_prune_conv
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import get_eval


def get_logger(filepath: Path) -> logging.Logger:
    os.makedirs(filepath.parent, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(message)s")
    file_handler = logging.FileHandler(filepath, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    return logger


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    # if torch.backends.mps.is_available():
    #     return "mps"
    return "cpu"


def linear_decay_fn(x, max_x):
    return 1 - x / max_x


# =================
# Model pruning
# =================


def experiment_cifar_vgg16_ln(logger, model, eval_fn, norm, max_prune=512, step=32):
    logs = {
        "pruned": [],
        "acc1": [],
        "acc5": [],
        "loss": [],
    }
    for amount in range(0, max_prune, step):
        if amount > 0:
            ln_prune(model.classifier[0], model.classifier[3], step, norm=norm)
            ln_prune(model.classifier[3], model.classifier[6], step, norm=norm)
        acc1, acc5, loss = eval_fn(model)
        logs["pruned"].append(amount)
        logs["acc1"].append(acc1)
        logs["acc5"].append(acc5)
        logs["loss"].append(loss)
        logger.info(f"{amount / max_prune:.2f} pruned: {acc1=:.4f}, {acc5=:.4f}, {loss=:.2f}")
    return logs


def experiment_imagenet_alexnet_ln(logger, model, eval_fn, norm, max_prune=4096, step=128):
    logs = {
        "pruned": [],
        "acc1": [],
        "acc5": [],
        "loss": [],
    }
    for amount in range(0, max_prune, step):
        if amount > 0:
            ln_prune(model.classifier[1], model.classifier[4], step, norm=norm)
            ln_prune(model.classifier[4], model.classifier[6], step, norm=norm)
        acc1, acc5, loss = eval_fn(model)
        logs["pruned"].append(amount)
        logs["acc1"].append(acc1)
        logs["acc5"].append(acc5)
        logs["loss"].append(loss)
        logger.info(f"{amount / max_prune:.2f} pruned: {acc1=:.4f}, {acc5=:.4f}, {loss=:.2f}")
    return logs


def experiment_cifar_vgg16_proj(logger, model, eval_fn, params, max_prune=512, step=32, decay_fn=linear_decay_fn):
    alpha, beta, gamma = params
    logs = {
        "pruned": [],
        "acc1": [],
        "acc5": [],
        "loss": [],
    }
    for amount in range(0, max_prune, step):
        if amount > 0:
            decay_coeff = decay_fn(amount, max_prune)

            W = model.classifier[0].weight
            b = model.classifier[0].bias
            V = model.classifier[3].weight
            a = model.classifier[3].bias
            U = model.classifier[6].weight

            W, b, V = proj_prune(
                W,
                b,
                V,
                step,
                decay_coeff * alpha,
                decay_coeff * beta,
                decay_coeff * gamma,
                lmbda=1e-3,
            )
            V, a, U = proj_prune(
                V,
                a,
                U,
                step,
                decay_coeff * alpha,
                decay_coeff * beta,
                decay_coeff * gamma,
                lmbda=1e-3,
            )

            model.classifier[0].weight = nn.Parameter(W)
            model.classifier[0].bias = nn.Parameter(b)
            model.classifier[3].weight = nn.Parameter(V)
            model.classifier[3].bias = nn.Parameter(a)
            model.classifier[6].weight = nn.Parameter(U)

        acc1, acc5, loss = eval_fn(model)
        logs["pruned"].append(amount)
        logs["acc1"].append(acc1)
        logs["acc5"].append(acc5)
        logs["loss"].append(loss)
        logger.info(f"{amount / max_prune:.2f} pruned: {acc1=:.4f}, {acc5=:.4f}, {loss=:.2f}")

    return logs


def experiment_cifar_vgg16_proj_conv(
    logger,
    m,
    eval_fn,
    params,
    scheme=[0, 0.1, 0.2, 0.3, 0.4, 0.5],
    decay_fn=linear_decay_fn,
    offset=7,
):
    alpha, beta, gamma = params
    logs = {
        "pruned": [],
        "acc1": [],
        "acc5": [],
        "loss": [],
    }
    for ramount in scheme:
        if ramount > 0:
            conv_ids = [i for i in range(0, len(m.features)) if isinstance(m.features[i], torch.nn.Conv2d)]
            for ci0, ci1 in zip(conv_ids[offset:-1], conv_ids[offset + 1 :]):
                conv0 = m.features[ci0]
                bn0 = m.features[ci0 + 1]
                conv1 = m.features[ci1]

                amount = int(0.1 * conv0.out_channels)
                coef = decay_fn(ramount, 1)
                conv0, bn0, conv1 = proj_prune_conv(
                    conv0,
                    bn0,
                    conv1,
                    amount,
                    coef * alpha,
                    coef * beta,
                    coef * gamma,
                    lmbda=1e-3,
                )

                m.features[ci0] = conv0
                m.features[ci0 + 1] = bn0
                m.features[ci1] = conv1

        acc1, acc5, loss = eval_fn(m)
        logs["pruned"].append(ramount)
        logs["acc1"].append(acc1)
        logs["acc5"].append(acc5)
        logs["loss"].append(loss)
        logger.info(f"{ramount:.2f} pruned: {acc1=:.4f}, {acc5=:.4f}, {loss=:.2f}")
    return logs


# =================
# Experiments
# =================


def run_experiments(
    dir: Path,
    data: torch.utils.data.Dataset,
    get_model,
    experiment_ln,
    experiment_proj,
    experiment_proj_conv,
):
    device = get_device()
    loader = DataLoader(data, batch_size=512)
    evaluate = get_eval(nn.CrossEntropyLoss(), loader, device)

    # L2
    if experiment_proj is not None:
        name = "l2"
        logger = get_logger(dir / f"{name}.log")
        model = get_model().to(device)
        logs = experiment_ln(logger, model, evaluate, 2)
        pd.DataFrame(logs).to_csv(dir / f"{name}.csv")

    # PROJ
    if experiment_proj is not None:
        for params, label in [
            ((0, 0, 0), "000"),
            ((5, 0, 0), "500"),
            ((5, 5, 0), "550"),
            ((5, 5, 5), "555"),
        ]:
            name = f"proj_{label}"
            logger = get_logger(dir / f"{name}.log")
            model = get_model().to(device)
            logs = experiment_proj(logger, model, evaluate, params)
            pd.DataFrame(logs).to_csv(dir / f"{name}.csv")

    # PROJ CONV
    if experiment_proj_conv is not None:
        for params, label in [
            ((0, 0, 0), "000"),
            ((5, 0, 0), "500"),
            ((5, 5, 0), "550"),
            ((5, 5, 5), "555"),
        ]:
            name = f"proj_conv_{label}"
            logger = get_logger(dir / f"{name}.log")
            model = get_model().to(device)
            logs = experiment_proj_conv(logger, model, evaluate, params)
            pd.DataFrame(logs).to_csv(dir / f"{name}.csv")


def experiment_cifar10_vgg16():
    cifar10 = get_cifar10(
        datapath=Path("data"),
        train=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        ),
    )
    run_experiments(
        Path("logs/cifar10_vgg16"),
        cifar10,
        get_cifar10_vgg16,
        experiment_cifar_vgg16_ln,
        experiment_cifar_vgg16_proj,
        experiment_cifar_vgg16_proj_conv,
    )


def experiment_cifar100_vgg16():
    cifar100 = get_cifar100(
        datapath=Path("data"),
        train=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5070, 0.4865, 0.4409],
                    std=[0.2673, 0.2564, 0.2761],
                ),
            ]
        ),
    )
    run_experiments(
        Path("logs/cifar100_vgg16"),
        cifar100,
        get_cifar100_vgg16,
        experiment_cifar_vgg16_ln,
        experiment_cifar_vgg16_proj,
        experiment_cifar_vgg16_proj_conv,
    )


def experiment_imagenet_vgg16():
    device = get_device()
    model = get_imagenet_vgg16().to(device)
    preprocessor = nn.Sequential(*[model.features, model.avgpool])
    featurepath = Path("data/imagenet_features_vgg16.pt")
    if not featurepath.exists():
        download_imagenet_features_vgg(preprocessor, device, Path("data"))

    feature_dataset = FeatureDataset(featurepath)

    def get_model():
        m = get_imagenet_vgg16().to(device)
        m = nn.Sequential(OrderedDict([("classifier", m.classifier)]))
        return m

    run_experiments(
        Path("logs/imagenet_vgg16"),
        feature_dataset,
        get_model,
        partial(experiment_cifar_vgg16_ln, max_prune=4096, step=256),
        partial(experiment_cifar_vgg16_proj, max_prune=4096, step=256),
        None,
    )


def experiment_imagenet_alexnet():
    device = get_device()
    model = get_imagenet_alexnet().to(device)
    preprocessor = nn.Sequential(*[model.features, model.avgpool])
    featurepath = Path("data/imagenet_features_alexnet.pt")
    if not featurepath.exists():
        download_imagenet_features_alexnet(preprocessor, device, Path("data"))

    feature_dataset = FeatureDataset(featurepath)

    def get_model():
        m = get_imagenet_alexnet().to(device)
        m = nn.Sequential(OrderedDict([("classifier", m.classifier)]))
        return m

    run_experiments(
        Path("logs/imagenet_alexnet"),
        feature_dataset,
        get_model,
        experiment_imagenet_alexnet_ln,
        None,
    )
