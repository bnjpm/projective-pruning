import os
import sys

import torch
from experiments import (
    experiment_cifar10_vgg16,
    experiment_cifar100_vgg16,
    experiment_imagenet_alexnet,
    experiment_imagenet_vgg16,
)


def create_dirs():
    for d in ["logs", "data"]:
        os.makedirs(d, exist_ok=True)


@torch.no_grad()
def main():
    create_dirs()
    arg = sys.argv[1]
    print("Arg:", arg)
    match arg:
        case "cifar10_vgg16":
            experiment_cifar10_vgg16()
        case "cifar100_vgg16":
            experiment_cifar100_vgg16()
        case "imagenet_vgg16":
            experiment_imagenet_vgg16()
        case "imagenet_alexnet":
            experiment_imagenet_alexnet()


if __name__ == "__main__":
    main()
