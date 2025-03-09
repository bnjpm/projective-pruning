from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageNet


def get_cifar10(
    datapath: Path = Path("data"),
    train: bool = False,
    transform: transforms.functional = None,
):
    return CIFAR10(
        root=(datapath / "cifar10"),
        train=train,
        download=True,
        transform=transform,
    )


def get_cifar100(
    datapath: Path = Path("data"),
    train: bool = False,
    transform: transforms.functional = None,
):
    return CIFAR100(
        root=(datapath / "cifar100"),
        train=train,
        download=True,
        transform=transform,
    )


def get_imagenet(
    datapath: Path = Path("data"),
    train: bool = False,
    transform: transforms.functional = None,
):
    return ImageNet(
        root=(datapath / "imagenet"),
        split=("train" if train else "val"),
        transform=transform,
    )


def download_imagenet_features_vgg(
    preprocessor: torch.nn.Module,
    device: torch.device,
    datapath: Path = Path("data"),
    filename: str = "imagenet_features_vgg16.pt",
):
    imagenet = get_imagenet(
        datapath=datapath,
        train=False,
        transform=transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        ),
    )
    loader = DataLoader(imagenet, batch_size=64)

    features = []
    labels = []
    with torch.no_grad():
        for inputs, target in loader:
            inputs = inputs.to(device)
            outputs = preprocessor(inputs)
            outputs = outputs.flatten(1)
            features.append(outputs.cpu())
            labels.append(target)
    features = torch.cat(features)
    labels = torch.cat(labels)
    torch.save(
        {"features": features, "labels": labels},
        datapath / filename,
    )


def download_imagenet_features_alexnet(
    preprocessor: torch.nn.Module,
    device: torch.device,
    datapath: Path = Path("data"),
):
    return download_imagenet_features_vgg(
        preprocessor=preprocessor,
        device=device,
        datapath=datapath,
        filename="imagenet_features_alexnet.pt",
    )


class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, feature_file):
        data = torch.load(feature_file, weights_only=True)
        self.features = data["features"]
        self.labels = data["labels"]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
