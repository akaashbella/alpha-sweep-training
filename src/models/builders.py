"""TorchVision model builders with approved CIFAR adaptations."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet50


def build_cifar_resnet50(*, num_classes: int) -> nn.Module:
    """Build scratch ResNet50 with only the approved CIFAR stem changes."""
    model = resnet50(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(
        3,
        64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    # Match TorchVision ResNet's initialization for the newly replaced stem.
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    model.maxpool = nn.Identity()
    return model


def build_model(architecture: str, *, num_classes: int) -> nn.Module:
    """Build a registered model; only the first approved path exists so far."""
    builders = {"resnet50": build_cifar_resnet50}
    try:
        builder = builders[architecture]
    except KeyError as error:
        raise ValueError(f"Unsupported architecture: {architecture}") from error
    return builder(num_classes=num_classes)

