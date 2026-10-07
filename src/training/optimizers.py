"""Optimizer and learning-rate scheduler construction."""

from __future__ import annotations

import math

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from src.config import TrainingConfig


def build_optimizer(model: torch.nn.Module, config: TrainingConfig) -> Optimizer:
    if config.optimizer.name != "sgd":
        raise ValueError("Only the specified ResNet50 SGD recipe is implemented")
    return torch.optim.SGD(
        model.parameters(),
        lr=config.optimizer.learning_rate,
        momentum=config.optimizer.momentum,
        weight_decay=config.optimizer.weight_decay,
    )


def build_scheduler(optimizer: Optimizer, config: TrainingConfig) -> LambdaLR:
    """Create linear epoch warmup followed by cosine decay."""
    total_epochs = config.epochs
    warmup_epochs = config.warmup_epochs

    def learning_rate_factor(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        cosine_epochs = total_epochs - warmup_epochs
        if cosine_epochs <= 0:
            return 1.0
        progress = float(epoch - warmup_epochs) / float(cosine_epochs)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=learning_rate_factor)

