"""Checkpoint serialization for resumable, inspectable runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from src.config import RunConfig


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    config: RunConfig,
    epoch: int,
    validation_accuracy: float,
    validation_loss: float,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": config.dataset.name,
        "architecture": config.architecture,
        "perturbation_alpha": config.perturbation_alpha,
        "seed": config.seed,
        "epoch": epoch,
        "validation_accuracy": validation_accuracy,
        "validation_loss": validation_loss,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "run_configuration": config.as_dict(),
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return destination


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    return checkpoint

