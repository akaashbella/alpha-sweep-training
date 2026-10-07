"""Evaluation using clean model weights and deterministic data transforms."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class Metrics:
    loss: float
    accuracy: float


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> Metrics:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_examples += batch_size
    if total_examples == 0:
        raise RuntimeError("Cannot evaluate an empty data loader")
    return Metrics(
        loss=total_loss / total_examples,
        accuracy=total_correct / total_examples,
    )

