"""Reproducibility helpers for clean and future perturbation runs."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch CPU/CUDA random generators."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(*, require_cuda: bool) -> torch.device:
    """Select CUDA when available and refuse an expected-GPU CPU fallback."""
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was required, but PyTorch cannot access a CUDA device. "
            "Check the Slurm GPU allocation and CUDA-enabled PyTorch installation."
        )
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

