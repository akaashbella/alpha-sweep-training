#!/usr/bin/env python3
"""Print Python, PyTorch, CUDA, host, and Slurm diagnostics."""

from __future__ import annotations

import argparse
import os
import platform
import socket
import sys
from pathlib import Path

import torch

SLURM_VARIABLES = (
    "SLURM_JOB_ID",
    "SLURM_JOB_NAME",
    "SLURM_JOB_ACCOUNT",
    "SLURM_JOB_PARTITION",
    "SLURM_JOB_NODELIST",
    "SLURM_CPUS_PER_TASK",
    "SLURM_MEM_PER_NODE",
    "SLURM_JOB_GPUS",
    "SLURM_GPUS",
    "SLURM_GPUS_ON_NODE",
    "CUDA_VISIBLE_DEVICES",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="exit nonzero instead of accepting a CPU-only environment",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"hostname: {socket.gethostname()}")
    print(f"working_directory: {Path.cwd()}")
    print(f"platform: {platform.platform()}")
    print(f"python_executable: {sys.executable}")
    print(f"python_version: {platform.python_version()}")
    print(f"pytorch_version: {torch.__version__}")
    print(f"pytorch_cuda_version: {torch.version.cuda}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    print(f"cuda_device_count: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        print(f"cuda_current_device: {current_device}")
        for device_index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(device_index)
            print(f"cuda_device_{device_index}_name: {properties.name}")
            print(
                f"cuda_device_{device_index}_memory_bytes: "
                f"{properties.total_memory}"
            )
    else:
        print("cuda_current_device: unavailable")

    print("slurm_environment:")
    for variable in SLURM_VARIABLES:
        print(f"  {variable}={os.environ.get(variable, '<unset>')}")

    gpu_expected = args.require_gpu or any(
        os.environ.get(variable)
        for variable in ("SLURM_JOB_GPUS", "SLURM_GPUS", "SLURM_GPUS_ON_NODE")
    )
    if gpu_expected and not torch.cuda.is_available():
        print(
            "ERROR: A GPU was requested/required, but PyTorch cannot access CUDA. "
            "Do not continue with training.",
            file=sys.stderr,
        )
        return 2

    print("environment_check: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

