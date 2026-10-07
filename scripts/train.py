#!/usr/bin/env python3
"""Run one configured clean baseline training job."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.config import load_config, replace_storage_paths
from src.training.trainer import run_clean_training
from src.utils.reproducibility import seed_everything, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--logs-root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    config = replace_storage_paths(
        config,
        data_root=args.data_root,
        checkpoints=args.checkpoint_root,
        results=args.results_root,
        logs=args.logs_root,
    )
    seed_everything(config.seed)
    device = select_device(require_cuda=args.require_cuda)
    run_clean_training(config, device=device, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

