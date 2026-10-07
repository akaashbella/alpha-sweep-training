"""End-to-end clean baseline training for one configured run."""

from __future__ import annotations

import logging
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torchvision
import yaml
from torch import nn
from torch.utils.data import DataLoader

from src.config import RunConfig
from src.data.cifar import build_cifar_dataloaders
from src.evaluation.evaluate import Metrics, evaluate
from src.logging.experiment_logger import write_csv, write_json
from src.models.builders import build_model
from src.training.checkpointing import load_checkpoint, save_checkpoint
from src.training.optimizers import build_optimizer, build_scheduler


def _configure_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"alpha_sweep.{log_path.parent.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Metrics:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        # This is the intentional future insertion boundary for the separately
        # tested perturbation mechanism. This stage always uses clean weights.
        logits = model(inputs)
        loss = criterion(logits, targets)
        if not math.isfinite(loss.item()):
            raise FloatingPointError(f"Non-finite training loss: {loss.item()}")
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_examples += batch_size
    if total_examples == 0:
        raise RuntimeError("Cannot train with an empty data loader")
    return Metrics(
        loss=total_loss / total_examples,
        accuracy=total_correct / total_examples,
    )


def _layer_statistics(model: nn.Module, config: RunConfig, epoch: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer_name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weight_std = module.weight.detach().std(correction=0).item()
            rows.append(
                {
                    "dataset": config.dataset.name,
                    "architecture": config.architecture,
                    "perturbation_alpha": config.perturbation_alpha,
                    "seed": config.seed,
                    "epoch": epoch,
                    "layer_name": layer_name,
                    "weight_std": weight_std,
                }
            )
    return rows


def run_clean_training(
    config: RunConfig,
    *,
    device: torch.device,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Train one non-perturbed run and evaluate its best-validation checkpoint."""
    if config.perturbation_alpha != 0.0:
        raise ValueError("Clean training requires perturbation_alpha=0.0")

    checkpoint_dir = config.paths.checkpoints / config.run_id
    result_dir = config.paths.results / config.run_id
    log_dir = config.paths.logs / config.run_id
    summary_path = result_dir / "summary.json"
    if summary_path.exists() and not overwrite:
        raise FileExistsError(
            f"Completed output already exists at {summary_path}; use --overwrite explicitly"
        )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    logger = _configure_logger(log_dir / "train.log")

    with (result_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.as_dict(), handle, sort_keys=False)

    logger.info("Starting clean run %s on %s", config.run_id, device)
    loaders = build_cifar_dataloaders(
        config.dataset,
        training_seed=config.seed,
        batch_size=config.training.batch_size,
    )
    model = build_model(config.architecture, num_classes=config.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config.training)
    scheduler = build_scheduler(optimizer, config.training)

    epoch_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    best_validation_accuracy = -math.inf
    best_validation_loss = math.inf
    best_epoch = -1
    best_path = checkpoint_dir / "best.pt"
    final_path = checkpoint_dir / "final.pt"
    started = time.monotonic()

    for epoch in range(1, config.training.epochs + 1):
        learning_rate = optimizer.param_groups[0]["lr"]
        training_metrics = _train_one_epoch(
            model, loaders.train, criterion, optimizer, device
        )
        validation_metrics = evaluate(
            model, loaders.validation, criterion, device
        )
        scheduler.step()

        epoch_rows.append(
            {
                "epoch": epoch,
                "train_loss": training_metrics.loss,
                "train_accuracy": training_metrics.accuracy,
                "validation_loss": validation_metrics.loss,
                "validation_accuracy": validation_metrics.accuracy,
                "learning_rate": learning_rate,
            }
        )
        layer_rows.extend(_layer_statistics(model, config, epoch))
        logger.info(
            "epoch=%d train_loss=%.6f train_accuracy=%.4f "
            "validation_loss=%.6f validation_accuracy=%.4f lr=%.8f",
            epoch,
            training_metrics.loss,
            training_metrics.accuracy,
            validation_metrics.loss,
            validation_metrics.accuracy,
            learning_rate,
        )

        # Strict comparison deliberately keeps the earlier epoch on an exact tie.
        if validation_metrics.accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_metrics.accuracy
            best_validation_loss = validation_metrics.loss
            best_epoch = epoch
            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                epoch=epoch,
                validation_accuracy=validation_metrics.accuracy,
                validation_loss=validation_metrics.loss,
            )

    final_row = epoch_rows[-1]
    save_checkpoint(
        final_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        epoch=config.training.epochs,
        validation_accuracy=float(final_row["validation_accuracy"]),
        validation_loss=float(final_row["validation_loss"]),
    )
    write_csv(result_dir / "epochs.csv", epoch_rows)
    write_csv(result_dir / "layer_statistics.csv", layer_rows)

    load_checkpoint(best_path, model=model, map_location=device)
    test_metrics = evaluate(model, loaders.test, criterion, device)
    elapsed_seconds = time.monotonic() - started
    summary = {
        "run_id": config.run_id,
        "run_label": config.label,
        "run_status": "completed",
        "infrastructure_smoke_test": config.label.startswith("infrastructure-smoke"),
        "dataset": config.dataset.name,
        "architecture": config.architecture,
        "perturbation_alpha": config.perturbation_alpha,
        "seed": config.seed,
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_validation_accuracy,
        "best_validation_loss": best_validation_loss,
        "test_accuracy": test_metrics.accuracy,
        "test_loss": test_metrics.loss,
        "final_epoch_training_accuracy": final_row["train_accuracy"],
        "final_epoch_training_loss": final_row["train_loss"],
        "final_epoch_validation_accuracy": final_row["validation_accuracy"],
        "final_epoch_validation_loss": final_row["validation_loss"],
        "checkpoint_best_path": str(best_path.resolve()),
        "checkpoint_final_path": str(final_path.resolve()),
        "elapsed_seconds": elapsed_seconds,
        "device": str(device),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
    }
    write_json(summary_path, summary)
    logger.info("Completed clean run; summary=%s", summary_path)
    return summary

