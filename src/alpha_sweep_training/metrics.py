"""Typed computation records and finite CSV/JSONL renderers for Step 5 commits.

No file in this module is an authoritative progress counter. CSV publication is
delegated with checkpoints; train.log is an append-only diagnostic event stream.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
import csv
from datetime import datetime, timezone
import io
import json
import math
import os
from typing import Iterable

import torch
from torch import nn

from .artifacts import RunHandle, RunDirectoryError
from .component_contracts import ComponentError, finite_number, integer_count
from .data import LoaderMetadata
from .identity import canonical_alpha, identity_metadata
from .numerical import NumericalFailure, require_finite
from .serialization import strict_json_text
from .spec import EPOCHS, TRAINING_ENGINE_VERSION, TRAINING_RECORD_VERSION, VALIDATION_SIZE
from .weights import discover_eligible_weights


@dataclass(frozen=True)
class Counters:
    completed_epochs: int = 0
    attempted_batches: int = 0
    processed_examples: int = 0
    successful_updates: int = 0
    skipped_updates: int = 0

    def validate(self, *, complete_boundary: bool) -> None:
        for name, value in asdict(self).items():
            integer_count(value, f"counter {name}")
        if self.completed_epochs > EPOCHS:
            raise ComponentError("Completed epochs exceed the original budget")
        updates = self.successful_updates + self.skipped_updates
        if updates > self.attempted_batches or (complete_boundary and updates != self.attempted_batches):
            raise ComponentError("Attempted batch and successful/skipped update counters disagree")


@dataclass(frozen=True)
class ClassificationMetrics:
    loss: float
    correct: int
    examples: int
    batches: int

    def __post_init__(self) -> None:
        if finite_number(self.loss, "classification loss") < 0:
            raise ComponentError("Cross-entropy loss cannot be negative")
        integer_count(self.examples, "metric examples", minimum=1)
        integer_count(self.batches, "metric batches", minimum=1)
        integer_count(self.correct, "metric correct")
        if self.correct > self.examples:
            raise ComponentError("Correct count exceeds processed examples")

    @property
    def accuracy(self) -> float:
        return self.correct / self.examples

    def to_dict(self) -> dict:
        return {**asdict(self), "accuracy": self.accuracy,
                "accuracy_unit": "fraction", "loss_unit": "mean_cross_entropy_nats_per_example"}


class MetricAccumulator:
    def __init__(self, device: torch.device):
        self.loss_sum = torch.zeros((), dtype=torch.float64, device=device)
        self.correct = torch.zeros((), dtype=torch.int64, device=device)
        self.examples = 0
        self.batches = 0

    def add(self, loss: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor) -> None:
        size = labels.numel()
        with torch.no_grad():
            self.loss_sum.add_(loss.detach().to(dtype=torch.float64) * size)
            self.correct.add_((logits.detach().argmax(dim=1) == labels).sum())
        self.examples += size
        self.batches += 1

    def finish(self) -> ClassificationMetrics:
        require_finite((("sample_weighted_loss_sum", self.loss_sum),), "epoch_metrics")
        if not self.examples:
            raise ComponentError("Cannot report an empty epoch/evaluation")
        return ClassificationMetrics(float(self.loss_sum.item()) / self.examples,
                                     int(self.correct.item()), self.examples, self.batches)


@dataclass(frozen=True)
class BestSelection:
    epoch_index: int
    correct: int
    examples: int = VALIDATION_SIZE

    def validate(self, completed_epochs: int) -> None:
        integer_count(self.epoch_index, "best epoch")
        integer_count(self.correct, "best correct")
        if (self.epoch_index >= completed_epochs or type(self.examples) is not int
                or self.examples != VALIDATION_SIZE or self.correct > self.examples):
            raise ComponentError("Invalid best-validation selection metadata")


@dataclass(frozen=True)
class AmpSkip:
    batch_index: int
    gradient_names: tuple[str, ...]
    scale_before: float
    scale_after: float


@dataclass(frozen=True)
class LayerStatistic:
    name: str
    module_type: str
    shape: tuple[int, ...]
    numel: int
    population_std: float


def clean_layer_statistics(model: nn.Module) -> tuple[LayerStatistic, ...]:
    if getattr(model, "_alpha_sweep_perturbation_active", False):
        raise ComponentError("Statistics require original clean weights after training updates")
    entries = discover_eligible_weights(model)
    if not entries or any(entry.parameter.dtype != torch.float32 for entry in entries):
        raise ComponentError("Clean statistics require the float32 eligible inventory")
    with torch.no_grad(), torch.autocast(device_type=entries[0].parameter.device.type, enabled=False):
        values = torch.stack([torch.std(entry.parameter.detach(), correction=0) for entry in entries])
        numbers = values.cpu().tolist()
    bad = tuple(entry.name for entry, value in zip(entries, numbers) if not math.isfinite(value))
    if bad:
        raise NumericalFailure("clean_layer_statistics", "Nonfinite clean population standard deviation", bad)
    return tuple(LayerStatistic(entry.name, entry.module_type, entry.shape, entry.numel, value)
                 for entry, value in zip(entries, numbers))


@dataclass(frozen=True)
class EpochRecord:
    epoch_index: int
    completed_epochs: int
    training_kind: str
    training: ClassificationMetrics
    validation: ClassificationMetrics
    attempted_batches: int
    successful_updates: int
    skipped_updates: int
    totals: Counters
    lr_first_attempt: float
    lr_last_attempt: float
    lr_at_boundary: float
    scheduler_position_before: int
    scheduler_position_after: int
    scheduler_interval: str
    duration_seconds: float
    selected_best: bool
    best: BestSelection
    amp_skips: tuple[AmpSkip, ...]
    status: str

    def to_dict(self) -> dict:
        value = asdict(self)
        value.update(training=self.training.to_dict(),
                     validation={**self.validation.to_dict(), "clean_weights": True})
        strict_json_text(value)
        return value


@dataclass(frozen=True)
class TestRecord:
    metrics: ClassificationMetrics
    best: BestSelection
    best_checkpoint: dict
    final_commit_id: str
    duration_seconds: float
    completed_epochs: int = EPOCHS
    status: str = "evaluating"

    def to_dict(self) -> dict:
        value = {**asdict(self), "metrics": self.metrics.to_dict(), "clean_weights": True,
                 "checkpoint_role": "best", "completion_pending_publication": True}
        strict_json_text(value)
        return value


@dataclass(frozen=True)
class FailureRecord:
    status: str
    phase: str
    reason: str
    exception_type: str
    epoch_index: int | None
    batch_index: int | None
    observed_counters: Counters
    durable_completed_epochs: int
    last_commit_id: str | None
    affected_tensors: tuple[str, ...] = ()
    numerical_stage: str | None = None
    # A failed/incomplete epoch has no fabricated aggregate metrics.
    training_loss: float | None = None
    validation_loss: float | None = None
    test_loss: float | None = None
    training_accuracy: float | None = None
    validation_accuracy: float | None = None
    test_accuracy: float | None = None
    commit_outcome: str = "no_commit_in_flight"


@dataclass(frozen=True)
class RecordContext:
    identity: dict

    @classmethod
    def from_run(cls, run: RunHandle, data: LoaderMetadata) -> "RecordContext":
        run.require_active()
        science = run.config.scientific
        result = {**identity_metadata(run.config), "result_format_version": TRAINING_RECORD_VERSION,
                  "engine_version": TRAINING_ENGINE_VERSION, "dataset": science["dataset"],
                  "architecture": science["architecture"], "alpha": science["alpha"],
                  "canonical_alpha": canonical_alpha(science["alpha"]), "training_seed": science["training_seed"],
                  "split_identity": data.split_identity, "preprocessing_identity": data.preprocessing_identity,
                  "normalization_identity": data.normalization_identity, "attempt_id": run.attempt_id,
                  "provenance": deepcopy(run.provenance)}
        strict_json_text(result)
        return cls(result)

    def envelope(self, kind: str, payload: dict) -> dict:
        result = {**deepcopy(self.identity), "kind": kind, "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                  "payload": deepcopy(payload)}
        strict_json_text(result)
        return result


def _json_cell(value: object) -> str:
    strict_json_text(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _identity_columns(context: RecordContext) -> dict:
    return {key if key != "provenance" else "provenance_json":
            (_json_cell(value) if isinstance(value, (dict, list, tuple)) else value)
            for key, value in context.identity.items()}


def _csv(rows: Iterable[dict], *, include_header: bool) -> str:
    rows = list(rows)
    if not rows:
        raise ComponentError("CSV rendering requires at least one record")
    strict_json_text(rows)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    if include_header:
        writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def epoch_csv(context: RecordContext, record: EpochRecord, *, include_header: bool) -> str:
    row = {**_identity_columns(context), **record.to_dict()}
    for section in ("training", "validation", "totals"):
        value = row.pop(section)
        row.update({f"{section}_{key}": item for key, item in value.items()})
    row["best_json"] = _json_cell(row.pop("best"))
    row["amp_skips_json"] = _json_cell(row.pop("amp_skips"))
    return _csv((row,), include_header=include_header)


def layer_statistics_csv(context: RecordContext, record: EpochRecord,
                         statistics: tuple[LayerStatistic, ...], *, include_header: bool) -> str:
    rows = [{**_identity_columns(context), "epoch_index": record.epoch_index,
             "completed_epochs": record.completed_epochs, "status": record.status,
             "boundary": "after_training_updates_before_clean_validation", **asdict(statistic),
             "shape": _json_cell(statistic.shape), "std_correction": 0, "weights": "clean"}
            for statistic in statistics]
    return _csv(rows, include_header=include_header)


def append_train_log(run: RunHandle, context: RecordContext, kind: str, payload: dict) -> None:
    """Diagnostic JSONL only. Checkpoint/metric commits, not this log, own progress.

    Failed appends may leave a partial last line; Step 5 must never derive resume
    position from this file or overwrite a durable checkpoint to match the log.
    """
    run.require_active()
    if context.identity["scientific_fingerprint"] != identity_metadata(run.config)["scientific_fingerprint"]:
        raise RunDirectoryError("Log context does not belong to this guarded run")
    path = run.paths.logs / "train.log"
    if path.is_symlink() or getattr(path, "is_junction", lambda: False)() or (path.exists() and not path.is_file()):
        raise RunDirectoryError(f"Refusing a linked/non-file diagnostic log: {path}")
    line = _json_cell(context.envelope(kind, payload)) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())
