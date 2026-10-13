"""Step 4 state ownership and the REQUIRED Step 5 persistence service contract.

This module contains records, validators and CPU snapshot copying. It contains
no checkpoint-file writer, loader, recovery implementation or no-op service.
"""

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Literal, Protocol, runtime_checkable

import torch

from .artifacts import RunHandle
from .component_contracts import ComponentError, integer_count
from .config import RunConfig
from .identity import identity_metadata
from .metrics import BestSelection, Counters, EpochRecord, FailureRecord, LayerStatistic, RecordContext
from .spec import EPOCHS


class PersistenceContractError(RuntimeError):
    """A service failed to acknowledge the required durable boundary."""


def _digest(value: str) -> None:
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise PersistenceContractError("Artifact references require actual lowercase SHA-256 digests")


@dataclass(frozen=True)
class CheckpointReference:
    role: Literal["latest", "best", "final"]
    relative_path: str  # Relative to run.paths.checkpoints; relocation-safe.
    sha256: str
    completed_epochs: int

    def validate(self, role: str, completed_epochs: int) -> None:
        integer_count(self.completed_epochs, "checkpoint completed epochs")
        if self.role != role or self.completed_epochs != completed_epochs:
            raise PersistenceContractError("Checkpoint role or completed epoch does not match the requested boundary")
        if type(self.relative_path) is not str or not self.relative_path:
            raise PersistenceContractError("Checkpoint reference needs a relative path")
        path = PurePosixPath(self.relative_path)
        if path.is_absolute() or ".." in path.parts or ":" in self.relative_path or "\\" in self.relative_path or path == PurePosixPath("."):
            raise PersistenceContractError("Checkpoint paths must remain relative to the guarded checkpoint directory")
        _digest(self.sha256)


@dataclass(frozen=True)
class CommitReceipt:
    identity: dict
    commit_id: str
    completed_epochs: int
    latest: CheckpointReference
    best: CheckpointReference | None
    final: CheckpointReference | None

    def validate(self, config: RunConfig, counters: Counters, best: BestSelection | None) -> None:
        counters.validate(complete_boundary=True)
        if (self.identity != identity_metadata(config) or type(self.commit_id) is not str or not self.commit_id
                or type(self.completed_epochs) is not int or self.completed_epochs != counters.completed_epochs):
            raise PersistenceContractError("Commit receipt does not acknowledge this run and completed boundary")
        self.latest.validate("latest", self.completed_epochs)
        if best is None:
            if self.completed_epochs or self.best is not None:
                raise PersistenceContractError("Every completed training epoch requires a persisted best selection")
        else:
            best.validate(self.completed_epochs)
            if self.best is None:
                raise PersistenceContractError("Commit did not persist the selected best model")
            self.best.validate("best", best.epoch_index + 1)
        if self.completed_epochs == EPOCHS:
            if self.final is None:
                raise PersistenceContractError("Epoch 200 requires a final clean checkpoint")
            self.final.validate("final", EPOCHS)
        elif self.final is not None:
            raise PersistenceContractError("Final means the complete original training budget")


@dataclass(frozen=True)
class ContinuationState:
    """Step 5 supplies this AFTER validating and restoring every live component."""
    counters: Counters
    best: BestSelection | None
    receipt: CommitReceipt
    data_continuation: dict


def owned_cpu_snapshot(value: Any) -> Any:
    """Copy all tensor storage now; state_dict alone aliases live training state."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        result = OrderedDict() if isinstance(value, OrderedDict) else {}
        result.update((key, owned_cpu_snapshot(child)) for key, child in value.items())
        if hasattr(value, "_metadata"):
            result._metadata = deepcopy(value._metadata)
        return result
    if isinstance(value, tuple):
        return tuple(owned_cpu_snapshot(child) for child in value)
    if isinstance(value, list):
        return [owned_cpu_snapshot(child) for child in value]
    return deepcopy(value)


@dataclass(frozen=True)
class TrainingSnapshot:
    config: dict
    provenance: dict
    clean_model: dict
    optimizer: dict
    scheduler: dict
    scaler: dict | None
    rng: dict
    data_continuation: dict
    counters: Counters
    best: BestSelection | None
    component_metadata: dict
    format_version: int = 1
    clean_weights: bool = True


@dataclass(frozen=True)
class EpochBoundary:
    context: RecordContext
    record: EpochRecord
    statistics: tuple[LayerStatistic, ...]
    snapshot: TrainingSnapshot
    previous_commit_id: str | None

    @property
    def save_best(self) -> bool:
        return self.record.selected_best

    @property
    def save_final(self) -> bool:
        return self.record.completed_epochs == EPOCHS


@dataclass(frozen=True)
class InitialBoundary:
    context: RecordContext
    snapshot: TrainingSnapshot


@dataclass(frozen=True)
class BestLoadReceipt:
    identity: dict
    checkpoint: CheckpointReference


@dataclass(frozen=True)
class CompletionReceipt:
    """Legacy v1 combined training + clean test receipt; never robustness proof."""
    identity: dict
    final_commit_id: str
    summary_sha256: str
    status: Literal["completed"] = "completed"

    def validate(self, config: RunConfig, final_commit: CommitReceipt) -> None:
        if (self.identity != identity_metadata(config) or self.final_commit_id != final_commit.commit_id
                or self.status != "completed" or final_commit.completed_epochs != EPOCHS):
            raise PersistenceContractError("Completion must acknowledge the final committed run and its clean best-checkpoint test")
        _digest(self.summary_sha256)


@dataclass(frozen=True)
class TrainingCompletionReceipt:
    identity: dict
    final_commit_id: str
    summary_sha256: str
    status: Literal["training_completed"] = "training_completed"

    def validate(self, config: RunConfig, final_commit: CommitReceipt) -> None:
        if (self.identity != identity_metadata(config) or self.final_commit_id != final_commit.commit_id
                or self.status != "training_completed" or final_commit.completed_epochs != EPOCHS
                or final_commit.best is None or final_commit.final is None):
            raise PersistenceContractError("Training completion requires the full budget and best/final/latest authority")
        _digest(self.summary_sha256)


@runtime_checkable
class TrainingPersistence(Protocol):
    """Required service; every method holds run.require_active() throughout use.

    A return receipt means DURABLE success, never queued work. A service may
    transfer the owned snapshot to a writer, but must wait for its transaction
    to commit before returning. Exceptions stop the trainer; Step 5 reconciles
    unknown commit outcomes. All paths come from run.paths. No log controls resume.
    """

    def commit_initial(self, run: RunHandle, boundary: InitialBoundary) -> CommitReceipt:
        """Durable fresh clean epoch-zero state; no records, best or final yet."""
        ...

    def commit_epoch(self, run: RunHandle, boundary: EpochBoundary) -> CommitReceipt:
        """Commit latest + metrics/statistics/selection together; best on strict
        improvement, final at epoch 200. Preserve previous best otherwise.
        Snapshot tensors already own CPU storage and contain no temporary noise.
        Publish derived results/epochs.csv and results/layer_statistics.csv from
        committed records using metrics.py renderers, with no duplicate rows on
        recovery. At epoch 200 acknowledge training_pending_completion; the
        training-only completion transaction follows independently of test.
        """
        ...

    def publish_training_completion(self, run: RunHandle, context: RecordContext,
                                    final_commit: CommitReceipt) -> TrainingCompletionReceipt:
        """Commit full-budget training/checkpoint/metric authority, without test.
        Evaluation has an independent namespace and completion authority.
        """
        ...

    def publish_failure(self, run: RunHandle, context: RecordContext, failure: FailureRecord) -> None:
        """Durably record failure/interruption and last ACKNOWLEDGED boundary.
        Preserve existing checkpoints; do not save the failed live state as latest
        or destroy evidence of an epoch/completion transaction with unknown outcome.
        Reconcile an unknown completion before changing its status: a committed
        completion must not be downgraded by a later diagnostic/publication error.
        """
        ...


def require_persistence(service: TrainingPersistence) -> None:
    if not isinstance(service, TrainingPersistence):
        raise ComponentError("Training requires commit_initial, commit_epoch, publish_training_completion and publish_failure")
