"""Read-only binding of a training task to verified immutable best authority."""

from copy import deepcopy
from dataclasses import asdict, dataclass

from .artifacts import RunPaths, effective_storage_config, resolve_run_paths, _check_existing
from .config import RunConfig
from .coverage import inspect_task, _stamp, _lock_paths
from .checkpoint_io import ArtifactReference, fail, read_document
from .checkpoint_journal import read_committed
from .checkpoints import read_checkpoint
from .evaluation_protocol import EvaluationProtocol, evaluation_protocol, prescribed_slots, perturbation_seed
from .identity import identity_metadata, canonical_alpha
from .manifests import Task, task_config
from .serialization import read_json
from .spec import EPOCHS


@dataclass(frozen=True)
class TrainedCheckpoint:
    config: RunConfig
    paths: RunPaths
    protocol: EvaluationProtocol
    binding: dict
    payload: dict
    training_transaction: dict
    legacy_clean: dict | None


def bind_evaluation_checkpoint(task: Task, *, base_dir, overrides=None, metadata_only=False) -> TrainedCheckpoint:
    """No source equality with current evaluator; no runtime/model/optimizer/RNG restore.

    Restricted CPU loading validates saved provenance against itself, pinned
    checkpoint/component formats and full-budget journal/record authority. Actual
    model construction later verifies its complete native metadata and inventory.
    metadata_only is for planning: payload contains validated metadata only, never
    executable model state. Execution callers retain the full payload by default.
    """
    observation = inspect_task(task, base_dir=base_dir, overrides=overrides)
    if observation.condition != "ok" or not observation.evaluation_eligible:
        fail(f"Evaluation ineligible: {observation.condition}, status={observation.stored_status}, {observation.diagnostic}")
    config = effective_storage_config(task_config(task, overrides), base_dir)
    paths = resolve_run_paths(config, base_dir)
    before = _stamp(paths)
    if any(path.exists() for path in _lock_paths(paths)):
        fail("Training writer lock appeared; inspect after it exits")
    stored = read_json(paths.metadata / "run_metadata.json")["provenance"]
    _check_existing(paths, config, "recover", stored)
    current = read_committed(paths, config, stored, metadata_only=metadata_only)
    if current.receipt.completed_epochs != EPOCHS:
        fail("Evaluation cannot bind an unfinished training boundary")
    best = current.receipt.best
    # Planning needs the already validated best metadata, never another tensor
    # load. Execution retains the full independently validated payload contract.
    payload = {"metadata": current.best_metadata} if metadata_only else read_checkpoint(paths, best, config, stored)
    data = payload["metadata"]["data_identity"]
    if any(data[k] != v for k, v in task.prepared_data.items()):
        fail("Best checkpoint prepared identities disagree with training manifest")
    if before != _stamp(paths) or any(path.exists() for path in _lock_paths(paths)):
        fail("Training authority changed during evaluation binding; no checkpoint was pinned")
    protocol = evaluation_protocol(config)
    science = config.scientific
    binding = {"evaluation_storage_version": 1, "training_identity": identity_metadata(config),
        "dataset": science["dataset"], "architecture": science["architecture"],
        "training_alpha": science["alpha"], "canonical_training_alpha": canonical_alpha(science["alpha"]),
        "training_seed": science["training_seed"], "best_checkpoint": asdict(best),
        "best_checkpoint_identity": {"generation": payload["metadata"]["generation"], "sha256": best.sha256},
        "best_validation": deepcopy(current.document["best"]), "data_identity": deepcopy(data),
        "training_provenance": deepcopy(stored), "evaluation_protocol": protocol.to_dict(),
        "evaluation_protocol_id": protocol.to_dict()["protocol_version"], "evaluation_protocol_fingerprint": protocol.fingerprint}
    seeds = [perturbation_seed(binding["best_checkpoint_identity"], slot) for slot in prescribed_slots()
             if slot.evaluation_alpha > 0]
    if len(set(seeds)) != len(seeds):
        fail("Evaluation seed derivation collided; protocol review required, never silently reseed")
    legacy = None
    if current.status == "completed":
        refs = current.document["artifacts"]["completion"]
        legacy = {"test_reference": deepcopy(refs["test"]), "summary_reference": deepcopy(refs["summary"]),
            "test": read_document(paths, ArtifactReference.from_dict(refs["test"])),
            "summary": read_document(paths, ArtifactReference.from_dict(refs["summary"]))}
    return TrainedCheckpoint(config, paths, protocol, binding, payload, current.reference.to_dict(), legacy)
