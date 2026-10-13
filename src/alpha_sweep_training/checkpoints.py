"""Restricted versioned clean checkpoints; no unrestricted pickle fallback."""

from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import math
import os
import random
import uuid

import numpy as np
import torch
from torch import nn

from .artifacts import RunHandle, RunPaths
from .checkpoint_io import (
    CHECKPOINT_FORMAT_VERSION, ArtifactReference, exact, fail, fsync_directory,
    relative_file, same,
)
from .config import RunConfig
from .component_contracts import require_runtime_versions
from .data_contract import CONTINUATION_VERSION, NORMALIZATION_VERSION, PREPARATION_VERSION, SOURCE_VERSION, SPLIT_VERSION
from .identity import identity_metadata
from .metrics import BestSelection, Counters
from .models import describe_model
from .numerical import require_finite
from .provenance import software_provenance
from .spec import (
    DATA_RANDOMNESS_VERSION, ELIGIBLE_WEIGHT_VERSION, EPOCHS, MODEL_BUILDER_VERSION,
    MODEL_INITIALIZATION_VERSION, OPTIMIZER_BUILDER_VERSION, PREPROCESSING_VERSION,
    RUNTIME_RANDOMNESS_VERSION, SCHEDULER_VERSION, TRAINING_ENGINE_VERSION, BATCH_SIZE, TRAIN_SIZE, VALIDATION_SIZE, TEST_SIZE,
)
from .training_contracts import CheckpointReference, TrainingSnapshot

COMPONENT_VERSIONS = {
    "model": MODEL_BUILDER_VERSION, "initialization": MODEL_INITIALIZATION_VERSION,
    "weights": ELIGIBLE_WEIGHT_VERSION, "optimizer": OPTIMIZER_BUILDER_VERSION,
    "scheduler": SCHEDULER_VERSION, "training": TRAINING_ENGINE_VERSION,
    "runtime_rng": RUNTIME_RANDOMNESS_VERSION, "data_rng": DATA_RANDOMNESS_VERSION,
    "preprocessing": PREPROCESSING_VERSION,
    "source": SOURCE_VERSION, "split": SPLIT_VERSION, "normalization": NORMALIZATION_VERSION,
    "preparation": PREPARATION_VERSION, "data_continuation": CONTINUATION_VERSION,
}
_DATA_FIELDS = ("dataset", "num_classes", "training_seed", "split_identity", "normalization_identity",
                "preprocessing_identity", "preparation_identity", "official_training_identity",
                "official_test_identity", "randomness_version", "counts", "drop_last", "data_rng_software")


def data_identity(metadata: dict) -> dict:
    return {key: deepcopy(metadata[key]) for key in _DATA_FIELDS}


def compatible_provenance(saved: dict, current: dict) -> None:
    for value in (saved, current):
        software_provenance(code=value["code"], software=value["software"], allocation=value.get("allocation"))
    same(saved["code"]["source_sha256"], current["code"]["source_sha256"], "source snapshot")
    same(saved["software"], current["software"], "software provenance")


def restricted_tree(value: object) -> object:
    """Only tensors and builtin scalar/containers accepted by weights_only."""
    if isinstance(value, torch.Tensor):
        if type(value) is not torch.Tensor or value.device.type != "cpu" or value.requires_grad or value.grad_fn is not None:
            fail("Checkpoint tensors must be owned detached CPU tensors, never Parameters/graphs")
        return value
    if type(value) in (str, int, bool, type(None)):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            fail("Checkpoint scalar is nonfinite")
        return value
    if isinstance(value, dict):
        if any(type(key) not in (str, int) for key in value):
            fail("Checkpoint dictionary keys must be builtin strings/integers")
        return {key: restricted_tree(child) for key, child in value.items()}
    if type(value) in (tuple, list):
        result = [restricted_tree(child) for child in value]
        return tuple(result) if type(value) is tuple else result
    fail(f"Unsupported checkpoint value type: {type(value).__name__}")


def encode_rng(state: dict) -> dict:
    result = deepcopy(state)
    name, keys, position, has_gauss, cached = state["numpy"]
    if name != "MT19937" or keys.dtype != np.uint32 or keys.shape != (624,):
        fail("Expected explicit NumPy MT19937 state")
    result["numpy"] = {"algorithm": name, "keys": keys.tolist(), "position": int(position),
                       "has_gauss": int(has_gauss), "cached_gaussian": float(cached)}
    return restricted_tree(result)


def decode_rng(state: dict) -> dict:
    exact(state, {"descriptor", "assigned_device", "python", "numpy", "torch_cpu", "torch_cuda", "weight_noise"}, "RNG state")
    value = exact(state["numpy"], {"algorithm", "keys", "position", "has_gauss", "cached_gaussian"}, "NumPy RNG")
    if (value["algorithm"] != "MT19937" or type(value["keys"]) is not list or len(value["keys"]) != 624
            or any(type(key) is not int or not 0 <= key < 2**32 for key in value["keys"])
            or type(value["position"]) is not int or not 0 <= value["position"] <= 624
            or type(value["has_gauss"]) is not int or value["has_gauss"] not in (0, 1)
            or type(value["cached_gaussian"]) is not float or not math.isfinite(value["cached_gaussian"])):
        fail("Invalid explicit NumPy RNG state")
    result = dict(state)
    result["numpy"] = ("MT19937", np.array(value["keys"], dtype=np.uint32), value["position"],
                       value["has_gauss"], value["cached_gaussian"])
    try:
        random.Random().setstate(result["python"])
        np.random.RandomState().set_state(result["numpy"])
        if torch.device(state["assigned_device"]).type != "cuda":
            fail("Saved RNG device is not CUDA")
        for key in ("torch_cpu", "torch_cuda", "weight_noise"):
            tensor = state[key]
            if type(tensor) is not torch.Tensor or tensor.dtype != torch.uint8 or tensor.ndim != 1 or not tensor.numel():
                fail(f"Invalid {key} RNG vector")
        torch.Generator(device="cpu").set_state(state["torch_cpu"])
    except (ValueError, TypeError, RuntimeError) as exc:
        fail(f"Invalid saved RNG state: {exc}")
    return result


def model_inventory(state: dict, parameter_names: list[str]) -> list[dict]:
    parameters = set(parameter_names)
    if not parameters or len(parameters) != len(parameter_names) or not parameters <= state.keys():
        fail("Model/optimizer parameter inventories disagree")
    return [{"name": name, "kind": "parameter" if name in parameters else "buffer",
             "shape": list(value.shape), "dtype": str(value.dtype), "numel": value.numel()}
            for name, value in sorted(state.items())]


def make_checkpoint(snapshot: TrainingSnapshot, run: RunHandle, role: str, generation: str,
                    artifact_state: dict) -> dict:
    run.require_active()
    if not snapshot.clean_weights or snapshot.format_version != 1:
        fail("Only clean owned Step 4 snapshots can be committed")
    config = RunConfig(snapshot.config)
    run.require_active(config)
    same(snapshot.provenance, run.provenance, "snapshot attempt provenance")
    model = dict(snapshot.clean_model)
    inventory = model_inventory(model, snapshot.component_metadata["optimizer"]["parameter_names"])
    metadata = {"format_version": CHECKPOINT_FORMAT_VERSION, "component_versions": COMPONENT_VERSIONS,
                "identity": identity_metadata(config), "config": config.to_dict(), "role": role,
                "generation": generation, "completed_epochs": snapshot.counters.completed_epochs,
                "original_epochs": EPOCHS, "next_epoch_index": snapshot.counters.completed_epochs,
                "best": None if snapshot.best is None else asdict(snapshot.best),
                "counters": asdict(snapshot.counters), "inventory": inventory, "clean_weights": True,
                "data_identity": data_identity(snapshot.component_metadata["data"]),
                "component_metadata": deepcopy(snapshot.component_metadata), "provenance": deepcopy(snapshot.provenance),
                "created_at_utc": datetime.now(timezone.utc).isoformat(), "attempt_id": run.attempt_id,
                "state_metadata": deepcopy(getattr(snapshot.clean_model, "_metadata", {}))}
    state = None
    if role == "latest":
        state = {"optimizer": snapshot.optimizer, "scheduler": snapshot.scheduler, "scaler": snapshot.scaler,
                 "rng": encode_rng(snapshot.rng), "data_continuation": snapshot.data_continuation,
                 "artifacts": deepcopy(artifact_state)}
    return restricted_tree({"metadata": metadata, "clean_model": model, "continuation": state})


def validate_checkpoint(payload: dict, config: RunConfig, provenance: dict, *, role: str,
                        completed_epochs: int) -> tuple[Counters, BestSelection | None]:
    require_runtime_versions(str(torch.__version__))
    exact(payload, {"metadata", "clean_model", "continuation"}, "checkpoint")
    restricted_tree(payload)
    meta = exact(payload["metadata"], {
        "format_version", "component_versions", "identity", "config", "role", "generation", "completed_epochs",
        "original_epochs", "next_epoch_index", "best", "counters", "inventory", "clean_weights", "data_identity",
        "component_metadata", "provenance", "created_at_utc", "attempt_id", "state_metadata"}, "checkpoint metadata")
    same(meta["format_version"], CHECKPOINT_FORMAT_VERSION, "checkpoint format")
    same(meta["component_versions"], COMPONENT_VERSIONS, "checkpoint component versions")
    same(meta["identity"], identity_metadata(config), "checkpoint identity")
    saved = RunConfig(meta["config"])
    same(saved.scientific, config.scientific, "checkpoint scientific configuration")
    compatible_provenance(meta["provenance"], provenance)
    same(meta["role"], role, "checkpoint role")
    same(meta["completed_epochs"], completed_epochs, "checkpoint boundary")
    same(meta["original_epochs"], EPOCHS, "original training budget")
    same(meta["next_epoch_index"], completed_epochs, "next epoch")
    same(meta["clean_weights"], True, "clean weights marker")
    if (type(meta["generation"]) is not str or len(meta["generation"]) != 32
            or any(c not in "0123456789abcdef" for c in meta["generation"])
            or type(meta["attempt_id"]) is not str or not meta["attempt_id"]
            or type(meta["created_at_utc"]) is not str or not meta["created_at_utc"]):
        fail("Invalid checkpoint generation/creation/attempt metadata")
    counters = Counters(**exact(meta["counters"], set(asdict(Counters())), "counters"))
    counters.validate(complete_boundary=True)
    same(counters.completed_epochs, completed_epochs, "counter epoch")
    best = None if meta["best"] is None else BestSelection(**exact(meta["best"], {"epoch_index", "correct", "examples"}, "best selection"))
    if best is None and completed_epochs or best is not None and not completed_epochs:
        fail("Best selection is required exactly after completed training epochs")
    if best is not None:
        best.validate(completed_epochs)
    if role == "best" and (best is None or best.epoch_index + 1 != completed_epochs):
        fail("Best checkpoint is not its own selected epoch")
    if role == "final" and completed_epochs != EPOCHS:
        fail("Final checkpoint requires the original 200 epochs")
    if role not in ("latest", "best", "final"):
        fail("Unknown checkpoint role")
    components = exact(meta["component_metadata"], {"model", "optimizer", "runtime", "randomness", "data", "scheduler"}, "component metadata")
    same(meta["data_identity"], data_identity(components["data"]), "checkpoint data identities")
    science = config.scientific
    data = meta["data_identity"]
    from .training_contracts import _digest
    for key in ("split_identity", "normalization_identity", "preprocessing_identity", "preparation_identity",
                "official_training_identity", "official_test_identity"):
        if type(data[key]) is not str or not data[key].startswith("sha256:"):
            fail(f"Missing data content identity: {key}")
        _digest(data[key][7:])
    for key, expected in (("dataset", science["dataset"]), ("architecture", science["architecture"]),
                          ("num_classes", science["model"]["num_classes"]), ("training_seed", science["training_seed"])):
        same(components["model"][key], expected, f"model {key}")
    for key, expected in (("dataset", science["dataset"]), ("num_classes", science["model"]["num_classes"]),
                          ("training_seed", science["training_seed"]), ("split_identity", science["data"]["split"]["identity"]),
                          ("drop_last", science["training"]["drop_last"]), ("randomness_version", DATA_RANDOMNESS_VERSION)):
        same(data[key], expected, f"data {key}")
    same(f"normalization-v1:{data['normalization_identity']}", science["data"]["normalization"]["source"], "normalization identity")
    counts = data["counts"]
    batches = TRAIN_SIZE // BATCH_SIZE if data["drop_last"] else (TRAIN_SIZE + BATCH_SIZE - 1) // BATCH_SIZE
    expected_counts = {"official_training": TRAIN_SIZE + VALIDATION_SIZE, "training_partition": TRAIN_SIZE, "validation_partition": VALIDATION_SIZE,
                       "test_partition": TEST_SIZE, "training_processed_per_epoch": batches * BATCH_SIZE if data["drop_last"] else TRAIN_SIZE,
                       "training_batches": batches, "validation_batches": (VALIDATION_SIZE + BATCH_SIZE - 1) // BATCH_SIZE,
                       "test_batches": (TEST_SIZE + BATCH_SIZE - 1) // BATCH_SIZE, "batch_size": BATCH_SIZE}
    same(counts, expected_counts, "complete CIFAR counts")
    same(counters.attempted_batches, completed_epochs * batches, "attempt counter")
    same(counters.processed_examples, completed_epochs * counts["training_processed_per_epoch"], "processed counter")
    model = payload["clean_model"]
    if type(model) is not dict or not model:
        fail("Missing clean model tensors")
    names = components["optimizer"]["parameter_names"]
    same(model_inventory(model, names), meta["inventory"], "parameter/buffer inventory")
    elements = sum(entry["numel"] for entry in meta["inventory"] if entry["kind"] == "parameter")
    same(components["model"]["total_parameters"], elements, "model parameter elements")
    same(components["model"]["trainable_parameters"], elements, "model trainable elements")
    for key, expected in (("builder_version", MODEL_BUILDER_VERSION), ("initialization_version", MODEL_INITIALIZATION_VERSION),
                          ("eligible_weight_version", ELIGIBLE_WEIGHT_VERSION)):
        same(components["model"][key], expected, f"model component {key}")
    same(components["optimizer"]["builder_version"], OPTIMIZER_BUILDER_VERSION, "optimizer component version")
    for entry in meta["inventory"]:
        value = model[entry["name"]]
        if value.layout != torch.strided or (entry["kind"] == "parameter" and value.dtype != torch.float32):
            fail("Unsupported clean tensor layout/dtype")
        if value.is_floating_point() and value.dtype != torch.float32:
            fail("Floating clean parameters and buffers must be float32")
    require_finite(model.items(), "checkpoint_clean_state")
    if role == "latest":
        state = exact(payload["continuation"], {"optimizer", "scheduler", "scaler", "rng", "data_continuation", "artifacts"}, "latest state")
        exact(state["artifacts"], {"records", "previous_commit_id"}, "latest artifact references")
        decode_rng(state["rng"])
        same(state["data_continuation"]["next_epoch_index"], completed_epochs, "data continuation epoch")
        from .checkpoint_validation import validate_latest_components
        validate_latest_components(payload, config, counters)
    elif payload["continuation"] is not None:
        fail("Best/final cannot contain latest continuation state")
    return counters, best


def write_checkpoint(run: RunHandle, snapshot: TrainingSnapshot, role: str, generation: str,
                     artifact_state: dict) -> CheckpointReference:
    run.require_active()
    if run.operation not in ("new", "continue", "overwrite"):
        fail("Publication-only finalization cannot write latest/best/final tensors")
    payload = make_checkpoint(snapshot, run, role, generation, artifact_state)
    validate_checkpoint(payload, run.config, run.provenance, role=role, completed_epochs=snapshot.counters.completed_epochs)
    relative = f"generations/{generation}.{role}.pt"
    target = relative_file(run.paths.checkpoints, relative, create_parents=True)
    if target.exists():
        fail("Immutable checkpoint generation already exists")
    temporary = target.with_name(target.name + f".part-{uuid.uuid4().hex}")
    # Leave failed staging evidence. Only proven superseded commits may be pruned.
    with temporary.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    fsync_directory(temporary.parent)
    staged = CheckpointReference(role, temporary.relative_to(run.paths.checkpoints).as_posix(),
                                _file_digest(temporary), snapshot.counters.completed_epochs)
    read_checkpoint(run.paths, staged, run.config, run.provenance)
    run.require_active()
    os.replace(temporary, target)
    fsync_directory(target.parent)
    return CheckpointReference(role, relative, staged.sha256, staged.completed_epochs)


def _file_digest(path) -> str:
    from .checkpoint_io import digest_file
    return digest_file(path)


def read_checkpoint(paths: RunPaths, reference: CheckpointReference, config: RunConfig, provenance: dict) -> dict:
    reference.validate(reference.role, reference.completed_epochs)
    path = relative_file(paths.checkpoints, reference.relative_path)
    if not path.is_file():
        fail(f"Missing/non-file checkpoint: {path}")
    # Hash and restricted-load the same open file, excluding a path swap between
    # the digest check and load. Cooperative run locks remain the writer boundary.
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        if digest.hexdigest() != reference.sha256:
            fail(f"Checkpoint content digest mismatch: {path}")
        stream.seek(0)
        try:
            payload = torch.load(stream, map_location="cpu", weights_only=True)
        except Exception as exc:
            fail(f"Restricted checkpoint loading failed: {exc}")
    validate_checkpoint(payload, config, provenance, role=reference.role, completed_epochs=reference.completed_epochs)
    return payload


def copy_clean_model(model: nn.Module, payload: dict) -> None:
    """Full read-only compatibility checks precede copying into existing bindings."""
    meta = payload["metadata"]
    same(describe_model(model).to_dict(), meta["component_metadata"]["model"], "constructed model metadata")
    current = model.state_dict()
    same(getattr(current, "_metadata", {}), meta["state_metadata"], "actual state component versions")
    names = meta["component_metadata"]["optimizer"]["parameter_names"]
    same(model_inventory(current, names), meta["inventory"], "actual model inventory")
    parameters, buffers = dict(model.named_parameters()), dict(model.named_buffers())
    state = OrderedDict(payload["clean_model"])
    state._metadata = deepcopy(meta["state_metadata"])
    model.load_state_dict(state, strict=True, assign=False)
    if any(dict(model.named_parameters())[name] is not value for name, value in parameters.items()) or any(
            dict(model.named_buffers())[name] is not value for name, value in buffers.items()):
        fail("Clean loading replaced original parameter/buffer bindings")
