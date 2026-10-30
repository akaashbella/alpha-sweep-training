"""Historical read-only run evidence; never continuation or publication authority."""

from dataclasses import asdict
import hashlib
import os

from .artifacts import _check_existing, _is_link, effective_storage_config, resolve_run_paths
from .coverage import _lock_paths
from .data_contract import file_digest
from .diagnostic_protocol import DiagnosticError, CORE_FILES
from .identity import identity_metadata
from .planning_inspection import PlanningInspection, training_paths
from .serialization import read_json, strict_json_text


def state_digest(state):
    import torch
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        owned = tensor.detach().to("cpu", copy=True).contiguous()
        digest.update(strict_json_text({"name": name, "shape": list(owned.shape), "dtype": str(owned.dtype)}).encode())
        digest.update(owned.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def audit_run(config, *, base_dir, source_root, initial_reference=None):
    """CPU restricted-load inspection of original evidence, with content guards.

    Source/software comparisons use actual current provenance separately from
    historical validation. A different source never becomes continuation permission.
    """
    from .checkpoint_journal import read_committed
    from .checkpoint_io import ArtifactReference, read_document
    from .checkpoints import read_checkpoint
    from .training_contracts import CheckpointReference
    from .execution_provenance import collect_provenance, source_inventory
    from .diagnostic_observer import tensor_summary
    from .metrics import Counters
    from .config import load_config
    config = effective_storage_config(config, base_dir)
    config.require_research_ready()
    if config.scientific["dataset"] != "cifar10":
        raise DiagnosticError("This audit never starts or diagnoses CIFAR-100")
    paths = resolve_run_paths(config, base_dir)
    inspection = PlanningInspection()

    def inspect():
        if any(p.exists() or _is_link(p) for p in _lock_paths(paths)):
            raise DiagnosticError("A run writer owns this evidence; inspect after it exits, never release its lock here")
        metadata = read_json(paths.metadata / "run_metadata.json")
        saved = metadata["provenance"]
        _check_existing(paths, config, "recover", saved)
        stored_config = load_config(paths.metadata / "resolved_config.yaml")
        stored_effective = effective_storage_config(stored_config, base_dir)
        if strict_json_text(stored_effective.to_dict()) != strict_json_text(config.to_dict()):
            raise DiagnosticError("Use the actual archived resolved config, including its execution/data bindings; do not substitute current declarations")
        current = read_committed(paths, config, saved)
        continuation = current.latest["continuation"]
        names = current.latest["metadata"]["component_metadata"]["optimizer"]["parameter_names"]
        groups = continuation["optimizer"]["param_groups"]
        ids = [identifier for group in groups for identifier in group["params"]]
        if len(ids) != len(names):
            raise DiagnosticError("Saved native optimizer IDs do not match the validated owned parameter inventory")
        states = continuation["optimizer"]["state"]
        native = {name: {key: tensor_summary(value) for key, value in states.get(identifier, {}).items()}
                  for name, identifier in zip(names, ids)}
        epochs = [{"reference": ref, "document": read_document(paths, ArtifactReference.from_dict(ref))}
                  for ref in current.document["records"]]
        export = {paths.metadata / "resolved_config.yaml", paths.metadata / "run_metadata.json",
                  paths.metadata / "state" / "HEAD.json", current.reference.path(paths)}
        if current.head["previous"] is not None:
            previous = ArtifactReference.from_dict(current.head["previous"])
            export.add(previous.path(paths))
            parent = read_document(paths, previous)
            for ref in parent["records"]:
                export.add(ArtifactReference.from_dict(ref).path(paths))
            for role in ("latest", "best", "final"):
                checkpoint = parent["receipt"][role]
                if checkpoint is not None:
                    export.add(paths.checkpoints / checkpoint["relative_path"])
            for role, refs in parent["artifacts"].items():
                if refs is None:
                    continue
                for ref in (refs.values() if role == "completion" else (refs,)):
                    reference = ArtifactReference.from_dict(ref)
                    document = read_document(paths, reference)
                    export.add(reference.path(paths))
                    for metric in document.get("metric_artifacts", {}).values():
                        export.add(ArtifactReference.from_dict(metric).verify(paths))
        for epoch in epochs:
            export.add(ArtifactReference.from_dict(epoch["reference"]).path(paths))
        # Include completion/failure documents and immutable metric views, not aliases.
        for role, refs in current.document["artifacts"].items():
            if refs is None:
                continue
            for ref in (refs.values() if role == "completion" else (refs,)):
                reference = ArtifactReference.from_dict(ref)
                document = read_document(paths, reference)
                export.add(reference.path(paths))
                for metric in document.get("metric_artifacts", {}).values():
                    export.add(ArtifactReference.from_dict(metric).verify(paths))
        checkpoints = {role: None if getattr(current.receipt, role) is None else asdict(getattr(current.receipt, role))
                       for role in ("latest", "best", "final")}
        for reference in checkpoints.values():
            if reference is not None:
                export.add(paths.checkpoints / reference["relative_path"])
        initial = None
        if initial_reference is not None:
            ref = CheckpointReference("latest", initial_reference["relative_path"], initial_reference["sha256"], 0)
            payload = read_checkpoint(paths, ref, config, saved)
            if payload["metadata"]["counters"] != asdict(Counters()) or payload["continuation"]["optimizer"]["state"]:
                raise DiagnosticError("Supplied epoch-zero checkpoint must precede all training/optimizer updates")
            initial = {"reference": asdict(ref), "metadata": payload["metadata"],
                       "model_state_sha256": state_digest(payload["clean_model"])}
            export.add(paths.checkpoints / ref.relative_path)
        return {"audit_version": 1, "kind": "historical_training_audit", "research_authority": False,
            "identity": identity_metadata(config), "saved_config": config.to_dict(), "saved_run_metadata": metadata,
            "archived_config_declaration": stored_config.to_dict(),
            "selected_effective_config_matches_archive": True,
            "head": current.head, "transaction_reference": current.reference.to_dict(), "transaction": current.document,
            "receipt": asdict(current.receipt), "checkpoints": checkpoints,
            "latest_metadata": current.latest["metadata"], "best_metadata": current.best_metadata,
            "latest_clean_model_state_sha256": state_digest(current.latest["clean_model"]),
            "native_optimizer_groups": [{k: v for k, v in group.items() if k != "params"} for group in groups],
            "native_optimizer_state_summaries": native, "scheduler_state": continuation["scheduler"],
            "scaler_state": continuation["scaler"], "data_continuation": continuation["data_continuation"],
            "epochs": epochs, "initial_checkpoint": initial,
            "initial_evidence": "supplied_verified_epoch_zero" if initial is not None else "unavailable; seeded reconstruction is not proof of original initialization",
            "export_files": [{"path": str(path), "sha256": file_digest(path)} for path in sorted(export)]}

    with inspection.activate():
        report = inspection.reuse("run-audit", training_paths(paths), inspect, track_dependencies=True)
    actual = collect_provenance(source_root, attempt={"lifecycle": "diagnostic_source_audit"})
    current_inventory = actual["code"]["inventory"]
    original = report["saved_run_metadata"]["provenance"]
    recorded_inventory = original["code"].get("inventory")
    differences = None if type(recorded_inventory) is not dict else [
        {"path": path, "saved": sha, "actual": current_inventory.get(path)}
        for path, sha in recorded_inventory.items() if sha != current_inventory.get(path)]
    report["source_comparison"] = {"actual_provenance": actual,
        "full_source_equal": actual["code"]["source_sha256"] == original["code"]["source_sha256"],
        "saved_inventory_available": type(recorded_inventory) is dict,
        "recorded_file_differences": differences,
        "core_file_differences": None if type(recorded_inventory) is not dict else [
            {"path": path, "saved": recorded_inventory.get(path), "actual": current_inventory.get(path)}
            for path in CORE_FILES if recorded_inventory.get(path) is None or recorded_inventory[path] != current_inventory.get(path)],
        "existing_python_differences": None if differences is None else [v for v in differences if v["path"].endswith(".py")],
        "added_inventory_paths": sorted(set(current_inventory) - set(recorded_inventory or {})),
        "software_equal": actual["software"] == original["software"]}
    import torch
    report["diagnostic_process_threads"] = {"torch_intraop": torch.get_num_threads(),
        "torch_interop": torch.get_num_interop_threads(), "environment": {key: os.environ.get(key) for key in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}}
    if source_inventory(source_root) != current_inventory:
        raise DiagnosticError("Diagnostic source changed during the audit; freeze the reviewed checkout")
    inspection.assert_unchanged()
    strict_json_text(report)
    return report
