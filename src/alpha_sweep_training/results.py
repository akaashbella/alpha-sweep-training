"""Read-only full-manifest collection through historical journal/slot validators."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from .artifacts import effective_storage_config, resolve_run_paths, _is_link
from .coverage import inspect_task, _stamp, _lock_paths
from .evaluation_protocol import evaluation_protocol, prescribed_slots, perturbation_seed
from .manifests import manifest_from_dict, task_config
from .serialization import read_json
from .reporting_contract import (REPORTING_VERSION, CONVENTIONS, STABILITY, ReportingError,
    reporting_provenance, sealed, task_identity, utc_now, validate_collection)


def protected_storage(manifest, *, base_dir, overrides=None, evaluation_root=None, workflow_root=None, extra_protected=()):
    """Conservative broad roots as well as run directories; no filesystem writes."""
    roots = {str(Path(base_dir).expanduser().absolute())}
    for task in manifest.tasks:
        config = effective_storage_config(task_config(task, overrides), base_dir)
        roots.update(str(p) for p in resolve_run_paths(config, base_dir).directories())
        storage = config.execution["storage"]
        for key in ("output_root", "checkpoint_root", "result_root", "log_root", "data_root"):
            if storage[key] is not None:
                roots.add(str(Path(storage[key]).absolute()))
        if storage["split_file"] is not None:
            roots.add(str(Path(storage["split_file"]).absolute().parent))
        roots.add(str(Path(evaluation_root).expanduser().absolute()) if evaluation_root is not None else
                  str(Path(storage["result_root"] or storage["output_root"]) / "evaluations"))
    for path in (workflow_root, *extra_protected):
        if path is not None:
            roots.add(str(Path(path).expanduser().absolute()))
    return sorted(roots)


def _evaluation_stamp(root):
    from .evaluation_coverage import _token
    paths = [root / "owner.json", root / ".namespace.lock", root / ".sweep.lock", root / "SWEEP_HEAD.json"]
    for slot in prescribed_slots():
        paths.extend([root / "slots" / slot.slot_id / "HEAD.json", root / "locks" / f"{slot.slot_id}.lock"])
    return tuple((str(path), _token(path)) for path in paths)


def _training_stamp(paths):
    try:
        return _stamp(paths)
    except Exception as exc:
        return (("stamp_inspection_error", f"{type(exc).__name__}: {exc}"),)


def _training_extras(paths):
    """Small reserved-root inspection; unknown files never become result rows."""
    known = {"run_metadata.json", "resolved_config.yaml", ".run_owner.json", "state", "generations", "records",
        "failures", "training_completions", "completions", "attempts", "status.json", "summary.json", "test.json",
        "epochs.csv", "layer_statistics.csv", "train.log", "latest.pt", "best.pt", "final.pt"}
    known.update(path.name for path in paths.directories())
    diagnostics = []
    for directory in paths.directories():
        try:
            if directory.is_dir() and not _is_link(directory):
                for path in directory.iterdir():
                    if _is_link(path) or path.name not in known:
                        diagnostics.append({"path": str(path), "condition": "unselected_or_unexpected_training_entry", "accepted": False})
        except Exception as exc:
            diagnostics.append({"path": str(directory), "condition": "training_extra_inspection_error", "reason": str(exc), "accepted": False})
    return diagnostics


def _manifest_extras(manifest, *, base_dir, overrides, evaluation_root):
    expected = {task.run_id for task in manifest.tasks}
    roots = set()
    for task in manifest.tasks:
        storage = effective_storage_config(task_config(task, overrides), base_dir).execution["storage"]
        roots.update(Path(storage[key]) for key in ("output_root", "checkpoint_root", "result_root", "log_root") if storage[key] is not None)
        roots.add(Path(evaluation_root).expanduser().absolute() if evaluation_root is not None else
                  Path(storage["result_root"] or storage["output_root"]) / "evaluations")
    diagnostics = []
    for root in sorted(roots, key=str):
        try:
            if any(_is_link(path) for path in (root, *root.parents)):
                diagnostics.append({"path": str(root), "condition": "linked_store", "accepted": False})
            elif root.is_dir():
                for path in root.iterdir():
                    if path.name not in expected and path.name != "evaluations":
                        diagnostics.append({"path": str(path), "condition": "outside_expected_manifest_membership", "accepted": False})
        except Exception as exc:
            diagnostics.append({"path": str(root), "condition": "store_inspection_error", "reason": str(exc), "accepted": False})
    return diagnostics


def _unexpected_evidence(root, inputs):
    """Diagnose extra namespaces/slots and unacknowledged raw successes; never join them."""
    from .evaluation_persistence import _read_reference, validate_result
    from .checkpoint_io import digest_file
    diagnostics = []
    expected = {slot.slot_id for slot in prescribed_slots()}
    if not root.exists():
        return diagnostics
    for path in root.iterdir():
        if _is_link(path) or path.name not in {"owner.json", "slots", "locks", "sweep_completions", "SWEEP_HEAD.json", ".namespace.lock", ".sweep.lock"}:
            diagnostics.append({"path": str(path), "condition": "unexpected_or_linked_artifact", "accepted": False})
    slots_root = root / "slots"
    if _is_link(slots_root):
        return [*diagnostics, {"path": str(slots_root), "condition": "linked_slot_root", "accepted": False}]
    if slots_root.exists():
        for path in slots_root.iterdir():
            if _is_link(path) or path.name not in expected:
                diagnostics.append({"path": str(path), "condition": "unexpected_slot", "accepted": False})
        for slot in prescribed_slots():
            directory = slots_root / slot.slot_id / "records"
            head = slots_root / slot.slot_id / "HEAD.json"
            accepted = set()
            reference = read_json(head)["current"] if head.is_file() and not _is_link(head) else None
            while reference is not None:
                if reference["relative_path"] in accepted:
                    raise ReportingError("Cycle in evaluation ancestry during reporting")
                accepted.add(reference["relative_path"])
                reference = _read_reference(root, reference)["previous_record"]
            if directory.exists() and not _is_link(directory):
                for path in directory.iterdir():
                    if _is_link(path) or path.relative_to(root).as_posix() not in accepted:
                        diagnostic = {"path": str(path), "slot_id": slot.slot_id,
                                      "condition": "unacknowledged_or_unexpected_record", "accepted": False}
                        if path.is_file() and not _is_link(path) and path.suffix == ".json":
                            try:
                                record = validate_result(read_json(path), inputs, slot)
                                diagnostic.update(sha256=digest_file(path), stored_status=record["status"], failure=record["failure"])
                            except Exception as exc:
                                diagnostic["reason"] = f"{type(exc).__name__}: {exc}"
                        diagnostics.append(diagnostic)
    # Other checkpoint/protocol namespaces under this expected training run are
    # diagnostics only; no glob of successes defines reporting membership.
    for parent in (root.parent, root.parent.parent, root.parent.parent.parent):
        if parent.exists() and not _is_link(parent):
            selected = root if parent == root.parent else root.parent if parent == root.parent.parent else root.parent.parent
            for path in parent.iterdir():
                if path != selected:
                    diagnostics.append({"path": str(path), "condition": "other_checkpoint_protocol_or_fingerprint", "accepted": False})
    return diagnostics


def _empty_slot(task, slot, protocol, reason):
    return {**task_identity(task), "logical_index": task.global_index * 28 + list(prescribed_slots()).index(slot),
        "slot": slot.to_dict(), "requested_protocol": protocol.to_dict(),
        "requested_protocol_fingerprint": protocol.fingerprint, "condition": "checkpoint_unavailable", "stored_status": None,
        "binding": None, "best_checkpoint_identity": None, "perturbation_seed": None,
        "namespace": None, "raw_reference": None, "raw_record": None, "successful": False,
        "failure": None, "orphan_failures": [], "diagnostic": reason, "robustness_completed": False}


def _collect_evaluation(task, slot_rows, *, base_dir, overrides, evaluation_root, training_transaction):
    from .evaluation_inputs import bind_evaluation_checkpoint
    from .evaluation_coverage import collect_evaluation_coverage
    from .evaluation_persistence import evaluation_directory, read_slot, _read_reference
    inputs = bind_evaluation_checkpoint(task, base_dir=base_dir, overrides=overrides)
    if inputs.training_transaction != training_transaction:
        raise ReportingError("Training transaction changed before evaluation binding")
    root = evaluation_directory(inputs, evaluation_root=evaluation_root)
    before = _evaluation_stamp(root)
    # Binding is already verified, independently of any slot availability.
    for target, slot in zip(slot_rows, prescribed_slots()):
        target.update(binding=deepcopy(inputs.binding), best_checkpoint_identity=deepcopy(inputs.binding["best_checkpoint_identity"]),
                      perturbation_seed=perturbation_seed(inputs.binding["best_checkpoint_identity"], slot), namespace=str(root))
    coverage = collect_evaluation_coverage(inputs, evaluation_root=evaluation_root)
    coverage["completion_document"] = None
    references, diagnostics = [], []
    if coverage["robustness_completed"]:
        reference = coverage["completion"]["reference"]
        coverage["completion_document"] = _read_reference(root, reference)
        references.append({"path": str(root / reference["relative_path"]), "sha256": reference["sha256"], "role": "robustness_completion"})
    for target, slot, observed in zip(slot_rows, prescribed_slots(), coverage["slots"]):
        target.update(condition=observed["condition"], stored_status=observed["stored_status"], failure=observed["failure"],
            orphan_failures=observed["orphan_failures"], diagnostic=observed["diagnostic"], robustness_completed=coverage["robustness_completed"])
        if observed["condition"] == "ok":
            accepted = read_slot(root, inputs, slot)
            if accepted is None or accepted[0]["current"] != observed["reference"]:
                raise ReportingError("Evaluation slot changed during raw-result read")
            target.update(raw_reference=accepted[0]["current"], raw_record=accepted[1], successful=accepted[1]["status"] == "evaluation_completed")
            references.append({"path": str(root / accepted[0]["current"]["relative_path"]),
                               "sha256": accepted[0]["current"]["sha256"], "role": "evaluation_raw"})
    try:
        diagnostics.extend(_unexpected_evidence(root, inputs))
    except Exception as exc:
        diagnostics.append({"path": str(root), "condition": "unexpected_evidence_inspection_error", "accepted": False,
                            "reason": f"{type(exc).__name__}: {exc}"})
    if before != _evaluation_stamp(root):
        raise ReportingError("Checkpoint evaluation authority changed during collection")
    return coverage, references, diagnostics


def collect_results(manifest, *, base_dir, source_root, overrides=None, evaluation_root=None,
                    workflow_root=None, extra_protected=()):
    """Return a finite sealed 240/6720-row snapshot; never acquire research writer locks.

    Each task is bracketed with the existing authority/lock stamps. A concurrent
    change invalidates its accepted metrics rather than merging generations.
    """
    from .checkpoint_io import ArtifactReference, read_document
    from .checkpoint_journal import read_committed
    manifest = manifest_from_dict(manifest.to_dict())
    training, evaluation, epochs, layers, diagnostics, references = [], [], [], [], [], []
    began = utc_now()
    provenance = reporting_provenance(source_root, "collector")
    for task in manifest.tasks:
        config = effective_storage_config(task_config(task, overrides), base_dir)
        paths = resolve_run_paths(config, base_dir)
        protocol = evaluation_protocol(config)
        before = _training_stamp(paths)
        observation = inspect_task(task, base_dir=base_dir, overrides=overrides)  # saved provenance, never reporter's digest
        row = {**task_identity(task), **observation.to_dict(), "training_run_id": task.run_id,
            "snapshot": None, "best_validation_metrics": None, "legacy_clean": None,
            "scientific_settings": task.config.scientific, "prepared_data": task.prepared_data,
            "evaluation_coverage": None, "paths": paths.to_dict()}
        slot_rows = [_empty_slot(task, slot, protocol, observation.diagnostic or observation.condition) for slot in prescribed_slots()]
        task_epochs, task_layers, task_refs, task_diagnostics = [], [], [], []
        try:
            if observation.condition == "ok":
                saved = read_json(paths.metadata / "run_metadata.json")["provenance"]
                current = read_committed(paths, config, saved)
                if current.reference.to_dict() != observation.transaction:
                    raise ReportingError("Training transaction changed after stable observation")
                snapshot_epochs = []
                for ref, (context, record, statistics) in zip(current.document["records"], current.records):
                    document = read_document(paths, ArtifactReference.from_dict(ref))
                    snapshot_epochs.append({"reference": ref, "document": document})
                    # Index the single embedded epoch document instead of repeating
                    # full source inventories for every layer tensor/CSV row.
                    index = {"training_run_id": task.run_id, "reference": ref, "epoch_label": record.epoch_index + 1}
                    task_epochs.append(index)
                    task_layers.append(index)
                    task_refs.append({"path": str(ArtifactReference.from_dict(ref).path(paths)), "sha256": ref["sha256"], "role": "committed_epoch"})
                best = current.document["best"]
                row["best_validation_metrics"] = None if best is None else snapshot_epochs[best["epoch_index"]]["document"]["record"]["validation"]
                row["snapshot"] = {"transaction_reference": current.reference.to_dict(), "transaction": current.document,
                    "receipt": asdict(current.receipt), "provenance": saved,
                    "component_metadata": current.latest["metadata"]["component_metadata"], "epochs": snapshot_epochs,
                    "completion_documents": {}}
                for role, reference in current.document["artifacts"].items():
                    if reference is None:
                        continue
                    role_refs = reference if role == "completion" else {"failure": reference}
                    for name, ref in role_refs.items():
                        document = read_document(paths, ArtifactReference.from_dict(ref))
                        row["snapshot"]["completion_documents"][name] = {"reference": ref, "document": document}
                        task_refs.append({"path": str(ArtifactReference.from_dict(ref).path(paths)), "sha256": ref["sha256"], "role": name})
                        for metric_name, metric_ref in document.get("metric_artifacts", {}).items():
                            verified = ArtifactReference.from_dict(metric_ref).verify(paths)
                            task_refs.append({"path": str(verified), "sha256": metric_ref["sha256"], "role": metric_name})
                task_refs.append({"path": str(current.reference.path(paths)), "sha256": current.reference.sha256, "role": "training_transaction"})
                for role in ("latest", "best", "final"):
                    checkpoint = getattr(current.receipt, role)
                    if checkpoint is not None:
                        task_refs.append({"path": str(Path(paths.checkpoints) / checkpoint.relative_path), "sha256": checkpoint.sha256, "role": role})
                if current.status == "completed":
                    legacy = row["snapshot"]["completion_documents"]
                    row["legacy_clean"] = {"test_reference": legacy["test"]["reference"], "summary_reference": legacy["summary"]["reference"],
                                           "test": legacy["test"]["document"], "summary": legacy["summary"]["document"]}
                task_diagnostics.extend(_training_extras(paths))
            if observation.condition == "ok" and observation.evaluation_eligible:
                try:
                    coverage, eval_refs, eval_diagnostics = _collect_evaluation(task, slot_rows, base_dir=base_dir, overrides=overrides,
                        evaluation_root=evaluation_root, training_transaction=observation.transaction)
                    row["evaluation_coverage"] = coverage
                    task_refs.extend(eval_refs)
                    task_diagnostics.extend(eval_diagnostics)
                except Exception as exc:
                    # Test availability/inspection cannot change validation
                    # eligibility. Only a changing training bracket below can.
                    condition = "unstable" if isinstance(exc, ReportingError) else "inspection_error"
                    reason = f"{type(exc).__name__}: {exc}"
                    for target in slot_rows:
                        target.update(condition=condition, diagnostic=reason, raw_reference=None, raw_record=None,
                                      successful=False, robustness_completed=False)
                    task_diagnostics.append({"lifecycle": "evaluation", "condition": condition, "reason": reason})
            if before != _training_stamp(paths) or any(p.exists() or _is_link(p) for p in _lock_paths(paths)):
                raise ReportingError("Training authority/locks changed during collection")
        except Exception as exc:
            # Preserve the original observation as diagnostics, remove all accepted
            # metrics for this bracket. Stored status is separate from condition.
            reason = f"{type(exc).__name__}: {exc}"
            changed = before != _training_stamp(paths)
            condition = "unstable" if changed or isinstance(exc, ReportingError) else "inspection_error"
            row.update(condition=condition, diagnostic=reason, snapshot=None, best_validation_metrics=None,
                       training_complete=False, evaluation_coverage=None, legacy_clean=None)
            for target in slot_rows:
                target.update(condition=condition, diagnostic=reason, binding=None, best_checkpoint_identity=None,
                    perturbation_seed=None, raw_reference=None, raw_record=None, successful=False, robustness_completed=False)
            task_epochs, task_layers, task_refs = [], [], []
            task_diagnostics.append({"training_run_id": task.run_id, "condition": condition, "reason": reason})
        training.append(row)
        evaluation.extend(slot_rows)
        epochs.extend(task_epochs)
        layers.extend(task_layers)
        references.extend(task_refs)
        diagnostics.extend({"training_run_id": task.run_id, **entry} for entry in task_diagnostics)
    diagnostics.extend(_manifest_extras(manifest, base_dir=base_dir, overrides=overrides, evaluation_root=evaluation_root))
    collection = sealed({"reporting_version": REPORTING_VERSION, "kind": "result_collection", "conventions": CONVENTIONS,
        "manifest": manifest.to_dict(), "grid_id": manifest.grid_id, "manifest_document_sha256": manifest.document_sha256,
        "stability": STABILITY, "observation_started_at_utc": began, "observed_at_utc": utc_now(),
        "collector_provenance": provenance, "protected_roots": protected_storage(manifest, base_dir=base_dir, overrides=overrides,
            evaluation_root=evaluation_root, workflow_root=workflow_root, extra_protected=extra_protected),
        "workflow_root": None if workflow_root is None else str(Path(workflow_root).expanduser().absolute()),
        "training": training, "evaluation": evaluation, "epochs": epochs, "layer_statistics": layers,
        "accepted_source_references": references, "diagnostics": diagnostics})
    validate_collection(collection)
    return collection
