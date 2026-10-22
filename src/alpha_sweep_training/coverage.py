"""Read-only manifest joins against journal authority; no CUDA/model construction."""

from dataclasses import asdict, dataclass
import sys

from .artifacts import (_check_existing, _is_link, _require_plain_file, _inspect_directory,
                        effective_storage_config, resolve_run_paths, RunDirectoryError)
from .config import load_config
from .data_contract import file_digest
from .identity import canonical_alpha, identity_metadata, scientific_fingerprint
from .manifests import Manifest, Task, manifest_from_dict, task_config
from .provenance import software_provenance
from .serialization import read_json, strict_json_text
from .spec import DATASETS, ARCHITECTURES, EPOCHS


class _Incompatible(ValueError):
    pass


@dataclass(frozen=True)
class TaskObservation:
    run_id: str
    scientific_fingerprint: str
    global_index: int
    phase: int
    phase_index: int
    dataset: str
    architecture: str
    alpha: float
    canonical_alpha: str
    training_seed: int
    condition: str
    stored_status: str | None = None
    completed_epochs: int | None = None
    training_complete: bool = False
    full_budget_committed: bool = False
    evaluation_eligible: bool = False
    training_finalization_pending: bool = False
    legacy_clean_test: bool = False
    terminal_numerical: bool = False
    execution_failed: bool = False
    locks: tuple[str, ...] = ()
    numerical_evidence: tuple[dict, ...] = ()
    execution_evidence: tuple[dict, ...] = ()
    evidence_failures: tuple[dict, ...] = ()
    failure: dict | None = None
    transaction: dict | None = None
    diagnostic: str | None = None

    def to_dict(self):
        return {**asdict(self), "training_alpha": self.alpha,
                "canonical_training_alpha": self.canonical_alpha}


def _lock_paths(paths):
    return tuple(root.parent / f".{root.name}.lock" for root in paths.managed_roots())


def _stamp(paths):
    """A bounded observation token, never a reader lock or completion heuristic."""
    names = [*paths.directories(), *_lock_paths(paths), paths.metadata / "state" / "HEAD.json",
             paths.metadata / "run_metadata.json", paths.metadata / "resolved_config.yaml"]
    result = []
    for path in names:
        if _is_link(path):
            result.append((str(path), "linked"))
        elif path.is_file():
            result.append((str(path), file_digest(path)))
        else:
            result.append((str(path), "directory" if path.exists() else "missing"))
    return tuple(result)


def inspect_task(task: Task, *, base_dir, overrides=None, provenance=None) -> TaskObservation:
    from .planning_inspection import active_inspection, training_paths, compatibility_key
    inspection = active_inspection()
    if inspection is None:
        return _inspect_task(task, base_dir=base_dir, overrides=overrides, provenance=provenance)
    config = effective_storage_config(task_config(task, overrides), base_dir)
    paths = resolve_run_paths(config, base_dir)
    key = ("observation", task.run_id, strict_json_text(task.prepared_data),
           strict_json_text(config.to_dict()), compatibility_key(provenance))
    return inspection.reuse(key, training_paths(paths), lambda: _inspect_task(
        task, base_dir=base_dir, overrides=overrides, provenance=provenance))


def _inspect_task(task: Task, *, base_dir, overrides=None, provenance=None) -> TaskObservation:
    """Use saved provenance for historical coverage, actual provenance for dispatch.

    Imports of checkpoint readers are lazy. They restricted-load CPU state and
    validate records, but never construct models, loaders or CUDA runtime. Locks
    suppress authority claims; changing tokens invalidate the whole observation.
    """
    config = task_config(task, overrides)
    science = config.scientific
    common = dict(run_id=task.run_id, scientific_fingerprint=task.fingerprint,
        global_index=task.global_index, phase=task.phase, phase_index=task.phase_index,
        dataset=science["dataset"], architecture=science["architecture"], alpha=science["alpha"],
        canonical_alpha=canonical_alpha(science["alpha"]),
        training_seed=science["training_seed"])
    paths = None
    before = None
    values = {}
    try:
        config = effective_storage_config(config, base_dir)
        paths = resolve_run_paths(config, base_dir)
        for path in paths.directories():
            _inspect_directory(path)
        before = _stamp(paths)
        locks = tuple(str(p) for p in _lock_paths(paths) if p.exists() or _is_link(p))
        if locks:
            return TaskObservation(**common, condition="locked", locks=locks,
                                   diagnostic="Inspect ownership with recovery.inspect_locks; no PID/age inference")
        if not any(p.exists() or _is_link(p) for p in paths.directories()):
            values = {"condition": "missing"}
        elif (not (paths.metadata / "state" / "HEAD.json").is_file()
              or not (paths.metadata / "resolved_config.yaml").is_file()
              or not (paths.metadata / "run_metadata.json").is_file()):
            values = {"condition": "partial", "diagnostic": "Existing run lacks durable initialization/ownership; preserve evidence"}
        else:
            _require_plain_file(paths.metadata / "run_metadata.json")
            _require_plain_file(paths.metadata / "resolved_config.yaml")
            metadata = read_json(paths.metadata / "run_metadata.json")
            if type(metadata) is not dict:
                raise ValueError("Invalid run metadata object")
            for key, value in identity_metadata(config).items():
                if strict_json_text(metadata.get(key)) != strict_json_text(value):
                    raise _Incompatible(f"Stored {key} does not match selected task")
            saved_config = load_config(paths.metadata / "resolved_config.yaml")
            if scientific_fingerprint(saved_config) != task.fingerprint:
                raise _Incompatible("Saved configuration disagrees with manifest")
            for directory in (paths.checkpoints, paths.results, paths.logs):
                if _inspect_directory(directory):
                    marker = directory / ".run_owner.json"
                    _require_plain_file(marker)
                    owner = read_json(marker)
                    if type(owner) is not dict:
                        raise ValueError(f"Invalid artifact-root owner: {marker}")
                    if any(strict_json_text(owner.get(key)) != strict_json_text(value)
                           for key, value in identity_metadata(config).items()):
                        raise _Incompatible(f"Artifact-root owner disagrees with selected task: {marker}")
            saved = metadata["provenance"]
            software_provenance(code=saved["code"], software=saved["software"], allocation=saved.get("allocation"))
            current_provenance = saved if provenance is None else software_provenance(
                code=provenance["code"], software=provenance["software"], allocation=provenance.get("allocation"))
            _check_existing(paths, config, "recover", saved)
            from .checkpoint_journal import numerical_failure_evidence, execution_failure_evidence, read_committed
            from .checkpoint_io import ArtifactReference, read_document
            current = read_committed(paths, config, saved, metadata_only=True)
            budget_complete = current.receipt.completed_epochs == EPOCHS
            if not budget_complete and (saved["code"]["source_sha256"] != current_provenance["code"]["source_sha256"]
                    or saved["software"] != current_provenance["software"]):
                raise _Incompatible("Training continuation source/software differs; read-only evaluation/finalization is a separate policy")
            current_provenance = saved
            stored_data = current.latest["metadata"]["data_identity"]
            if any(stored_data[key] != value for key, value in task.prepared_data.items()):
                raise _Incompatible("Stored prepared-data identities disagree with manifest")
            evidence = () if budget_complete else numerical_failure_evidence(paths, config, current_provenance)
            execution_evidence = () if budget_complete else execution_failure_evidence(
                paths, config, current_provenance, training_commit_id=current.receipt.commit_id)
            failure_ref = current.document["artifacts"]["failure"]
            failure = None if failure_ref is None else read_document(paths, ArtifactReference.from_dict(failure_ref))["payload"]
            training_complete = current.status in ("completed", "training_completed")
            # read_committed already validates completion, immutable CSV bytes,
            # best/final bindings and ancestry. Do not load the journal a second
            # time through recovery.inspect_completion; the after stamp remains.
            count = current.receipt.completed_epochs
            values = {"condition": "ok", "stored_status": current.status, "completed_epochs": count,
                "training_complete": training_complete, "full_budget_committed": budget_complete,
                "evaluation_eligible": budget_complete,
                "training_finalization_pending": budget_complete and not training_complete,
                "legacy_clean_test": current.status == "completed",
                "terminal_numerical": not budget_complete and (current.status == "failed_numerical" or bool(evidence)),
                "execution_failed": not budget_complete and (current.status == "failed_execution" or bool(execution_evidence)),
                "numerical_evidence": tuple(ref.to_dict() for ref in evidence),
                "execution_evidence": tuple(ref.to_dict() for ref in execution_evidence),
                "evidence_failures": tuple({"reference": ref.to_dict(), "failure": read_document(
                    paths, ref)["payload"]} for ref in (*evidence, *execution_evidence)),
                "failure": failure, "transaction": current.reference.to_dict()}
    except _Incompatible as exc:
        values = {"condition": "incompatible", "diagnostic": str(exc)}
    except RunDirectoryError as exc:
        values = {"condition": "partial", "diagnostic": f"Ownership/publication rejected: {exc}"}
    except (ValueError, KeyError, TypeError) as exc:
        values = {"condition": "corrupt", "diagnostic": f"{type(exc).__name__}: {exc}"}
    except Exception as exc:
        # PersistenceContractError is a RuntimeError; bounded stable validation
        # rejection differs from environmental/I/O errors or a changing writer.
        contracts = sys.modules.get(__package__ + ".training_contracts")
        contract_error = getattr(contracts, "PersistenceContractError", ())
        corrupt = isinstance(exc, contract_error)
        values = {"condition": "corrupt" if corrupt else "inspection_error",
                  "diagnostic": f"{type(exc).__name__}: {exc}"}
    if paths is None:
        return TaskObservation(**common, **values)
    if before is None:
        return TaskObservation(**common, **values)
    try:
        after = _stamp(paths)
        if before != after or any(p.exists() or _is_link(p) for p in _lock_paths(paths)):
            return TaskObservation(**common, condition="unstable", diagnostic="Authority/locks changed; inspect after writer exits")
    except Exception as exc:
        return TaskObservation(**common, condition="inspection_error", diagnostic=f"Cannot establish stable read: {exc}")
    return TaskObservation(**common, **values)


def _counts(rows):
    completed = sum(r.condition == "ok" and r.training_complete for r in rows)
    numeric = sum(r.condition == "ok" and r.terminal_numerical for r in rows)
    execution = sum(r.condition == "ok" and not r.terminal_numerical and r.execution_failed for r in rows)
    return {"expected": len(rows), "training_completed": completed, "training_failed": numeric + execution,
        "failed_numerical": numeric, "failed_execution": execution,
        "training_unfinished": len(rows) - completed - numeric - execution,
        "interrupted": sum(r.condition == "ok" and r.stored_status == "interrupted"
                           and not r.terminal_numerical and not r.execution_failed for r in rows),
        "evaluation_eligible": sum(r.evaluation_eligible for r in rows),
        "training_finalization_pending": sum(r.training_finalization_pending for r in rows),
        **{name: sum(r.condition == name for r in rows) for name in
           ("missing", "locked", "partial", "incompatible", "corrupt", "unstable", "inspection_error")}}


def collect_coverage(manifest: Manifest, *, base_dir, overrides=None) -> dict:
    manifest = manifest_from_dict(manifest.to_dict())
    task_config(manifest.tasks[0], overrides)  # Invalid global overrides reject the command itself.
    from .planning_inspection import active_inspection
    inspection = active_inspection()
    rows = []
    for index, task in enumerate(manifest.tasks, 1):
        if inspection is not None:
            inspection.progress(f"Training authority {index}/{len(manifest.tasks)}: {task.run_id}")
        rows.append(inspect_task(task, base_dir=base_dir, overrides=overrides))
    rows = tuple(rows)
    phases = []
    for phase, dataset in enumerate(DATASETS, start=1):
        selected = tuple(r for r in rows if r.phase == phase)
        counts = _counts(selected)
        phases.append({"phase": phase, "dataset": dataset, **counts,
            "training_successfully_completed": counts["training_completed"] == counts["expected"],
            "terminal_scientific_failures": counts["failed_numerical"],
            "training_settled_with_scientific_failures": counts["training_completed"] + counts["failed_numerical"] == counts["expected"],
            "training_eligible": phase == 1 or phases[phase - 2]["training_successfully_completed"],
            "automatic_candidates": [r.run_id for r in selected if r.condition == "missing" or
                (r.condition == "ok" and not r.training_complete and not r.terminal_numerical and not r.execution_failed)]})
    return {"coverage_version": 2, "lifecycle": "training", "grid_id": manifest.grid_id, "manifest_document_sha256": manifest.document_sha256,
        "counts": _counts(rows), "phases": phases,
        "datasets": {name: _counts(tuple(r for r in rows if r.dataset == name)) for name in DATASETS},
        "architectures": {name: _counts(tuple(r for r in rows if r.architecture == name)) for name in ARCHITECTURES},
        "dataset_architectures": {dataset: {architecture: _counts(tuple(r for r in rows
            if r.dataset == dataset and r.architecture == architecture)) for architecture in ARCHITECTURES} for dataset in DATASETS},
        "tasks": [r.to_dict() for r in rows],
        "observation_semantics": "Independent stable task observations, not an atomic grid snapshot; historical provenance, no repair"}
