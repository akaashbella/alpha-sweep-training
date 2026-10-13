"""Read-only inspection, explicit publication recovery and confirmed lock release."""

from dataclasses import dataclass
from pathlib import Path
import uuid

from .artifacts import _check_existing, _is_link, effective_storage_config, resolve_run_paths
from .checkpoint_io import ArtifactReference, digest_file, fail, fsync_directory, relative_file, same
from .checkpoint_journal import HEAD_PATH, read_committed
from .identity import identity_metadata
from .provenance import software_provenance
from .serialization import atomic_write_json, read_json
from .training_contracts import CommitReceipt, CompletionReceipt, TrainingCompletionReceipt


def _provenance(value: dict) -> dict:
    return software_provenance(code=value["code"], software=value["software"], allocation=value.get("allocation"))


@dataclass(frozen=True)
class CompletionInspection:
    receipt: CommitReceipt
    completion: CompletionReceipt | TrainingCompletionReceipt
    summary: dict
    transaction: ArtifactReference
    lifecycle: str = "training"

    @property
    def legacy_clean_test(self) -> bool:
        return isinstance(self.completion, CompletionReceipt)


def inspect_completion(config, *, base_dir, provenance) -> CompletionInspection | None:
    """Read TRAINING completion (including legacy clean evidence), never robustness.

    Missing tasks return None. Existing partial/corrupt/incompatible runs raise;
    a stale/missing derived summary never authorizes recomputation or replacement.
    Concurrent mutation can cause a conservative read failure: retry inspection
    after its writer exits, rather than deriving progress from partial files.
    """
    config = effective_storage_config(config, base_dir)
    paths = resolve_run_paths(config, base_dir)
    current_provenance = _provenance(provenance)
    if not any(path.exists() or _is_link(path) for path in paths.directories()):
        return None
    _check_existing(paths, config, "recover", current_provenance)
    current = read_committed(paths, config, current_provenance)
    if current.status not in ("completed", "training_completed"):
        return None
    from .checkpoint_io import read_document
    reference = ArtifactReference.from_dict(current.document["artifacts"]["completion"]["summary"])
    summary = read_document(paths, reference)
    receipt_type = CompletionReceipt if current.status == "completed" else TrainingCompletionReceipt
    return CompletionInspection(current.receipt, receipt_type(identity_metadata(config),
        current.receipt.commit_id, reference.sha256), summary, current.reference)


def inspect_training_completion(config, *, base_dir, provenance):
    """Explicit lifecycle adapter for the legacy-named inspect_completion API."""
    return inspect_completion(config, base_dir=base_dir, provenance=provenance)


@dataclass(frozen=True)
class LockEvidence:
    path: str
    sha256: str
    owner: dict


def inspect_locks(config, *, base_dir) -> tuple[LockEvidence, ...]:
    """No local PID inference: recorded host/PID/token are diagnostics only."""
    paths = resolve_run_paths(config, base_dir)
    result = []
    for root in sorted(paths.managed_roots(), key=str):
        path = root.parent / f".{root.name}.lock"
        if path.exists() or _is_link(path):
            if _is_link(path) or not path.is_file():
                fail(f"Linked/non-file lock requires manual evidence-preserving recovery: {path}")
            owner = read_json(path)
            for key, value in identity_metadata(config).items():
                same(owner[key], value, f"lock {key}")
            if type(owner.get("token")) is not str or not owner["token"] or not owner.get("hostname"):
                fail("Lock ownership is incomplete; do not automatically release it")
            result.append(LockEvidence(str(path), digest_file(path), owner))
    return tuple(result)


def release_confirmed_locks(config, *, base_dir, expected: tuple[LockEvidence, ...],
                            inactive_owners_confirmed: bool, reason: str) -> tuple[str, ...]:
    """Explicit operator recovery after ALL recorded owners/allocations stopped.

    Confirmation is supplied by the later caller, never inferred from PID age or
    elapsed time. Match all paths, owners and bytes and preserve the lock files
    by rename beside their original locations, with an audit record. A failure
    midway preserves evidence; re-inspect before doing anything further.
    """
    if inactive_owners_confirmed is not True or type(reason) is not str or not reason:
        fail("Lock recovery requires explicit confirmation of inactive owners across nodes and an audit reason")
    actual = inspect_locks(config, base_dir=base_dir)
    if not actual or actual != expected:
        fail("Lock evidence changed or was incomplete; re-inspect all selected storage-root locks")
    paths = resolve_run_paths(config, base_dir)
    allowed = {str(root.parent / f".{root.name}.lock") for root in paths.managed_roots()}
    recovered = []
    for evidence in expected:
        if evidence.path not in allowed:
            fail("Lock recovery path is outside the selected managed storage roots")
        path = Path(evidence.path)
        same(read_json(path), evidence.owner, "lock owner at release")
        same(digest_file(path), evidence.sha256, "lock bytes at release")
        archive = path.with_name(path.name + f".recovered-{uuid.uuid4().hex}")
        if archive.exists() or _is_link(archive):
            fail("Lock evidence archive already exists")
        audit = archive.with_name(archive.name + ".audit.json")
        atomic_write_json(audit, {**identity_metadata(config), "reason": reason,
            "inactive_owners_confirmed": True, "original_path": evidence.path,
            "original_sha256": evidence.sha256, "owner": evidence.owner, "archive": str(archive)})
        path.rename(archive)
        fsync_directory(path.parent)
        recovered.append(str(archive))
    return tuple(recovered)


def recovery_diagnostics(config, *, base_dir, provenance) -> dict:
    """Report partial initialization, durable outcome and ambiguous evidence.

    No repair, lock stealing, checkpoint fallback or file deletion occurs here.
    Publication recovery requires open_run(operation='recover', continuation_plan)
    followed by DurableTrainingPersistence.repair_publication while holding it.
    """
    config = effective_storage_config(config, base_dir)
    paths = resolve_run_paths(config, base_dir)
    result = {**identity_metadata(config), "compatible_committed_state": False, "status": None,
              "completed_epochs": None, "diagnostic": None, "locks": [], "evidence": []}
    try:
        result["locks"] = [evidence.__dict__ for evidence in inspect_locks(config, base_dir=base_dir)]
        _check_existing(paths, config, "recover", _provenance(provenance))
        current = read_committed(paths, config, _provenance(provenance))
        result.update(compatible_committed_state=True, status=current.status,
                      completed_epochs=current.receipt.completed_epochs, transaction=current.reference.to_dict())
    except Exception as exc:
        result["diagnostic"] = f"{type(exc).__name__}: {exc}"
    for root_name, relative in (("metadata", "state/transactions"), ("checkpoints", "generations"),
                                ("results", "records"), ("results", "completions"), ("results", "failures")):
        root = getattr(paths, root_name)
        if not root.is_dir() or _is_link(root):
            continue
        try:
            directory = relative_file(root, relative + "/probe").parent
            if directory.is_dir():
                result["evidence"].extend({"root": root_name, "relative_path": path.relative_to(root).as_posix(),
                                           "linked": _is_link(path)} for path in directory.iterdir())
        except Exception as exc:
            result["evidence"].append({"root": root_name, "diagnostic": str(exc)})
    return result
