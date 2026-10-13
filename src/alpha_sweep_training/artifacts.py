"""Guarded run initialization and fully preflighted continuation/recovery leases.

All writers must cooperate with these locks and keep open_run's context active
throughout their use of a run. No artifact is considered disposable because a
summary is missing. Explicit overwrite archives selected directories, never
deletes them. No helpers here are called during repository implementation.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from copy import deepcopy
from datetime import datetime, timezone
import os
from pathlib import Path
import socket
from typing import TYPE_CHECKING, Iterator, Literal, Mapping
import uuid

from .config import RunConfig, load_config
from .identity import identity_metadata, run_identity, scientific_fingerprint
from .provenance import software_provenance
from .serialization import (
    atomic_write_json, atomic_write_text, durable_mkdir, fsync_directory,
    read_json, resolved_config_text, strict_json_text,
)

RunOperation = Literal["new", "continue", "recover", "overwrite"]

if TYPE_CHECKING:
    from .continuation import ContinuationPlan


class RunDirectoryError(RuntimeError):
    """Output reuse would be ambiguous, incompatible, or unsafe."""


@dataclass(frozen=True)
class RunPaths:
    metadata: Path
    checkpoints: Path
    results: Path
    logs: Path

    def directories(self) -> tuple[Path, ...]:
        return (self.metadata, self.checkpoints, self.results, self.logs)

    def managed_roots(self) -> tuple[Path, ...]:
        directories = self.directories()
        return tuple(path for path in directories
                     if not any(parent != path and parent in path.parents for parent in directories))

    def to_dict(self) -> dict:
        return {name: str(getattr(self, name)) for name in ("metadata", "checkpoints", "results", "logs")}


@dataclass
class _RunLease:
    active: bool = True


@dataclass(frozen=True)
class RunHandle:
    config: RunConfig
    paths: RunPaths
    operation: RunOperation
    attempt_id: str
    archived_directories: tuple[Path, ...]
    provenance: dict = field(default_factory=dict)
    _lease: _RunLease | None = field(default=None, repr=False, compare=False)

    def require_active(self, config: RunConfig | None = None) -> None:
        """Reject use after open_run exits; this extends the existing run lock."""
        if self._lease is None or not self._lease.active:
            raise RunDirectoryError("Hold open_run's context throughout training and every artifact write")
        if config is not None and scientific_fingerprint(config) != scientific_fingerprint(self.config):
            raise RunDirectoryError("Run handle and component configuration have different scientific identities")


def effective_storage_config(config: RunConfig, base_dir: str | Path) -> RunConfig:
    """Make storage paths absolute for a portable saved resolved configuration."""
    base_dir = Path(base_dir).expanduser().resolve()
    document = config.to_dict()
    for name, value in document["execution"]["storage"].items():
        if value is not None:
            path = Path(value).expanduser()
            document["execution"]["storage"][name] = str((base_dir / path).resolve())
    return RunConfig(document)


def resolve_run_paths(config: RunConfig, base_dir: str | Path) -> RunPaths:
    config = effective_storage_config(config, base_dir)
    run_id = run_identity(config)
    storage = config.execution["storage"]
    metadata = Path(storage["output_root"]) / run_id
    paths = RunPaths(
        metadata=metadata,
        checkpoints=(Path(storage["checkpoint_root"]) / run_id
                     if storage["checkpoint_root"] else metadata / "checkpoints"),
        results=(Path(storage["result_root"]) / run_id
                 if storage["result_root"] else metadata / "results"),
        logs=(Path(storage["log_root"]) / run_id
              if storage["log_root"] else metadata / "logs"),
    )
    directories = paths.directories()
    if len(set(directories)) != len(directories):
        raise RunDirectoryError("Metadata, checkpoint, result, and log directories must be distinct")
    for path in directories:
        for parent in directories:
            if parent != path and parent in path.parents:
                if parent != metadata or path.parent != metadata:
                    raise RunDirectoryError(f"Unsupported overlapping run directories: {parent}, {path}")
    return paths


def _is_link(path: Path) -> bool:
    # Junction detection is available on newer Windows Python versions.
    return path.is_symlink() or getattr(path, "is_junction", lambda: False)()


def _inspect_directory(path: Path) -> bool:
    if _is_link(path):
        raise RunDirectoryError(f"Refusing a linked run directory: {path}")
    if path.exists():
        if not path.is_dir():
            raise RunDirectoryError(f"Expected a run directory, found a file: {path}")
        return next(path.iterdir(), None) is not None
    return False


def _require_plain_file(path: Path) -> None:
    if _is_link(path) or not path.is_file():
        raise RunDirectoryError(f"Missing or linked identity/configuration artifact: {path}")


def _compatible_record(record: Mapping, expected: dict, path: Path) -> None:
    if not isinstance(record, Mapping):
        raise RunDirectoryError(f"Expected an identity record: {path}")
    for key, value in expected.items():
        if type(record.get(key)) is not type(value) or record.get(key) != value:
            raise RunDirectoryError(f"Incompatible {key} in {path}; refusing output reuse")


def _check_existing(paths: RunPaths, config: RunConfig, operation: RunOperation, provenance: dict) -> None:
    """Read every configured run directory before any run-specific file changes."""
    occupied = {path: _inspect_directory(path) for path in paths.directories()}
    if operation == "new":
        existing = [str(path) for path, present in occupied.items() if present]
        if existing:
            raise RunDirectoryError("Existing artifacts require explicit continue or overwrite: " + ", ".join(existing))
        return
    if operation == "overwrite":
        # No compatibility requirement: the explicit selected identity is checked
        # by open_run, and all occupied roots will be archived before fresh writes.
        if not any(path.exists() for path in paths.directories()):
            raise RunDirectoryError("No existing selected run directory; use operation='new'")
        return

    expected = identity_metadata(config)
    metadata_file = paths.metadata / "run_metadata.json"
    config_file = paths.metadata / "resolved_config.yaml"
    _require_plain_file(metadata_file)
    _require_plain_file(config_file)
    record = read_json(metadata_file)
    _compatible_record(record, expected, metadata_file)
    existing_config = load_config(config_file)
    existing_config.require_research_ready()
    if scientific_fingerprint(existing_config) != expected["scientific_fingerprint"]:
        raise RunDirectoryError(f"Saved configuration disagrees with scientific identity: {config_file}")
    previous_provenance = record.get("provenance")
    if (not isinstance(previous_provenance, dict)
            or not isinstance(previous_provenance.get("code"), dict)
            or not isinstance(previous_provenance.get("software"), dict)):
        raise RunDirectoryError(f"Missing original provenance: {metadata_file}")
    if (previous_provenance.get("code", {}).get("source_sha256") != provenance["code"]["source_sha256"]
            or previous_provenance.get("software") != provenance["software"]):
        raise RunDirectoryError("Continuation code/software provenance differs; explicit migration belongs to Step 5")
    for directory, present in occupied.items():
        if directory == paths.metadata or not present:
            continue
        marker = directory / ".run_owner.json"
        _require_plain_file(marker)
        _compatible_record(read_json(marker), expected, marker)
    # HEAD preflight owns completion and immutable result validation. A stale or
    # damaged derived summary must not veto committed completion or prevent
    # explicit publication repair. Root/config markers still guard ownership;
    # the service checks linked targets before writing any derived views.
    # Attempt history is the only pre-existing directory written inside metadata.
    _inspect_directory(paths.metadata / "attempts")


@contextmanager
def _run_locks(paths: RunPaths, expected: dict) -> Iterator[None]:
    acquired = []
    token = uuid.uuid4().hex
    try:
        # One lock per independent storage root guards users choosing different
        # metadata roots while pointing at the same checkpoint/result/log roots.
        for root in sorted(paths.managed_roots(), key=str):
            durable_mkdir(root.parent)
            lock = root.parent / f".{root.name}.lock"
            if _is_link(lock):
                raise RunDirectoryError(f"Refusing a linked lock file: {lock}")
            try:
                descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError as exc:
                raise RunDirectoryError(f"Run lock exists: {lock}; inspect ownership before manual recovery") from exc
            acquired.append(lock)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(strict_json_text({**expected, "token": token,
                                               "hostname": socket.gethostname(), "pid": os.getpid(),
                                               "created_at_utc": datetime.now(timezone.utc).isoformat()}))
                stream.flush()
                os.fsync(stream.fileno())
            fsync_directory(lock.parent)
        yield
    finally:
        for lock in reversed(acquired):
            # Never remove a replacement lock belonging to another process.
            if lock.is_file() and not _is_link(lock):
                try:
                    owner = read_json(lock)
                except (OSError, ValueError):
                    # A damaged lock remains protected for manual recovery.
                    continue
                if isinstance(owner, dict) and owner.get("token") == token:
                    lock.unlink()
                    fsync_directory(lock.parent)


def _archive_selected(paths: RunPaths, attempt_id: str) -> tuple[Path, ...]:
    moves = []
    for root in paths.managed_roots():
        if root.exists():
            archive = root.with_name(f"{root.name}.archive-{attempt_id}")
            if archive.exists() or _is_link(archive):
                raise RunDirectoryError(f"Archive target already exists: {archive}")
            moves.append((root, archive))
    completed = []
    try:
        for root, archive in moves:
            root.rename(archive)
            completed.append((root, archive))
            fsync_directory(root.parent)
    except OSError:
        # Best-effort rollback of multi-root archiving; no old content is deleted.
        for root, archive in reversed(completed):
            if not root.exists():
                archive.rename(root)
                fsync_directory(root.parent)
        raise
    return tuple(archive for _, archive in completed)


@contextmanager
def open_run(
    config: RunConfig,
    *,
    base_dir: str | Path,
    provenance: Mapping,
    operation: RunOperation = "new",
    overwrite_run_id: str | None = None,
    continuation_plan: "ContinuationPlan | None" = None,
) -> Iterator[RunHandle]:
    """Explicit new/compatible-continue/publication-recover/selected-overwrite.

    Readiness, provenance, path safety, and existing files are checked before
    run-specific writes. Hold the context throughout future training/writes.
    Continue/recover must supply a complete plan, preflighted before writes and
    repeated under locks. The plan/service separately restore or repair; entering
    this context itself never restores components or marks completion.
    """
    if operation not in ("new", "continue", "recover", "overwrite"):
        raise RunDirectoryError(f"Unknown operation: {operation!r}")
    config.require_research_ready()
    config = effective_storage_config(config, base_dir)
    paths = resolve_run_paths(config, base_dir)
    expected = identity_metadata(config)
    if operation == "overwrite":
        if overwrite_run_id != expected["run_id"]:
            raise RunDirectoryError("Overwrite requires the exact selected overwrite_run_id")
    elif overwrite_run_id is not None:
        raise RunDirectoryError("overwrite_run_id is only valid for operation='overwrite'")
    if not isinstance(provenance, Mapping) or not {"code", "software"}.issubset(provenance):
        raise RunDirectoryError("Provide actual code/software provenance before initializing a research run")
    current_provenance = software_provenance(code=provenance["code"], software=provenance["software"],
                                             allocation=provenance.get("allocation"))
    # Fail serialization before touching existing runs.
    yaml_text = resolved_config_text(config)
    attempt_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex
    # Preflight even before creating lock files, then repeat under locks to close
    # the race with another cooperating writer between inspection and locking.
    if operation in ("continue", "recover"):
        from .continuation import ContinuationPlan
        if not isinstance(continuation_plan, ContinuationPlan):
            raise RunDirectoryError("Continue/recover requires a complete ContinuationPlan preflight before any run-specific writes")
        continuation_plan.preflight(paths, config, current_provenance, recovery=operation == "recover")
    elif continuation_plan is not None:
        raise RunDirectoryError("A continuation plan is only used with continue/recover")
    _check_existing(paths, config, operation, current_provenance)
    with _run_locks(paths, expected):
        _check_existing(paths, config, operation, current_provenance)
        if operation in ("continue", "recover"):
            continuation_plan.preflight(paths, config, current_provenance, recovery=operation == "recover")
        archives = _archive_selected(paths, attempt_id) if operation == "overwrite" else ()
        for directory in paths.directories():
            durable_mkdir(directory)
        if operation in ("new", "overwrite"):
            atomic_write_json(paths.metadata / "run_metadata.json", {
                **expected, "status": "initialized", "provenance": current_provenance,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            })
        for directory in (paths.checkpoints, paths.results, paths.logs):
            marker = directory / ".run_owner.json"
            if not marker.exists():
                atomic_write_json(marker, expected)
        attempt = paths.metadata / "attempts" / attempt_id
        durable_mkdir(attempt.parent)
        attempt.mkdir(exist_ok=False)
        fsync_directory(attempt.parent)
        atomic_write_text(attempt / "resolved_config.yaml", yaml_text)
        atomic_write_json(attempt / "metadata.json", {
            **expected, "operation": operation, "provenance": current_provenance,
            "directories": paths.to_dict(), "archives": [str(path) for path in archives],
        })
        atomic_write_text(paths.metadata / "resolved_config.yaml", yaml_text)
        lease = _RunLease()
        try:
            yield RunHandle(config=config, paths=paths, operation=operation,
                            attempt_id=attempt_id, archived_directories=archives,
                            provenance=deepcopy(current_provenance), _lease=lease)
        finally:
            lease.active = False
