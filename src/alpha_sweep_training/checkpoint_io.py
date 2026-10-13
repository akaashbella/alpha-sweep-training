"""Storage-relative references and durable immutable writes under run leases."""

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path, PurePosixPath
from typing import Literal

from .artifacts import RunHandle, RunPaths, _is_link
from .serialization import atomic_write_json, atomic_write_text, fsync_directory, read_json
from .training_contracts import PersistenceContractError, _digest

TRANSACTION_VERSION = 2  # v1 combined clean-test completion remains read-only compatible.
CHECKPOINT_FORMAT_VERSION = 1


def fail(message: str) -> None:
    raise PersistenceContractError(message)


def exact(value: object, keys: set[str], label: str) -> dict:
    if type(value) is not dict or set(value) != keys:
        fail(f"Invalid {label} fields")
    return value


def same(actual: object, expected: object, label: str) -> None:
    # Finite canonical JSON distinguishes bools from numbers, and permits saved
    # tuple/list metadata equivalence without relying on tensor equality.
    from .serialization import strict_json_text
    if strict_json_text(actual) != strict_json_text(expected):
        fail(f"Incompatible {label}")


def digest_file(path: Path) -> str:
    if _is_link(path) or not path.is_file():
        fail(f"Missing or linked referenced file: {path}")
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_file(root: Path, relative: str, *, create_parents: bool = False) -> Path:
    if type(relative) is not str or not relative:
        fail("Missing storage-relative file path")
    parsed = PurePosixPath(relative)
    if (parsed.is_absolute() or ".." in parsed.parts or ":" in relative or "\\" in relative
            or str(parsed) != relative or parsed == PurePosixPath(".")):
        fail("References must be canonical relative POSIX paths within their selected root")
    current = root
    if _is_link(current) or not current.is_dir():
        fail(f"Missing/linked guarded storage root: {root}")
    for part in parsed.parts[:-1]:
        child = current / part
        if _is_link(child) or (child.exists() and not child.is_dir()):
            fail(f"Linked/non-directory storage component: {child}")
        if create_parents and not child.exists():
            child.mkdir()
            fsync_directory(current)
        current = child
    target = current / parsed.name
    if _is_link(target):
        fail(f"Linked referenced file: {target}")
    return target


@dataclass(frozen=True)
class ArtifactReference:
    root: Literal["metadata", "checkpoints", "results"]
    relative_path: str
    sha256: str

    def to_dict(self) -> dict:
        return asdict(self)

    def path(self, paths: RunPaths) -> Path:
        if self.root not in ("metadata", "checkpoints", "results"):
            fail("Unknown reference storage root")
        _digest(self.sha256)
        return relative_file(getattr(paths, self.root), self.relative_path)

    def verify(self, paths: RunPaths) -> Path:
        path = self.path(paths)
        if digest_file(path) != self.sha256:
            fail(f"Content digest mismatch: {path}")
        return path

    @classmethod
    def from_dict(cls, value: dict) -> "ArtifactReference":
        return cls(**exact(value, {"root", "relative_path", "sha256"}, "artifact reference"))


def read_document(paths: RunPaths, reference: ArtifactReference) -> dict:
    value = read_json(reference.verify(paths))
    if type(value) is not dict:
        fail("Referenced document must be a JSON object")
    return value


def write_document(run: RunHandle, root: str, relative: str, value: dict) -> ArtifactReference:
    run.require_active()
    paths = run.paths
    path = relative_file(getattr(paths, root), relative, create_parents=True)
    if path.exists():
        fail(f"Immutable generation already exists: {path}")
    atomic_write_json(path, value)
    return ArtifactReference(root, relative, digest_file(path))


def write_text(run: RunHandle, root: str, relative: str, value: str) -> ArtifactReference:
    run.require_active()
    paths = run.paths
    path = relative_file(getattr(paths, root), relative, create_parents=True)
    if path.exists():
        fail(f"Immutable generation already exists: {path}")
    atomic_write_text(path, value)
    return ArtifactReference(root, relative, digest_file(path))
