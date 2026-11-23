"""Sealed records, exclusive publication and process-scoped checkpoint leases."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile

from alpha_sweep_training.serialization import durable_mkdir, fsync_directory, read_json, strict_json_text

from . import VERSION

CHECKOUT = Path(__file__).resolve().parents[2]


class Stage2Error(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise Stage2Error(message)


def identity(document):
    encoded = json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def seal(kind, contents):
    record = {"stage2_version": VERSION, "kind": kind, **contents}
    return {**record, "document_sha256": identity(record)}


def plain(value):
    path = Path(value).expanduser().absolute()
    for part in (path, *path.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        require(not stat.S_ISLNK(info.st_mode) and not
                (getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)),
                f"Linked storage path: {part}")
    return path.resolve()


def disjoint(output, protected, *, existing=False):
    path = plain(output)
    for item in (CHECKOUT, *protected):
        root = plain(item)
        require(root != path and root not in path.parents and path not in root.parents,
                f"Output overlaps protected storage: {root}")
    require(not path.exists() or (existing and path.is_dir()), f"Output already exists: {path}")
    return path


def sha_file(path):
    digest = hashlib.sha256()
    with plain(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path, kind):
    document = read_json(plain(path))
    require(type(document) is dict and type(document.get("stage2_version")) is int and
            document["stage2_version"] == VERSION and document.get("kind") == kind,
            f"Incompatible {kind} record: {path}")
    require(document.get("document_sha256") == identity({k: v for k, v in document.items() if k != "document_sha256"}),
            f"Record checksum mismatch: {path}")
    return document


def install(path, document):
    """Publish a whole fsynced record exactly once, using a same-directory hard link."""
    path = plain(path)
    durable_mkdir(path.parent)
    fd, staged = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".staging")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(strict_json_text(document))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(staged, path)
        fsync_directory(path.parent)
    finally:
        Path(staged).unlink(missing_ok=True)


def finish(root, kind, files):
    refs = {Path(name).as_posix(): sha_file(root / name) for name in sorted(files)}
    install(root / "COMPLETE.json", seal("publication", {"publication_kind": kind, "files": refs}))


def published(root, kind, primary):
    root = plain(root)
    marker = read(root / "COMPLETE.json", "publication")
    require(marker["publication_kind"] == kind and primary in marker["files"], f"Invalid {kind} publication marker")
    for name, digest in marker["files"].items():
        path = plain(root / name)
        require(root in path.parents and path.relative_to(root).as_posix() == name and sha_file(path) == digest,
                f"Publication file mismatch: {name}")
    return read(root / primary, kind)


def publish_document(output, protected, kind, document):
    root = disjoint(output, protected)
    root.mkdir(parents=True)
    install(root / f"{kind}.json", document)
    finish(root, kind, [f"{kind}.json"])


@contextmanager
def lease(path):
    """OS releases the advisory lock if a worker exits. Keep the inode in place."""
    path = plain(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt

            def lock():
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)

            def unlock():
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            def lock():
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)

            def unlock():
                fcntl.flock(stream, fcntl.LOCK_UN)
        try:
            lock()
        except OSError as exc:
            raise Stage2Error(f"Checkpoint lease unavailable: {path}") from exc
        try:
            yield
        finally:
            unlock()


def source_digest():
    from alpha_sweep_training.execution_provenance import source_inventory

    files = source_inventory(CHECKOUT)
    files = {"experiment": files, "stage2": {p.name: sha_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))}}
    return identity(files)
