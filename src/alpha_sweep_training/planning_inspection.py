"""Operation-local read reuse, never persistent scientific authority.

Every reuse checks content, directory membership, ancestor identities and locks.
Timestamps are change hints, never proof of unchanged content. Instrumented readers
track validated dependencies; other readers conservatively hash their whole scope.
Operation-wide entries are tensor-free. This is not a reader lease/grid snapshot.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import hashlib
import stat
import sys


class InspectionChanged(RuntimeError):
    pass


_ACTIVE = ContextVar("planning_inspection", default=None)
_READS = ContextVar("planning_dependency_reads", default=None)
_CONTROLS = {"HEAD.json", "SWEEP_HEAD.json", "run_metadata.json", "resolved_config.yaml",
             ".run_owner.json", "owner.json"}


def _control(path):
    return path.name in _CONTROLS or path.name.endswith(".lock")


def _content_digest(path, metrics, prefix="fingerprint"):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise InspectionChanged(f"Missing/plain-file dependency required: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            if metrics is not None:
                metrics[f"{prefix}_read_bytes"] += len(block)
    if metrics is not None:
        metrics[f"{prefix}_files_hashed"] += 1
    return digest.hexdigest()


def record_dependency(path, digest=None, *, content=None):
    """Bind a validated read to its bytes; a supplied digest must be from that read.

    Outside cache validation this is a no-op. Existing hash readers supply their
    digest or exact bytes without another read. Other readers take a conservative probe.
    Nested cache hits propagate their freshly checked dependencies to the caller.
    """
    reads = _READS.get()
    if reads is None:
        return
    dependencies, metrics = reads
    path = Path(path).absolute()
    if digest is None:
        digest = (hashlib.sha256(content).hexdigest() if content is not None
                  else _content_digest(path, metrics, "dependency_probe"))
    if path in dependencies and dependencies[path] != digest:
        raise InspectionChanged(f"Dependency changed between validated reads: {path}")
    dependencies[path] = digest


def active_inspection():
    return _ACTIVE.get()


def compatibility_key(provenance):
    """Eligibility compares source/software, independent of inspection timestamps.

    Still validate the complete supplied provenance before considering reuse.
    Actual persisted provenance is never changed or replaced with this key.
    """
    if provenance is None:
        return None
    from .provenance import software_provenance
    from .serialization import strict_json_text
    software_provenance(code=provenance["code"], software=provenance["software"],
                        allocation=provenance.get("allocation"))
    return (provenance["code"]["source_sha256"], strict_json_text(provenance["software"]))


def training_paths(paths):
    roots = tuple(paths.managed_roots())
    return (*roots, *(root.parent / f".{root.name}.lock" for root in roots))


def evidence_stamp(paths, *, content_paths=None, digests=None, metrics=None):
    """Content proof plus structural evidence, independent of timestamp precision.

    Default: hash every regular file. Tracked callers pass their validated read set
    instead, so unrelated checkpoint history receives structure/type checks only.
    Control/ownership files are always hashed. Child/ancestor links are not followed.
    A digest map may be shared within one final guard, never across reuse checks.
    """
    rows = []
    content_paths = None if content_paths is None else set(content_paths)
    digests = {} if digests is None else digests

    def visit(path, descend):
        try:
            info = path.lstat()
        except FileNotFoundError:
            rows.append((str(path), None))
            return False
        attributes = getattr(info, "st_file_attributes", 0)
        linked = stat.S_ISLNK(info.st_mode) or bool(attributes & 0x400)
        identity = (info.st_dev, info.st_ino, info.st_mode, attributes)
        token = (*identity, info.st_size, info.st_mtime_ns, info.st_ctime_ns) if descend else identity
        rows.append((str(path), token))
        if (descend and stat.S_ISREG(info.st_mode) and not linked
                and (content_paths is None or path in content_paths or _control(path))):
            if path not in digests:
                digests[path] = _content_digest(path, metrics)
            rows.append((str(path), "content", digests[path]))
        if descend and stat.S_ISDIR(info.st_mode) and not linked:
            for child in sorted(path.iterdir()):
                visit(child, True)
        return linked

    for path in sorted(set(Path(p).absolute() for p in paths)):
        blocked = False
        for parent in reversed(path.parents):
            if visit(parent, False):
                blocked = True
                break
        if blocked:
            rows.append((str(path), "linked-ancestor"))
        else:
            visit(path, True)
    return tuple(rows)


def _contents(stamp):
    return {Path(row[0]): row[2] for row in stamp if len(row) == 3 and row[1] == "content"}


def _structure_and_controls(stamp):
    return tuple(row for row in stamp if len(row) != 3 or _control(Path(row[0])))


def stderr_progress(message):
    print(f"[workflow] {message}", file=sys.stderr, flush=True)


class PlanningInspection:
    """Discard after one CLI operation; no serialized cache or cross-process reuse."""

    def __init__(self, progress=None):
        self.progress = progress or (lambda message: None)
        self._entries = {}
        self.hits = 0
        self.validations = 0
        self.read_metrics = {"fingerprint_read_bytes": 0, "fingerprint_files_hashed": 0,
                             "dependency_probe_read_bytes": 0, "dependency_probe_files_hashed": 0}

    @contextmanager
    def activate(self):
        token = _ACTIVE.set(self)
        try:
            yield self
        finally:
            _ACTIVE.reset(token)

    def reuse(self, key, paths, validate, *, track_dependencies=False):
        """Tracking is opt-in only for validators whose content readers register reads.

        Otherwise hash all files conservatively. Neither mode trusts unchanged stat
        values. A fresh content/structure proof brackets the successful validation.
        """
        paths = tuple(Path(path).absolute() for path in paths)
        entry = self._entries.get(key)
        scope = entry[3] if entry is not None and track_dependencies else (() if track_dependencies else None)
        before = evidence_stamp(paths, content_paths=scope, metrics=self.read_metrics)
        if entry is not None and entry[0] == paths and entry[4] == track_dependencies and entry[1] == before:
            self.hits += 1
            for path, digest in entry[3].items():
                record_dependency(path, digest)
            return entry[2]
        # Drop old authority even if its replacement fails validation.
        self._entries.pop(key, None)
        self.validations += 1
        dependencies = {}
        token = _READS.set((dependencies, self.read_metrics))
        try:
            result = validate()
        finally:
            _READS.reset(token)
        if any(not any(path == root or root in path.parents for root in paths) for path in dependencies):
            raise InspectionChanged("Validated dependency escaped the declared observation scope; expand scope before reuse")
        after = evidence_stamp(paths, content_paths=dependencies if track_dependencies else None,
                               metrics=self.read_metrics)
        contents = _contents(after)
        stable = (_structure_and_controls(before) == _structure_and_controls(after)
                  if track_dependencies else before == after)
        if not stable or any(contents.get(path) != digest for path, digest in dependencies.items()):
            raise InspectionChanged("Authority/artifacts/ownership changed during validation; replan after writers exit")
        self._entries[key] = (paths, after, result, contents, track_dependencies)
        for path, digest in contents.items():
            record_dependency(path, digest)
        return result

    def assert_unchanged(self):
        """Fresh final guard immediately before reserving a submission intent."""
        stamps = {}
        digests = {}  # One fresh read per shared dependency in this guard only.
        for paths, expected, _, dependencies, tracked in self._entries.values():
            key = (paths, frozenset(dependencies) if tracked else None)
            if key not in stamps:
                stamps[key] = evidence_stamp(paths, content_paths=dependencies if tracked else None,
                                             digests=digests, metrics=self.read_metrics)
            if stamps[key] != expected:
                raise InspectionChanged("Planning evidence changed before submission intent; replan (no submission reserved)")
