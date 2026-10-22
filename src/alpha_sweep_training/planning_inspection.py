"""Operation-local read reuse, never persistent scientific authority.

Only fully validated, stable reads enter the cache. Every reuse checks all files,
directories, ancestor identities and ownership locks again. Operation-wide entries
are tensor-free; temporary checkpoint caches are discarded after each deep read.
Requires coherent filesystem stat/change-time observations, as do immutable-file
publication and the existing read-only observation contract; this is not a lease.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import hashlib
import os
import stat
import sys


class InspectionChanged(RuntimeError):
    pass


_ACTIVE = ContextVar("planning_inspection", default=None)


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


def evidence_stamp(paths):
    """Metadata change detectors for already content-validated immutable files.

    Include ctime so an in-place edit with restored mtime/size invalidates reuse;
    include device/inode for replacements, reparse flags for Windows links, all
    descendants for uncommitted failure evidence, and parents for redirected roots.
    Missing evidence is a token, never a successful validation or an entitlement.
    Symlinks/junctions are recorded without following them.
    """
    rows = []

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
        # Windows ctime is creation time on supported Python versions, not a
        # reliable content-change time. Conservatively hash there; production
        # POSIX Slurm storage uses stat change-time and avoids these rereads.
        authority = path.name in {"HEAD.json", "SWEEP_HEAD.json", "run_metadata.json",
                                 "resolved_config.yaml", ".run_owner.json", "owner.json"} or path.name.endswith(".lock")
        if descend and (os.name == "nt" or authority) and stat.S_ISREG(info.st_mode) and not linked:
            with path.open("rb") as stream:
                digest = hashlib.sha256()
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            token = (*token, digest.hexdigest())
        rows.append((str(path), token))
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


def stderr_progress(message):
    print(f"[workflow] {message}", file=sys.stderr, flush=True)


class PlanningInspection:
    """Discard after one CLI operation; no serialized cache or cross-process reuse."""

    def __init__(self, progress=None):
        self.progress = progress or (lambda message: None)
        self._entries = {}
        self.hits = 0
        self.validations = 0

    @contextmanager
    def activate(self):
        token = _ACTIVE.set(self)
        try:
            yield self
        finally:
            _ACTIVE.reset(token)

    def reuse(self, key, paths, validate):
        paths = tuple(paths)
        before = evidence_stamp(paths)
        entry = self._entries.get(key)
        if entry is not None and entry[0] == paths and entry[1] == before:
            self.hits += 1
            return entry[2]
        # Drop old authority even if its replacement fails validation.
        self._entries.pop(key, None)
        self.validations += 1
        result = validate()
        if before != evidence_stamp(paths):
            raise InspectionChanged("Authority/artifacts/ownership changed during validation; replan after writers exit")
        self._entries[key] = (paths, before, result)
        return result

    def assert_unchanged(self):
        """Fresh final guard immediately before reserving a submission intent."""
        stamps = {}
        for paths, expected, _ in self._entries.values():
            if paths not in stamps:
                stamps[paths] = evidence_stamp(paths)
            if stamps[paths] != expected:
                raise InspectionChanged("Planning evidence changed before submission intent; replan (no submission reserved)")
