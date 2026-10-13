"""Provenance records supplied by later launchers, separate from scientific hash.

This step does not probe the environment or execute Git/project commands. A
launcher must collect the actual software versions and code state at execution.
"""

from copy import deepcopy
from datetime import datetime, timezone
from typing import Mapping

from .serialization import strict_json_text


def software_provenance(
    *,
    code: Mapping,
    software: Mapping,
    allocation: Mapping | None = None,
) -> dict:
    """Validate the minimum future provenance contract without invoking programs.

    code: commit (null for uncommitted code), dirty (bool), source_sha256 (the
    digest of the source snapshot actually executed). software: python, torch,
    torchvision, cuda, cudnn versions; cuda/cudnn can be null on CPU.
    Additional JSON-compatible provenance is preserved.
    """
    if not isinstance(code, Mapping) or not isinstance(software, Mapping):
        raise ValueError("code and software provenance must be mappings")
    if allocation is not None and not isinstance(allocation, Mapping):
        raise ValueError("allocation provenance must be a mapping or null")
    code = deepcopy(dict(code))
    software = deepcopy(dict(software))
    if not {"commit", "dirty", "source_sha256"}.issubset(code):
        raise ValueError("code provenance requires commit, dirty, source_sha256")
    if code["commit"] is not None and (type(code["commit"]) is not str or not code["commit"]):
        raise ValueError("code.commit must be a nonempty string or null")
    if type(code["dirty"]) is not bool:
        raise ValueError("code.dirty must be bool")
    digest = code["source_sha256"]
    if type(digest) is not str or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("code.source_sha256 must be a lowercase SHA-256 digest")
    for name in ("python", "torch", "torchvision", "cuda", "cudnn"):
        if name not in software:
            raise ValueError(f"software provenance requires {name}")
        value = software[name]
        if name in ("cuda", "cudnn") and value is None:
            continue
        if type(value) is not str or not value.strip():
            raise ValueError(f"software.{name} must be a nonempty version string")
    result = {
        "provenance_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "code": code, "software": software,
        "allocation": deepcopy(dict(allocation or {})),
    }
    strict_json_text(result)
    return result
