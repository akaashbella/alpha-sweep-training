"""Atomic, finite serialization shared by configuration and result writers."""

import json
import os
from pathlib import Path
import tempfile
from typing import Any

from .config import RunConfig


def strict_json_text(value: Any) -> str:
    """Reject nonfinite numbers; failure metrics must be null with explicit status."""
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                      allow_nan=False) + "\n"


def read_json(path: str | Path) -> Any:
    def invalid_constant(token: str):
        raise ValueError(f"{path}: invalid JSON numeric token {token}")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{path}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream, parse_constant=invalid_constant, object_pairs_hook=unique_object)
    # Also catches otherwise valid JSON such as 1e999 that overflows a Python float.
    strict_json_text(value)
    return value


def atomic_write_text(path: str | Path, text: str) -> None:
    """Replace a single file atomically using a same-directory temporary file.

    This low-level writer does not authorize output reuse. Run-specific callers
    must first hold the run lock and pass open_run's artifact checks.
    """
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"Refusing to replace a symbolic link: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        # Directory fsync is supported on POSIX HPC storage, not on Windows.
        if os.name == "posix":
            descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_write_json(path: str | Path, value: Any) -> None:
    atomic_write_text(path, strict_json_text(value))


def resolved_config_text(config: RunConfig) -> str:
    import yaml

    # Reconstruct through the accepted full schema before serializing.
    return yaml.safe_dump(RunConfig(config.to_dict()).to_dict(), sort_keys=False, allow_unicode=True)


def save_resolved_config(path: str | Path, config: RunConfig) -> None:
    atomic_write_text(path, resolved_config_text(config))
