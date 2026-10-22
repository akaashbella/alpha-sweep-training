"""Versioned CIFAR artifact and preprocessing contracts, without compute imports."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal, Mapping

from .spec import DATA_RANDOMNESS_VERSION, PREPROCESSING_VERSION
from .serialization import read_json

SOURCE_VERSION = 1
SPLIT_VERSION = 1
NORMALIZATION_VERSION = 1
PREPARATION_VERSION = 1
CONTINUATION_VERSION = 1
SPLIT_ALGORITHM = "numpy-pcg64-permutation-v1"
ARTIFACT_FOLDER = ".alpha_sweep_prepared"
PREPROCESSING = {
    "version": PREPROCESSING_VERSION,
    "input": "RGB_uint8_HWC_32x32",
    "tensor": "CPU_float32_CHW_uint8_div_255",
    "training": {"crop_size": 32, "padding": 4, "padding_mode": "constant",
                 "padding_value": 0, "horizontal_flip_probability": 0.5},
    "evaluation": {"random_augmentation": False, "resize": False},
    "normalization": "per_RGB_channel_same_constants_all_partitions",
    "randomness_version": DATA_RANDOMNESS_VERSION,
}


class DatasetArtifactError(ValueError):
    """Missing, incomplete, corrupted, or incompatible shared CIFAR artifacts."""


def content_identity(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    value = digest.hexdigest()
    from .planning_inspection import record_dependency
    record_dependency(path, value)
    return value


def read_artifact_json(path: Path) -> Any:
    try:
        return read_json(path)
    except (OSError, ValueError) as exc:
        raise DatasetArtifactError(f"Invalid or unreadable preparation artifact {path}: {exc}") from exc


def require_keys(record: Any, keys: set[str], label: str) -> dict:
    if type(record) is not dict or set(record) != keys:
        raise DatasetArtifactError(f"{label}: expected exactly {sorted(keys)}")
    return record


def require_version(value: Any, expected: int, label: str) -> None:
    if type(value) is not int or value != expected:
        raise DatasetArtifactError(f"{label}: unsupported format version {value!r}; expected {expected}")


def require_seed(value: Any, label: str) -> int:
    if type(value) is not int or not 0 <= value <= 2**32 - 1:
        raise DatasetArtifactError(f"{label}: supply an explicit unsigned 32-bit integer")
    return value


def rgb_values(values: Any, label: str, *, positive: bool = False) -> list[float]:
    if not isinstance(values, (tuple, list)) or len(values) != 3:
        raise DatasetArtifactError(f"{label}: supply exactly three RGB numbers")
    result = []
    for value in values:
        if type(value) not in (int, float):
            raise DatasetArtifactError(f"{label}: values must be finite numbers")
        try:
            number = float(value)
        except (ValueError, OverflowError) as exc:
            raise DatasetArtifactError(f"{label}: values must be finite numbers") from exc
        if not math.isfinite(number) or (positive and number <= 0):
            raise DatasetArtifactError(f"{label}: values must be finite" + (" and positive" if positive else ""))
        result.append(0.0 if number == 0 else number)
    return result


def require_provenance(value: Any, label: str) -> None:
    if not isinstance(value, Mapping) or not isinstance(value.get("software"), Mapping):
        raise DatasetArtifactError(f"{label}: missing preparation software provenance")
    require_version(value.get("format_version"), 1, f"{label} provenance")
    for name in ("python", "torch", "torchvision", "numpy"):
        version = value["software"].get(name)
        if type(version) is not str or not version.strip():
            raise DatasetArtifactError(f"{label}: missing {name} version")
    implementation = value.get("implementation_sha256")
    if (type(implementation) is not dict or not implementation
            or any(type(name) is not str or type(digest) is not str or len(digest) != 64
                   or any(character not in "0123456789abcdef" for character in digest)
                   for name, digest in implementation.items())):
        raise DatasetArtifactError(f"{label}: missing or invalid preparation implementation digests")


@dataclass(frozen=True)
class NormalizationRequest:
    policy: Literal["explicit", "derive_training"]
    mean: tuple[float, float, float] | None = None
    std: tuple[float, float, float] | None = None
    source: str | None = None

    def validate(self) -> None:
        if self.policy == "explicit":
            rgb_values(self.mean, "normalization mean")
            rgb_values(self.std, "normalization std", positive=True)
            if type(self.source) is not str or not self.source.strip():
                raise DatasetArtifactError("Explicit normalization requires a documented --normalization-source")
        elif self.policy == "derive_training":
            if any(value is not None for value in (self.mean, self.std, self.source)):
                raise DatasetArtifactError("Training-derived normalization cannot also supply constants or a source")
        else:
            raise DatasetArtifactError("Explicitly select normalization policy 'explicit' or 'derive_training'")
