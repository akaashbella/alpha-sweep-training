"""Small runtime-component validators; configuration loading stays compute-free."""

import math
from typing import Any

from .config import RunConfig
from .spec import PINNED_TORCH_VERSION, PINNED_TORCHVISION_VERSION


class ComponentError(ValueError):
    """A runtime component cannot honor its declared construction contract."""


def require_config(config: Any, component: str) -> RunConfig:
    if not isinstance(config, RunConfig):
        raise ComponentError(f"{component}: provide a validated RunConfig")
    return config


def finite_number(value: Any, label: str) -> float:
    if type(value) not in (int, float):
        raise ComponentError(f"{label}: require a finite number, not a bool/string")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ComponentError(f"{label}: require a finite number") from exc
    if not math.isfinite(number):
        raise ComponentError(f"{label}: NaN/infinity are forbidden")
    return number


def integer_count(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ComponentError(f"{label}: require an integer >= {minimum}")
    return value


def require_runtime_versions(torch_version: str, vision_version: str | None = None) -> None:
    # CUDA wheel suffixes are operational build identifiers, not another release.
    if torch_version.split("+", 1)[0] != PINNED_TORCH_VERSION:
        raise ComponentError(f"Construction requires pinned torch {PINNED_TORCH_VERSION}; found {torch_version}")
    if vision_version is not None and vision_version.split("+", 1)[0] != PINNED_TORCHVISION_VERSION:
        raise ComponentError(f"Construction requires pinned torchvision {PINNED_TORCHVISION_VERSION}; found {vision_version}")
