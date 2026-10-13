"""Scientific configuration and artifact foundations for the alpha sweep.

Training, datasets, model builders, and experiment execution belong to later steps.
"""

from .config import ConfigError, RunConfig, UnresolvedConfigError, load_config
from .identity import canonical_alpha, run_identity, scientific_fingerprint

__all__ = [
    "ConfigError",
    "RunConfig",
    "UnresolvedConfigError",
    "load_config",
    "canonical_alpha",
    "run_identity",
    "scientific_fingerprint",
]
