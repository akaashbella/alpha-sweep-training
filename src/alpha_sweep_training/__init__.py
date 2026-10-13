"""Configuration/artifact foundations and full CIFAR dataset implementation.

Dataset APIs live in data.py and prepare.py; compute dependencies are not imported
by this initializer. Models, training, and experiment execution are later steps.
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
