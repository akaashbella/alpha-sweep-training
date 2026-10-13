"""Configuration/artifacts, CIFAR data, and model/optimization source interfaces.

Compute APIs live in data.py, prepare.py, models.py, weights.py, optimizers.py,
and scheduler.py; compute dependencies are not imported by this initializer.
Training, checkpoint files, and experiment execution remain later steps.
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
