"""Configuration/artifacts, CIFAR data, model/optimization and training sources.

Compute APIs live in data.py, prepare.py, models.py, weights.py, optimizers.py,
and scheduler.py, plus training.py and its runtime/evaluation/record contracts.
Checkpoint/persistence/continuation/recovery APIs live in their own modules.
Compute dependencies are not imported by this initializer. Production manifests,
task/evaluation execution, workflow orchestration and reporting have separate
entry-point modules; all nine steps are implemented in source, runtime unverified.
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
