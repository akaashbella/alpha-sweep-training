"""Human-readable run names and independent scientific compatibility hashes."""

from decimal import Decimal
import hashlib
import json
import math

from .config import ConfigError, RunConfig
from .spec import FINGERPRINT_VERSION, SCHEMA_VERSION


def canonical_alpha(alpha: int | float) -> str:
    """Shortest float representation, expanded as decimal without rounding.

    Equivalent numeric inputs (including signed zero) have the same name. This
    does not round distinct floating-point values into arbitrary decimal bins.
    The configuration validator separately restricts the main sweep to its grid.
    """
    if type(alpha) not in (int, float):
        raise ConfigError("alpha must be numeric, not a bool or string")
    if type(alpha) is float and not math.isfinite(alpha):
        raise ConfigError("alpha must be finite")
    if alpha < 0:
        raise ConfigError("alpha must be nonnegative")
    if alpha == 0:
        return "0"
    value = format(Decimal(str(alpha)), "f")
    return value.rstrip("0").rstrip(".") if "." in value else value


def run_identity(config: RunConfig) -> str:
    """A research run is one dataset/architecture/alpha/training-seed tuple."""
    config.require_research_ready()
    science = config.scientific
    return (
        f"{science['dataset']}__{science['architecture']}"
        f"__alpha-{canonical_alpha(science['alpha'])}__seed-{science['training_seed']}"
    )


def scientific_fingerprint(config: RunConfig) -> str:
    """Hash all effective scientific fields; omit execution and code provenance."""
    config.require_research_ready()
    payload = {
        "fingerprint_version": FINGERPRINT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "scientific": config.scientific,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def identity_metadata(config: RunConfig) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "specification_version": config.scientific["specification_version"],
        "fingerprint_version": FINGERPRINT_VERSION,
        "run_id": run_identity(config),
        "scientific_fingerprint": scientific_fingerprint(config),
    }


def training_alpha(config: RunConfig) -> float:
    """Explicit adapter: schema-4 scientific.alpha always means TRAINING alpha.

    The historical evaluation block describes clean validation/checkpoint
    selection and the mandatory clean baseline, not the independent sweep.
    Preserve its bytes and fingerprint version 4; protocol identity is separate.
    """
    return config.scientific["alpha"]
