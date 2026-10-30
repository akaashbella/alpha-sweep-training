"""Explicit diagnostic identities and isolated output guards; no research defaults."""

from copy import deepcopy
import math
from pathlib import Path
import os
import uuid

from .artifacts import _is_link, effective_storage_config, resolve_run_paths
from .data_contract import content_identity
from .identity import scientific_fingerprint
from .serialization import strict_json_text, atomic_write_json, read_json


class DiagnosticError(ValueError):
    pass


CORE_FILES = tuple(f"src/alpha_sweep_training/{name}.py" for name in (
    "training", "perturbation", "models", "optimizers", "scheduler", "data", "runtime",
    "numerical", "randomness", "weights", "evaluation", "evaluation_state", "evaluation_protocol",
    "spec", "config", "aggregation", "metrics", "prepared", "cifar_source", "dataset_artifacts"))


def require_replay_source(audit, review):
    """Diagnostic-only reviewed auxiliary differences; never a continuation waiver."""
    comparison = audit["source_comparison"]
    if comparison["core_file_differences"] != [] or not comparison["software_equal"]:
        raise DiagnosticError("Match original core computation bytes/software before controlled diagnosis")
    differences = comparison["recorded_file_differences"]
    if differences is None or type(review) is not dict or set(review) != {v["path"] for v in differences}:
        raise DiagnosticError("Explicitly review every recorded auxiliary file difference, including packaging/declarations")
    for difference in differences:
        if difference["path"] in CORE_FILES:
            raise DiagnosticError("A source review cannot waive a core computation change")
        entry = review[difference["path"]]
        if (type(entry) is not dict or set(entry) != {"saved", "actual", "reason"}
                or entry["saved"] != difference["saved"] or entry["actual"] != difference["actual"]
                or type(entry["reason"]) is not str or not entry["reason"].strip()):
            raise DiagnosticError("Source review must bind exact saved/actual hashes and a nonempty audit reason")


def mobile_baseline(config):
    config.require_research_ready()
    s = config.scientific
    if (s["dataset"], s["architecture"], s["alpha"]) != ("cifar10", "mobilenetv3_large", 0):
        raise DiagnosticError("This diagnostic requires an original CIFAR-10 MobileNet alpha-zero config")
    return s


def positive(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise DiagnosticError(f"{label} must be an explicit finite positive number")
    return value


def validate_protocol(value, config):
    s = mobile_baseline(config)
    keys = {"protocol_version", "kind", "base_scientific_fingerprint", "arm", "epsilon",
            "learning_rate", "precision_mode", "epochs", "max_batches_per_epoch", "trace_every",
            "activation_modules", "initial_checkpoint", "source_difference_review"}
    if type(value) is not dict or set(value) != keys:
        raise DiagnosticError(f"Diagnostic protocol requires exactly {sorted(keys)}")
    if type(value["protocol_version"]) is not int or value["protocol_version"] != 1 or value["kind"] != "optimization_diagnostic_v1":
        raise DiagnosticError("Unsupported diagnostic protocol; never a main-v1 research configuration")
    if value["base_scientific_fingerprint"] != scientific_fingerprint(config):
        raise DiagnosticError("Protocol must bind the actual original saved scientific configuration")
    if type(value["source_difference_review"]) is not dict:
        raise DiagnosticError("source_difference_review must explicitly list any audited auxiliary hash differences")
    original = (s["optimizer"]["epsilon"], s["optimizer"]["learning_rate"], s["precision"]["mode"])
    actual = (positive(value["epsilon"], "epsilon"), positive(value["learning_rate"], "learning_rate"), value["precision_mode"])
    changed = [i for i, (a, b) in enumerate(zip(original, actual)) if a != b]
    expected = {"baseline": [], "epsilon": [0], "lower_lr": [1], "float32": [2]}
    if value["arm"] not in expected or changed != expected[value["arm"]]:
        raise DiagnosticError("Change exactly the named factor relative to the original saved recipe")
    if value["arm"] == "lower_lr" and not actual[1] < original[1]:
        raise DiagnosticError("lower_lr requires an explicitly smaller base LR")
    if actual[2] not in ("amp_bfloat16", "float32") or (value["arm"] == "float32" and actual[2] != "float32"):
        raise DiagnosticError("Compare original bfloat16 against float32; no implicit precision changes")
    if s["precision"]["mode"] != "amp_bfloat16":
        raise DiagnosticError("The reported instability comparison requires the original bfloat16 policy")
    if value["learning_rate"] < s["training"]["cosine_min_lr"]:
        raise DiagnosticError("Diagnostic base LR cannot be below the unchanged schedule minimum")
    for key, minimum, maximum in (("epochs", 1, 200), ("trace_every", 0, 10**9)):
        if type(value[key]) is not int or not minimum <= value[key] <= maximum:
            raise DiagnosticError(f"Invalid explicit {key}")
    limit = value["max_batches_per_epoch"]
    if limit is not None and (type(limit) is not int or limit < 1):
        raise DiagnosticError("max_batches_per_epoch must be null or positive; truncation is diagnostic only")
    if limit is not None and value["epochs"] != 1:
        raise DiagnosticError("A truncated diagnostic must be one original-epoch prefix; do not advance across omitted batches")
    modules = value["activation_modules"]
    if type(modules) is not list or any(type(m) is not str or not m for m in modules) or len(set(modules)) != len(modules):
        raise DiagnosticError("activation_modules must be unique exact module names (or ['*'])")
    if "*" in modules and modules != ["*"]:
        raise DiagnosticError("Use either ['*'] or exact module names")
    if modules and value["trace_every"] == 0:
        raise DiagnosticError("Activation hooks require explicitly enabled tracing")
    ref = value["initial_checkpoint"]
    if ref is not None:
        from pathlib import PurePosixPath
        if type(ref) is not dict or set(ref) != {"relative_path", "sha256"}:
            raise DiagnosticError("initial_checkpoint requires its exact relative path and SHA-256")
        path, digest = ref["relative_path"], ref["sha256"]
        if (type(path) is not str or PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts
                or ":" in path or "\\" in path or not path.startswith("generations/")
                or type(digest) is not str or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
            raise DiagnosticError("Invalid immutable initial checkpoint reference")
    strict_json_text(value)
    return deepcopy(value)


def diagnostic_identity(protocol):
    return content_identity(protocol).replace("sha256:", "diagnostic-v1-")


def protected_config_roots(config, base_dir, source_root, extras=()):
    cfg = effective_storage_config(config, base_dir)
    roots = [Path(base_dir), Path(source_root), *resolve_run_paths(cfg, base_dir).directories(), *map(Path, extras)]
    for key in ("output_root", "checkpoint_root", "result_root", "log_root", "data_root"):
        value = cfg.execution["storage"][key]
        if value is not None:
            roots.append(Path(value))
    split = cfg.execution["storage"]["split_file"]
    if split is not None:
        roots.append(Path(split).parent)
    return roots


class DiagnosticOutput:
    """An exclusive new directory with immutable ownership checked on every write."""
    def __init__(self, root, owner):
        self.root, self.owner = root, owner

    def guard(self):
        if any(_is_link(p) for p in (self.root, *self.root.parents)):
            raise DiagnosticError("Diagnostic ownership path became linked")
        marker = self.root / "diagnostic_owner.json"
        if _is_link(marker) or not marker.is_file():
            raise DiagnosticError("Missing/plain-file diagnostic owner required")
        if read_json(marker) != self.owner:
            raise DiagnosticError("Diagnostic output ownership changed")

    def __truediv__(self, name):
        self.guard()
        if type(name) is not str or Path(name).name != name or name in (".", "..", "diagnostic_owner.json"):
            raise DiagnosticError("Use a plain diagnostic artifact filename; ownership is immutable")
        target = self.root / name
        if _is_link(target) or (target.exists() and not target.is_file()):
            raise DiagnosticError("Linked/non-file diagnostic artifact target")
        return target

    def __fspath__(self):
        return str(self.root)

    def __str__(self):
        return str(self.root)


def new_output(path, protected):
    """Exclusive new diagnostic directory, disjoint from every protected namespace."""
    target = Path(path).expanduser().absolute()
    if any(_is_link(p) for p in (target, *target.parents)):
        raise DiagnosticError("Linked diagnostic output paths are forbidden")
    markers = (".run_owner.json", "run_metadata.json", "OWNER.json", "owner.json", "diagnostic_owner.json")
    if any((parent / marker).exists() for parent in target.parents for marker in markers):
        raise DiagnosticError("Diagnostic output cannot nest within an existing owned research/report/diagnostic namespace")
    target = target.resolve()
    for root in protected:
        root = Path(root).expanduser().absolute().resolve()
        if target == root or root in target.parents or target in root.parents:
            raise DiagnosticError(f"Diagnostic output overlaps protected storage: {root}")
    if target.exists():
        raise DiagnosticError("Diagnostic output already exists; no overwrite, repair or automatic continuation")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir()  # Exclusive claim; never an open_run or research writer lease.
    owner = {"kind": "isolated_diagnostic_namespace", "version": 1, "token": uuid.uuid4().hex,
             "pid": os.getpid(), "research_authority": False}
    atomic_write_json(target / "diagnostic_owner.json", owner)
    return DiagnosticOutput(target, owner)
