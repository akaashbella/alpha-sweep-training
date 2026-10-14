"""Versioned operational policy, independent of every scientific identity."""

from copy import deepcopy
from dataclasses import dataclass
import re
import math
import os
from pathlib import PurePosixPath

from .config import ConfigError, _read_yaml
from .data_contract import content_identity

POLICY_VERSION = 1
PUBLISHED_POLICY = {"checked_on": "2026-10-13", "partition": "dgxh",
    "source": "https://it.engineering.oregonstate.edu/hpc/slurm-howto",
    "max_wall_hours": 48, "max_gpus": 8, "max_cpus": 224, "max_memory_gb": 2000,
    "gpu_hours": 48, "cpu_hours": 64 * 24, "memory_gb_hours": 500 * 24}
PATH_KEYS = {"checkout", "base_dir", "workflow_root", "slurm_log_root", "evaluation_root",
             "output_root", "checkpoint_root", "result_root", "log_root", "data_root", "split_file"}
LIMIT_KEYS = {"max_wall_hours", "max_gpus", "max_cpus", "max_memory_gb",
              "gpu_hours", "cpu_hours", "memory_gb_hours"}


class WorkflowError(ConfigError):
    pass


def exact(value, keys, label):
    if type(value) is not dict or set(value) != set(keys):
        raise WorkflowError(f"{label}: require exactly {sorted(keys)}")
    return value


def same(actual, expected, label):
    if content_identity(actual) != content_identity(expected):
        raise WorkflowError(f"Incompatible {label}")


def integer(value, label, minimum=1, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise WorkflowError(f"{label}: expected integer in [{minimum}, {maximum if maximum is not None else 'unbounded'}]")
    return value


def text(value, label, *, optional=False):
    if optional and value is None:
        return value
    if type(value) is not str or not value or any(ord(c) < 32 for c in value):
        raise WorkflowError(f"{label}: expected nonempty text without control characters")
    return value


def shared_path(value, label, *, optional=False):
    if value is None and optional:
        return None
    text(value, label)
    parsed = PurePosixPath(value)
    if not parsed.is_absolute() or ".." in parsed.parts or str(parsed) != value or value == "/" or "%" in value:
        raise WorkflowError(f"{label}: supply a canonical absolute shared Linux path")
    return value


def wall_seconds(value):
    text(value, "wall_time")
    match = re.fullmatch(r"(?:(\d+)-)?(\d+):([0-5]\d):([0-5]\d)", value)
    if match is None:
        raise WorkflowError("wall_time: expected [days-]hours:mm:ss")
    days, hours, minutes, seconds = (int(v or 0) for v in match.groups())
    result = days * 86400 + hours * 3600 + minutes * 60 + seconds
    if result <= 0 or result > 48 * 3600:
        raise WorkflowError("dgxh allocation must be positive and at most the published two days")
    return result


@dataclass(frozen=True)
class OperationalPolicy:
    _document: dict

    def __post_init__(self):
        doc = deepcopy(self._document)
        exact(doc, {"policy_version", "resources", "paths", "environment", "signals", "continuation",
                    "commands", "effective_limits"}, "workflow policy")
        same(doc["policy_version"], POLICY_VERSION, "operational policy version")
        resources = exact(doc["resources"], {"partition", "account", "qos", "constraint", "nodes", "gpus",
            "cpus", "memory_gb", "wall_time", "max_concurrent", "num_workers", "native_threads"}, "resources")
        for key, fixed in (("partition", "dgxh"), ("nodes", 1), ("gpus", 1)):
            same(resources[key], fixed, f"resources.{key}")
        for key in ("account", "qos", "constraint"):
            if resources[key] is not None:
                if not re.fullmatch(r"[A-Za-z0-9_.-]+", text(resources[key], key)):
                    raise WorkflowError(f"resources.{key}: use one explicit Slurm identifier/feature")
        integer(resources["cpus"], "cpus", maximum=PUBLISHED_POLICY["max_cpus"])
        integer(resources["memory_gb"], "memory_gb", maximum=PUBLISHED_POLICY["max_memory_gb"])
        integer(resources["max_concurrent"], "max_concurrent", maximum=PUBLISHED_POLICY["max_gpus"])
        integer(resources["num_workers"], "num_workers", minimum=0)
        integer(resources["native_threads"], "native_threads")
        if resources["num_workers"] + resources["native_threads"] > resources["cpus"]:
            raise WorkflowError("num_workers + native_threads must fit cpus (workers use one torch thread)")
        duration = wall_seconds(resources["wall_time"])
        paths = exact(doc["paths"], PATH_KEYS, "shared paths")
        for key, value in paths.items():
            shared_path(value, f"paths.{key}", optional=True)
        operational = [paths[k] for k in ("workflow_root", "slurm_log_root", "evaluation_root") if paths[k] is not None]
        for index, value in enumerate(operational):
            for other in operational[index + 1:]:
                if PurePosixPath(value) == PurePosixPath(other) or PurePosixPath(value) in PurePosixPath(other).parents or PurePosixPath(other) in PurePosixPath(value).parents:
                    raise WorkflowError("Workflow, Slurm logs and evaluation roots must not overlap")
        environment = exact(doc["environment"], {"python_executable", "activation_script", "modules", "variables"}, "environment")
        for key in ("python_executable", "activation_script"):
            shared_path(environment[key], f"environment.{key}", optional=True)
        if type(environment["modules"]) is not list or any(type(m) is not str or not re.fullmatch(r"[A-Za-z0-9_./+-]+", m) for m in environment["modules"]):
            raise WorkflowError("environment.modules: explicit module/version names only")
        variables = environment["variables"]
        if type(variables) is not dict or set(variables) - {"CUBLAS_WORKSPACE_CONFIG"}:
            raise WorkflowError("Only the optional CUBLAS_WORKSPACE_CONFIG environment declaration is supported")
        if "CUBLAS_WORKSPACE_CONFIG" in variables and variables["CUBLAS_WORKSPACE_CONFIG"] not in (":4096:8", ":16:8"):
            raise WorkflowError("Use an approved cuBLAS workspace value; no scientific default is selected")
        signals = exact(doc["signals"], {"lead_seconds", "warning_policy", "termination_policy"}, "signals")
        integer(signals["lead_seconds"], "lead_seconds", maximum=min(65535, duration - 1))
        if signals["warning_policy"] not in ("discard_epoch", "finish_epoch") or signals["termination_policy"] != "discard_epoch":
            raise WorkflowError("Warning supports discard/finish; termination must discard")
        exact(doc["continuation"], {"max_allocations_per_work", "controller_wait_seconds"}, "continuation")
        integer(doc["continuation"]["max_allocations_per_work"], "max_allocations_per_work")
        integer(doc["continuation"]["controller_wait_seconds"], "controller_wait_seconds", maximum=3600)
        commands = exact(doc["commands"], {"sbatch", "squeue", "sacct", "scontrol", "sacctmgr", "timeout_seconds"}, "commands")
        for key in ("sbatch", "squeue", "sacct", "scontrol", "sacctmgr"):
            value = text(commands[key], key)
            if not re.fullmatch(r"[A-Za-z0-9_./+-]+", value):
                raise WorkflowError(f"commands.{key}: one executable, not a shell command")
        integer(commands["timeout_seconds"], "command timeout", maximum=300)
        limits = doc["effective_limits"]
        if limits is not None:
            exact(limits, LIMIT_KEYS | {"max_allocations", "inspection_report_path", "inspection_report_sha256"}, "reviewed account/QOS limits")
            shared_path(limits["inspection_report_path"], "inspection_report_path")
            digest = text(limits["inspection_report_sha256"], "inspection_report_sha256")
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise WorkflowError("inspection_report_sha256 must be a complete SHA-256 digest")
            for key in LIMIT_KEYS:
                if key in ("max_gpus", "max_cpus"):
                    integer(limits[key], f"effective_limits.{key}")
                elif type(limits[key]) not in (int, float) or not math.isfinite(limits[key]) or limits[key] <= 0:
                    raise WorkflowError(f"effective_limits.{key}: require actual finite positive reviewed bound")
            integer(limits["max_allocations"], "effective_limits.max_allocations (reviewed job/submit-count allowance)")
        object.__setattr__(self, "_document", doc)
        resource_budget(self, resources["max_concurrent"])

    def to_dict(self):
        return deepcopy(self._document)

    @property
    def fingerprint(self):
        return content_identity(self._document)[7:]

    def require_ready(self, *, submission=False):
        if os.name != "posix":
            raise WorkflowError("Operational execution requires the shared Linux HPC environment; declarations remain valid on other platforms")
        doc = self.to_dict()
        # Each dataset's finalized manifest supplies its own split. Retain the
        # legacy key for reading sealed policies, but never execute a global
        # split override against the two-dataset grid.
        if doc["paths"]["split_file"] is not None:
            raise WorkflowError("Leave paths.split_file null: the full workflow inherits each dataset's finalized manifest split; a global split override is unsafe. Replan, or explicitly adopt the corrected policy after quiescence.")
        pending = [f"paths.{k}" for k, v in doc["paths"].items() if v is None and k != "split_file"]
        if doc["resources"]["account"] is None:
            pending.append("resources.account (actual Slurm research/class account, never an ONID fallback)")
        if doc["environment"]["python_executable"] is None:
            pending.append("environment.python_executable (existing compatible shared environment)")
        if submission and doc["effective_limits"] is None:
            pending.append("effective_limits (inspect-restrictions, then review actual account/QOS restrictions and report digest)")
        if pending:
            raise WorkflowError("Finalize operational inputs: " + ", ".join(pending))

    def execution_overrides(self):
        self.require_ready()
        doc = self.to_dict()
        resources, paths = doc["resources"], doc["paths"]
        return {"execution.device": "cuda:0", "execution.num_workers": resources["num_workers"],
            **{f"execution.storage.{k}": paths[k] for k in ("output_root", "checkpoint_root", "result_root", "log_root", "data_root")},
            **{f"execution.slurm.{k}": resources[k] for k in ("partition", "account", "gpus", "cpus", "memory_gb", "wall_time", "max_concurrent")}}


def load_policy(path):
    return OperationalPolicy(_read_yaml(path))


def resource_budget(policy, allocations):
    """Full requested overlap, including pending/unknown allocations, never measured use."""
    doc = policy.to_dict()
    integer(allocations, "managed overlapping allocations", minimum=0)
    r = doc["resources"]
    if allocations > r["max_concurrent"]:
        raise WorkflowError("Shared workflow concurrency cap exceeded")
    hours = wall_seconds(r["wall_time"]) / 3600
    requested = {"max_wall_hours": hours if allocations else 0,
        "max_gpus": allocations, "max_cpus": allocations * r["cpus"], "max_memory_gb": allocations * r["memory_gb"],
        "gpu_hours": allocations * hours, "cpu_hours": allocations * r["cpus"] * hours,
        "memory_gb_hours": allocations * r["memory_gb"] * hours}
    effective = doc["effective_limits"]
    if effective and allocations > effective["max_allocations"]:
        raise WorkflowError("Combined requested allocations exceed reviewed account/QOS job/submit-count allowance")
    allowed = {k: min(PUBLISHED_POLICY[k], effective[k]) if effective else PUBLISHED_POLICY[k] for k in LIMIT_KEYS}
    rejected = [f"{k}: requested {v}, allowed {allowed[k]}" for k, v in requested.items() if v > allowed[k]]
    if rejected:
        raise WorkflowError("Combined allocation policy rejected: " + "; ".join(rejected))
    return {"allocations": allocations, "requested_overlap": requested, "limits": allowed,
            "published_policy": deepcopy(PUBLISHED_POLICY), "effective_restrictions_reviewed": effective is not None,
            "unmeasured_planning_requests": True, "unrelated_jobs_controlled": False}
