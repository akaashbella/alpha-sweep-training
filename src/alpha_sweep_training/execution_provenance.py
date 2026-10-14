"""Actual provenance collection, called only by future one-task execution."""

import hashlib
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys

from .data_contract import file_digest
from .provenance import software_provenance
from .serialization import strict_json_text
from .spec import ARCHITECTURES, DATASETS

SOURCE_INVENTORY_VERSION = "checkout-python-and-declarations-v2"
ALLOCATION_ENV = ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_JOB_PARTITION",
    "SLURM_JOB_ACCOUNT", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE", "SLURM_JOB_NODELIST",
    "SLURM_RESTART_COUNT", "CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG")


def source_inventory(source_root) -> dict:
    """Inventory only executable package .py and named contract/declaration files.

    Never recursively scan a checkout for arbitrary files or include output/data.
    Require execution from this checkout (editable install/PYTHONPATH), rather
    than claiming a different installed package matches the inspected tree.
    """
    root = Path(source_root).expanduser().resolve()
    package = root / "src" / "alpha_sweep_training"
    if Path(__file__).resolve() != package / "execution_provenance.py":
        raise ValueError("source_root must be the checkout actually imported; use its editable install/PYTHONPATH")
    files = list(package.rglob("*.py"))
    files += [root / "pyproject.toml", root / "configs" / "research.yaml", root / "configs" / "workflow.yaml",
              root / "docs" / "scientific-specification.md"]
    files += [root / "configs" / "datasets" / f"{name}.yaml" for name in DATASETS]
    files += [root / "configs" / "models" / f"{name}.yaml" for name in ARCHITECTURES]
    result = {}
    for path in sorted(files, key=lambda p: p.relative_to(root).as_posix()):
        if "__pycache__" in path.parts:
            continue
        if not path.is_file() or any(p.is_symlink() or getattr(p, "is_junction", lambda: False)()
                                     for p in (path, *path.parents) if p != root.parent):
            raise ValueError(f"Missing or linked source inventory file: {path}")
        if root not in path.resolve().parents:
            raise ValueError("Source inventory escaped the selected checkout")
        result[path.relative_to(root).as_posix()] = file_digest(path)
    if not result or "src/alpha_sweep_training/training.py" not in result:
        raise ValueError("Incomplete execution source inventory")
    return result


def collect_launch_inputs(policy) -> dict:
    """Pin external launch bytes without executing activation/module commands.

    A venv interpreter may be a symlink; pin both its configured path and its
    resolved plain target. Activation scripts and their ancestors must be plain.
    Module names remain declarations; site module/transitive environment freezing
    and CUDA compatibility still require explicit operator verification.
    """
    from .artifacts import _is_link
    from .hpc_policy import WorkflowError
    policy.require_ready()
    env = policy.to_dict()["environment"]

    def plain(path):
        if not path.is_file() or any(_is_link(p) for p in (path, *path.parents)):
            raise WorkflowError(f"Missing or linked launch input: {path}")
        return {"path": str(path), "sha256": file_digest(path)}

    configured = Path(env["python_executable"])
    try:
        resolved = configured.resolve(strict=True)
    except OSError as exc:
        raise WorkflowError(f"Cannot resolve configured interpreter: {configured}") from exc
    interpreter = plain(resolved)
    interpreter["configured_path"] = str(configured)
    activation = None if env["activation_script"] is None else plain(Path(env["activation_script"]))
    return {"launch_inputs_version": 1, "python": interpreter,
            "activation_script": activation, "modules": list(env["modules"])}


def require_launch_inputs(execution_pin, policy) -> dict:
    """New execution rejects historical unpinned plans; reading remains possible."""
    from .hpc_policy import WorkflowError, exact, same
    if type(execution_pin) is not dict or "launch_inputs" not in execution_pin:
        raise WorkflowError("Historical work has no external launch-input pin; preserve it for reconciliation and create a new reviewed plan/bundle before execution")
    pin = execution_pin["launch_inputs"]
    exact(pin, {"launch_inputs_version", "python", "activation_script", "modules"}, "launch inputs")
    same(pin["launch_inputs_version"], 1, "launch inputs version")
    same(pin, collect_launch_inputs(policy), "fixed planned/current external launch inputs")
    return pin


def collect_provenance(source_root, *, attempt: dict) -> dict:
    """No device probes; runtime properties are subsequently recorded by the engine.

    These subprocesses execute only on the future production execution path.
    Unavailable Git is explicit; versions/digests/checkout association must exist.
    """
    import torch
    import torchvision
    import numpy as np
    import yaml

    root = Path(source_root).expanduser().resolve()
    inventory = source_inventory(root)
    code = {"commit": None, "dirty": None, "inventory_version": SOURCE_INVENTORY_VERSION,
            "inventory": inventory, "source_sha256": hashlib.sha256(strict_json_text({
                "version": SOURCE_INVENTORY_VERSION, "files": inventory}).encode("utf-8")).hexdigest()}
    try:
        def git(*args):
            return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True,
                                  text=True, timeout=30).stdout.strip()
        code["git_root"] = git("rev-parse", "--show-toplevel")
        # Unborn HEAD has no commit, but a verifiable dirty state.
        try:
            code["commit"] = git("rev-parse", "--verify", "HEAD")
        except subprocess.CalledProcessError:
            code["commit"] = None
            code["commit_unavailable"] = "HEAD could not be resolved (for example an unborn repository)"
        code["dirty"] = bool(git("status", "--porcelain=v1", "--untracked-files=normal"))
    except (OSError, subprocess.SubprocessError) as exc:
        code["commit"], code["dirty"] = None, None
        code["git_unavailable"] = f"{type(exc).__name__}: Git state could not be established"
    if source_inventory(root) != inventory:
        raise ValueError("Source changed during provenance collection; do not launch from a changing checkout")
    cudnn = torch.backends.cudnn.version()
    software = {"python": platform.python_version(), "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__), "numpy": str(np.__version__),
        "PyYAML": str(yaml.__version__), "cuda": torch.version.cuda,
        "cudnn": None if cudnn is None else str(cudnn)}
    allocation = {"environment": {key: os.environ[key] for key in ALLOCATION_ENV if key in os.environ},
                  "hostname": socket.gethostname(), "pid": os.getpid(), "executable": sys.executable,
                  "attempt": attempt}
    return software_provenance(code=code, software=software, allocation=allocation)
