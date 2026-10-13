"""Explicit CUDA precision/backend policy; no construction or probes on import."""

from copy import deepcopy
from dataclasses import dataclass
import os
from typing import ContextManager

import torch

from .component_contracts import ComponentError, require_config, require_runtime_versions
from .config import RunConfig
from .numerical import NumericalFailure, require_finite
from .spec import BATCH_SIZE, TRAINING_ENGINE_VERSION


def _backend_settings() -> dict:
    return {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


@dataclass
class RuntimePolicy:
    device: torch.device
    precision: dict
    effective_backend: dict
    device_metadata: dict
    scaler: torch.amp.GradScaler | None

    def autocast(self) -> ContextManager:
        mode = self.precision["mode"]
        return torch.autocast(device_type="cuda", enabled=mode != "float32",
                              dtype=torch.float16 if mode == "amp_float16" else torch.bfloat16)

    def require_active(self, config: RunConfig | None = None) -> None:
        if config is not None and config.scientific["precision"] != self.precision:
            raise ComponentError("Runtime precision differs from RunConfig")
        if torch.cuda.current_device() != self.device.index or _backend_settings() != self.effective_backend:
            raise ComponentError("Assigned CUDA device or precision/backend settings changed during this run")
        if self.precision["mode"] == "amp_float16":
            if self.scaler is None or not self.scaler.is_enabled():
                raise ComponentError("amp_float16 requires the enabled pinned CUDA GradScaler")
            state = self.scaler.state_dict()
            if (state["growth_factor"] != 2.0 or state["backoff_factor"] != 0.5
                    or state["growth_interval"] != 2000):
                raise ComponentError("Preserve the pinned native GradScaler growth/backoff policy")
            if type(state["_growth_tracker"]) is not int or not 0 <= state["_growth_tracker"] < 2000:
                raise ComponentError("GradScaler growth tracker is incompatible with its native growth interval")
            scale = state["scale"]
            if not isinstance(scale, (int, float)) or not 0 < scale < float("inf"):
                raise NumericalFailure("grad_scaler", "Gradient scale must remain finite and positive")
        elif self.scaler is not None:
            raise ComponentError("Float32 and bfloat16 policies do not use gradient scaling")

    def metadata(self) -> dict:
        self.require_active()
        return {"engine_version": TRAINING_ENGINE_VERSION, "device": str(self.device),
                "device_properties": deepcopy(self.device_metadata), "precision": deepcopy(self.precision),
                "effective_backend": deepcopy(self.effective_backend), "master_dtype": "float32",
                "evaluation_precision": "same_autocast_policy_as_training; no_scaling",
                "grad_scaler": None if self.scaler is None else {
                    "implementation": "torch.amp.GradScaler", "initial_scale": 65536.0,
                    "growth_factor": 2.0, "backoff_factor": 0.5, "growth_interval": 2000}}

    def batch(self, batch: object, num_classes: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(batch, (tuple, list)) or len(batch) != 2:
            raise ComponentError("Expected the prepared loader's (images, labels) batch")
        images, labels = batch
        if (not isinstance(images, torch.Tensor) or not isinstance(labels, torch.Tensor)
                or images.device.type != "cpu" or labels.device.type != "cpu"
                or images.dtype != torch.float32 or labels.dtype != torch.int64
                or images.ndim != 4 or tuple(images.shape[1:]) != (3, 32, 32)
                or not 1 <= images.shape[0] <= BATCH_SIZE or labels.shape != (images.shape[0],)):
            raise ComponentError("Expected CPU float32 NCHW CIFAR images and int64 labels with the actual batch size")
        require_finite((("input_images", images),), "input")
        if bool(((labels < 0) | (labels >= num_classes)).any()):
            raise ComponentError("Batch labels are outside the dataset classifier range")
        return images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)


def build_runtime(config: RunConfig) -> RuntimePolicy:
    """Apply a resolved policy once in the owning training process, at runtime.

    This does not seed RNGs, construct models, select a precision default, or
    fall back to CPU. Build/move the model before building its optimizer.
    """
    config = require_config(config, "CUDA runtime")
    precision = config.scientific["precision"]
    pending = [f"scientific.precision.{key}" for key, value in precision.items() if value is None]
    if pending:
        raise ComponentError("Runtime requires explicit decisions for " + ", ".join(pending))
    require_runtime_versions(torch.__version__)
    expected_tf32 = precision["float32_matmul_precision"] != "highest"
    if precision["allow_tf32"] != expected_tf32:
        raise ComponentError("Pinned CUDA matmul requires highest with allow_tf32=False, or high/medium with allow_tf32=True; finalize a consistent policy")
    if (precision["deterministic_algorithms"]
            and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8")):
        raise ComponentError("Deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8 set before launching Python; no late environment repair is performed")
    try:
        requested = torch.device(config.execution["device"])
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ComponentError("execution.device must name the assigned CUDA device") from exc
    if requested.type != "cuda":
        raise ComponentError("The main experiment requires CUDA; CPU fallback is forbidden")
    if torch.version.cuda is None or not torch.cuda.is_available():
        raise ComponentError("Requested CUDA runtime is unavailable; use a compatible CUDA build and assigned GPU")
    try:
        index = torch.cuda.current_device() if requested.index is None else requested.index
        if not 0 <= index < torch.cuda.device_count():
            raise ComponentError("Requested CUDA index is outside the visible assigned devices")
        device = torch.device("cuda", index)
        torch.cuda.set_device(device)
        properties = torch.cuda.get_device_properties(device)
        if precision["mode"] == "amp_bfloat16" and not torch.cuda.is_bf16_supported(including_emulation=False):
            raise ComponentError("amp_bfloat16 requires native bfloat16 support on the assigned CUDA device")
        # Inspect advertised cubin/PTX targets without launching a kernel. This
        # preflight is not evidence of driver/kernel runtime correctness.
        arches = torch.cuda.get_arch_list()
        cubins = [int(a.split("_")[1].rstrip("a")) for a in arches if a.startswith("sm_")]
        ptx = [int(a.split("_")[1].rstrip("a")) for a in arches if a.startswith("compute_")]
        capability = properties.major * 10 + properties.minor
        if not any(a // 10 == properties.major and a <= capability for a in cubins) and not any(a <= capability for a in ptx):
            raise ComponentError(f"CUDA device capability {capability} is incompatible with wheel architectures {arches}")
    except (AssertionError, RuntimeError) as exc:
        raise ComponentError(f"Cannot initialize the requested CUDA device: {exc}") from exc
    torch.backends.cuda.matmul.allow_tf32 = precision["allow_tf32"]
    # Set this LAST: the boolean TF32 setter maps True to 'high', losing 'medium'.
    torch.set_float32_matmul_precision(precision["float32_matmul_precision"])
    torch.backends.cudnn.allow_tf32 = precision["allow_tf32"]
    torch.backends.cudnn.deterministic = precision["cudnn_deterministic"]
    torch.backends.cudnn.benchmark = precision["cudnn_benchmark"]
    torch.use_deterministic_algorithms(precision["deterministic_algorithms"], warn_only=False)
    effective = _backend_settings()
    for key in ("float32_matmul_precision", "deterministic_algorithms", "cudnn_deterministic", "cudnn_benchmark"):
        if effective[key] != precision[key]:
            raise ComponentError(f"Runtime could not honor precision.{key}")
    if effective["allow_tf32_matmul"] != precision["allow_tf32"] or effective["allow_tf32_cudnn"] != precision["allow_tf32"]:
        raise ComponentError("Runtime could not honor the declared TF32 policy")
    scaler = torch.amp.GradScaler("cuda") if precision["mode"] == "amp_float16" else None
    runtime = RuntimePolicy(device, precision, effective,
                            {"name": properties.name, "capability": [properties.major, properties.minor],
                             "total_memory_bytes": properties.total_memory, "wheel_architectures": arches,
                             "torch": str(torch.__version__), "cuda": torch.version.cuda,
                             "cudnn": torch.backends.cudnn.version()}, scaler)
    runtime.require_active(config)
    return runtime
