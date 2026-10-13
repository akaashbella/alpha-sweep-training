"""Independent initialization/data/runtime/noise ownership and in-memory state."""

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import random
from typing import Iterator, Mapping

import numpy as np
import torch

from .component_contracts import ComponentError, require_config, require_runtime_versions
from .config import RunConfig
from .spec import RUNTIME_RANDOMNESS_VERSION


_STREAMS = ("python", "numpy", "model_runtime_cpu", "model_runtime_cuda", "weight_noise")


def runtime_seeds(config: RunConfig) -> dict[str, int]:
    science = require_config(config, "runtime RNG").scientific
    if science["training_seed"] is None:
        raise ComponentError("Runtime RNG requires an explicitly selected scientific.training_seed")
    seeds = {}
    for stream in _STREAMS:
        payload = {"version": RUNTIME_RANDOMNESS_VERSION, "stream": stream,
                   "dataset": science["dataset"], "architecture": science["architecture"],
                   "training_seed": science["training_seed"]}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False).encode("utf-8")
        seeds[stream] = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little")
    seeds["numpy"] %= 2**32  # Legacy global NumPy MT19937's accepted seed width.
    return seeds


class RuntimeStreams:
    """One owner of process-global model RNGs and one private CUDA noise stream.

    Construct fresh streams once AFTER model construction/device placement.
    Continuation uses initialize=False and load_state_dict, never reseeding.
    Training and artifact callbacks must not concurrently consume these globals.
    """

    def __init__(self, config: RunConfig, device: torch.device, *, initialize: bool):
        require_runtime_versions(torch.__version__)
        if type(initialize) is not bool or device.type != "cuda" or device.index is None:
            raise ComponentError("RuntimeStreams requires explicit initialize bool and an indexed CUDA device")
        science = config.scientific
        self.device = device
        self._descriptor = {"format_version": 1, "randomness_version": RUNTIME_RANDOMNESS_VERSION,
                            "dataset": science["dataset"], "architecture": science["architecture"],
                            "training_seed": science["training_seed"], "seeds": runtime_seeds(config),
                            "torch": str(torch.__version__), "numpy": np.__version__, "device_type": "cuda"}
        self._noise = torch.Generator(device=device)
        self._origin: str | None = None
        self._claimed = False
        if initialize:
            seeds = self._descriptor["seeds"]
            random.seed(seeds["python"])
            np.random.seed(seeds["numpy"])
            cpu = torch.Generator(device="cpu").manual_seed(seeds["model_runtime_cpu"])
            cuda = torch.Generator(device=device).manual_seed(seeds["model_runtime_cuda"])
            torch.set_rng_state(cpu.get_state())
            torch.cuda.set_rng_state(cuda.get_state(), device=device)
            self._noise.manual_seed(seeds["weight_noise"])
            self._origin = "fresh"

    def claim(self, config: RunConfig, device: torch.device, *, continuation: bool) -> None:
        science = config.scientific
        if (self.device != device or self._descriptor["seeds"] != runtime_seeds(config)
                or self._descriptor["dataset"] != science["dataset"]
                or self._descriptor["architecture"] != science["architecture"]):
            raise ComponentError("Runtime RNG streams are incompatible with this model/config/device")
        if self._claimed or self._origin != ("restored" if continuation else "fresh"):
            raise ComponentError("Use fresh streams once for a new run, or restore streams once for continuation")
        self._claimed = True

    @property
    def noise_generator(self) -> torch.Generator:
        if self._origin is None:
            raise ComponentError("Restore runtime RNG state before using continuation streams")
        return self._noise

    def metadata(self) -> dict:
        return {**deepcopy(self._descriptor), "assigned_device": str(self.device),
                "ownership": "one_process; global_model_rngs_and_private_weight_noise"}

    def state_dict(self) -> dict:
        if self._origin is None:
            raise ComponentError("Runtime RNG continuation state has not been restored")
        return {"descriptor": deepcopy(self._descriptor), "assigned_device": str(self.device),
                "python": deepcopy(random.getstate()), "numpy": deepcopy(np.random.get_state()),
                "torch_cpu": torch.get_rng_state().clone(),
                "torch_cuda": torch.cuda.get_rng_state(self.device).clone(),
                "weight_noise": self._noise.get_state().clone()}

    def validate_state_dict(self, state: Mapping) -> None:
        keys = {"descriptor", "assigned_device", "python", "numpy", "torch_cpu", "torch_cuda", "weight_noise"}
        if not isinstance(state, Mapping) or set(state) != keys or state["descriptor"] != self._descriptor:
            raise ComponentError("Incompatible runtime RNG descriptor/state; retain stream, seed and software identities")
        try:
            if torch.device(state["assigned_device"]).type != "cuda":
                raise ValueError("saved device was not CUDA")
            # Validate on local generators before changing ANY live global state.
            random.Random().setstate(state["python"])
            np.random.RandomState().set_state(state["numpy"])
            for key, device in (("torch_cpu", torch.device("cpu")),
                                ("torch_cuda", self.device), ("weight_noise", self.device)):
                value = state[key]
                if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype != torch.uint8 or value.ndim != 1:
                    raise ValueError(f"{key} must be a CPU uint8 RNG-state vector")
                torch.Generator(device=device).set_state(value)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ComponentError(f"Invalid runtime RNG state: {exc}") from exc

    def _apply_state(self, state: Mapping) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"])
        torch.cuda.set_rng_state(state["torch_cuda"], self.device)
        self._noise.set_state(state["weight_noise"])

    def load_state_dict(self, state: Mapping) -> None:
        if self._claimed:
            raise ComponentError("Restore into new unclaimed continuation streams before entering the trainer")
        self.validate_state_dict(state)
        self._apply_state(state)
        self._origin = "restored"

    @contextmanager
    def preserve(self) -> Iterator[None]:
        """Artifact/logging callbacks cannot advance training/noise randomness."""
        state = self.state_dict()
        try:
            yield
        finally:
            self._apply_state(state)
