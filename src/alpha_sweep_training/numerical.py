"""Finite-value diagnostics without one host synchronization per layer."""

from collections import defaultdict
import math
from typing import Any, Iterable, Iterator

import torch
from torch import nn
from torch.optim import Optimizer

from .component_contracts import ComponentError
from .weights import named_parameters_once


class NumericalFailure(RuntimeError):
    def __init__(self, stage: str, reason: str, tensors: Iterable[str] = ()):
        self.stage = stage
        self.reason = reason
        self.tensors = tuple(tensors)
        super().__init__(f"{stage}: {reason}" + (f"; tensors={self.tensors}" if self.tensors else ""))


def nonfinite_names(tensors: Iterable[tuple[str, torch.Tensor]]) -> tuple[str, ...]:
    """Reduce on-device, transfer only the stacked scalar flags once per device."""
    groups = defaultdict(list)
    with torch.no_grad():
        for name, tensor in tensors:
            if tensor.layout != torch.strided:
                raise ComponentError(f"Unsupported non-dense tensor at {name}")
            if tensor.is_floating_point() or tensor.is_complex():
                groups[tensor.device].append((name, torch.isfinite(tensor.detach()).all()))
        bad = []
        for entries in groups.values():
            flags = torch.stack([flag for _, flag in entries]).cpu().tolist()
            bad.extend(name for (name, _), finite in zip(entries, flags) if not finite)
    return tuple(bad)


def require_finite(tensors: Iterable[tuple[str, torch.Tensor]], stage: str) -> None:
    bad = nonfinite_names(tensors)
    if bad:
        raise NumericalFailure(stage, "Nonfinite tensor values", bad)


def require_clean_model(model: nn.Module, device: torch.device, *, stage: str) -> None:
    if getattr(model, "_alpha_sweep_perturbation_active", False):
        raise ComponentError("Clean state inspection is forbidden during a functional perturbed forward")
    parameters = named_parameters_once(model)
    if not parameters or any(p.dtype != torch.float32 or p.device != device or not p.requires_grad
                             for _, p in parameters):
        raise ComponentError("All clean/master parameters must remain trainable float32 on the assigned device")
    buffers = tuple(model.named_buffers())
    if any(b.device != device or (b.is_floating_point() and b.dtype != torch.float32) for _, b in buffers):
        raise ComponentError("Model buffers must stay on the assigned device with float32 floating buffers")
    require_finite((*parameters, *((f"buffer:{name}", value) for name, value in buffers)), stage)


def gradient_nonfinite_names(model: nn.Module) -> tuple[str, ...]:
    parameters = named_parameters_once(model)
    missing = tuple(name for name, p in parameters if p.requires_grad and p.grad is None)
    if missing:
        raise ComponentError(f"Trainable clean parameters did not receive gradients: {missing}")
    return nonfinite_names((name, p.grad) for name, p in parameters if p.grad is not None)


def _state_tensors(value: Any, prefix: str) -> Iterator[tuple[str, torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from _state_tensors(child, f"{prefix}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            yield from _state_tensors(child, f"{prefix}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise NumericalFailure("optimizer_state", "Nonfinite optimizer scalar", (prefix,))


def require_finite_optimizer(optimizer: Optimizer, model: nn.Module) -> None:
    names = {id(parameter): name for name, parameter in named_parameters_once(model)}
    tensors = []
    for parameter, state in optimizer.state.items():
        name = names.get(id(parameter))
        if name is None:
            raise ComponentError("Optimizer state contains a key outside the clean parameter inventory")
        tensors.extend(_state_tensors(state, name))
    require_finite(tensors, "optimizer_state")
