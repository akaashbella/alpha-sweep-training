"""Functional training forwards with gradients to original clean parameters."""

import torch
from torch import nn
from torch.func import functional_call

from .component_contracts import ComponentError, finite_number
from .models import model_construction
from .numerical import require_finite
from .spec import ALPHAS
from .weights import discover_eligible_weights, named_parameters_once


def gaussian_weight_offset(shape, *, strength, clean_population_std, generator, device):
    """Shared float32 detached noise mathematics; callers own scale/lifetime.

    Training computes all current scales each mini-batch; evaluation computes
    clean-checkpoint scales once per fixed realization. No global RNG calls.
    """
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
        epsilon = torch.randn(shape, generator=generator, dtype=torch.float32, device=device)
        return epsilon.mul(strength * clean_population_std).detach()


def training_forward(model: nn.Module, inputs: torch.Tensor, *, alpha: float,
                     noise_generator: torch.Generator) -> torch.Tensor:
    """Exactly one forward. Autocast belongs to the caller's precision policy.

    The functional call temporarily binds tensors, restores bindings on exit,
    and uses the ORIGINAL buffers so native in-place BatchNorm updates persist.
    No in-place addition/subtraction is applied to clean weights.
    """
    alpha = finite_number(alpha, "perturbation alpha")
    if alpha not in ALPHAS:
        raise ComponentError("Use an approved fixed run-level alpha")
    model_construction(model)
    if not all(module.training for module in model.modules()):
        raise ComponentError("Training forward requires normal training mode on every module")
    if getattr(model, "_alpha_sweep_perturbation_active", False):
        raise ComponentError("Nested/concurrent perturbed forwards on one model are unsupported")
    if alpha == 0:
        # No inventory/std/noise work and no generator-state access on this path.
        return model(inputs)

    original = dict(named_parameters_once(model))
    buffers = dict(model.named_buffers())
    eligible = discover_eligible_weights(model)
    if not eligible:
        raise ComponentError("Approved model has no eligible weights")
    device = eligible[0].parameter.device
    if device.type != "cuda" or noise_generator.device != device:
        raise ComponentError("Perturbation requires a dedicated generator on the assigned CUDA device")
    if any(p.dtype != torch.float32 or p.device != device for p in original.values()):
        raise ComponentError("Perturbation requires float32 clean/master parameters on one device")
    parameters = dict(original)
    try:
        with torch.autocast(device_type="cuda", enabled=False):
            with torch.no_grad():
                scales = tuple((entry.name, torch.std(entry.parameter.detach(), correction=0))
                               for entry in eligible)
                require_finite(scales, "perturbation_scale")
            for entry, (_, scale) in zip(eligible, scales):
                offset = gaussian_weight_offset(entry.shape, strength=alpha, clean_population_std=scale,
                                                generator=noise_generator, device=device)
                # Keep this addition tracked: d temporary / d clean = identity.
                parameters[entry.name] = entry.parameter + offset
            require_finite(((entry.name, parameters[entry.name]) for entry in eligible),
                           "perturbed_weights")
        model._alpha_sweep_perturbation_active = True
        return functional_call(model, (parameters, buffers), (inputs,), strict=True, tie_weights=True)
    finally:
        model._alpha_sweep_perturbation_active = False
        # The pinned functional_call restores bindings even when forward raises.
        current = dict(model.named_parameters())
        if set(current) != set(original) or any(current[name] is not p for name, p in original.items()):
            raise ComponentError("Functional call did not restore original clean parameter ownership")
        current_buffers = dict(model.named_buffers())
        if set(current_buffers) != set(buffers) or any(current_buffers[name] is not b for name, b in buffers.items()):
            raise ComponentError("Functional call changed original buffer ownership")
        parameters.clear()
        buffers.clear()
