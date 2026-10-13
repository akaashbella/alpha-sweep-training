"""Authoritative, side-effect-free owned Conv2d/Linear weight discovery."""

from dataclasses import dataclass, field

import torch
from torch import nn

from .component_contracts import ComponentError


class WeightInventoryError(ComponentError):
    """Unsupported parameter ownership/aliasing prevents unambiguous discovery."""


def named_parameters_once(model: nn.Module) -> tuple[tuple[str, nn.Parameter], ...]:
    """Deterministic names; reject shared objects and shared backing storage.

    All approved builders own independent dense parameters. Reject aliasing
    explicitly rather than double-noising, double-updating, or overlooking a
    weight shared with a bias/normalization parameter.
    """
    if not isinstance(model, nn.Module):
        raise WeightInventoryError("Weight inventory requires an nn.Module")
    parameters = sorted(model.named_parameters(remove_duplicate=False), key=lambda pair: pair[0])
    owners = {}
    storage_owners = {}
    for name, parameter in parameters:
        if not isinstance(parameter, nn.Parameter) or parameter.layout != torch.strided:
            raise WeightInventoryError(f"Unsupported dense parameter ownership at {name}")
        if id(parameter) in owners:
            raise WeightInventoryError(f"Shared parameter aliases are unsupported: {owners[id(parameter)]}, {name}")
        owners[id(parameter)] = name
        if parameter.device.type == "meta":
            raise WeightInventoryError(f"Unmaterialized parameter is unsupported: {name}")
        if parameter.numel():
            storage = (str(parameter.device), parameter.untyped_storage().data_ptr())
            if storage in storage_owners:
                raise WeightInventoryError(f"Parameters share backing storage: {storage_owners[storage]}, {name}")
            storage_owners[storage] = name
    return tuple(parameters)


@dataclass(frozen=True)
class EligibleWeight:
    name: str
    module_name: str
    module_type: str
    shape: tuple[int, ...]
    numel: int
    requires_grad: bool
    parameter: nn.Parameter = field(repr=False, compare=False)

    def to_dict(self) -> dict:
        return {"name": self.name, "module_name": self.module_name, "module_type": self.module_type,
                "shape": list(self.shape), "numel": self.numel, "requires_grad": self.requires_grad}


def discover_eligible_weights(model: nn.Module) -> tuple[EligibleWeight, ...]:
    """Every owned Conv2d.weight/Linear.weight, independent of rank/size/location.

    No statistics, noise, parameter changes, gradient changes, or mode changes.
    Normalization, biases, buffers, and other parameters stay outside this list.
    """
    parameters = dict(named_parameters_once(model))
    entries = []
    seen = set()
    for module_name, module in sorted(model.named_modules(remove_duplicate=False), key=lambda pair: pair[0]):
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue
        name = f"{module_name}.weight" if module_name else "weight"
        weight = module.weight
        if not isinstance(weight, nn.Parameter) or parameters.get(name) is not weight:
            raise WeightInventoryError(f"Expected an owned weight Parameter at {name}; parametrizations are unsupported")
        if id(weight) in seen:
            raise WeightInventoryError(f"An eligible weight has multiple owners: {name}")
        seen.add(id(weight))
        entries.append(EligibleWeight(name, module_name,
                       f"{type(module).__module__}.{type(module).__qualname__}",
                       tuple(weight.shape), weight.numel(), weight.requires_grad, weight))
    return tuple(sorted(entries, key=lambda entry: entry.name))
