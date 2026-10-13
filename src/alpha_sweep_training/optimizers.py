"""One clean/master parameter group using each fixed native optimizer recipe."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any
import weakref

import torch
from torch import nn
from torch.optim import AdamW, Optimizer, RMSprop, SGD

from .component_contracts import ComponentError, finite_number, require_config, require_runtime_versions
from .config import RunConfig
from .models import model_construction
from .spec import OPTIMIZER_BUILDER_VERSION, optimizer_recipe
from .weights import named_parameters_once


class OptimizerConstructionError(ComponentError):
    """The optimizer no longer matches its clean model and fixed recipe."""


def _same_option(actual: Any, expected: Any) -> bool:
    # Reject bool-as-number and tensor-valued flags/LRs introduced by foreign
    # state. Native tuple betas remain tuples in optimizer state dictionaries.
    if type(expected) in (int, float):
        return type(actual) in (int, float) and finite_number(actual, "optimizer option") == expected
    if type(expected) is tuple:
        return type(actual) is tuple and len(actual) == len(expected) and all(
            _same_option(item, reference) for item, reference in zip(actual, expected))
    return type(actual) is type(expected) and actual == expected


def _same_options(actual: dict, expected: dict) -> bool:
    return set(actual) == set(expected) and all(_same_option(actual[key], value)
                                               for key, value in expected.items())


@dataclass(frozen=True)
class _OptimizerConstruction:
    architecture: str
    base_learning_rate: float
    recipe: dict
    options: dict
    names: tuple[str, ...]
    parameters: tuple[nn.Parameter, ...] = field(repr=False, compare=False)
    model: weakref.ReferenceType = field(repr=False, compare=False)


@dataclass(frozen=True)
class OptimizerMetadata:
    architecture: str
    recipe: dict
    base_learning_rates: tuple[float, ...]
    effective_options: dict
    current_learning_rates: tuple[float, ...]
    parameter_names: tuple[str, ...]
    parameter_count: int
    parameter_elements: int

    def to_dict(self) -> dict:
        options = deepcopy(self.effective_options)
        if "betas" in options:
            options["betas"] = list(options["betas"])
        return {"builder_version": OPTIMIZER_BUILDER_VERSION,
                "torch_version": str(torch.__version__), "architecture": self.architecture,
                "declared_recipe": deepcopy(self.recipe),
                "base_learning_rates": list(self.base_learning_rates),
                "effective_options": options, "current_learning_rates": list(self.current_learning_rates),
                "parameter_names": list(self.parameter_names), "parameter_count": self.parameter_count,
                "parameter_elements": self.parameter_elements, "parameter_groups": 1,
                "weight_decay_policy": "all_trainable_parameters_including_bias_and_normalization",
                "execution_selection": "native; None flags do not identify the eventual execution path"}


def build_optimizer(model: nn.Module, config: RunConfig) -> Optimizer:
    """Call after placing the clean model on its final training device.

    No perturbation tensor copies, decay exclusions, or training RNG operations.
    Only recipe-declared options are supplied; other options use pinned defaults.
    """
    science = require_config(config, "optimizer").scientific
    require_runtime_versions(torch.__version__)
    model_info = model_construction(model)
    if (model_info.architecture != science["architecture"] or model_info.dataset != science["dataset"]
            or model_info.num_classes != science["model"]["num_classes"]
            or (science["training_seed"] is not None and model_info.training_seed != science["training_seed"])):
        raise OptimizerConstructionError("Optimizer config must match the constructed model's architecture, dataset, classes, and selected seed")
    named = named_parameters_once(model)
    if not named or any(not parameter.requires_grad for _, parameter in named):
        raise OptimizerConstructionError("Approved scratch models must retain every parameter as trainable; do not freeze biases, normalization, or other clean parameters")
    recipe = optimizer_recipe(science["architecture"])
    parameters = tuple(parameter for _, parameter in named)
    common = {"lr": recipe["learning_rate"], "weight_decay": recipe["weight_decay"]}
    if recipe["name"] == "sgd":
        optimizer = SGD(parameters, momentum=recipe["momentum"], **common)
    elif recipe["name"] == "rmsprop":
        optimizer = RMSprop(parameters, momentum=recipe["momentum"],
                            alpha=recipe["rmsprop_alpha"], eps=recipe["epsilon"], **common)
    else:
        optimizer = AdamW(parameters, betas=tuple(recipe["betas"]), **common)
    options = {key: deepcopy(value) for key, value in optimizer.param_groups[0].items()
               if key not in ("params", "lr")}
    optimizer._alpha_sweep_construction = _OptimizerConstruction(
        science["architecture"], recipe["learning_rate"], recipe, options,
        tuple(name for name, _ in named), parameters, weakref.ref(model))
    validate_optimizer(optimizer, config)
    return optimizer


def validate_optimizer(optimizer: Optimizer, config: RunConfig | None = None) -> _OptimizerConstruction:
    """Read-only guard reused by the scheduler before any LR mutation.

    Parameter replacement/device placement after optimizer construction is
    rejected when it changes the model's parameter objects. Move the model first.
    """
    if not isinstance(optimizer, Optimizer):
        raise OptimizerConstructionError("Expected an optimizer returned by build_optimizer")
    info = getattr(optimizer, "_alpha_sweep_construction", None)
    if not isinstance(info, _OptimizerConstruction):
        raise OptimizerConstructionError("Expected an optimizer returned by build_optimizer")
    expected_type = {"sgd": SGD, "rmsprop": RMSprop, "adamw": AdamW}[info.recipe["name"]]
    if type(optimizer) is not expected_type or len(optimizer.param_groups) != 1:
        raise OptimizerConstructionError("Optimizer must retain its native type and exactly one parameter group")
    group = optimizer.param_groups[0]
    if set(group) != set(info.options) | {"params", "lr"}:
        raise OptimizerConstructionError("Optimizer parameter-group options differ from the constructed native recipe")
    if not _same_options({key: value for key, value in group.items() if key not in ("params", "lr")}, info.options):
        raise OptimizerConstructionError("Optimizer options changed; the fixed native recipe must be preserved")
    if not _same_options(optimizer.defaults, {"lr": info.base_learning_rate, **info.options}):
        raise OptimizerConstructionError("Optimizer defaults must preserve the original recipe and base learning rate")
    actual = group["params"]
    if (type(actual) not in (list, tuple) or len(actual) != len(info.parameters)
            or any(a is not b for a, b in zip(actual, info.parameters))):
        raise OptimizerConstructionError("Optimizer must own every original clean/master parameter exactly once, in original order")
    model = info.model()
    if model is None:
        raise OptimizerConstructionError("Keep the clean model alive throughout optimizer use")
    current = named_parameters_once(model)
    if (tuple(name for name, _ in current) != info.names or len(current) != len(info.parameters)
            or any(parameter is not original or not parameter.requires_grad
                   for (_, parameter), original in zip(current, info.parameters))):
        raise OptimizerConstructionError("Clean model parameters changed or were frozen after optimizer construction")
    if finite_number(group["lr"], "optimizer current learning rate") < 0:
        raise OptimizerConstructionError("Optimizer learning rate cannot be negative")
    if config is not None:
        science = require_config(config, "optimizer").scientific
        model_info = model_construction(model)
        if (science["architecture"] != info.architecture or science["optimizer"] != info.recipe
                or science["dataset"] != model_info.dataset
                or (science["training_seed"] is not None and science["training_seed"] != model_info.training_seed)):
            raise OptimizerConstructionError("Optimizer/config identities or recipe are incompatible")
    return info


def describe_optimizer(optimizer: Optimizer) -> OptimizerMetadata:
    info = validate_optimizer(optimizer)
    return OptimizerMetadata(info.architecture, deepcopy(info.recipe), (info.base_learning_rate,),
                             deepcopy(info.options), (float(optimizer.param_groups[0]["lr"]),),
                             info.names, len(info.parameters), sum(p.numel() for p in info.parameters))
