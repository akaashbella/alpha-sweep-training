"""Pinned scratch builders with isolated CPU initialization and CIFAR stems."""

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import threading
from typing import Iterator

import torch
from torch import nn
import torchvision
from torchvision import models
from torchvision.models.convnext import ConvNeXt, LayerNorm2d
from torchvision.models.efficientnet import EfficientNet
from torchvision.models.mobilenetv3 import MobileNetV3
from torchvision.models.resnet import ResNet
from torchvision.ops import StochasticDepth
from torchvision.ops.misc import Conv2dNormActivation

from .component_contracts import ComponentError, require_config, require_runtime_versions
from .config import RunConfig
from .spec import MODEL_BUILDER_VERSION, MODEL_INITIALIZATION_VERSION, ELIGIBLE_WEIGHT_VERSION
from .weights import discover_eligible_weights, named_parameters_once


class ModelConstructionError(ComponentError):
    """A selected seed or the expected pinned architecture is unavailable."""


@dataclass(frozen=True)
class _Construction:
    dataset: str
    architecture: str
    num_classes: int
    training_seed: int
    initialization_seed: int
    adaptations: tuple[str, ...]
    torch_version: str
    torchvision_version: str


@dataclass(frozen=True)
class ModelMetadata:
    """Serializable observations of a constructed model, without a forward pass."""

    construction: _Construction
    total_parameters: int
    trainable_parameters: int
    eligible_weights: tuple[dict, ...]
    native_regularization: tuple[dict, ...]

    def to_dict(self) -> dict:
        info = self.construction
        return {
            "builder_version": MODEL_BUILDER_VERSION,
            "initialization_version": MODEL_INITIALIZATION_VERSION,
            "eligible_weight_version": ELIGIBLE_WEIGHT_VERSION,
            "dataset": info.dataset, "architecture": info.architecture,
            "num_classes": info.num_classes, "training_seed": info.training_seed,
            "initialization_seed": info.initialization_seed,
            "construction_device": "cpu", "construction_dtype": "float32",
            "torch_version": info.torch_version, "torchvision_version": info.torchvision_version,
            "weights": None, "adaptations": list(info.adaptations),
            "total_parameters": self.total_parameters,
            "trainable_parameters": self.trainable_parameters,
            "eligible_weights": [dict(entry) for entry in self.eligible_weights],
            "native_regularization": [dict(entry) for entry in self.native_regularization],
            "classifier_check": "structural_only_no_forward",
        }


_INITIALIZATION_LOCK = threading.RLock()


def initialization_seed(config: RunConfig) -> int:
    """Stable uint64 seed; experiment alpha and data randomness are absent."""
    science = require_config(config, "model initialization").scientific
    seed = science["training_seed"]
    if seed is None:
        raise ModelConstructionError("Model construction requires scientific.training_seed; select a seed explicitly (the three-seed decision may remain pending)")
    payload = {"version": MODEL_INITIALIZATION_VERSION, "stream": "model_initialization",
               "dataset": science["dataset"], "architecture": science["architecture"],
               "training_seed": seed}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little")


@contextmanager
def _cpu_initialization(seed: int) -> Iterator[None]:
    # Only CPU APIs: torch.manual_seed/fork_rng would touch accelerator RNGs.
    # Serialize these builders; unrelated concurrent global RNG/default mutations
    # are unsupported. The launcher constructs models before training threads.
    with _INITIALIZATION_LOCK:
        previous_state = torch.get_rng_state()
        previous_dtype = torch.get_default_dtype()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        try:
            torch.set_rng_state(generator.get_state())
            torch.set_default_dtype(torch.float32)
            with torch.device("cpu"):
                yield
        finally:
            torch.set_default_dtype(previous_dtype)
            torch.set_rng_state(previous_state)


def _expect(condition: bool, description: str) -> None:
    if not condition:
        raise ModelConstructionError(f"Pinned TorchVision structure mismatch: {description}; inspect the pinned source before changing the builder")


def _conv(module: nn.Module, *, channels: tuple[int, int], kernel: int,
          stride: int, padding: int, bias: bool) -> nn.Conv2d:
    _expect(type(module) is nn.Conv2d, "stem must be a Conv2d")
    _expect((module.in_channels, module.out_channels) == channels
            and module.kernel_size == (kernel, kernel) and module.stride == (stride, stride)
            and module.padding == (padding, padding) and module.dilation == (1, 1)
            and module.groups == 1 and module.padding_mode == "zeros"
            and (module.bias is not None) == bias, "native stem convolution settings")
    return module


def _linear(module: nn.Module, inputs: int, outputs: int) -> None:
    _expect(type(module) is nn.Linear and module.in_features == inputs
            and module.out_features == outputs and module.bias is not None,
            "native classifier Linear settings")


def _adapt(model: nn.Module, architecture: str, classes: int) -> tuple[str, ...]:
    if architecture == "resnet50":
        _expect(type(model) is ResNet, "resnet50 model type")
        _expect(all(hasattr(model, name) for name in ("conv1", "maxpool", "fc")),
                "ResNet stem/pool/classifier attributes")
        stem = _conv(model.conv1, channels=(3, 64), kernel=7, stride=2, padding=3, bias=False)
        pool = model.maxpool
        _expect(type(pool) is nn.MaxPool2d and pool.kernel_size == 3
                and pool.stride == 2 and pool.padding == 1 and pool.dilation == 1
                and not pool.ceil_mode and not pool.return_indices, "ResNet initial max pool")
        _linear(model.fc, 2048, classes)
        replacement = nn.Conv2d(stem.in_channels, stem.out_channels, 3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(replacement.weight, mode="fan_out", nonlinearity="relu")
        model.conv1 = replacement
        model.maxpool = nn.Identity()
        return ("conv1: 7x7/stride2/pad3 -> 3x3/stride1/pad1, bias=False",
                "maxpool: MaxPool2d -> Identity")

    expected_type = {"mobilenetv3_large": MobileNetV3, "efficientnet_b0": EfficientNet,
                     "convnext_tiny": ConvNeXt}[architecture]
    _expect(type(model) is expected_type and type(getattr(model, "features", None)) is nn.Sequential
            and len(model.features) > 0 and hasattr(model, "classifier"),
            f"{architecture} model/features type")
    stem_block = model.features[0]
    _expect(type(stem_block) is Conv2dNormActivation, "native Conv2dNormActivation stem")
    if architecture == "convnext_tiny":
        _expect(len(stem_block) == 2 and type(stem_block[1]) is LayerNorm2d
                and tuple(stem_block[1].normalized_shape) == (96,)
                and stem_block[1].eps == 1e-6, "ConvNeXt stem normalization")
        stem = _conv(stem_block[0], channels=(3, 96), kernel=4, stride=4, padding=0, bias=True)
        _expect(type(model.classifier) is nn.Sequential and len(model.classifier) == 3
                and type(model.classifier[0]) is LayerNorm2d
                and tuple(model.classifier[0].normalized_shape) == (768,)
                and model.classifier[0].eps == 1e-6
                and type(model.classifier[1]) is nn.Flatten
                and model.classifier[1].start_dim == 1, "ConvNeXt native classifier")
        _linear(model.classifier[2], 768, classes)
        replacement = nn.Conv2d(stem.in_channels, stem.out_channels, 3, stride=1, padding=1, bias=True)
        nn.init.trunc_normal_(replacement.weight, std=0.02)
        nn.init.zeros_(replacement.bias)
        stem_block[0] = replacement
        return ("features.0.0: 4x4/stride4/pad0 -> 3x3/stride1/pad1, bias=True",)

    mobile = architecture == "mobilenetv3_large"
    _expect(len(stem_block) == 3 and type(stem_block[1]) is nn.BatchNorm2d
            and type(stem_block[2]) is (nn.Hardswish if mobile else nn.SiLU),
            "native stem BatchNorm/activation")
    stem = _conv(stem_block[0], channels=(3, 16 if mobile else 32), kernel=3,
                 stride=2, padding=1, bias=False)
    norm = stem_block[1]
    _expect(norm.num_features == stem.out_channels
            and norm.eps == (0.001 if mobile else 1e-5)
            and norm.momentum == (0.01 if mobile else 0.1), "native stem BatchNorm settings")
    classifier = model.classifier
    _expect(type(classifier) is nn.Sequential and len(classifier) == (4 if mobile else 2),
            "native classifier structure")
    dropout = classifier[2 if mobile else 0]
    _expect(type(dropout) is nn.Dropout and dropout.p == 0.2 and dropout.inplace,
            "native classifier dropout")
    if mobile:
        _linear(classifier[0], 960, 1280)
        _expect(type(classifier[1]) is nn.Hardswish, "MobileNet classifier activation")
    _linear(classifier[-1], 1280, classes)
    stem.stride = (1, 1)
    return ("features.0.0: stride2 -> stride1; other stem settings preserved",)


def build_model(config: RunConfig) -> nn.Module:
    """Construct CPU float32 clean weights; no data/HPC/scheduler readiness gate."""
    science = require_config(config, "model").scientific
    seed = initialization_seed(config)
    require_runtime_versions(torch.__version__, torchvision.__version__)
    architecture = science["architecture"]
    classes = science["model"]["num_classes"]
    constructor = {"resnet50": models.resnet50, "mobilenetv3_large": models.mobilenet_v3_large,
                   "efficientnet_b0": models.efficientnet_b0, "convnext_tiny": models.convnext_tiny}[architecture]
    with _cpu_initialization(seed):
        model = constructor(weights=None, num_classes=classes, progress=False)
        adaptations = _adapt(model, architecture, classes)
        parameters = named_parameters_once(model)
        _expect(bool(parameters) and all(p.device.type == "cpu" and p.dtype == torch.float32
                                        and p.requires_grad for _, p in parameters),
                "all scratch parameters must be trainable CPU float32")
        _expect(all(b.device.type == "cpu" and (not b.is_floating_point() or b.dtype == torch.float32)
                    for b in model.buffers()), "CPU float32 floating buffers")
        discover_eligible_weights(model)
    model._alpha_sweep_construction = _Construction(
        science["dataset"], architecture, classes, science["training_seed"], seed,
        adaptations, str(torch.__version__), str(torchvision.__version__))
    return model


def model_construction(model: nn.Module) -> _Construction:
    """Require a supported builder's structural identity (no forward validation)."""
    if not isinstance(model, nn.Module):
        raise ModelConstructionError("Expected a model returned by build_model")
    info = getattr(model, "_alpha_sweep_construction", None)
    if not isinstance(info, _Construction):
        raise ModelConstructionError("Expected a model returned by build_model")
    return info


def describe_model(model: nn.Module) -> ModelMetadata:
    info = model_construction(model)
    parameters = named_parameters_once(model)
    regularization = []
    for name, module in sorted(model.named_modules(), key=lambda pair: pair[0]):
        if isinstance(module, nn.Dropout):
            regularization.append({"name": name, "type": "Dropout", "p": module.p, "inplace": module.inplace})
        elif isinstance(module, StochasticDepth):
            regularization.append({"name": name, "type": "StochasticDepth", "p": module.p, "mode": module.mode})
    return ModelMetadata(info, sum(p.numel() for _, p in parameters),
                         sum(p.numel() for _, p in parameters if p.requires_grad),
                         tuple(entry.to_dict() for entry in discover_eligible_weights(model)),
                         tuple(regularization))
