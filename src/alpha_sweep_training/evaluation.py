"""Shared full-set evaluation with clean or fixed functional weights."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, SequentialSampler

from .component_contracts import ComponentError
from .config import RunConfig
from .data import EvaluationView
from .interruption import StopRequest
from .metrics import ClassificationMetrics, MetricAccumulator
from .models import model_construction
from .numerical import require_clean_model, require_finite
from .runtime import RuntimePolicy
from .spec import BATCH_SIZE, TEST_SIZE, VALIDATION_SIZE


class EvaluationFailure(RuntimeError):
    """Keep split/batch location and the original numerical/execution cause."""
    def __init__(self, split: str, batch_index: int | None, cause: Exception):
        self.split = split
        self.batch_index = batch_index
        self.cause = cause
        super().__init__(f"{split} evaluation at batch {batch_index}: {cause}")


def classification_loss(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Unsmoothed mean cross-entropy in float32, with one finite-flag transfer."""
    if (not isinstance(logits, torch.Tensor) or logits.shape != (labels.numel(), num_classes)
            or not logits.is_floating_point() or logits.device != labels.device):
        raise ComponentError("Expected one classifier-logit row per example and the dataset's class count")
    with torch.autocast(device_type="cuda", enabled=False):
        loss = F.cross_entropy(logits.float(), labels, reduction="mean", label_smoothing=0.0)
    require_finite((("logits", logits), ("cross_entropy", loss)), "forward_loss")
    return loss


def evaluate_clean(model: nn.Module, loader: DataLoader, *, split: str,
                   config: RunConfig, runtime: RuntimePolicy, stop: StopRequest | None = None) -> ClassificationMetrics:
    """Evaluate one full deterministic split; never accepts a noise generator."""
    return _evaluate_split(model, loader, split=split, config=config, runtime=runtime, stop=stop, forward=model)


def _evaluate_split(model, loader, *, split, config, runtime, stop, forward):
    """Shared full-set metrics/precision loop; forward lifetime belongs to caller."""
    if split not in ("validation", "test"):
        raise ComponentError("Clean evaluator supports only full validation or official test")
    expected = VALIDATION_SIZE if split == "validation" else TEST_SIZE
    if (not isinstance(loader, DataLoader) or not isinstance(loader.dataset, EvaluationView)
            or not isinstance(loader.sampler, SequentialSampler) or loader.drop_last
            or loader.batch_size != BATCH_SIZE or len(loader.dataset) != expected):
        raise ComponentError(f"{split} evaluation requires its full deterministic prepared loader ({expected} examples)")
    runtime.require_active(config)
    info = model_construction(model)
    science = config.scientific
    if (info.dataset != science["dataset"] or info.architecture != science["architecture"]
            or info.num_classes != science["model"]["num_classes"]
            or info.training_seed != science["training_seed"]):
        raise ComponentError("Clean evaluation model/config identity mismatch")
    require_clean_model(model, runtime.device, stage=f"{split}_entry")
    modes = tuple((module, module.training) for module in model.modules())
    accumulator = MetricAccumulator(runtime.device)
    batch_index = None
    try:
        model.eval()
        with torch.inference_mode():
            for batch_index, batch in enumerate(loader):
                if stop is not None:
                    stop.check(at_boundary=False, phase=f"clean_{split}_batch")
                images, labels = runtime.batch(batch, info.num_classes)
                with runtime.autocast():
                    logits = forward(images)
                loss = classification_loss(logits, labels, info.num_classes)
                accumulator.add(loss, logits, labels)
                if accumulator.examples > expected:
                    raise ComponentError(f"{split} evaluation exceeded expected full coverage")
        result = accumulator.finish()
        if result.examples != expected or result.batches != len(loader):
            raise ComponentError(f"Incomplete {split} evaluation: {result.examples}/{expected} examples")
        require_clean_model(model, runtime.device, stage=f"{split}_exit")
        return result
    except Exception as exc:
        raise EvaluationFailure(split, batch_index, exc) from exc
    finally:
        # Assign flags individually, preserving mixed prior submodule modes.
        for module, training in modes:
            module.training = training


def evaluate_realization(model, loader, *, config, runtime, slot, seed, baseline, stop=None):
    """One fixed temporary full network across the entire official test pass.

    Alpha zero delegates to the ordinary clean evaluator and never constructs
    a perturbation generator. No training/noise RNG state is consumed/restored.
    """
    from torch.func import functional_call
    from .evaluation_protocol import EvaluationSlot
    from .weights import discover_eligible_weights, named_parameters_once
    from .evaluation_state import CleanStatePreservationError
    from .perturbation import gaussian_weight_offset
    if not isinstance(slot, EvaluationSlot):
        raise ComponentError("Evaluation requires a prescribed EvaluationSlot")
    if not runtime.evaluation_only or baseline.model is not model:
        raise ComponentError("Use evaluation-only runtime and this model's owned clean baseline")
    with baseline.realization():
        runtime.require_active(config)
        require_clean_model(model, runtime.device, stage="robustness_realization_entry")
        if slot.evaluation_alpha == 0:
            if seed is not None:
                raise ComponentError("Clean baseline has no perturbation seed")
            return evaluate_clean(model, loader, split="test", config=config, runtime=runtime, stop=stop)
        if type(seed) is not int or not 0 <= seed < 2**64:
            raise ComponentError("Positive realization requires its derived uint64 seed")
        generator = torch.Generator(device=runtime.device).manual_seed(seed)
        parameters = dict(named_parameters_once(model))
        # Private learned-buffer copies also protect originals during a failing forward.
        buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
        buffer_baseline = {name: value.detach().to("cpu", copy=True) for name, value in buffers.items()}
        try:
            with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
                eligible = discover_eligible_weights(model)
                if not eligible:
                    raise ComponentError("Approved evaluation model has no owned eligible weights")
                scales = tuple(torch.std(entry.parameter.detach(), correction=0) for entry in eligible)
                require_finite(((entry.name, scale) for entry, scale in zip(eligible, scales)), "evaluation_scales")
                for entry, scale in zip(eligible, scales):
                    offset = gaussian_weight_offset(entry.shape, strength=slot.evaluation_alpha,
                        clean_population_std=scale, generator=generator, device=runtime.device)
                    parameters[entry.name] = entry.parameter.detach() + offset
                require_finite(((entry.name, parameters[entry.name]) for entry in eligible), "evaluation_fixed_weights")
            def forward(images):
                return functional_call(model, (parameters, buffers), (images,), strict=True, tie_weights=True)
            result = _evaluate_split(model, loader, split="test", config=config, runtime=runtime, stop=stop, forward=forward)
            if any(not torch.equal(value.detach().cpu(), buffer_baseline[name]) for name, value in buffers.items()):
                raise CleanStatePreservationError("Fixed evaluation forward changed learned normalization buffers")
            return result
        finally:
            parameters.clear()
            buffers.clear()
            buffer_baseline.clear()
