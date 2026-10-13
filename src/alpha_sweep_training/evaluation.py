"""Full clean validation/test using learned parameters and learned buffers."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, SequentialSampler

from .component_contracts import ComponentError
from .config import RunConfig
from .data import EvaluationView
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
                   config: RunConfig, runtime: RuntimePolicy) -> ClassificationMetrics:
    """Evaluate one full deterministic split; never accepts a noise generator."""
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
                images, labels = runtime.batch(batch, info.num_classes)
                with runtime.autocast():
                    logits = model(images)
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
