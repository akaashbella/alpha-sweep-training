"""Full reverse-over-reverse HVPs with one fixed vector per dataset pass."""

from contextlib import contextmanager
import math

import torch

from alpha_sweep_training.evaluation import classification_loss
from alpha_sweep_training.evaluation_state import CleanStateBaseline
from alpha_sweep_training.numerical import require_finite
from alpha_sweep_training.weights import named_parameters_once

from .protocol import probe_seed, decomposition
from .storage import require


class DerivativeError(RuntimeError):
    """The requested derivatives were unavailable or numerically invalid."""


def hvp(loss, parameters, vector):
    require(loss.ndim == 0 and loss.requires_grad and not torch.is_inference_mode_enabled(),
            "HVP requires a differentiable scalar loss outside inference mode")
    require(len(parameters) == len(vector) and len(parameters) > 0 and
            all(v.shape == p.shape and v.dtype == p.dtype and v.device == p.device and not v.requires_grad
                for p, v in zip(parameters, vector)), "Probe tensors must be detached and match the differentiated parameters")
    try:
        grads = torch.autograd.grad(loss, parameters, create_graph=True, allow_unused=False)
        require_finite(((str(i), g) for i, g in enumerate(grads)), "stage3_gradient")
        # A detached derivative can signal an unsupported double backward. The
        # scientific path rejects it even when its observed value appears constant.
        if any(not g.requires_grad for g in grads):
            raise DerivativeError("A first derivative is detached. Its second derivative cannot be verified")
        directional = sum((g*v).sum() for g, v in zip(grads, vector))
        require_finite((("directional_derivative", directional),), "stage3_directional_derivative")
        products = torch.autograd.grad(directional, parameters, retain_graph=False, allow_unused=False)
        require_finite(((str(i), h) for i, h in enumerate(products)), "stage3_hessian_vector")
        return products
    except (RuntimeError, NotImplementedError) as exc:
        raise DerivativeError(f"Full double backward failed or a derivative is unavailable: {exc}") from exc


def vectors(entries, scales, checkpoint, probe):
    result = []
    for entry in entries:
        p = entry.parameter
        generator = torch.Generator(device=p.device).manual_seed(probe_seed(checkpoint, probe, entry.name))
        with torch.no_grad():
            signs = torch.randint(0, 2, p.shape, generator=generator, device=p.device, dtype=torch.int8)
            result.append(signs.to(dtype=p.dtype).mul_(2).sub_(1).mul_(scales[entry.name]).detach())
    return tuple(result)


class StateGuard(CleanStateBaseline):
    """Extend the original guard to verify and restore pre-existing gradients."""

    def __init__(self, model):
        super().__init__(model)
        self.gradients = [(p, p.grad, None if p.grad is None else p.grad.detach().to("cpu", copy=True))
                          for _, p in named_parameters_once(model)]

    def verify(self):
        super().verify()
        for p, previous, value in self.gradients:
            require(p.grad is previous and (previous is None or torch.equal(previous.detach().cpu(), value)),
                    "Curvature computation changed a .grad binding or value")

    def restore(self):
        super().restore()
        with torch.no_grad():
            for p, previous, value in self.gradients:
                p.grad = previous
                if previous is not None:
                    previous.copy_(value)


@contextmanager
def evaluation(guard, device):
    require(not torch.is_inference_mode_enabled(), "Stage 3 requires autograd outside inference mode")
    with guard.realization(), torch.enable_grad(), torch.autocast(device_type=device.type, enabled=False):
        guard.model.eval()
        require(not any(module.training for module in guard.model.modules()), "Model could not enter evaluation mode")
        yield
        require(not any(module.training for module in guard.model.modules()), "Forward mutated evaluation modes")


def batch_measure(model, prepare, batch, classes, parameters, vector):
    inputs, labels = prepare(batch)
    logits = model(inputs)
    require(logits.dtype == torch.float32, "Stage 3 requires an FP32 differentiable forward")
    loss = classification_loss(logits, labels, classes)
    products = hvp(loss, parameters, vector)
    # Products may be FP32, but reductions and the dataset accumulator are FP64.
    quadratics = torch.stack([(v*h).sum(dtype=torch.float64) for v, h in zip(vector, products)])
    require_finite((("batch quadratics", quadratics),), "stage3_quadratic_form")
    return quadratics, loss.detach(), (logits.detach().argmax(1) == labels).sum(), labels.numel()


def measure_probe(model, loader, prepare, entries, scales, layout, checkpoint, probe, *, classes, expected, batches, guard):
    parameters = tuple(entry.parameter for entry in entries)
    require(all(p.dtype == torch.float32 and p.requires_grad for p in parameters), "Eligible weights must be trainable FP32 leaves")
    device = parameters[0].device
    vector = vectors(entries, scales, checkpoint, probe)
    accumulated = torch.zeros(len(parameters), dtype=torch.float64, device=device)
    loss_total = torch.zeros((), dtype=torch.float64, device=device)
    correct_total = torch.zeros((), dtype=torch.int64, device=device)
    seen, processed_batches = 0, 0
    with evaluation(guard, device):
        for index, batch in enumerate(loader, 1):
            try:
                q, loss, correct, count = batch_measure(model, prepare, batch, classes, parameters, vector)
                accumulated.add_(q, alpha=count)
                loss_total.add_(loss.to(torch.float64), alpha=count)
                correct_total.add_(correct)
                seen += count
                processed_batches += 1
                require(seen <= expected, "Unexpected extra examples")
            except Exception as exc:
                raise DerivativeError(f"Probe {probe}, batch {index}: {type(exc).__name__}: {exc}") from exc
        require(seen == expected and processed_batches == batches,
                f"Incomplete dataset HVP pass: {seen}/{expected} examples, {processed_batches}/{batches} batches")
    tensors = {entry.name: value/seen for entry, value in zip(entries, accumulated.cpu().tolist())}
    clean_loss = loss_total.item()/seen
    require(math.isfinite(clean_loss) and clean_loss >= 0, "Invalid mean clean loss")
    correct = int(correct_total.item())
    return {"contributions": decomposition(layout, tensors), "examples": seen, "batches": processed_batches,
            "clean_loss": clean_loss, "clean_correct": correct, "clean_accuracy": correct/seen}
