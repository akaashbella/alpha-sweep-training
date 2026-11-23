"""One fixed functional Gaussian realization over a complete ordered split."""

from contextlib import nullcontext
import math

import torch
from torch.func import functional_call

from alpha_sweep_training.evaluation import classification_loss
from alpha_sweep_training.evaluation_state import CleanStateBaseline
from alpha_sweep_training.metrics import MetricAccumulator
from alpha_sweep_training.numerical import require_finite
from alpha_sweep_training.perturbation import gaussian_weight_offset
from alpha_sweep_training.weights import discover_eligible_weights, named_parameters_once

from .planning import tensor_seed
from .storage import require


def inventory(model, layout):
    eligible = discover_eligible_weights(model)
    require([e.to_dict() for e in eligible] == layout["tensors"], "Model eligible weights differ from audited inventory")
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
        scales = {entry.name: torch.std(entry.parameter.detach(), correction=0) for entry in eligible}
        require_finite(scales.items(), "stage2_clean_population_std")
    return {**layout, "tensors": [{**e.to_dict(), "population_std": float(scales[e.name].cpu())} for e in eligible]}, scales


def parameters_for(model, scales, selected, checkpoint, alpha, realization):
    parameters = dict(named_parameters_once(model))
    if alpha == 0:
        require(realization == 0 and not selected, "Clean baseline cannot select perturbations")
        return parameters, {}
    require(alpha > 0 and type(realization) is int and realization > 0 and
            len(set(selected)) == len(selected) and selected and set(selected) <= set(scales), "Invalid perturbation target")
    seeds = {name: tensor_seed(checkpoint, alpha, realization, name) for name in selected}
    require(len(set(seeds.values())) == len(seeds), "Per-tensor RNG seed collision")
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
        for name, seed in seeds.items():
            clean = parameters[name]
            require(clean.dtype == torch.float32, "Expected clean float32 master weights")
            generator = torch.Generator(device=clean.device).manual_seed(seed)
            offset = gaussian_weight_offset(clean.shape, strength=alpha, clean_population_std=scales[name],
                                            generator=generator, device=clean.device)
            parameters[name] = clean.detach() + offset
        require_finite(((name, parameters[name]) for name in selected), "stage2_fixed_weights")
    return parameters, seeds


def measure(model, loader, *, baseline, device, classes, expected, scales, selected, checkpoint,
            alpha, realization, prepare_batch=lambda batch: batch, autocast=nullcontext):
    """Supports deterministic synthetic loaders and the separately verified CIFAR loader."""
    require(baseline.model is model, "Clean baseline belongs to another model")
    accumulator = MetricAccumulator(torch.device(device))
    predictions, labels = bytearray(), bytearray()
    with baseline.realization():
        model.eval()
        parameters, seeds = parameters_for(model, scales, selected, checkpoint, alpha, realization)
        buffers = {name: value.detach().clone() for name, value in model.named_buffers()} if alpha else {}
        before = {name: tensor.detach().to("cpu", copy=True) for name, tensor in buffers.items()}
        with torch.inference_mode():
            for batch in loader:
                images, batch_labels = prepare_batch(batch)
                with autocast():
                    if alpha:
                        logits = functional_call(model, (parameters, buffers), (images,), strict=True, tie_weights=True)
                    else:
                        logits = model(images)
                loss = classification_loss(logits, batch_labels, classes)
                accumulator.add(loss, logits, batch_labels)
                predictions.extend(logits.argmax(dim=1).cpu().tolist())
                labels.extend(batch_labels.cpu().tolist())
                require(len(labels) <= expected, "Pass exceeded the declared split")
        require(all(torch.equal(buffer.detach().cpu(), before[name]) for name, buffer in buffers.items()),
                "Perturbed forward changed private normalization buffers")
        result = accumulator.finish().to_dict()
        require(result["examples"] == expected and len(labels) == expected and result["batches"] == len(loader),
                f"Incomplete pass: {len(labels)}/{expected} examples")
    return result, bytes(labels), bytes(predictions), seeds


def changes(clean_metrics, metrics, labels, clean, perturbed):
    n = len(labels)
    require(n and n == len(clean) == len(perturbed) == clean_metrics["examples"] == metrics["examples"],
            "Clean and perturbed predictions must cover the same complete examples")
    flips = sum(a != b for a, b in zip(clean, perturbed))
    lost = sum(a == y and b != y for y, a, b in zip(labels, clean, perturbed))
    gained = sum(a != y and b == y for y, a, b in zip(labels, clean, perturbed))
    require(clean_metrics["correct"] == sum(a == y for a, y in zip(clean, labels)) and
            metrics["correct"] == sum(b == y for b, y in zip(perturbed, labels)), "Prediction/metric correct counts disagree")
    return {"delta_loss": metrics["loss"]-clean_metrics["loss"],
            "accuracy_change_pp": 100*(gained-lost)/n, "prediction_flip_rate": flips/n,
            "prediction_flips": flips, "correct_to_incorrect": lost, "incorrect_to_correct": gained}


def normalization(inv, selected, alpha, delta_loss):
    counts = {r["name"]: (r["numel"], r["population_std"]) for r in inv["tensors"]}
    total = sum(n for n, _ in counts.values())
    count = sum(counts[name][0] for name in selected)
    energy = alpha**2 * math.fsum(counts[name][0]*counts[name][1]**2 for name in selected)
    return {"eligible_parameters": total, "perturbed_parameters": count, "fraction_eligible_perturbed": count/total,
            "expected_squared_perturbation_norm": energy,
            "delta_loss_per_million_parameters": 1e6*delta_loss/count if count else None,
            "delta_loss_per_expected_squared_norm": delta_loss/energy if energy else None}
