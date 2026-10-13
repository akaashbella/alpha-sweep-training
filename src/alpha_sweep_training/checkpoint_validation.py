"""Saved component validation without mutating model, optimizer or global RNGs."""

import math

import torch

from .checkpoint_io import exact, fail, same
from .metrics import Counters
from .data_contract import CONTINUATION_VERSION
from .numerical import require_finite
from .scheduler import saved_schedule_rates
from .spec import EPOCHS, RUNTIME_RANDOMNESS_VERSION


def validate_scaler(state: dict | None, mode: str, *, fresh: bool = False) -> None:
    if mode != "amp_float16":
        if state is not None:
            fail("Float32/bfloat16 checkpoint must not carry scaler state")
        return
    value = exact(state, {"scale", "growth_factor", "backoff_factor", "growth_interval", "_growth_tracker"}, "scaler state")
    for key, expected in (("growth_factor", 2.0), ("backoff_factor", 0.5), ("growth_interval", 2000)):
        same(value[key], expected, f"scaler {key}")
    if type(value["scale"]) not in (int, float) or not math.isfinite(value["scale"]) or value["scale"] <= 0:
        fail("Scaler scale must be finite and positive")
    if type(value["_growth_tracker"]) is not int or not 0 <= value["_growth_tracker"] < 2000:
        fail("Invalid scaler growth tracker")
    if fresh and (value["scale"] != 65536.0 or value["_growth_tracker"] != 0):
        fail("Epoch-zero checkpoint must carry the pinned fresh scaler state")


def validate_optimizer_state(state: dict, metadata: dict, inventory: list[dict], config, counters: Counters,
                             learning_rates: tuple[float, ...]) -> None:
    exact(state, {"state", "param_groups"}, "optimizer state")
    names = metadata["parameter_names"]
    parameters = {entry["name"]: entry for entry in inventory if entry["kind"] == "parameter"}
    same(sorted(parameters), names, "optimizer parameter order")
    same(metadata["declared_recipe"], config.scientific["optimizer"], "optimizer recipe")
    same(metadata["parameter_groups"], 1, "optimizer group count")
    same(metadata["base_learning_rates"], [config.scientific["optimizer"]["learning_rate"]], "base optimizer LR")
    same(metadata["current_learning_rates"], list(learning_rates), "saved optimizer metadata LR")
    groups = state["param_groups"]
    if type(groups) is not list or len(groups) != 1:
        fail("Saved optimizer must have exactly one group")
    group = exact(groups[0], set(metadata["effective_options"]) | {"params", "lr"}, "optimizer group")
    same({key: value for key, value in group.items() if key not in ("params", "lr")}, metadata["effective_options"], "optimizer options")
    same(group["params"], list(range(len(names))), "optimizer parameter IDs/order")
    same([group["lr"]], list(learning_rates), "saved optimizer LR")
    recipe_fields = config.scientific["optimizer"]
    same(group["weight_decay"], recipe_fields["weight_decay"], "saved weight decay recipe")
    for option, field in (("momentum", "momentum"), ("alpha", "rmsprop_alpha"),
                          ("eps", "epsilon"), ("betas", "betas")):
        if recipe_fields[field] is not None:
            same(group[option], recipe_fields[field], f"saved optimizer recipe {option}")
    same(metadata["parameter_count"], len(names), "optimizer parameter count")
    same(metadata["parameter_elements"], sum(entry["numel"] for entry in parameters.values()), "optimizer elements")
    if group.get("capturable", False) or group.get("fused") not in (None, False) or group.get("differentiable", False):
        fail("Only pinned native noncapturable, unfused, nondifferentiable state is supported")
    states = state["state"]
    if type(states) is not dict or any(type(key) is not int for key in states):
        fail("Saved optimizer state IDs must be integers")
    expected_ids = set(range(len(names))) if counters.successful_updates else set()
    if set(states) != expected_ids:
        fail("Saved optimizer required state disagrees with successful updates")
    recipe = config.scientific["optimizer"]["name"]
    required = {"sgd": {"momentum_buffer"}, "rmsprop": {"step", "square_avg", "momentum_buffer"},
                "adamw": {"step", "exp_avg", "exp_avg_sq"}}[recipe]
    tensors = []
    for index, value in states.items():
        exact(value, required, f"optimizer state for {names[index]}")
        entry = parameters[names[index]]
        for key, tensor in value.items():
            if type(tensor) is not torch.Tensor or tensor.device.type != "cpu" or tensor.layout != torch.strided:
                fail("Optimizer checkpoint state must contain dense CPU tensors")
            if key == "step":
                if tensor.shape != () or tensor.dtype not in (torch.float32, torch.float64) or float(tensor.item()) != counters.successful_updates:
                    fail("Native CPU optimizer step counter disagrees with confirmed updates")
            elif list(tensor.shape) != entry["shape"] or str(tensor.dtype) != entry["dtype"]:
                fail(f"Optimizer tensor shape/dtype mismatch: {names[index]}.{key}")
            tensors.append((f"{names[index]}.{key}", tensor))
    require_finite(tensors, "saved_optimizer_state")


def validate_latest_components(payload: dict, config, counters: Counters) -> None:
    from .checkpoints import decode_rng
    from .randomness import runtime_seeds
    meta, state = payload["metadata"], payload["continuation"]
    components = meta["component_metadata"]
    data = meta["data_identity"]
    rates = saved_schedule_rates(config, data["counts"]["training_batches"], state["scheduler"],
                                 counters.completed_epochs, counters.successful_updates)
    for key, value in state["scheduler"].items():
        same(components["scheduler"][key], value, f"scheduler component metadata {key}")
    same(components["scheduler"]["current_learning_rates"], list(rates), "scheduler metadata LR")
    validate_optimizer_state(state["optimizer"], components["optimizer"], meta["inventory"], config, counters, rates)
    validate_scaler(state["scaler"], config.scientific["precision"]["mode"], fresh=counters.completed_epochs == 0)
    same(components["runtime"]["precision"], config.scientific["precision"], "runtime precision")
    same(components["runtime"]["master_dtype"], "float32", "master dtype")
    same(components["runtime"]["evaluation_precision"], "same_autocast_policy_as_training; no_scaling", "evaluation precision")
    rng = decode_rng(state["rng"])
    same(rng["descriptor"], {"format_version": 1, "randomness_version": RUNTIME_RANDOMNESS_VERSION,
         "dataset": config.scientific["dataset"], "architecture": config.scientific["architecture"],
         "training_seed": config.scientific["training_seed"], "seeds": runtime_seeds(config),
         "torch": components["randomness"]["torch"], "numpy": components["randomness"]["numpy"], "device_type": "cuda"}, "RNG descriptor")
    same({key: value for key, value in components["randomness"].items() if key not in ("assigned_device", "ownership")},
         rng["descriptor"], "RNG component metadata")
    continuation = state["data_continuation"]
    expected = {"format_version": CONTINUATION_VERSION, "boundary": "completed_epoch", "next_epoch_index": counters.completed_epochs,
                "original_epochs": EPOCHS, "dataset": data["dataset"], "training_seed": data["training_seed"],
                "split_identity": data["split_identity"], "normalization_identity": data["normalization_identity"],
                "preprocessing_identity": data["preprocessing_identity"], "official_training_identity": data["official_training_identity"],
                "official_test_identity": data["official_test_identity"], "randomness_version": data["randomness_version"],
                "batch_size": data["counts"]["batch_size"], "drop_last": data["drop_last"], "data_rng_software": data["data_rng_software"]}
    same(continuation, expected, "saved data continuation")
