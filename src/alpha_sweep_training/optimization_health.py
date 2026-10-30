"""Descriptive CIFAR-10 health observations, separate from scientific completion."""

from collections import OrderedDict
from copy import deepcopy
import math
import statistics

from .data_contract import content_identity
from .diagnostic_protocol import DiagnosticError, positive


def validate_health_policy(policy):
    keys = {"near_chance_margin", "extreme_loss", "early_best_epochs", "weak_final_accuracy", "seed_accuracy_range"}
    if type(policy) is not dict or set(policy) != keys:
        raise DiagnosticError(f"Supply explicit descriptive thresholds: {sorted(keys)}")
    for key in ("near_chance_margin", "weak_final_accuracy", "seed_accuracy_range"):
        if type(policy[key]) not in (int, float) or not math.isfinite(policy[key]) or not 0 <= policy[key] <= 1:
            raise DiagnosticError(f"{key} must be a finite accuracy fraction in [0,1]")
    positive(policy["extreme_loss"], "extreme_loss")
    if type(policy["early_best_epochs"]) is not int or not 1 <= policy["early_best_epochs"] <= 200:
        raise DiagnosticError("early_best_epochs is an explicit 1-based epoch threshold")
    return deepcopy(policy)


def accuracy(metric):
    return metric["correct"] / metric["examples"]


def analyze_run(row, policy):
    """Pure analysis of an already authority-validated training row; no filtering."""
    identity = {key: row[key] for key in ("training_run_id", "dataset", "architecture", "training_alpha", "training_seed")}
    result = {**identity, "condition": row["condition"], "stored_status": row["stored_status"],
              "training_complete": row["training_complete"], "evaluation_coverage": deepcopy(row.get("evaluation_coverage")),
              "research_authority": False, "health_assessment": "unavailable", "flags": None,
              "best_epoch_label": None, "best_validation_accuracy": None, "epochs": []}
    snapshot = row["snapshot"]
    if snapshot is None or not snapshot["epochs"]:
        result["unavailable_reason"] = "No accepted committed epoch snapshot; no health values imputed"
        return result
    epochs = [entry["document"] for entry in snapshot["epochs"]]
    best = snapshot["transaction"]["best"]
    last = epochs[-1]["record"]
    classes = row["scientific_settings"]["model"]["num_classes"]
    extremes = []
    medians = []
    for index, document in enumerate(epochs):
        record = document["record"]
        for split in ("training", "validation"):
            if record[split]["loss"] > policy["extreme_loss"]:
                extremes.append({"epoch_label": record["epoch_index"] + 1, "split": split, "loss": record[split]["loss"]})
        values = [stat["population_std"] for stat in document.get("statistics", [])]
        medians.append(statistics.median(values) if values else document.get("exported_median_population_std"))
        result["epochs"].append({"epoch_label": record["epoch_index"] + 1,
            "source_reference": snapshot["epochs"][index].get("reference", document.get("exported_reference")),
            "original_epoch_record": deepcopy(record),
            "training": deepcopy(record["training"]), "clean_validation": deepcopy(record["validation"]),
            "median_eligible_tensor_population_std": medians[-1]})
    best_epoch = best["epoch_index"] + 1
    result.update(health_assessment="descriptive_only", best_epoch_label=best_epoch,
        best_validation_accuracy=best["correct"] / best["examples"],
        final_training_accuracy=accuracy(last["training"]), final_clean_validation_accuracy=accuracy(last["validation"]),
        extreme_loss_observations=extremes, first_extreme_loss=None if not extremes else extremes[0],
        median_weight_sd_last_to_first_recorded_epoch_ratio=(None if medians[0] in (None, 0.) or medians[-1] is None else medians[-1] / medians[0]),
        median_weight_sd_ratio_numerator_epoch_label=last["epoch_index"] + 1,
        median_weight_sd_ratio_denominator_epoch_label=epochs[0]["record"]["epoch_index"] + 1,
        median_weight_sd_ratio_boundary="post_training_epoch; denominator is not initialization",
        flags={"near_chance_final_training": accuracy(last["training"]) <= 1 / classes + policy["near_chance_margin"],
               "near_chance_final_clean_validation": accuracy(last["validation"]) <= 1 / classes + policy["near_chance_margin"],
               "extreme_finite_loss": bool(extremes),
               "early_best_and_weak_final": best_epoch <= policy["early_best_epochs"]
                    and accuracy(last["validation"]) <= policy["weak_final_accuracy"]})
    return result


def health_report(collection, policy):
    from .reporting_contract import validate_collection
    policy = validate_health_policy(policy)
    manifest = validate_collection(collection)
    selected = [row for row in collection["training"] if row["dataset"] == "cifar10"]
    expected = [task for task in manifest.tasks if task.config.scientific["dataset"] == "cifar10"]
    if len(selected) != len(expected) or len(selected) != 120:
        raise DiagnosticError("Health reporting must preserve all 120 CIFAR-10 identities")
    rows = [analyze_run(row, policy) for row in selected]
    groups = OrderedDict()
    for row in rows:
        groups.setdefault((row["architecture"], row["training_alpha"]), []).append(row)
    seeds = list(expected[0].config.scientific["training_seeds"])
    summaries = []
    for (architecture, alpha), members in groups.items():
        complete = len(members) == 3 and set(m["training_seed"] for m in members) == set(seeds) and all(
            m["training_complete"] and m["best_validation_accuracy"] is not None for m in members)
        values = [m["best_validation_accuracy"] for m in members] if complete else []
        summaries.append({"architecture": architecture, "training_alpha": alpha, "prescribed_seeds": seeds,
            "individuals": [{"training_seed": m["training_seed"], "best_validation_accuracy": m["best_validation_accuracy"]} for m in members],
            "complete_three_seed_validation": complete,
            "mean_validation_accuracy": statistics.mean(values) if complete else None,
            "sample_sd": statistics.stdev(values) if complete else None,
            "accuracy_range": max(values) - min(values) if complete else None,
            "severe_seed_variation": max(values) - min(values) >= policy["seed_accuracy_range"] if complete else None})
    return {"health_version": 1, "kind": "descriptive_optimization_health", "research_authority": False,
        "source_collection_sha256": collection["document_sha256"], "health_policy": policy,
        "health_policy_identity": content_identity(policy), "training_alpha_selection": "unchanged; not performed here",
        "score_filtering": False, "early_stopping": False, "individual_runs": rows, "seed_groups": summaries}
