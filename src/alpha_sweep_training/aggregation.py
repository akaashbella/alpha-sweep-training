"""Complete prescribed summaries, validation-only selection and paired descriptions."""

from copy import deepcopy
import math
import statistics

from .data_contract import content_identity
from .identity import canonical_alpha
from .reporting_contract import CONVENTIONS, REPORTING_VERSION, require, sealed, validate_collection
from .spec import ALPHAS, ARCHITECTURES, DATASETS, EVALUATION_ALPHAS


def _summary(observations, expected):
    values = [row["value"] for row in observations if row["value"] is not None]
    complete = len(values) == expected
    return {"expected_count": expected, "eligible_count": len(values), "complete": complete,
        "individuals": observations, "exclusions": [row for row in observations if row["value"] is None],
        "official": None if not complete else {"mean": math.fsum(value / len(values) for value in values),
            "sample_sd": statistics.stdev(values) if len(values) >= 2 else None, "count": len(values)}}


def _unavailable(row):
    return row.get("diagnostic") or f"condition={row['condition']}; stored_status={row['stored_status']}"


def _signature(row):
    if not row["successful"]:
        return None
    record = row["raw_record"]
    data = deepcopy(record["data_identity"])
    data.pop("training_seed", None)  # Distinct prescribed seeds are the replication unit, not an incompatibility.
    return content_identity({"data": data, "protocol": row["requested_protocol"],
        "precision": record["runtime"]["precision"], "backend": record["runtime"]["effective_backend"],
        "torch": record["evaluation_provenance"]["software"]["torch"],
        "torchvision": record["evaluation_provenance"]["software"]["torchvision"]})


def _compatible_summary(rows, metric, expected):
    signatures = {_signature(row) for row in rows if row["successful"]}
    compatible = len(signatures) <= 1
    return _summary([{"training_run_id": row["training_run_id"], "training_seed": row["training_seed"],
        "logical_index": row["logical_index"], "raw_reference": row["raw_reference"],
        "observed_value": row["raw_record"]["test_metrics"][metric] if row["successful"] else None,
        "value": row["raw_record"]["test_metrics"][metric] if row["successful"] and compatible else None,
        "reason": None if row["successful"] and compatible else "incompatible data/protocol/runtime/software" if not compatible else _unavailable(row)}
        for row in rows], expected)


def _paired(left, right):
    if not left["successful"] or not right["successful"]:
        return None, f"positive/target: {_unavailable(left)}; baseline: {_unavailable(right)}"
    if _signature(left) != _signature(right):
        return None, "Incompatible data/protocol/runtime/software for paired comparison"
    return 100 * (left["raw_record"]["test_metrics"]["accuracy"] - right["raw_record"]["test_metrics"]["accuracy"]), None


def validation_selection(training_groups):
    """Only validation fields are read; exact co-winners never consult any test row."""
    selections = []
    for dataset in DATASETS:
        for architecture in ARCHITECTURES:
            available = {g["canonical_training_alpha"]: {"canonical_training_alpha": g["canonical_training_alpha"],
                "validation_accuracy": g["validation_accuracy"]} for g in training_groups
                if g["dataset"] == dataset and g["architecture"] == architecture}
            require(set(available) == {canonical_alpha(alpha) for alpha in ALPHAS}, "Validation selection requires the approved candidate grid")
            candidates = [available[canonical_alpha(alpha)] for alpha in ALPHAS]
            complete = [g for g in candidates if g["validation_accuracy"]["complete"]]
            ordered = sorted(complete, key=lambda g: -g["validation_accuracy"]["official"]["mean"])
            all_complete = len(complete) == len(ALPHAS)
            largest = None if not ordered else ordered[0]["validation_accuracy"]["official"]["mean"]
            ties = [g["canonical_training_alpha"] for g in candidates if g in complete and g["validation_accuracy"]["official"]["mean"] == largest]
            selections.append({"dataset": dataset, "architecture": architecture, "selection_available": all_complete,
                "status": "complete_grid_validation_selection" if all_complete else "incomplete_alpha_grid",
                "co_winners": ties if all_complete else [], "winning_mean_validation_accuracy": largest if all_complete else None,
                "missing_candidates": [{"canonical_training_alpha": g["canonical_training_alpha"],
                                        "exclusions": g["validation_accuracy"]["exclusions"]} for g in candidates if g not in complete],
                "provisional_complete_candidate_ranking": [{"canonical_training_alpha": g["canonical_training_alpha"],
                    "mean_validation_accuracy": g["validation_accuracy"]["official"]["mean"]} for g in ordered],
                "provisional_co_winners": ties if not all_complete else [], "uses_test_results": False})
    return selections


def summarize_results(collection):
    """No partial official summaries; all individual observations remain in the collection."""
    manifest = validate_collection(collection)
    training, evaluation = collection["training"], collection["evaluation"]
    slots = {(row["training_run_id"], row["slot"]["canonical_evaluation_alpha"], row["slot"]["evaluation_replicate"]): row for row in evaluation}
    clean = {row["training_run_id"]: row for row in evaluation if row["slot"]["evaluation_alpha"] == 0}
    training_groups, checkpoint_groups, robustness_groups = [], [], []
    clean_pairs, noise_pairs, pair_groups = [], [], []
    for dataset in DATASETS:
        for architecture in ARCHITECTURES:
            base = {row["training_seed"]: row for row in training if row["dataset"] == dataset and
                    row["architecture"] == architecture and row["canonical_training_alpha"] == canonical_alpha(0)}
            for alpha in ALPHAS:
                key = canonical_alpha(alpha)
                runs = [row for row in training if row["dataset"] == dataset and row["architecture"] == architecture and row["canonical_training_alpha"] == key]
                header = {"dataset": dataset, "architecture": architecture, "training_alpha": alpha, "canonical_training_alpha": key}
                validation = _summary([{"training_run_id": row["training_run_id"], "training_seed": row["training_seed"],
                    "value": row["best_validation_metrics"]["accuracy"] if row["condition"] == "ok" and row["training_complete"] else None,
                    "reason": None if row["condition"] == "ok" and row["training_complete"] else _unavailable(row)} for row in runs], 3)
                if validation["complete"]:
                    # All denominators are the verified fixed 5,000 validation
                    # images. Integer counts make exact mean ties order-independent.
                    validation["official"]["mean"] = sum(row["best_validation_metrics"]["correct"] for row in runs) / sum(row["best_validation_metrics"]["examples"] for row in runs)
                training_groups.append({**header, "training_run_ids": [row["training_run_id"] for row in runs],
                    "validation_accuracy": validation,
                    "clean_accuracy": _compatible_summary([clean[row["training_run_id"]] for row in runs], "accuracy", 3),
                    "clean_loss": _compatible_summary([clean[row["training_run_id"]] for row in runs], "loss", 3)})
                pairs = []
                for row in runs:
                    current, baseline = clean[row["training_run_id"]], clean[base[row["training_seed"]]["training_run_id"]]
                    settings, base_settings = deepcopy(row["scientific_settings"]), deepcopy(base[row["training_seed"]]["scientific_settings"])
                    del settings["alpha"], base_settings["alpha"]
                    difference, reason = _paired(current, baseline)
                    if settings != base_settings:
                        difference, reason = None, "Incompatible training settings for alpha-zero seed pair"
                    pair = {**header, "training_run_id": row["training_run_id"], "training_seed": row["training_seed"],
                        "baseline_run_id": baseline["training_run_id"], "value": difference, "reason": reason,
                        "unit": "percentage_points", "target_reference": current["raw_reference"], "baseline_reference": baseline["raw_reference"],
                        "target_accuracy": None if not current["successful"] else current["raw_record"]["test_metrics"]["accuracy"],
                        "baseline_accuracy": None if not baseline["successful"] else baseline["raw_record"]["test_metrics"]["accuracy"],
                        "target_examples": None if not current["successful"] else current["raw_record"]["test_metrics"]["examples"],
                        "baseline_examples": None if not baseline["successful"] else baseline["raw_record"]["test_metrics"]["examples"],
                        "eligible_pair_count": int(difference is not None)}
                    clean_pairs.append(pair)
                    pairs.append(pair)
                pair_groups.append({**header, "comparison": "clean_minus_training_alpha_zero", "summary": _summary(pairs, 3)})
                for eval_alpha in EVALUATION_ALPHAS:
                    eval_key = canonical_alpha(eval_alpha)
                    checkpoint_rows = []
                    for run in runs:
                        replicas = (0,) if eval_alpha == 0 else (1, 2, 3)
                        individual = [slots[(run["training_run_id"], eval_key, rep)] for rep in replicas]
                        accuracy = _compatible_summary(individual, "accuracy", len(replicas))
                        loss = _compatible_summary(individual, "loss", len(replicas))
                        changes = []
                        if eval_alpha > 0:
                            for raw in individual:
                                difference, reason = _paired(raw, clean[run["training_run_id"]])
                                pair = {"training_run_id": run["training_run_id"], "training_seed": run["training_seed"],
                                    "canonical_evaluation_alpha": eval_key, "evaluation_replicate": raw["slot"]["evaluation_replicate"],
                                    "value": difference, "reason": reason, "unit": "percentage_points",
                                    "target_reference": raw["raw_reference"], "baseline_reference": clean[run["training_run_id"]]["raw_reference"],
                                    "eligible_pair_count": int(difference is not None),
                                    "target_examples": None if not raw["successful"] else raw["raw_record"]["test_metrics"]["examples"],
                                    "baseline_examples": None if not clean[run["training_run_id"]]["successful"] else clean[run["training_run_id"]]["raw_record"]["test_metrics"]["examples"]}
                                changes.append(pair)
                                noise_pairs.append(pair)
                        checkpoint = {**header, "training_run_id": run["training_run_id"], "training_seed": run["training_seed"],
                            "evaluation_alpha": eval_alpha, "canonical_evaluation_alpha": eval_key,
                            "accuracy": accuracy, "loss": loss,
                            "noise_change_pp": _summary(changes, 3) if changes else None,
                            "compatibility_signature": _signature(individual[0]) if accuracy["complete"] else None,
                            "best_checkpoint_identity": individual[0]["best_checkpoint_identity"]}
                        checkpoint_groups.append(checkpoint)
                        checkpoint_rows.append(checkpoint)
                    signatures = {row["compatibility_signature"] for row in checkpoint_rows if row["accuracy"]["complete"]}
                    compatible = len(signatures) <= 1
                    group = {**header, "evaluation_alpha": eval_alpha, "canonical_evaluation_alpha": eval_key,
                        "within_checkpoint_sample_sds": [{"training_seed": c["training_seed"],
                            "realization_count": c["accuracy"]["eligible_count"],
                            "sample_sd": None if c["accuracy"]["official"] is None else c["accuracy"]["official"]["sample_sd"]} for c in checkpoint_rows]}
                    for name in ("accuracy", "loss", "noise_change_pp"):
                        if name == "noise_change_pp" and eval_alpha == 0:
                            group[name] = None
                            continue
                        group[name] = _summary([{"training_run_id": c["training_run_id"], "training_seed": c["training_seed"],
                            "value": c[name]["official"]["mean"] if compatible and c[name]["complete"] else None,
                            "reason": None if compatible and c[name]["complete"] else "incomplete checkpoint realization set" if compatible else "incompatible cross-seed evaluation settings"}
                            for c in checkpoint_rows], 3)
                    robustness_groups.append(group)
    coverage = {"expected_training": len(training), "expected_evaluation_slots": len(evaluation),
        "training_completed": sum(r["condition"] == "ok" and r["training_complete"] for r in training),
        "successful_slots": sum(r["successful"] for r in evaluation),
        "robustness_completed_checkpoints": sum(bool(r["evaluation_coverage"] and r["evaluation_coverage"]["robustness_completed"]) for r in training),
        "terminal_numerical_training": sum(r["terminal_numerical"] for r in training),
        "execution_failed_training": sum(r["execution_failed"] for r in training),
        "training_finalization_pending": sum(r["condition"] == "ok" and r["training_finalization_pending"] for r in training),
        "orphan_evaluation_failure_slots": sum(bool(r["orphan_failures"]) for r in evaluation),
        "checkpoint_completion_diagnostics": [{"training_run_id": r["training_run_id"],
            **{key: r["evaluation_coverage"][key] for key in ("robustness_completed", "all_slots_successful",
                "completion_publication_pending", "completion_inspection_deferred", "completion_deferred_reason", "completion_inspection_error")}}
            for r in training if r["evaluation_coverage"] is not None],
        "training_conditions": {k: sum(r["condition"] == k for r in training) for k in sorted({r["condition"] for r in training})},
        "training_statuses": {str(k): sum(r["stored_status"] == k for r in training) for k in sorted({r["stored_status"] for r in training}, key=str)},
        "evaluation_conditions": {k: sum(r["condition"] == k for r in evaluation) for k in sorted({r["condition"] for r in evaluation})},
        "evaluation_statuses": {str(k): sum(r["stored_status"] == k for r in evaluation) for k in sorted({r["stored_status"] for r in evaluation}, key=str)},
        "source_diagnostics": collection["diagnostics"]}
    return sealed({"reporting_version": REPORTING_VERSION, "kind": "result_summaries", "conventions": CONVENTIONS,
        "grid_id": manifest.grid_id, "manifest_document_sha256": manifest.document_sha256,
        "collection_sha256": collection["document_sha256"], "training_groups": training_groups,
        "checkpoint_robustness": checkpoint_groups, "robustness_groups": robustness_groups,
        "clean_seed_pairs": clean_pairs, "noise_realization_pairs": noise_pairs, "paired_training_groups": pair_groups,
        "validation_selections": validation_selection(training_groups), "coverage": coverage})
