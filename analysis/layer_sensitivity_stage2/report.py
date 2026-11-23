"""Complete-only realization means, training-seed summaries and paired residuals."""

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from .planning import planned_slots
from .records import inspect_checkpoint, run_root, slot_id
from .storage import disjoint, finish, identity, install, seal, source_digest

METRICS = ("delta_loss", "accuracy_change_pp", "prediction_flip_rate", "correct_to_incorrect",
           "incorrect_to_correct", "delta_loss_per_million_parameters", "delta_loss_per_expected_squared_norm",
           "fraction_eligible_perturbed", "expected_squared_perturbation_norm", "clean_accuracy", "perturbed_accuracy",
           "clean_loss", "perturbed_loss")


def descriptive(values, expected):
    valid = [value for value in values if value is not None]
    complete = len(valid) == expected
    return {"count": len(valid), "expected": expected, "complete": complete,
            "mean": math.fsum(v/expected for v in valid) if complete else None,
            "sample_sd": statistics.stdev(valid) if complete and expected >= 2 else None}


def stable_ratio(delta, full, clean_loss):
    threshold = max(1e-8, 1e-6*abs(clean_loss))
    return delta/full if abs(full) > threshold else None


def compare_training(summaries):
    """Compare complete, compatible seed-level means while retaining exclusions."""
    metrics = {"delta_loss", "accuracy_change_pp", "prediction_flip_rate"}
    by_key = {}
    for row in summaries:
        if row["metric"] in metrics and row["evaluation_alpha"] > 0:
            dims = tuple(row[k] for k in ("dataset", "architecture", "training_seed", "target_id", "evaluation_alpha", "metric"))
            by_key[dims, row["training_alpha"]] = row
    comparisons = []
    for (dims, alpha), row in by_key.items():
        if alpha == 0:
            continue
        baseline = by_key.get((dims, 0.))
        reason = None
        if baseline is None or not baseline["complete"] or not row["complete"]:
            reason = "Missing complete checkpoint realization mean"
        elif baseline["compatibility_signature"] != row["compatibility_signature"]:
            reason = "Incompatible training recipe, prepared data or evaluation runtime"
        comparisons.append({k: row[k] for k in ("dataset", "architecture", "training_seed", "training_alpha", "target_id", "scope",
                            "target_name", "evaluation_alpha", "metric", "checkpoint_id")} |
                           {"clean_training_checkpoint_id": None if baseline is None else baseline["checkpoint_id"],
                            "clean_training_mean": None if baseline is None else baseline["mean"], "noise_training_mean": row["mean"],
                            "noise_minus_clean_training": None if reason else row["mean"]-baseline["mean"], "reason": reason})
    return comparisons


def collect(plan, plan_dir, results):
    root = run_root(plan, plan_dir, results, create=False)
    raw, summaries, residuals, adds, coverage = [], [], [], [], []
    bins = defaultdict(list)
    for row in plan["checkpoints"]:
        key = {**row["key"], "checkpoint_id": row["checkpoint_id"]}
        if row["status"] != "eligible":
            coverage.append({**key, "status": "excluded", "reason": row["reason"]})
            continue
        try:
            records, details, _, marker, evidence = inspect_checkpoint(plan, row, root)
        except Exception as exc:
            # Reject the entire checkpoint's analytical evidence if any declared
            # authority is corrupt. Retain its prescribed slots as unavailable.
            records, marker, evidence = {}, False, {}
            details = [{**slot, "status": "rejected", "reason": f"{type(exc).__name__}: {exc}"}
                       for slot in planned_slots(plan, row)]
        coverage.append({**key, "status": "complete" if marker else "publication_pending" if len(records) == row["passes"] else "incomplete",
                         "expected_passes": row["passes"], "successful_passes": len(records), "slots": details,
                         **evidence})
        cohort_runtime = next(iter(records.values()))["runtime"] if records else None
        science = {k: v for k, v in row["evidence"]["scientific"].items() if k not in ("alpha", "training_seed")}
        compatibility = identity({"science_except_alpha_and_seed": science,
                                  "prepared": row["evidence"]["prepared_data"], "runtime": cohort_runtime})
        pairs = {(r["slot"]["target_id"], r["slot"]["evaluation_alpha"], r["slot"]["realization"]): r
                 for r in records.values()}
        count = plan["realizations_per_positive_alpha"]
        by_slot = {}
        for (target, alpha, rep), record in pairs.items():
            observation = {**key, **record["slot"], **record["changes"], **record["normalization"],
                           "run_id": row["run_id"], "best_checkpoint": record["binding"]["checkpoint"],
                           "scientific_fingerprint": record["binding"]["training_identity"]["scientific_fingerprint"],
                           "recipe_version": row["evidence"]["scientific"].get("specification_version"),
                           "data_split": record["data"]["split"], "examples": record["metrics"]["examples"],
                           "duration_seconds": record["duration_seconds"],
                           "clean_accuracy": record["clean_metrics"]["accuracy"], "perturbed_accuracy": record["metrics"]["accuracy"],
                           "clean_loss": record["clean_metrics"]["loss"], "perturbed_loss": record["metrics"]["loss"],
                           "record_sha256": record["document_sha256"]}
            full = pairs.get(("network:all", alpha, rep))
            observation["group_fraction_of_full_delta_loss"] = None if full is None else stable_ratio(
                observation["delta_loss"], full["changes"]["delta_loss"], observation["clean_loss"])
            raw.append(observation)
            by_slot[target, alpha, rep] = observation
        target_defs = [{"id": "clean", "scope": "clean", "name": "clean"}, *row["targets"]]
        for target in target_defs:
            alphas = [0.] if target["scope"] == "clean" else plan["evaluation_alphas"][1:]
            for alpha in alphas:
                selected = [by_slot.get((target["id"], alpha, rep)) for rep in (range(1, count+1) if alpha else (0,))]
                valid = [r for r in selected if r is not None]
                metadata = {**key, "target_id": target["id"], "scope": target["scope"], "target_name": target["name"],
                            "evaluation_alpha": alpha, "eligible_realizations": len(valid), "expected_realizations": len(selected),
                            "compatibility_signature": compatibility if valid else None}
                for metric in (*METRICS, "group_fraction_of_full_delta_loss"):
                    summary = descriptive([r[metric] for r in valid], len(selected))
                    entry = {**metadata, "metric": metric, **summary}
                    summaries.append(entry)
                    bins[(key["dataset"], key["architecture"], key["training_alpha"], target["id"], target["scope"],
                          target["name"], alpha, metric)].append((entry, compatibility))
        stages = [t["id"] for t in row["targets"] if t["scope"] == "stage"]
        if stages:
            for alpha in plan["evaluation_alphas"][1:]:
                residual_values = []
                for rep in range(1, count+1):
                    full = by_slot.get(("network:all", alpha, rep))
                    isolated = [by_slot.get((stage, alpha, rep)) for stage in stages]
                    a = math.fsum(r["delta_loss"] for r in isolated) if all(r is not None for r in isolated) else None
                    b = full["delta_loss"] if full else None
                    difference = b-a if a is not None and b is not None else None
                    residual_values.append(difference)
                    residuals.append({**key, "evaluation_alpha": alpha, "realization": rep, "sum_isolated": a,
                                      "full_network": b, "interaction_residual": difference,
                                      "missing_targets": [name for name, item in zip(stages, isolated) if item is None]
                                                         + (["network:all"] if full is None else [])})
                adds.append({**key, "evaluation_alpha": alpha, **descriptive(residual_values, count)})
    across = []
    for group, members in sorted(bins.items()):
        keys = ("dataset", "architecture", "training_alpha", "target_id", "scope", "target_name", "evaluation_alpha", "metric")
        seed_values = {r["training_seed"]: r["mean"] for r, _ in members}
        expected = plan["selected_seeds"]
        runtimes = [rt for r, rt in members if r["mean"] is not None]
        compatible = all(rt == runtimes[0] for rt in runtimes) if runtimes else True
        values = descriptive([seed_values.get(seed) for seed in expected], len(expected))
        # A one-seed pilot reports its checkpoint mean with no cross-seed SD.
        if not compatible:
            values.update(complete=False, mean=None, sample_sd=None)
        across.append({**dict(zip(keys, group)), "compatible": compatible, **values,
                       "seed_means": {str(seed): seed_values.get(seed) for seed in expected}})
    return {"raw": raw, "checkpoint_summaries": summaries, "cross_seed_summaries": across, "training_pairs": compare_training(summaries), "residuals": residuals,
            "residual_summaries": adds, "coverage": coverage}


def write_csv(path, rows, fallback_fields):
    with path.open("x", newline="", encoding="utf-8") as stream:
        fields = list(dict.fromkeys(key for row in rows for key in row)) or fallback_fields
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True, allow_nan=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def findings(result, plan):
    coverage = result["coverage"]
    accepted = sum(row.get("successful_passes", 0) for row in coverage)
    excluded = sum(row["status"] == "excluded" for row in coverage)
    lines = ["# Stage 2 layer sensitivity", "",
             f"The analysis used {accepted} of {plan['estimates']['passes']} executable passes on the "
             f"{plan['split']} split. The audit excluded {excluded} of {len(coverage)} requested checkpoints.", "",
             "Positive loss change indicates a higher cross-entropy loss. Negative accuracy change indicates lower accuracy. "
             "Prediction flips include changes between two incorrect classes.", "",
             "| Dataset | Architecture | Training alpha | Evaluation alpha | Largest observed complete raw stage mean | Delta CE |",
             "| --- | --- | --- | --- | --- | --- |"]
    groups = defaultdict(list)
    for row in result["cross_seed_summaries"]:
        if row["metric"] == "delta_loss" and row["scope"] == "stage" and row["complete"]:
            groups[(row["dataset"], row["architecture"], row["training_alpha"], row["evaluation_alpha"])].append(row)
    for key, rows in sorted(groups.items()):
        maximum = max(r["mean"] for r in rows)
        tied = ", ".join(r["target_name"] for r in rows if r["mean"] == maximum)
        lines.append(f"| {key[0]} | {key[1]} | {key[2]:g} | {key[3]:g} | {tied} | {maximum:.6g} |")
    if not groups:
        lines.extend(["", "No complete stage means were available. See coverage.json."])
    complete_pairs = [r for r in result["training_pairs"] if r["metric"] == "delta_loss" and r["scope"] == "stage"
                      and r["noise_minus_clean_training"] is not None]
    complete_residuals = [r for r in result["residuals"] if r["interaction_residual"] is not None]
    lines.extend(["", f"Compatible seed-level training comparisons are available for {len(complete_pairs)} stage/alpha cells. "
                  f"{sum(r['noise_minus_clean_training'] < 0 for r in complete_pairs)} have lower signed loss change after noise training. "
                  "The comparisons are stratified in training_pairs.csv. Counts combine repeated observations from the same checkpoint.", "",
                  f"Paired whole-network residuals are available for {len(complete_residuals)} realizations. "
                  "The residual tables retain each evaluation alpha, signed residual and missing stage identity."])
    lines.extend(["", "Stage comparisons are conditional on the checkpoint, data split, perturbation scale and target sizes. "
                  "The validation pilot is exploratory because these examples also selected the best checkpoints. "
                  "Layer sensitivity alone cannot establish a layer's causal importance for all model behavior.", "",
                  "Stage 2 uses paired draws to compare the sum of isolated stages with the whole-network response. "
                  "The residual contains Monte Carlo error and may reflect nonlinear effects. Estimating a particular layer interaction "
                  "or a direct Hessian trace requires additional analysis.", "",
                  "Within-checkpoint sample SD describes perturbation realizations. Cross-seed SD describes checkpoint means "
                  "where the plan includes multiple eligible seeds. Each architecture has its own stage and optimizer definitions.", ""])
    return "\n".join(lines)


def summarize(plan, plan_dir, results, output):
    target = disjoint(output, [plan_dir, results, *plan["protected_roots"]])
    result = collect(plan, plan_dir, results)
    target.mkdir(parents=True)
    files = []
    for table in ("raw", "checkpoint_summaries", "cross_seed_summaries", "training_pairs", "residuals", "residual_summaries"):
        name = f"{table}.csv"
        write_csv(target / name, result[table], ["dataset", "architecture", "training_alpha", "evaluation_alpha"])
        files.append(name)
    install(target / "coverage.json", seal("coverage", {"rows": result["coverage"]}))
    (target / "findings.md").write_text(findings(result, plan), encoding="utf-8")
    # Only these compact tables are required by plotting.
    document = seal("summary", {"plan_sha256": plan["document_sha256"], "source_sha256": source_digest(),
                   "protected_roots": plan["protected_roots"],
                   "plan_dir": str(Path(plan_dir).resolve()), "results": str(Path(results).resolve()), "split": plan["split"],
                   "cross_seed_summaries": result["cross_seed_summaries"],
                   "checkpoint_summaries": result["checkpoint_summaries"], "training_pairs": result["training_pairs"], "residuals": result["residuals"]})
    install(target / "summary.json", document)
    finish(target, "summary", [*files, "coverage.json", "summary.json", "findings.md"])
    return {"output": str(target), "checkpoints": len(result["coverage"]), "accepted_passes": len(result["raw"])}
