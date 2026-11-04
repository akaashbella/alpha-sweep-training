"""Read-only all-30/840 reporting. Derived reports never establish completion."""

import argparse
import csv
from pathlib import Path
import statistics

import launcher as op


def collect(plan, manifest):
    from alpha_sweep_training.coverage import inspect_task, _stamp, _lock_paths
    from alpha_sweep_training.artifacts import resolve_run_paths
    from alpha_sweep_training.evaluation_inputs import bind_evaluation_checkpoint
    from alpha_sweep_training.evaluation_coverage import collect_evaluation_coverage
    from alpha_sweep_training.evaluation_protocol import prescribed_slots, perturbation_seed
    from alpha_sweep_training.evaluation_persistence import _read_reference, validate_result, read_slot
    from alpha_sweep_training.planning_inspection import evidence_stamp
    training, evaluations = [], []
    for task in manifest.tasks:
        science = task.config.scientific
        row = {"global_index": task.global_index, "run_id": task.run_id,
               "scientific_fingerprint": task.fingerprint, "training_alpha": science["alpha"],
               "training_seed": science["training_seed"], "training_completion_verified": False,
               "robustness_completion_verified": False, "best_validation": None,
               "actual_binding": None, "error": None, "evaluation_coverage": None}
        slots = [{"training_global_index": task.global_index, "training_run_id": task.run_id,
                  "training_alpha": science["alpha"], "training_seed": science["training_seed"],
                  **slot.to_dict(), "condition": "unbound", "stored_status": None,
                  "best_checkpoint_identity": None, "perturbation_seed": None, "reference": None,
                  "test_metrics": None, "raw_history": [], "orphan_failures": [], "raw_records": [],
                  "attempts": [], "unparsed_evidence": [], "diagnostic": None}
                 for slot in prescribed_slots()]
        paths, before = None, None
        try:
            observation = inspect_task(task, base_dir=plan["native"]["base_dir"])
            row["training_observation"] = observation.to_dict()
            op.require(observation.condition == "ok" and observation.training_complete,
                       "Training authority is incomplete or unstable")
            paths = resolve_run_paths(task.config, plan["native"]["base_dir"])
            before = _stamp(paths)
            op.require(not any(p.exists() or p.is_symlink() for p in _lock_paths(paths)), "Training writer present")
            inputs = bind_evaluation_checkpoint(task, base_dir=plan["native"]["base_dir"], metadata_only=True)
            op.same(inputs.training_transaction, observation.transaction, "report training boundary")
            op.same(inputs.binding, plan["native"]["bindings"][task.global_index], "actual/frozen report binding")
            row.update(training_completion_verified=True, actual_binding=inputs.binding,
                       best_validation=inputs.binding["best_validation"])
            coverage = collect_evaluation_coverage(inputs, evaluation_root=plan["evaluation_root"])
            namespace = Path(coverage["namespace"])
            # Include records/attempts beyond HEAD ancestry, without promoting
            # uncommitted success, orphan failures or malformed staging evidence.
            evidence_before = evidence_stamp((namespace,))
            bound = []
            for slot, observed, initial in zip(prescribed_slots(), coverage["slots"], slots):
                item = {**initial, **observed, "best_checkpoint_identity": inputs.binding["best_checkpoint_identity"],
                        "perturbation_seed": perturbation_seed(inputs.binding["best_checkpoint_identity"], slot)}
                history, raw_records, attempts, unparsed = [], [], [], []
                if observed["condition"] == "ok":
                    current = read_slot(namespace, inputs, slot)
                    op.require(current is not None, "Slot HEAD disappeared")
                    op.same(current[0]["current"], observed["reference"], "slot HEAD snapshot")
                    reference, seen = observed["reference"], set()
                    while reference is not None:
                        op.require(reference["relative_path"] not in seen, "Result ancestry cycle")
                        seen.add(reference["relative_path"])
                        document = validate_result(_read_reference(namespace, reference), inputs, slot)
                        history.append({"reference": reference, "document": document})
                        reference = document["previous_record"]
                    if current[1]["status"] == "evaluation_completed":
                        item["test_metrics"] = current[1]["test_metrics"]
                for kind, target in (("records", raw_records), ("attempts", attempts)):
                    directory = op.plain(namespace / "slots" / slot.slot_id / kind)
                    if directory.is_dir():
                        for path in sorted(directory.iterdir()):
                            relative = path.relative_to(namespace).as_posix()
                            try:
                                op.require(path.is_file(), "Non-file evidence")
                                document = op.read(path)
                                if kind == "records":
                                    validate_result(document, inputs, slot)
                                target.append({"relative_path": relative, "sha256": op.digest(path), "document": document})
                            except Exception as exc:
                                try:
                                    sha = op.digest(path) if path.is_file() else None
                                except Exception:
                                    sha = None
                                unparsed.append({"relative_path": relative, "sha256": sha,
                                                 "error": f"{type(exc).__name__}: {exc}"})
                item.update(raw_history=history, raw_records=raw_records, attempts=attempts, unparsed_evidence=unparsed)
                bound.append(item)
            op.same(collect_evaluation_coverage(inputs, evaluation_root=plan["evaluation_root"]), coverage,
                    "stable evaluation coverage")
            op.require(evidence_before == evidence_stamp((namespace,)), "Raw evidence changed during read")
            op.require(before == _stamp(paths) and not any(p.exists() or p.is_symlink() for p in _lock_paths(paths)),
                       "Training changed during evaluation read")
            row.update(evaluation_coverage=coverage, robustness_completion_verified=coverage["robustness_completed"])
            slots = bound
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            if paths is not None and before is not None:
                try:
                    if before != _stamp(paths) or any(p.exists() or p.is_symlink() for p in _lock_paths(paths)):
                        row["training_completion_verified"] = False
                        row["best_validation"] = None
                except Exception:
                    row["training_completion_verified"] = False
                    row["best_validation"] = None
            for item in slots:
                item["diagnostic"] = row["error"]
        training.append(row)
        evaluations.extend(slots)
    op.require(len(training) == 30 and len(evaluations) == 840, "Preserve all expected observations")
    return op.seal({"version": "mobile-all30-evaluation-report-v1", "scientific_completion_authority": False,
                    "plan_sha256": plan["document_sha256"], "operator_pin": plan["operator_pin"],
                    "grid_id": manifest.grid_id, "evaluation_root": plan["evaluation_root"],
                    "training": training, "evaluation": evaluations,
                    "summaries": summaries(training, evaluations), "selection_uses_test": False,
                    "robustness_study_complete": all(r["robustness_completion_verified"] for r in training),
                    "stability": "Per-checkpoint stable historical reads; no global atomic snapshot or writer locks"})


def summaries(training, evaluations):
    candidates, seed_groups, groups, paired = [], [], [], []
    for alpha in op.ALPHAS:
        selected = [r for r in training if r["training_alpha"] == alpha]
        complete = len(selected) == 3 and {r["training_seed"] for r in selected} == {0, 1, 2} and all(
            r["training_completion_verified"] and r["best_validation"] is not None for r in selected)
        values = [r["best_validation"]["correct"] / r["best_validation"]["examples"] for r in selected] if complete else []
        candidates.append({"training_alpha": alpha, "complete": complete,
                           "mean_validation_accuracy": statistics.mean(values) if complete else None,
                           "sample_sd_validation_accuracy": statistics.stdev(values) if complete else None})
        for evaluation_alpha in op.ALPHAS:
            for seed in (0, 1, 2):
                rows = [r for r in evaluations if r["training_alpha"] == alpha
                        and r["evaluation_alpha"] == evaluation_alpha and r["training_seed"] == seed]
                expected = [0] if evaluation_alpha == 0 else [1, 2, 3]
                complete = sorted(r["evaluation_replicate"] for r in rows) == expected and all(
                    r["condition"] == "ok" and r["stored_status"] == "evaluation_completed"
                    and r["test_metrics"] is not None for r in rows)
                group = {"training_alpha": alpha, "evaluation_alpha": evaluation_alpha,
                         "training_seed": seed, "complete": complete, "expected_realizations": len(expected),
                         "individual": [{"replicate": r["evaluation_replicate"], "metrics": r["test_metrics"]} for r in rows]}
                for metric in ("accuracy", "loss"):
                    values = [r["test_metrics"][metric] for r in rows] if complete else []
                    group["mean_" + metric] = statistics.mean(values) if complete else None
                    group["realization_sample_sd_" + metric] = statistics.stdev(values) if complete and len(values) > 1 else None
                seed_groups.append(group)
            seeds = seed_groups[-3:]
            group = {"training_alpha": alpha, "evaluation_alpha": evaluation_alpha,
                     "complete_three_seeds": all(r["complete"] for r in seeds)}
            for metric in ("accuracy", "loss"):
                values = [r["mean_" + metric] for r in seeds] if group["complete_three_seeds"] else []
                group["mean_" + metric] = statistics.mean(values) if values else None
                group["between_seed_sample_sd_" + metric] = statistics.stdev(values) if values else None
            groups.append(group)
    for group in seed_groups:
        baseline = next(s for s in seed_groups if s["training_alpha"] == group["training_alpha"]
                        and s["training_seed"] == group["training_seed"] and s["evaluation_alpha"] == 0)
        item = {k: group[k] for k in ("training_alpha", "evaluation_alpha", "training_seed")}
        valid = group["complete"] and baseline["complete"]
        item["accuracy_difference_from_own_clean_pp"] = 100 * (group["mean_accuracy"] - baseline["mean_accuracy"]) if valid else None
        item["accuracy_retention_from_own_clean"] = group["mean_accuracy"] / baseline["mean_accuracy"] if valid and baseline["mean_accuracy"] > 0 else None
        paired.append(item)
    complete = len(candidates) == 10 and all(c["complete"] for c in candidates)
    maximum = max(c["mean_validation_accuracy"] for c in candidates) if complete else None
    return {"accuracy_unit": "fraction", "sd": "sample_ddof_1", "validation_candidates": candidates,
            "all_ten_validation_candidates_complete": complete,
            "validation_selected_training_alphas": [c["training_alpha"] for c in candidates
                if complete and c["mean_validation_accuracy"] == maximum],
            "per_training_seed": seed_groups, "across_training_seeds": groups, "paired_clean_relative": paired,
            "aggregation": "Average realizations within each checkpoint, then equally weight the three training seeds",
            "relative_formulas": {"difference_pp": "100 * (seed_mean_accuracy(beta) - same_checkpoint_clean_accuracy)",
                                  "retention": "seed_mean_accuracy(beta) / same_checkpoint_clean_accuracy; null if clean=0"}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    plan, manifest = op.load_plan(args.plan, bind=False)
    protected = [op.NATIVE, op.STUDY, op.HOME, op.OPS, op.EVALUATIONS, op.OLD_OPERATOR, op.OLD_PLAN,
                 *plan["protected_roots"],
                 plan["native"]["base_dir"], plan["export"]["path"], plan["inspection"]["path"],
                 *(x["path"] for x in plan["inspection"]["evidence"].values())]
    root, _ = op.claim(args.output_dir, protected, "derived_evaluation_report")
    result = collect(plan, manifest)
    op.publish(root / "report.json", result)
    fields = ["training_global_index", "training_run_id", "training_alpha", "training_seed", "evaluation_alpha",
              "evaluation_replicate", "perturbation_seed", "condition", "stored_status", "checkpoint_sha256",
              "examples", "correct", "accuracy", "loss", "diagnostic"]
    with (root / "slots.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in result["evaluation"]:
            metrics = row["test_metrics"] or {}
            checkpoint = row["best_checkpoint_identity"] or {}
            writer.writerow({**{k: row[k] for k in fields if k in row}, "checkpoint_sha256": checkpoint.get("sha256"),
                             **{k: metrics.get(k) for k in ("examples", "correct", "accuracy", "loss")}})
        stream.flush()
        op.os.fsync(stream.fileno())
    op.publish(root / "CHECKSUMS.json", {name: op.digest(root / name) for name in ("report.json", "slots.csv")})
    print(op.canonical({"report": str(root / "report.json"), "rows": 840,
                        "robustness_study_complete": result["robustness_study_complete"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
