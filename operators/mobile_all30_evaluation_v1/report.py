"""Read-only all-30/840 reporting. Derived reports never establish completion."""

import argparse
import csv
from pathlib import Path
import statistics

import launcher as op


class UnstableInspection(RuntimeError):
    """Observed authority/evidence changed while it was being read."""


def inspection_error(exc, phase):
    kind = ("unstable" if isinstance(exc, UnstableInspection) else
            "missing" if isinstance(exc, FileNotFoundError) else
            "unreadable" if isinstance(exc, OSError) else
            "incompatible" if "binding" in phase else "unreadable")
    return {"phase": phase, "kind": kind, "error": f"{type(exc).__name__}: {exc}"}


def stable(ok, message):
    if not ok:
        raise UnstableInspection(message)


def verified_measurement(row):
    """Fail closed for old reports and retained, unverified raw measurements."""
    return (row.get("binding_verified") is True and row.get("observation_verified") is True
            and row.get("measurement_verified") is True and row.get("condition") == "ok"
            and row.get("stored_status") == "evaluation_completed" and type(row.get("test_metrics")) is dict)


def inspect_slot(item, inputs, slot, namespace):
    """Append evidence in place. A failed section never discards earlier reads."""
    from alpha_sweep_training.evaluation_persistence import _read_reference, validate_result, read_slot
    if item["condition"] == "ok":
        phase = "slot_head"
        try:
            current = read_slot(namespace, inputs, slot)
            if current is None:
                raise FileNotFoundError("Slot HEAD disappeared")
            item.update(observed_head=current[0], observed_result=current[1],
                        test_metrics=current[1].get("test_metrics"))
            stable(op.canonical(current[0]["current"]) == op.canonical(item["reference"]), "Slot HEAD changed")
            phase = "slot_history"
            reference, seen = item["reference"], set()
            while reference is not None:
                op.require(reference["relative_path"] not in seen, "Result ancestry cycle")
                seen.add(reference["relative_path"])
                document = _read_reference(namespace, reference)
                entry = {"reference": reference, "document": document, "validated": False}
                item["raw_history"].append(entry)
                validate_result(document, inputs, slot)
                entry["validated"] = True
                reference = document["previous_record"]
        except Exception as exc:
            item["inspection_errors"].append(inspection_error(exc, phase))
    # Raw directories are independent of the HEAD/ancestry read. Preserve an
    # available attempt or orphan record even if another section cannot be read.
    for kind, target in (("records", item["raw_records"]), ("attempts", item["attempts"])):
        try:
            directory = op.plain(namespace / "slots" / slot.slot_id / kind)
            paths = sorted(directory.iterdir()) if directory.is_dir() else []
        except Exception as exc:
            item["inspection_errors"].append(inspection_error(exc, kind))
            continue
        for path in paths:
            relative = path.relative_to(namespace).as_posix()
            try:
                op.require(path.is_file(), "Non-file evidence")
                document = op.read(path)
                entry = {"relative_path": relative, "document": document, "sha256": None,
                         "validated": False}
                target.append(entry)
                entry["sha256"] = op.digest(path)
                if kind == "records":
                    validate_result(document, inputs, slot)
                    entry["validated"] = True
            except Exception as exc:
                try:
                    sha = op.digest(path) if path.is_file() else None
                except Exception:
                    sha = None
                error = {"relative_path": relative, "sha256": sha, **inspection_error(exc, kind)}
                item["unparsed_evidence"].append(error)
                item["inspection_errors"].append(error)


def collect(plan, manifest):
    from alpha_sweep_training.coverage import inspect_task, _stamp, _lock_paths
    from alpha_sweep_training.artifacts import resolve_run_paths
    from alpha_sweep_training.evaluation_inputs import bind_evaluation_checkpoint
    from alpha_sweep_training.evaluation_coverage import collect_evaluation_coverage
    from alpha_sweep_training.evaluation_protocol import prescribed_slots, perturbation_seed
    from alpha_sweep_training.planning_inspection import evidence_stamp
    training, evaluations = [], []
    for task in manifest.tasks:
        science = task.config.scientific
        row = {"global_index": task.global_index, "run_id": task.run_id,
               "scientific_fingerprint": task.fingerprint, "training_alpha": science["alpha"],
               "training_seed": science["training_seed"], "training_completion_verified": False,
               "robustness_completion_verified": False, "best_validation": None,
               "actual_binding": None, "error": None, "evaluation_coverage": None, "final_evaluation_coverage": None,
               "binding_verified": False, "training_snapshot_verified": False,
               "evaluation_snapshot_verified": False, "inspection_errors": []}
        slots = [{"training_global_index": task.global_index, "training_run_id": task.run_id,
                  "training_alpha": science["alpha"], "training_seed": science["training_seed"],
                  **slot.to_dict(), "condition": "unbound", "stored_status": None,
                  "best_checkpoint_identity": None, "perturbation_seed": None, "reference": None,
                  "test_metrics": None, "raw_history": [], "orphan_failures": [], "raw_records": [],
                  "attempts": [], "unparsed_evidence": [], "diagnostic": None,
                  "binding_verified": False, "observation_verified": False,
                  "measurement_verified": False, "inspection_state": "not_inspected",
                  "inspection_errors": [], "checkpoint_errors": []}
                 for slot in prescribed_slots()]
        paths, before, phase = None, None, "training_observation"
        try:
            observation = inspect_task(task, base_dir=plan["native"]["base_dir"])
            row["training_observation"] = observation.to_dict()
            op.require(observation.condition == "ok" and observation.training_complete,
                       "Training authority is incomplete or unstable")
            paths = resolve_run_paths(task.config, plan["native"]["base_dir"])
            before = _stamp(paths)
            stable(not any(p.exists() or p.is_symlink() for p in _lock_paths(paths)), "Training writer present")
            phase = "checkpoint_binding"
            inputs = bind_evaluation_checkpoint(task, base_dir=plan["native"]["base_dir"], metadata_only=True)
            # Retain the actual observation before comparing it with the plan.
            row["actual_binding"] = inputs.binding
            row["best_validation"] = inputs.binding["best_validation"]
            for slot, item in zip(prescribed_slots(), slots):
                item.update(best_checkpoint_identity=inputs.binding["best_checkpoint_identity"],
                            perturbation_seed=perturbation_seed(inputs.binding["best_checkpoint_identity"], slot))
            op.same(inputs.training_transaction, observation.transaction, "report training boundary")
            op.same(inputs.binding, plan["native"]["bindings"][task.global_index], "actual/frozen report binding")
            row.update(training_completion_verified=True, binding_verified=True)
            for item in slots:
                item["binding_verified"] = True
            phase = "evaluation_coverage"
            coverage = collect_evaluation_coverage(inputs, evaluation_root=plan["evaluation_root"])
            row["evaluation_coverage"] = coverage
            namespace = Path(coverage["namespace"])
            op.require(len(coverage["slots"]) == len(slots), "Require all 28 native coverage observations")
            # Save every coverage observation before reading any individual slot.
            for slot, observed, item in zip(prescribed_slots(), coverage["slots"], slots):
                item["coverage_observation"] = observed
                op.same({k: observed[k] for k in slot.to_dict()}, slot.to_dict(), "native coverage slot mapping")
                item.update(observed)
            phase = "evidence_snapshot"
            evidence_before = evidence_stamp((namespace,))
            phase = "slot_inspection"
            for slot, item in zip(prescribed_slots(), slots):
                inspect_slot(item, inputs, slot, namespace)
            phase = "evaluation_stability"
            final_coverage = collect_evaluation_coverage(inputs, evaluation_root=plan["evaluation_root"])
            row["final_evaluation_coverage"] = final_coverage
            stable(op.canonical(final_coverage) == op.canonical(coverage), "Native evaluation coverage changed")
            stable(evidence_before == evidence_stamp((namespace,)), "Raw evidence changed during read")
            phase = "training_stability"
            stable(before == _stamp(paths) and not any(p.exists() or p.is_symlink() for p in _lock_paths(paths)),
                   "Training changed during evaluation read")
            row.update(training_snapshot_verified=True, evaluation_snapshot_verified=True)
            for item in slots:
                item["observation_verified"] = not item["inspection_errors"] and item["condition"] in ("ok", "pending")
                item["measurement_verified"] = (item["observation_verified"] and item["condition"] == "ok"
                    and item["stored_status"] == "evaluation_completed" and item["test_metrics"] is not None)
                item["inspection_state"] = (item["inspection_errors"][0]["kind"] if item["inspection_errors"] else
                    "missing" if item["condition"] == "pending" else
                    "verified" if item["observation_verified"] else item["condition"])
                if item["inspection_errors"]:
                    item["diagnostic"] = "; ".join(e["error"] for e in item["inspection_errors"])
            row["robustness_completion_verified"] = (coverage["robustness_completed"] is True
                and all(verified_measurement(item) for item in slots))
        except Exception as exc:
            error = inspection_error(exc, phase)
            row["error"] = error["error"]
            row["inspection_errors"].append(error)
            if paths is not None and before is not None:
                try:
                    row["training_snapshot_verified"] = (before == _stamp(paths)
                        and not any(p.exists() or p.is_symlink() for p in _lock_paths(paths)))
                except Exception as training_exc:
                    row["inspection_errors"].append(inspection_error(training_exc, "training_stability"))
            row["training_completion_verified"] = (row["training_completion_verified"]
                and row["training_snapshot_verified"] and phase != "training_stability")
            if not row["training_completion_verified"]:
                row["binding_verified"] = False
            for item in slots:
                item["binding_verified"] = row["binding_verified"]
                item["observation_verified"] = item["measurement_verified"] = False
                item["inspection_state"] = error["kind"]
                item["checkpoint_errors"].append(error)
                item["diagnostic"] = row["error"]
        training.append(row)
        evaluations.extend(slots)
    op.require(len(training) == 30 and len(evaluations) == 840, "Preserve all expected observations")
    return op.seal({"version": "mobile-all30-evaluation-report-v1", "verification_version": 1,
                    "scientific_completion_authority": False,
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
            r.get("training_completion_verified") is True and r.get("training_snapshot_verified") is True
            and r.get("binding_verified") is True and r["best_validation"] is not None for r in selected)
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
                    verified_measurement(r) for r in rows)
                group = {"training_alpha": alpha, "evaluation_alpha": evaluation_alpha,
                         "training_seed": seed, "complete": complete, "expected_realizations": len(expected),
                         "individual": [{"replicate": r["evaluation_replicate"],
                             "measurement_verified": verified_measurement(r),
                             "metrics": r["test_metrics"] if verified_measurement(r) else None} for r in rows]}
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
              "binding_verified", "observation_verified", "measurement_verified", "inspection_state",
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
