"""External all-thirty historical report. Read-only native authority; no models/test passes."""

import argparse
from dataclasses import asdict
from pathlib import Path
import statistics

import launcher as op


def collect(plan, manifest, evaluation_root):
    from alpha_sweep_training.artifacts import resolve_run_paths
    from alpha_sweep_training.coverage import inspect_task, _stamp, _lock_paths
    from alpha_sweep_training.checkpoint_journal import read_committed
    from alpha_sweep_training.checkpoint_journal import _failure_evidence
    from alpha_sweep_training.checkpoint_io import ArtifactReference, read_document
    from alpha_sweep_training.serialization import read_json
    from alpha_sweep_training.evaluation_inputs import bind_evaluation_checkpoint
    from alpha_sweep_training.evaluation_coverage import collect_evaluation_coverage
    from alpha_sweep_training.evaluation_persistence import read_slot, _read_reference, validate_result
    from alpha_sweep_training.evaluation_protocol import prescribed_slots, perturbation_seed
    from alpha_sweep_training.planning_inspection import evidence_stamp
    rows, evaluations = [], []
    for task in manifest.tasks:
        observation = inspect_task(task, base_dir=plan["native"]["base_dir"])
        row = {"global_index": task.global_index, "run_id": task.run_id, "scientific_fingerprint": task.fingerprint,
            "training_alpha": task.config.training_alpha, "training_seed": task.config.scientific["training_seed"],
            "observation": observation.to_dict(), "training_completion_verified": False,
            "health_assessment": "unreviewed; no new thresholds", "epochs": [], "best": None,
            "final": None, "last_committed": None, "selected_epoch": None,
            "failures": list(observation.evidence_failures), "robustness_completion_verified": False}
        slots = [{"training_global_index": task.global_index, **slot.to_dict(), "perturbation_seed": None,
                  "best_checkpoint_identity": None, "condition": "unbound", "raw_history": []} for slot in prescribed_slots()]
        paths, stamp = None, None
        try:
            op.require(observation.condition == "ok", f"No stable native authority: {observation.condition}")
            paths = resolve_run_paths(task.config, plan["native"]["base_dir"])
            stamp = _stamp(paths)
            op.require(not any(p.exists() or p.is_symlink() for p in _lock_paths(paths)), "Writer lock appeared")
            saved = read_json(paths.metadata / "run_metadata.json")["provenance"]
            failure_stamp = evidence_stamp((paths.results / "failures",))
            current = read_committed(paths, task.config, saved)
            op.same(current.reference.to_dict(), observation.transaction, "report training transaction")
            epochs = [{"reference": ref, "epoch": record.epoch_index + 1, "record": record.to_dict(),
                       "weight_statistics": [asdict(s) for s in stats]}
                      for ref, (_, record, stats) in zip(current.document["records"], current.records)]
            # Include historical interruptions/retried execution failures too;
            # latest-boundary dispatch evidence alone intentionally omits them.
            failures = {ref.relative_path: ref for state in ("failed_numerical", "failed_execution", "interrupted")
                        for ref in _failure_evidence(paths, task.config, saved, status=state)}
            failure = current.document["artifacts"]["failure"]
            if failure is not None:
                reference = ArtifactReference.from_dict(failure)
                failures[reference.relative_path] = reference
            row["historical_failure_records"] = [{"reference": ref.to_dict(), "document": read_document(paths, ref)}
                                                  for ref in failures.values()]
            op.require(evidence_stamp((paths.results / "failures",)) == failure_stamp, "Failure evidence changed")
            op.require(stamp == _stamp(paths) and not any(p.exists() or p.is_symlink() for p in _lock_paths(paths)), "Training authority changed")
            best = current.document["best"]
            row.update(epochs=epochs, receipt=asdict(current.receipt), provenance=saved,
                transaction=current.reference.to_dict(), training_completion_verified=observation.training_complete,
                selected_epoch=None if best is None else best["epoch_index"] + 1,
                best=None if best is None else epochs[best["epoch_index"]],
                final=epochs[-1] if current.receipt.completed_epochs == 200 else None,
                last_committed=epochs[-1] if epochs else None)
        except Exception as exc:
            row.update(training_completion_verified=False, training_inspection_error=f"{type(exc).__name__}: {exc}")
        if "receipt" in row and row["receipt"]["completed_epochs"] == 200 and "training_inspection_error" not in row:
            try:
                inputs = bind_evaluation_checkpoint(task, base_dir=plan["native"]["base_dir"], metadata_only=True)
                op.same(inputs.training_transaction, observation.transaction, "evaluation/training snapshot")
                coverage = collect_evaluation_coverage(inputs, evaluation_root=evaluation_root)
                bound = []
                for slot, observed in zip(prescribed_slots(), coverage["slots"]):
                    history = []
                    if observed["condition"] == "ok":
                        current_slot = read_slot(Path(coverage["namespace"]), inputs, slot)
                        op.require(current_slot is not None, "Slot HEAD disappeared")
                        op.same(current_slot[0]["current"], observed["reference"], "slot snapshot/HEAD")
                        reference, seen = observed["reference"], set()
                        while reference is not None:
                            op.require(reference["relative_path"] not in seen, "Evaluation history cycle")
                            seen.add(reference["relative_path"])
                            document = validate_result(_read_reference(Path(coverage["namespace"]), reference), inputs, slot)
                            history.append({"reference": reference, "document": document})
                            reference = document["previous_record"]
                    bound.append({"training_global_index": task.global_index, **observed,
                        "best_checkpoint_identity": inputs.binding["best_checkpoint_identity"],
                        "perturbation_seed": perturbation_seed(inputs.binding["best_checkpoint_identity"], slot) if slot.evaluation_alpha > 0 else None,
                        "raw_history": history})
                op.same(collect_evaluation_coverage(inputs, evaluation_root=evaluation_root), coverage, "stable evaluation observation")
                op.require(stamp == _stamp(paths) and not any(p.exists() or p.is_symlink() for p in _lock_paths(paths)), "Training changed during evaluation inspection")
                row["evaluation_coverage"] = coverage
                row["robustness_completion_verified"] = coverage["robustness_completed"]
                slots = bound
            except Exception as exc:
                row["evaluation_inspection_error"] = f"{type(exc).__name__}: {exc}"
                # Evaluation inspection does not relabel stable training authority.
                try:
                    changed = stamp != _stamp(paths) or any(p.exists() or p.is_symlink() for p in _lock_paths(paths))
                except Exception:
                    changed = True
                if changed:
                    row.update(training_completion_verified=False, training_inspection_error="Training authority changed during evaluation read")
        for slot in slots:
            if slot["condition"] == "unbound":
                slot["diagnostic"] = row.get("evaluation_inspection_error", row.get("training_inspection_error", observation.condition))
        rows.append(row)
        evaluations.extend(slots)
    op.require(len(rows) == 30 and len(evaluations) == 840, "Retain all 30/840 expected observations")
    return op.sealed({"version": "external-mobile-study-report-v1", "scientific_completion_authority": False,
        "operator_plan_sha256": plan["document_sha256"], "grid_id": manifest.grid_id,
        "manifest_document_sha256": manifest.document_sha256, "operator_pin": op.operator_pin(),
        "evaluation_root": str(op.plain(evaluation_root)),
        "training": rows, "evaluation": evaluations, "summaries": summaries(rows),
        "training_study_complete": all(r["training_completion_verified"] for r in rows),
        "robustness_study_complete": all(r["robustness_completion_verified"] for r in rows),
        "health_assessment": "unreviewed; completion/finiteness do not imply healthy learning",
        "selection_uses_test": False,
        "stability": "Independent stable historical reads, not an atomic study snapshot; no writer locks or repair"})


def summaries(rows):
    groups, paired = [], []
    baseline = {r["training_seed"]: r for r in rows if r["training_alpha"] == 0}
    for alpha in op.ALPHAS:
        selected = [r for r in rows if r["training_alpha"] == alpha]
        complete = len(selected) == 3 and {r["training_seed"] for r in selected} == {0, 1, 2} and all(r["training_completion_verified"] for r in selected)
        values = [r["best"]["record"]["validation"]["accuracy"] for r in selected] if complete else None
        group = {"training_alpha": alpha, "complete_prescribed_seeds": complete,
                 "best_validation_mean": statistics.mean(values) if complete else None,
                 "best_validation_sample_sd": statistics.stdev(values) if complete else None}
        groups.append(group)
        differences = []
        for row in selected:
            base = baseline.get(row["training_seed"])
            difference = None
            if row["training_completion_verified"] and base and base["training_completion_verified"]:
                difference = 100 * (row["best"]["record"]["validation"]["accuracy"] - base["best"]["record"]["validation"]["accuracy"])
            differences.append({"seed": row["training_seed"], "difference_pp": difference})
        valid = len(differences) == 3 and all(d["difference_pp"] is not None for d in differences)
        paired.append({"training_alpha": alpha, "individual_paired_validation_differences": differences,
                       "mean_difference_pp": statistics.mean(d["difference_pp"] for d in differences) if valid else None})
    eligible = len(groups) == 10 and all(g["complete_prescribed_seeds"] for g in groups)
    maximum = max(g["best_validation_mean"] for g in groups) if eligible else None
    return {"accuracy_unit": "fraction", "standard_deviation": "sample_ddof_1", "groups": groups,
            "all_ten_candidates_complete": eligible,
            "best_training_alphas": [g["training_alpha"] for g in groups if eligible and g["best_validation_mean"] == maximum],
            "paired_descriptive_only": paired}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    plan, manifest = op.load_plan(args.plan, require_pilots=False)
    _, _, protected = op.native_inputs(require_pilots=False)
    # Claim only a separate derived namespace; raw storage remains read-only.
    root, _ = op.claim(args.output_dir, [*protected, plan["root"], args.evaluation_root,
                                      *plan["extra_protected_roots"]], "derived_all30_report")
    result = collect(plan, manifest, args.evaluation_root)
    op.publish(root / "report.json", result)
    print(op.canonical({"path": str(root / "report.json"), "document_sha256": result["document_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
