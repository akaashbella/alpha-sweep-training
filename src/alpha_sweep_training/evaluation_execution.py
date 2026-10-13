"""Independent best-checkpoint evaluation, complete-realization resumption only."""

from copy import deepcopy
from dataclasses import asdict, dataclass
import time

from .checkpoint_io import same, fail
from .evaluation_inputs import bind_evaluation_checkpoint
from .evaluation_protocol import (prescribed_slots, perturbation_seed,
                                 select_evaluation_slots, EvaluationProtocolError)
from .evaluation_persistence import (PRESERVATION, SweepFinalizationDeferred,
                                     open_evaluation_slot, publish_robustness_completion)
from .evaluation_coverage import collect_evaluation_coverage, observe_evaluation_slot
from .execution_provenance import collect_provenance
from .manifests import manifest_from_dict, select_task
from .spec import TOTAL_EVALUATION_SLOTS, EVALUATION_SLOTS_PER_CHECKPOINT


@dataclass(frozen=True)
class EvaluationOutcome:
    training_run_id: str
    evaluation_protocol_fingerprint: str
    outcome: str
    exit_code: int
    slots: tuple[dict, ...]
    coverage: dict
    lifecycle: str = "evaluation"

    def to_dict(self):
        return asdict(self)


def inspect_evaluation_eligibility(task, *, base_dir, overrides=None, evaluation_root=None):
    from .coverage import inspect_task
    observation = inspect_task(task, base_dir=base_dir, overrides=overrides)
    result = {"training": observation.to_dict(), "evaluation_eligible": observation.evaluation_eligible,
              "evaluation": None, "diagnostic": None}
    if observation.condition == "ok" and observation.evaluation_eligible:
        try:
            inputs = bind_evaluation_checkpoint(task, base_dir=base_dir, overrides=overrides)
            result["evaluation"] = collect_evaluation_coverage(inputs, evaluation_root=evaluation_root)
            result["legacy_clean_candidate"] = None if inputs.legacy_clean is None else {
                "test_reference": inputs.legacy_clean["test_reference"],
                "summary_reference": inputs.legacy_clean["summary_reference"],
                "reusable_state_proof": inputs.legacy_clean["summary"].get("evaluation_state_preservation") == PRESERVATION,
                "actual_test_runtime_present": type(inputs.legacy_clean["summary"].get("evaluation_runtime")) is dict}
        except Exception as exc:
            result.update(evaluation_eligible=False, diagnostic=f"{type(exc).__name__}: {exc}")
    return result


def collect_experiment_coverage(manifest, *, base_dir, overrides=None, evaluation_root=None):
    """Every training identity and all logical slots remain visible, even untrained."""
    from .coverage import collect_coverage
    manifest = manifest_from_dict(manifest.to_dict())
    training = collect_coverage(manifest, base_dir=base_dir, overrides=overrides)
    observations = []
    for task in manifest.tasks:
        row = inspect_evaluation_eligibility(task, base_dir=base_dir, overrides=overrides, evaluation_root=evaluation_root)
        if row["evaluation"] is None:
            row["expected_unbound_slots"] = [slot.to_dict() for slot in prescribed_slots()]
        observations.append(row)
    evaluation_counts = {"expected": TOTAL_EVALUATION_SLOTS, "successful": 0, "failed": 0,
        "interrupted": 0, "pending": 0, "locked": 0, "incompatible": 0, "inspection_error": 0,
        "partial": 0, "unstable": 0, "unacknowledged_failures": 0,
        "clean_points_successful": 0, "positive_realizations_successful": 0}
    for row in observations:
        coverage = row["evaluation"]
        if coverage is None:
            evaluation_counts["pending"] += EVALUATION_SLOTS_PER_CHECKPOINT
        else:
            for key in ("successful", "failed", "interrupted", "pending", "locked", "incompatible", "inspection_error",
                        "partial", "unstable", "unacknowledged_failures"):
                evaluation_counts[key] += coverage["counts"][key]
            evaluation_counts["clean_points_successful"] += int(coverage["clean_point_successful"])
            evaluation_counts["positive_realizations_successful"] += coverage["positive_realizations_successful"]
    return {"experiment_coverage_version": 1, "grid_id": manifest.grid_id, "training": training,
        "expected_evaluation_slots": TOTAL_EVALUATION_SLOTS, "evaluation_counts": evaluation_counts,
        "evaluation_eligible_checkpoints": sum(row["evaluation_eligible"] for row in observations),
        "robustness_completed_checkpoints": sum(row["evaluation"] is not None and
            row["evaluation"]["robustness_completed"] for row in observations), "checkpoints": observations}


def _legacy_clean(inputs, actual_provenance):
    """Optional qualified reuse; never invent an exact-state check for old evidence.

    The standard v1 engine recorded no exact entire-state proof. Such results
    stay visible as historical clean evidence and need one new independent clean
    pass. A compatible immutable legacy summary with that proof can be imported.
    """
    legacy = inputs.legacy_clean
    if legacy is None:
        raise EvaluationProtocolError("No verified legacy clean result is available")
    summary, record = legacy["summary"], legacy["test"]
    same(summary.get("evaluation_state_preservation"), PRESERVATION,
         "legacy exact entire-state evidence (standard v1 lacks this; run the independent clean baseline)")
    same(record["payload"]["best_checkpoint"], inputs.binding["best_checkpoint"], "legacy best checkpoint")
    same(record["payload"]["clean_weights"], True, "legacy clean marker")
    same(summary["scientific"]["data"], inputs.config.scientific["data"], "legacy preprocessing/normalization")
    for key in ("torch", "torchvision"):
        same(record["provenance"]["software"][key], actual_provenance["software"][key], f"legacy {key} evaluation compatibility")
    from .checkpoint_records import classification
    metrics = classification(record["payload"]["metrics"])
    runtime = deepcopy(summary.get("evaluation_runtime"))
    if type(runtime) is not dict:
        raise EvaluationProtocolError("Legacy result has no actual test-runtime evidence; execute the independent clean baseline")
    same(runtime["precision"], inputs.protocol.to_dict()["precision"], "legacy evaluation precision")
    original_runtime = deepcopy(runtime)
    # Scaler did not participate in the legacy clean evaluation forward/loss.
    runtime["grad_scaler"] = None
    return metrics, runtime, {"kind": "legacy_import", "test_reference": legacy["test_reference"],
        "summary_reference": legacy["summary_reference"], "original_test_provenance": record["provenance"],
        "original_duration_seconds": record["payload"]["duration_seconds"], "original_recorded_at_utc": record["recorded_at_utc"],
        "original_runtime": original_runtime}


def execute_evaluation(manifest, task, *, base_dir, source_root, overrides=None, evaluation_root=None,
                       evaluation_alphas=None, evaluation_replicates=None, retry_failed=False,
                       reuse_legacy_clean=False, stop=None) -> EvaluationOutcome:
    """Run selected missing slots, skip verified successes, restart partial passes.

    Exactly one checkpoint/process/GPU, optionally a group of realizations. Never
    constructs optimizer/scheduler/scaler/training streams or invokes train_run.
    Failed slots require explicit retry; interruptions restart at the same seed.
    """
    manifest = manifest_from_dict(manifest.to_dict())
    selected = select_task(manifest, run_id=task.run_id)
    same(selected.to_dict(), task.to_dict(), "evaluation training-task membership")
    if type(retry_failed) is not bool or type(reuse_legacy_clean) is not bool:
        raise EvaluationProtocolError("Evaluation retry/import options must be explicit bools")
    slots = select_evaluation_slots(evaluation_alphas=evaluation_alphas, evaluation_replicates=evaluation_replicates)
    from .interruption import StopRequest, TrainingInterrupted
    if stop is not None and not isinstance(stop, StopRequest):
        raise EvaluationProtocolError("Evaluation stop must be the existing StopRequest")
    stop = stop if stop is not None else StopRequest()
    inputs = bind_evaluation_checkpoint(selected, base_dir=base_dir, overrides=overrides)
    provenance = collect_provenance(source_root, attempt={"lifecycle": "evaluation", "training_run_id": selected.run_id,
        "grid_id": manifest.grid_id, "evaluation_protocol_fingerprint": inputs.protocol.fingerprint,
        "best_checkpoint_identity": inputs.binding["best_checkpoint_identity"]})
    outcomes, data, runtime, runtime_metadata, model, baseline = [], None, None, None, None, None
    interrupted = False
    for slot in slots:
        try:
            stop.check(at_boundary=True, phase="evaluation_realization_entry")
        except TrainingInterrupted as exc:
            interrupted = True
            outcomes.append({**slot.to_dict(), "outcome": "not_started_stop_requested", "diagnostic": str(exc), "exit_code": 10})
            break
        observation = observe_evaluation_slot(inputs, slot, evaluation_root=evaluation_root, evaluator_provenance=provenance)
        if observation["condition"] == "ok" and observation["stored_status"] == "evaluation_completed":
            outcomes.append({**slot.to_dict(), "outcome": "already_evaluation_completed", "exit_code": 0,
                             "reference": observation["reference"]})
            continue
        if observation["condition"] not in ("ok", "pending"):
            outcomes.append({**slot.to_dict(), "outcome": "rejected", "exit_code": 22, "inspection": observation})
            continue
        failed = observation["stored_status"] in ("failed_numerical", "failed_execution") or observation["orphan_failures"]
        if failed and not retry_failed:
            numeric = observation["stored_status"] == "failed_numerical" or any(
                evidence["status"] == "failed_numerical" for evidence in observation["orphan_failures"])
            outcomes.append({**slot.to_dict(), "outcome": "failed_slot_requires_explicit_retry", "exit_code": 20 if numeric else 21,
                             "inspection": observation})
            continue
        outcome_start = len(outcomes)
        try:
            with open_evaluation_slot(inputs, slot, evaluation_root=evaluation_root,
                                      provenance=provenance, retry_failed=retry_failed) as lease:
                current = lease.current()
                if current is not None and current[1]["status"] == "evaluation_completed":
                    outcomes.append({**slot.to_dict(), "outcome": "already_evaluation_completed", "exit_code": 0})
                    continue
                if current is not None and current[1]["status"] in ("failed_numerical", "failed_execution") and not retry_failed:
                    fail("A concurrent failure now requires an explicit retry choice")
                started = time.perf_counter()
                phase = "component_construction"
                try:
                    if slot.evaluation_alpha == 0 and reuse_legacy_clean:
                        phase = "legacy_clean_import"
                        metrics, metadata, origin = _legacy_clean(inputs, provenance)
                    else:
                        from .data import build_test_loader
                        from .runtime import build_runtime
                        from .models import build_model
                        from .checkpoints import copy_clean_model
                        from .evaluation_state import CleanStateBaseline
                        from .evaluation import evaluate_realization
                        import torch
                        if data is None:
                            phase = "test_data"
                            data = build_test_loader(inputs.config, base_dir=base_dir)
                            for key, value in selected.prepared_data.items():
                                same(data.metadata[key], value, f"actual evaluation prepared {key}")
                        if runtime is None:
                            phase = "evaluation_runtime"
                            runtime = build_runtime(inputs.config, evaluation_only=True)
                            runtime_metadata = runtime.metadata()
                        if model is None:
                            phase = "evaluation_model"
                            model = build_model(inputs.config).to(device=runtime.device, dtype=torch.float32)
                            copy_clean_model(model, inputs.payload)
                            baseline = CleanStateBaseline(model)
                        phase = "fixed_network_full_test"
                        metrics = evaluate_realization(model, data.loader, config=inputs.config, runtime=runtime, slot=slot,
                            seed=perturbation_seed(inputs.binding["best_checkpoint_identity"], slot), baseline=baseline, stop=stop)
                        metadata, origin = runtime_metadata, {"kind": "executed"}
                    stop.check(at_boundary=False, phase="evaluation_realization_before_publication")
                    phase = "evaluation_result_commit"
                    receipt = lease.publish(status="evaluation_completed", metrics=metrics, runtime=metadata,
                        duration_seconds=time.perf_counter() - started, preservation=PRESERVATION, origin=origin)
                    outcomes.append({**slot.to_dict(), "outcome": "evaluation_completed", "exit_code": 0, "receipt": asdict(receipt)})
                except BaseException as exc:
                    # Cached metadata only: failure publication makes no CUDA/RNG calls.
                    from .evaluation import EvaluationFailure
                    from .numerical import NumericalFailure
                    cause = exc.cause if isinstance(exc, EvaluationFailure) else exc
                    status = "interrupted" if isinstance(cause, (TrainingInterrupted, KeyboardInterrupt, SystemExit)) else (
                        "failed_numerical" if isinstance(cause, NumericalFailure) else "failed_execution")
                    failure = {"phase": phase, "reason": str(exc) or type(cause).__name__, "exception_type": type(cause).__name__,
                        "batch_index": exc.batch_index if isinstance(exc, EvaluationFailure) else None,
                        "numerical_stage": cause.stage if isinstance(cause, NumericalFailure) else None,
                        "affected_tensors": list(cause.tensors) if isinstance(cause, NumericalFailure) else [],
                        "partial_metrics_discarded": True}
                    publication_error = None
                    try:
                        receipt = lease.publish(status=status, metrics=None, runtime=runtime_metadata,
                            duration_seconds=time.perf_counter() - started, failure=failure)
                        # A completed HEAD dominates an unknown success acknowledgement.
                        reconciled = lease.current()
                        if reconciled is not None and reconciled[1]["status"] == "evaluation_completed":
                            outcomes.append({**slot.to_dict(), "outcome": "evaluation_completed_after_lost_ack", "exit_code": 0,
                                             "reference": reconciled[0]["current"], "diagnostic": str(exc)})
                            continue
                    except BaseException as publication_exc:
                        publication_error = f"{type(publication_exc).__name__}: {publication_exc}"
                        try:
                            reconciled = lease.current()
                            if reconciled is not None and reconciled[1]["status"] == "evaluation_completed":
                                outcomes.append({**slot.to_dict(), "outcome": "evaluation_completed_after_lost_ack", "exit_code": 0,
                                                 "reference": reconciled[0]["current"], "diagnostic": publication_error})
                                continue
                        except Exception:
                            pass
                    outcomes.append({**slot.to_dict(), "outcome": status, "exit_code": {
                        "interrupted": 10, "failed_numerical": 20, "failed_execution": 21}[status],
                        "failure": failure, "failure_publication_error": publication_error})
                    data, model, baseline, runtime, runtime_metadata = None, None, None, None, None
                    if status == "interrupted":
                        interrupted = True
                        break
        except Exception as exc:
            if len(outcomes) > outcome_start and outcomes[-1]["exit_code"] == 0:
                # The durable success survives a subsequent lease-cleanup error.
                # Coverage still exposes any lock that cleanup could not remove.
                outcomes[-1]["lease_diagnostic"] = f"{type(exc).__name__}: {exc}"
                continue
            outcomes.append({**slot.to_dict(), "outcome": "rejected", "exit_code": 22, "diagnostic": f"{type(exc).__name__}: {exc}"})
    try:
        publish_robustness_completion(inputs, evaluation_root=evaluation_root, provenance=provenance)
    except SweepFinalizationDeferred as exc:
        outcomes.append({"outcome": "completion_finalization_deferred", "exit_code": 0,
                         "diagnostic": str(exc), "lock_path": exc.lock_path})
    except Exception as exc:
        outcomes.append({"outcome": "completion_publication_error", "exit_code": 21,
                         "diagnostic": f"{type(exc).__name__}: {exc}"})
    coverage = collect_evaluation_coverage(inputs, evaluation_root=evaluation_root, evaluator_provenance=provenance)
    if coverage["robustness_completed"]:
        for row in outcomes:
            if row["outcome"] == "completion_publication_error":
                row.update(outcome="robustness_completed_after_lost_ack", exit_code=0)
    elif coverage["completion_inspection_error"] is not None and not any(
            row["outcome"] == "completion_publication_error" for row in outcomes):
        outcomes.append({"outcome": "completion_inspection_error", "exit_code": 21,
                         "diagnostic": coverage["completion_inspection_error"]})
    codes = {row["exit_code"] for row in outcomes}
    code = next((c for c in (22, 20, 21, 10) if c in codes), 0)
    finalization_deferred = coverage["completion_inspection_error"] is None and (
        coverage["completion_inspection_deferred"] or any(
            row["outcome"] == "completion_finalization_deferred" for row in outcomes))
    outcome = "robustness_completed" if coverage["robustness_completed"] else (
        "evaluation_interrupted" if interrupted else (
            "sweep_finalization_deferred" if finalization_deferred else "selected_slots_processed"))
    return EvaluationOutcome(selected.run_id, inputs.protocol.fingerprint, outcome, code, tuple(outcomes), coverage)
