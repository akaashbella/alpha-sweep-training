"""Deterministic one-task/one-checkpoint work, linked to full manifest authority."""

from copy import deepcopy

from .data_contract import content_identity
from .hpc_policy import WorkflowError, exact, integer, same, resource_budget, wave_capacity
from .manifests import manifest_from_dict, select_task

WORK_VERSION = 1


def seal(document):
    return {**deepcopy(document), "document_sha256": content_identity(document)[7:]}


def check_seal(document):
    if type(document) is not dict or "document_sha256" not in document:
        raise WorkflowError("Missing immutable work document seal")
    same(document["document_sha256"], content_identity({k: v for k, v in document.items() if k != "document_sha256"})[7:], "work document digest")


def workflow_identity(manifest):
    return content_identity({"workflow_version": 1, "grid_id": manifest.grid_id,
                             "manifest_document_sha256": manifest.document_sha256})[7:]


def unit_key(task, lifecycle, binding=None):
    identity = {"lifecycle": lifecycle, "training_run_id": task.run_id, "training_fingerprint": task.fingerprint}
    if lifecycle == "evaluation":
        identity.update(best_checkpoint_identity=binding["best_checkpoint_identity"],
                        evaluation_protocol_fingerprint=binding["evaluation_protocol_fingerprint"])
    return content_identity(identity)[7:]


def history_for(state, key):
    history = []
    for wave in state.get("waves", []):
        for index, item in enumerate(wave["items"]):
            if item["unit_key"] == key:
                history.append({"wave_id": wave["wave_id"], "status": wave["status"],
                    "outcome": wave.get("outcomes", {}).get(str(index)), "diagnostics": wave.get("diagnostics", {}).get(str(index), []),
                    "allocation_ordinal": item["allocation_ordinal"]})
    return history


def wave_reservations(state, policy):
    """Unknown/pending elements retain queue slots and their wave's enforced throttle.

    Sum all uncertain waves if historical accounting reopens more than one;
    reporting that excess must never create permission to submit a replacement.
    """
    waves = [w for w in state["waves"] if w["status"] not in ("terminal", "confirmed_not_submitted")]
    queued = sum(len(w["items"]) for w in waves)
    if policy.to_dict()["policy_version"] == 1:
        return {"reserved_allocations": queued}
    running = sum(min(len(w["items"]), w["policy"]["resources"]["max_concurrent"]) for w in waves)
    return {"reserved_queued_tasks": queued, "reserved_running_tasks": running}


def build_work_plan(manifest, policy, *, manifest_path, state=None, retry_run_ids=(), retry_reason=None,
                    inspection=None, progress=None):
    from .planning_inspection import PlanningInspection
    inspection = inspection or PlanningInspection(progress)
    inspection.progress("Planning: validating training authority and eligibility")
    with inspection.activate():
        result = _build_work_plan(manifest, policy, manifest_path=manifest_path, state=state,
                                  retry_run_ids=retry_run_ids, retry_reason=retry_reason)
    inspection.progress(f"Planning complete: {len(result['items'])} eligible work units; "
                        f"{inspection.hits} unchanged reads reused, {inspection.validations} validations")
    return result


def _build_work_plan(manifest, policy, *, manifest_path, state=None, retry_run_ids=(), retry_reason=None):
    """Future read-only planning; no sbatch, builders, iterations, or repair.

    Each candidate uses actual science authority and recorded allocation outcome.
    Existing unclassified unfinished runs need an explicit operator decision.
    A wave must be conclusively terminal before any next wave can be eligible.
    """
    from .coverage import collect_coverage, inspect_task
    from .evaluation_coverage import collect_evaluation_coverage
    from .evaluation_inputs import bind_evaluation_checkpoint
    from .execution_provenance import collect_provenance, collect_launch_inputs
    from .evaluation_protocol import prescribed_slots
    manifest = manifest_from_dict(manifest.to_dict())
    policy.require_ready()
    state = state or {"waves": [], "revision": 0}
    retries = list(retry_run_ids)
    if len(set(retries)) != len(retries) or any(type(value) is not str for value in retries):
        raise WorkflowError("Retry selections must be unique exact training run IDs")
    for run_id in retries:
        select_task(manifest, run_id=run_id)
    if retries and (type(retry_reason) is not str or not retry_reason.strip()):
        raise WorkflowError("Explicit execution-failure retry/adoption requires an audit reason")
    doc = policy.to_dict()
    overrides = policy.execution_overrides()
    base, source = doc["paths"]["base_dir"], doc["paths"]["checkout"]
    provenance = collect_provenance(source, attempt={"lifecycle": "workflow_planning", "grid_id": manifest.grid_id})
    # The plan pins executable inputs, not the submit process PID/time/allocation.
    # Full actual provenance is still recorded by the owning execution engine.
    execution_pin = {"code": {key: deepcopy(provenance["code"][key]) for key in
                             ("source_sha256", "inventory_version", "inventory")},
                     "software": deepcopy(provenance["software"]),
                     "launch_inputs": collect_launch_inputs(policy)}
    coverage = collect_coverage(manifest, base_dir=base, overrides=overrides)
    phase2 = coverage["phases"][0]["training_successfully_completed"]
    active = [wave["wave_id"] for wave in state["waves"] if wave["status"] not in ("terminal", "confirmed_not_submitted")]
    items, observations = [], []

    def candidate(task, lifecycle, observation, binding=None, operation="auto", retry_failed=False):
        key = unit_key(task, lifecycle, binding)
        prior = history_for(state, key)
        latest = prior[-1] if prior else None
        row = {"training_run_id": task.run_id, "training_global_index": task.global_index,
               "lifecycle": lifecycle, "unit_key": key, "classification": "eligible", "diagnostic": None}
        approved = task.run_id in retries
        if active:
            row.update(classification="wave_unresolved", diagnostic="Previous wave is active/unknown; shared capacity remains reserved")
        elif len(prior) >= doc["continuation"]["max_allocations_per_work"]:
            row.update(classification="continuation_exhausted", diagnostic="Bounded allocation-intent budget exhausted")
        elif latest is not None and latest["status"] != "confirmed_not_submitted":
            record = latest["outcome"]
            result = None if record is None else record["task_outcome"]
            code = None if record is None else record["exit_code"]
            clean_stop = code == 10 and record["safe_interruption"] is True
            deferred = code == 0 and result is not None and result["outcome"] == "sweep_finalization_deferred"
            if code == 20:
                row.update(classification="terminal_numerical", diagnostic="No automatic or managed numerical retry")
            elif (latest["diagnostics"] or (not clean_stop and not deferred)) and not approved:
                row.update(classification="operator_decision_required", diagnostic="Allocation outcome is failed/missing/unclassified; scheduler terminal state alone is insufficient")
        elif latest is None and lifecycle == "training" and observation.condition != "missing" and not observation.full_budget_committed and not approved:
            row.update(classification="operator_decision_required", diagnostic="Adopt an existing unfinished task explicitly; do not infer a prior safe allocation")
        observations.append(row)
        if row["classification"] != "eligible":
            return
        item = {"work_version": WORK_VERSION, "unit_key": key, "lifecycle": lifecycle,
            "training_run_id": task.run_id, "training_fingerprint": task.fingerprint,
            "training_global_index": task.global_index, "training_phase": task.phase,
            "training_phase_index": task.phase_index, "selector": {"run_id": task.run_id},
            "operation": operation, "retry_failed": retry_failed,
            "operator_retry_reason": retry_reason if approved else None,
            "allocation_ordinal": len(prior) + 1, "training_transaction": observation.transaction,
            "evaluation_binding": deepcopy(binding),
            "logical_evaluation_slots": [] if lifecycle == "training" else [s.to_dict() for s in prescribed_slots()]}
        items.append(seal(item))

    # Training candidates always retain the manifest's global indices and order.
    for task in manifest.tasks:
        obs = inspect_task(task, base_dir=base, overrides=overrides, provenance=provenance)
        if obs.condition == "ok" and obs.training_complete:
            observations.append({"training_run_id": task.run_id, "lifecycle": "training", "classification": "training_completed"})
        elif task.phase == 2 and not phase2 and not obs.full_budget_committed:
            observations.append({"training_run_id": task.run_id, "lifecycle": "training", "classification": "phase1_training_incomplete"})
        elif obs.condition not in ("ok", "missing") or obs.terminal_numerical:
            observations.append({"training_run_id": task.run_id, "lifecycle": "training", "classification": "unsafe_or_terminal", "inspection": obs.to_dict()})
        elif obs.execution_failed and task.run_id not in retries:
            observations.append({"training_run_id": task.run_id, "lifecycle": "training", "classification": "execution_retry_requires_operator", "inspection": obs.to_dict()})
        else:
            candidate(task, "training", obs, operation="continue" if obs.execution_failed else "auto")
    # One saved best checkpoint/all unfinished slots per allocation, never 6,720 jobs.
    for task in manifest.tasks:
        obs = inspect_task(task, base_dir=base, overrides=overrides)
        if obs.condition != "ok" or not obs.evaluation_eligible:
            observations.append({"training_run_id": task.run_id, "lifecycle": "evaluation", "classification": "checkpoint_not_eligible"})
            continue
        try:
            from .planning_inspection import active_inspection
            inspection = active_inspection()
            inspection.progress(f"Evaluation eligibility: {task.run_id}")
            inputs = bind_evaluation_checkpoint(task, base_dir=base, overrides=overrides, metadata_only=True)
            evaluation = collect_evaluation_coverage(inputs, evaluation_root=doc["paths"]["evaluation_root"], evaluator_provenance=provenance)
            counts = evaluation["counts"]
            if evaluation["robustness_completed"]:
                classification = "robustness_completed"
            elif any(counts[k] for k in ("locked", "partial", "incompatible", "unstable", "inspection_error")) or evaluation["completion_inspection_error"] or evaluation["completion_inspection_deferred"]:
                classification = "evaluation_unsafe_or_deferred_owner"
            elif any(row["stored_status"] == "failed_numerical" or any(e["status"] == "failed_numerical" for e in row["orphan_failures"]) for row in evaluation["slots"]):
                classification = "terminal_numerical"
            elif (counts["failed"] or counts["unacknowledged_failures"]) and task.run_id not in retries:
                classification = "execution_retry_requires_operator"
            else:
                candidate(task, "evaluation", obs, inputs.binding, retry_failed=bool(counts["failed"] or counts["unacknowledged_failures"]))
                continue
            observations.append({"training_run_id": task.run_id, "lifecycle": "evaluation", "classification": classification, "coverage": evaluation})
        except Exception as exc:
            observations.append({"training_run_id": task.run_id, "lifecycle": "evaluation", "classification": "inspection_error", "diagnostic": f"{type(exc).__name__}: {exc}"})
    return seal({"plan_version": doc["policy_version"], "kind": "workflow_work_plan", "workflow_id": workflow_identity(manifest),
        "manifest_path": str(manifest_path), "manifest_grid_id": manifest.grid_id,
        "manifest_document_sha256": manifest.document_sha256, "policy": doc, "policy_fingerprint": policy.fingerprint,
        "execution_provenance": execution_pin,
        "registry_revision": state["revision"], "phase2_training_eligible": phase2,
        "unresolved_waves": active,
        **wave_reservations(state, policy),
        "capacity_available_for_submission": not active,
        "resource_policy": resource_budget(policy, min(len(items), wave_capacity(policy))),
        "items": items, "observations": observations})


def validate_item(item, manifest):
    check_seal(item)
    exact(item, {"work_version", "unit_key", "lifecycle", "training_run_id", "training_fingerprint",
        "training_global_index", "training_phase", "training_phase_index", "selector", "operation", "retry_failed",
        "operator_retry_reason", "allocation_ordinal", "training_transaction", "evaluation_binding",
        "logical_evaluation_slots", "document_sha256"}, "work item")
    same(item["work_version"], WORK_VERSION, "work version")
    if item["lifecycle"] not in ("training", "evaluation"):
        raise WorkflowError("Unknown allocation lifecycle")
    task = select_task(manifest, run_id=item["training_run_id"])
    for key, expected in (("training_fingerprint", task.fingerprint), ("training_global_index", task.global_index),
                          ("training_phase", task.phase), ("training_phase_index", task.phase_index), ("selector", {"run_id": task.run_id})):
        same(item[key], expected, f"work membership {key}")
    integer(item["allocation_ordinal"], "allocation ordinal")
    if item["operation"] not in ("auto", "continue") or type(item["retry_failed"]) is not bool:
        raise WorkflowError("Work supports only validated auto/explicit execution continuation")
    if item["operator_retry_reason"] is not None and (type(item["operator_retry_reason"]) is not str or not item["operator_retry_reason"].strip()):
        raise WorkflowError("Malformed operator retry reason")
    if item["lifecycle"] == "training":
        same(item["evaluation_binding"], None, "training work has no evaluation binding")
        same(item["logical_evaluation_slots"], [], "training work evaluation slots")
        same(item["retry_failed"], False, "training work cannot retry numerical evaluation")
    else:
        from .evaluation_protocol import prescribed_slots, validate_protocol
        binding = item["evaluation_binding"]
        validate_protocol(binding["evaluation_protocol"], task.config)
        same(binding["training_identity"]["run_id"], task.run_id, "bound evaluation training run")
        same(binding["training_identity"]["scientific_fingerprint"], task.fingerprint, "bound evaluation training fingerprint")
        same(item["logical_evaluation_slots"], [s.to_dict() for s in prescribed_slots()], "default full prescribed evaluation group")
        same(item["operation"], "auto", "evaluation never invokes training continuation")
    if (item["operation"] == "continue" or item["retry_failed"]) and item["operator_retry_reason"] is None:
        raise WorkflowError("Explicit execution-failure retry requires an immutable reason")
    same(item["unit_key"], unit_key(task, item["lifecycle"], item["evaluation_binding"]), "allocation unit identity")
    return task


def validate_plan(plan, manifest, policy):
    check_seal(plan)
    version = policy.to_dict()["policy_version"]
    reservations = {"reserved_allocations"} if version == 1 else {"reserved_queued_tasks", "reserved_running_tasks"}
    exact(plan, {"plan_version", "kind", "workflow_id", "manifest_path", "manifest_grid_id", "manifest_document_sha256",
        "policy", "policy_fingerprint", "registry_revision", "phase2_training_eligible", "unresolved_waves", "resource_policy",
        "capacity_available_for_submission", "execution_provenance", "items", "observations", "document_sha256"}
        | reservations, "work plan")
    same(plan["plan_version"], version, "plan/policy version")
    same(plan["kind"], "workflow_work_plan", "plan kind")
    same(plan["workflow_id"], workflow_identity(manifest), "workflow identity")
    same(plan["manifest_grid_id"], manifest.grid_id, "work manifest grid")
    same(plan["manifest_document_sha256"], manifest.document_sha256, "work full manifest digest")
    same(plan["policy"], policy.to_dict(), "immutable execution policy")
    same(plan["policy_fingerprint"], policy.fingerprint, "execution policy digest")
    if type(plan["items"]) is not list:
        raise WorkflowError("Work mapping must be a list")
    seen = set()
    for item in plan["items"]:
        validate_item(item, manifest)
        if item["unit_key"] in seen:
            raise WorkflowError("Overlapping/duplicate work in plan")
        seen.add(item["unit_key"])
    return plan
