"""Exactly one registered work item in one Slurm allocation, no submission engine."""

import argparse
from copy import deepcopy
import os
from pathlib import Path
import re
import socket
import sys
import uuid

from .allocation_signals import allocation_signal_handlers
from .hpc_policy import OperationalPolicy, WorkflowError, same
from .interruption import StopRequest, TrainingInterrupted
from .serialization import read_json, strict_json_text


def allocated_identity(policy):
    r = policy.to_dict()["resources"]
    required = {"SLURM_JOB_NUM_NODES": 1, "SLURM_NTASKS": 1, "SLURM_CPUS_PER_TASK": r["cpus"], "SLURM_GPUS_ON_NODE": 1}
    for key, value in required.items():
        same(os.environ.get(key), str(value), f"actual allocation {key}")
    root, index, job_id = (os.environ.get(k) for k in ("SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_JOB_ID"))
    if any(v is None or re.fullmatch(r"\d+", v) is None for v in (root, index, job_id)) or int(root) < 1 or int(job_id) < 1:
        raise WorkflowError("Runner requires the explicit managed Slurm array identity")
    if os.environ.get("SLURM_RESTART_COUNT", "0") != "0":
        raise WorkflowError("Externally requeued allocations need explicit reconciliation; no duplicate computation")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible or "," in visible:
        raise WorkflowError("Require one Slurm-assigned visible GPU, with process-local cuda:0; do not select a physical device")
    same(os.environ.get("SLURM_JOB_ACCOUNT"), r["account"], "actual allocation account")
    same(os.environ.get("SLURM_JOB_PARTITION"), "dgxh", "actual allocation partition")
    if r["qos"] is not None:
        same(os.environ.get("SLURM_JOB_QOS"), r["qos"], "actual explicitly requested QOS")
    cluster = os.environ.get("SLURM_CLUSTER_NAME")
    if cluster is not None and not re.fullmatch(r"[A-Za-z0-9_.-]+", cluster):
        raise WorkflowError("Malformed actual Slurm cluster identity")
    return {"array_job_id": root, "array_index": int(index), "scheduler_id": f"{root}_{index}",
            "job_id": job_id, "cluster": cluster}


def _safe_interruption(manifest, task, policy, result):
    """Only complete compatible authority plus explicit stop evidence permits auto retry."""
    paths = policy.to_dict()["paths"]
    overrides = policy.execution_overrides()
    if result["outcome"] == "not_started_stop_requested":
        from .coverage import inspect_task
        obs = inspect_task(task, base_dir=paths["base_dir"], overrides=overrides)
        return obs.condition == "missing" or (obs.condition == "ok" and not obs.terminal_numerical and not obs.execution_failed)
    if result["lifecycle"] == "training":
        from .coverage import inspect_task
        from .execution_provenance import collect_provenance
        provenance = collect_provenance(paths["checkout"], attempt={"lifecycle": "allocation_stop_inspection"})
        obs = inspect_task(task, base_dir=paths["base_dir"], overrides=overrides, provenance=provenance)
        return (result["outcome"] == "interrupted" and result.get("failure_published") is True
                and obs.condition == "ok" and obs.stored_status == "interrupted"
                and not obs.terminal_numerical and not obs.execution_failed)
    from .evaluation_inputs import bind_evaluation_checkpoint
    from .evaluation_coverage import collect_evaluation_coverage
    inputs = bind_evaluation_checkpoint(task, base_dir=paths["base_dir"], overrides=overrides)
    coverage = collect_evaluation_coverage(inputs, evaluation_root=paths["evaluation_root"])
    return (result["outcome"] == "evaluation_interrupted" and not any(
        row.get("failure_publication_error") for row in result["slots"])
        and not any(coverage["counts"][k] for k in ("failed", "unacknowledged_failures", "locked", "partial", "incompatible", "unstable", "inspection_error"))
        and coverage["completion_inspection_error"] is None)


def run_allocation(mapping_path):
    stop = StopRequest()
    # Install before heavy imports/CPU checkpoint readers. Pending warnings are
    # retained while the approved warning policy is loaded from the mapping.
    with allocation_signal_handlers(stop):
        try:
            return _run_owned_allocation(mapping_path, stop)
        except Exception as exc:
            try:
                exc.allocation_diagnostic_reference = _publish_runner_diagnostic(mapping_path, exc, stop)
            except Exception as publication_error:
                exc.allocation_diagnostic_publication_error = f"{type(publication_error).__name__}: {publication_error}"
            raise


def _publish_runner_diagnostic(mapping_path, exc, stop):
    """Independent preclaim/cleanup evidence only after registered mapping checks."""
    from .checkpoint_io import relative_file
    from .evaluation_persistence import _exclusive_lock
    from .manifests import load_manifest
    from .slurm_client import utc_now
    from .work_plans import seal, workflow_identity
    from .work_render import validate_bundle
    from .workflow_registry import guarded_root, _read_state, read_reference, immutable_json
    bundle = read_json(mapping_path)
    policy = OperationalPolicy(bundle["policy"])
    manifest = load_manifest(bundle["manifest_path"])
    validate_bundle(bundle, manifest, policy)
    root = guarded_root(policy)
    _, state = _read_state(root, manifest)
    wave = next((w for w in state["waves"] if w["wave_id"] == bundle["wave_id"]), None)
    if wave is None:
        raise WorkflowError("Unregistered/invalid mappings cannot authorize diagnostic writes")
    same(read_reference(root, wave["bundle_reference"]), bundle, "registered diagnostic mapping")
    index_text, job_root = os.environ.get("SLURM_ARRAY_TASK_ID", ""), os.environ.get("SLURM_ARRAY_JOB_ID", "")
    if not re.fullmatch(r"\d+", index_text) or not re.fullmatch(r"[1-9]\d*", job_root):
        raise WorkflowError("Diagnostics require the actual managed array selector")
    index = int(index_text)
    if not 0 <= index < len(bundle["items"]):
        raise WorkflowError("Unknown diagnostic work selector")
    if wave["scheduler"] is not None:
        same(job_root, wave["scheduler"]["job_id"], "diagnostic allocation root")
    attempt = uuid.uuid4().hex
    relative = f"diagnostics/{bundle['wave_id']}/{index}/{attempt}"
    lock = relative_file(root, relative + ".lock", create_parents=True)
    with _exclusive_lock(lock, {"workflow_id": workflow_identity(manifest), "wave_id": bundle["wave_id"], "array_index": index}, atomic_owner=True):
        return immutable_json(root, relative + ".json", seal({"allocation_diagnostic_version": 1,
            "wave_id": bundle["wave_id"], "array_index": index,
            "work_item_sha256": bundle["items"][index]["document_sha256"], "scheduler_id": f"{job_root}_{index}",
            "exception_type": type(exc).__name__, "reason": str(exc), "recorded_at_utc": utc_now(),
            "stop_request": {"policy": stop.policy, "reason": stop.reason}, "hostname": socket.gethostname(), "pid": os.getpid(),
            "scientific_completion_asserted": False}))


def _run_owned_allocation(mapping_path, stop):
    from .checkpoint_io import digest_file
    from .manifests import load_manifest
    from .slurm_client import utc_now
    from .work_plans import seal, validate_item
    from .work_render import batch_source, validate_bundle
    from .workflow_registry import claim_allocation, guarded_root, publish_allocation_outcome
    bundle = read_json(mapping_path)
    policy = OperationalPolicy(bundle["policy"])
    policy.require_ready(submission=True)
    paths, resources = policy.to_dict()["paths"], policy.to_dict()["resources"]
    same(str(Path(sys.executable).resolve()), str(Path(policy.to_dict()["environment"]["python_executable"]).resolve()), "configured allocation interpreter")
    manifest = load_manifest(bundle["manifest_path"])
    validate_bundle(bundle, manifest, policy)
    root = guarded_root(policy)
    expected_path = root / "bundles" / bundle["wave_id"] / "mapping.json"
    same(str(Path(mapping_path).absolute()), str(expected_path), "registered mapping path")
    script_path = expected_path.with_name("batch.sh")
    same(digest_file(script_path), bundle["script_sha256"], "allocation batch source digest")
    same(script_path.read_text(encoding="utf-8"), batch_source(policy, expected_path), "allocation script/policy")
    scheduler = allocated_identity(policy)
    index = scheduler["array_index"]
    if not 0 <= index < len(bundle["items"]):
        raise WorkflowError("Array index is absent from immutable work mapping")
    item = bundle["items"][index]
    task = validate_item(item, manifest)
    signals = policy.to_dict()["signals"]
    provenance = {"environment": {key: value for key, value in os.environ.items() if key.startswith("SLURM_") or key == "CUDA_VISIBLE_DEVICES"},
                  "hostname": socket.gethostname(), "pid": os.getpid(), "executable": sys.executable,
                  "policy_fingerprint": policy.fingerprint, "scheduler": scheduler}
    with allocation_signal_handlers(stop, warning_policy=signals["warning_policy"], termination_policy=signals["termination_policy"]):
        with claim_allocation(manifest, policy, bundle, index, scheduler, provenance, stop=stop) as (claim, wave):
            result, diagnostic, safe_stop = None, None, False
            try:
                stop.check(at_boundary=True, phase="allocation_before_components")
                # Bound native pools before any model/loaders are constructed.
                for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
                    same(os.environ.get(key), str(resources["native_threads"]), f"allocated native thread policy {key}")
                import torch
                torch.set_num_threads(resources["native_threads"])
                torch.set_num_interop_threads(1)
                from .execution_provenance import collect_provenance
                actual = collect_provenance(paths["checkout"], attempt={"lifecycle": "allocation", "wave_id": bundle["wave_id"], "array_index": index})
                same(actual["code"]["source_sha256"], bundle["execution_provenance"]["code"]["source_sha256"], "fixed planned/imported source")
                same(actual["software"], bundle["execution_provenance"]["software"], "fixed planned/allocation software")
                stop.check(at_boundary=True, phase="allocation_before_work")
                if item["lifecycle"] == "training":
                    from .coverage import inspect_task, collect_coverage
                    from .task_execution import execute_task
                    obs = inspect_task(task, base_dir=paths["base_dir"], overrides=policy.execution_overrides(), provenance=actual)
                    if not obs.training_complete:
                        same(obs.transaction, item["training_transaction"], "planned training continuation boundary")
                        if task.phase == 2 and not collect_coverage(manifest, base_dir=paths["base_dir"], overrides=policy.execution_overrides())["phases"][0]["training_successfully_completed"]:
                            raise WorkflowError("All 120 CIFAR-10 training completions must validate before phase 2")
                    stop.check(at_boundary=True, phase="allocation_training_dispatch")
                    result = execute_task(manifest, task, base_dir=paths["base_dir"], source_root=paths["checkout"],
                        overrides=policy.execution_overrides(), operation=item["operation"], stop=stop).to_dict()
                else:
                    from .evaluation_execution import execute_evaluation
                    stop.check(at_boundary=True, phase="allocation_evaluation_dispatch")
                    result = execute_evaluation(manifest, task, base_dir=paths["base_dir"], source_root=paths["checkout"],
                        overrides=policy.execution_overrides(), evaluation_root=paths["evaluation_root"],
                        retry_failed=item["retry_failed"], expected_binding=item["evaluation_binding"], stop=stop).to_dict()
                code = int(result["exit_code"])
            except TrainingInterrupted as exc:
                code = 10
                result = {"lifecycle": item["lifecycle"], "outcome": "not_started_stop_requested", "exit_code": 10}
                diagnostic = str(exc)
            except Exception as exc:
                code = 22 if isinstance(exc, WorkflowError) else 21
                diagnostic = f"{type(exc).__name__}: {exc}"
            if code == 10 and result is not None:
                try:
                    safe_stop = _safe_interruption(manifest, task, policy, result)
                except Exception as exc:
                    diagnostic = (diagnostic or "") + f"; stop authority inspection unavailable: {type(exc).__name__}: {exc}"
            record = seal({"allocation_outcome_version": 1, "wave_id": bundle["wave_id"], "array_index": index,
                "work_item_sha256": item["document_sha256"], "claim_id": claim["claim_id"],
                "scheduler_id": scheduler["scheduler_id"], "exit_code": code, "task_outcome": deepcopy(result),
                "safe_interruption": safe_stop, "diagnostic": diagnostic,
                "stop_request": {"policy": stop.policy, "reason": stop.reason}, "allocation_provenance": provenance,
                "recorded_at_utc": utc_now()})
            publish_allocation_outcome(root, wave, index, claim, record)
            return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", required=True)
    args = parser.parse_args(argv)
    try:
        result = run_allocation(args.mapping)
        print(strict_json_text(result), end="")
        return result["exit_code"]
    except Exception as exc:
        code = 10 if isinstance(exc, TrainingInterrupted) else (22 if isinstance(exc, WorkflowError) else 21)
        print(strict_json_text({"allocation_runner_error": f"{type(exc).__name__}: {exc}", "exit_code": code,
                               "diagnostic_reference": getattr(exc, "allocation_diagnostic_reference", None),
                               "diagnostic_publication_error": getattr(exc, "allocation_diagnostic_publication_error", None),
                               "scientific_completion_asserted": False}), end="", file=sys.stderr)
        return code


if __name__ == "__main__":
    raise SystemExit(main())
