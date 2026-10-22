"""One-shot planning/rendering/submission/reconciliation, never a submit-node trainer."""

import argparse
from pathlib import Path
import sys
import uuid

from .artifacts import _is_link
from .checkpoint_io import digest_file, relative_file
from .data_contract import content_identity
from .hpc_policy import WorkflowError, load_policy, same, shared_path, validate_array_evidence
from .manifests import load_manifest, select_task, task_config
from .serialization import atomic_write_json, durable_mkdir, fsync_directory, read_json, strict_json_text
from .slurm_client import SlurmClient, utc_now, validate_restriction_report
from .work_plans import build_work_plan, workflow_identity
from .work_render import render_work, write_bundle, validate_bundle, submission_arguments, batch_source
from .workflow_registry import (open_workflow, read_workflow, guarded_root, read_reference,
                                immutable_json, _read_state, validate_operational_lock)


def verify_effective_evidence(policy):
    policy.require_ready(submission=True)
    limits = policy.to_dict()["effective_limits"]
    path = Path(limits["inspection_report_path"])
    if any(_is_link(p) for p in (path, *path.parents)):
        raise WorkflowError("Linked restriction evidence is rejected")
    same(digest_file(path), limits["inspection_report_sha256"], "actual restriction inspection evidence")
    report = read_json(path)
    validate_restriction_report(report, policy.to_dict()["resources"])
    validate_array_evidence(report, policy)
    return report


def submit_bundle(workflow, bundle, *, execute=False):
    """Only submission entry point: guarded fresh selection, intent, then sbatch."""
    if execute is not True:
        raise WorkflowError("Submission requires the explicit --execute/--submit operational action")
    workflow.guard()
    policy, manifest = workflow.policy, workflow.manifest
    validate_bundle(bundle, manifest, policy)
    from .execution_provenance import require_launch_inputs
    require_launch_inputs(bundle["execution_provenance"], policy)
    verify_effective_evidence(policy)
    client = SlurmClient(policy)
    workflow.reconcile(client)
    retries = [item["training_run_id"] for item in bundle["items"] if item["operator_retry_reason"] is not None]
    reasons = {item["operator_retry_reason"] for item in bundle["items"] if item["operator_retry_reason"] is not None}
    if len(reasons) > 1:
        raise WorkflowError("A wave must have one coherent audited operator retry decision")
    current = build_work_plan(manifest, policy, manifest_path=bundle["manifest_path"], state=workflow.state,
                              retry_run_ids=retries, retry_reason=next(iter(reasons), None))
    current_items = {item["unit_key"]: item for item in current["items"]}
    for item in bundle["items"]:
        if item["unit_key"] not in current_items:
            raise WorkflowError("Rendered work is no longer eligible; inspect allocation/science authority and replan")
        same(item, current_items[item["unit_key"]], "fresh reserved work/continuation/checkpoint binding")
    same(current["execution_provenance"]["code"]["source_sha256"], bundle["execution_provenance"]["code"]["source_sha256"], "fixed planned submission source")
    same(current["execution_provenance"]["software"], bundle["execution_provenance"]["software"], "fixed planned submission software")
    same(current["execution_provenance"]["launch_inputs"], bundle["execution_provenance"]["launch_inputs"], "fixed planned submission launch inputs")
    relative = f"bundles/{bundle['wave_id']}/mapping.json"
    target = relative_file(workflow.root, relative)
    same(read_json(target), bundle, "registered bundle content")
    script = relative_file(workflow.root, f"bundles/{bundle['wave_id']}/batch.sh")
    same(digest_file(script), bundle["script_sha256"], "submission script content")
    same(script.read_text(encoding="utf-8"), batch_source(policy, target, execution_pin=bundle["execution_provenance"]), "submission launch/policy")
    # Slurm opens logs before running the shell: prepare persistent directories now.
    logs = Path(policy.to_dict()["paths"]["slurm_log_root"])
    if any(_is_link(p) for p in (logs, *logs.parents)):
        raise WorkflowError("Linked Slurm log paths are rejected")
    durable_mkdir(logs)
    argv = submission_arguments(bundle, policy)
    reference = {"relative_path": relative, "sha256": digest_file(target)}
    wave = workflow.reserve(bundle, reference, argv)
    try:
        scheduler = client.submit(argv, workflow=workflow, wave_id=bundle["wave_id"])
        workflow.record_acknowledgement(bundle["wave_id"], scheduler)
        return {"outcome": "submitted", "wave_id": bundle["wave_id"], "scheduler": scheduler}
    except Exception as exc:
        # The intent was already durable. Even a command error is conservatively
        # ambiguous until exact ID/admin evidence resolves whether sbatch accepted.
        diagnostic = f"{type(exc).__name__}: {exc}"
        try:
            wave = workflow.wave(bundle["wave_id"])
            wave["submission_diagnostic"] = diagnostic
            workflow.commit({"kind": "submission_acknowledgement_uncertain", "wave_id": bundle["wave_id"]})
        except Exception as recording_error:
            diagnostic += f"; registry acknowledgement unavailable: {type(recording_error).__name__}: {recording_error}"
        return {"outcome": "submission_uncertain", "wave_id": bundle["wave_id"], "diagnostic": diagnostic,
                "replacement_submission_allowed": False}


def resolve_submission(workflow, wave_id, *, job_id=None, cluster=None, confirmed_absent=False, evidence=None, reason=None):
    wave = workflow.wave(wave_id)
    if wave["scheduler"] is not None or wave["status"] != "submission_uncertain":
        raise WorkflowError("Only unresolved submission intent needs scheduler identity attachment/non-submission resolution")
    if job_id is not None:
        workflow.policy.require_recovery_ready(scheduler=True)
        scheduler = SlurmClient(workflow.policy).verify_submission_identity(wave, job_id, cluster)
        workflow.record_acknowledgement(wave_id, scheduler)
        return {"outcome": "submission_identity_reconciled", "scheduler": scheduler}
    if confirmed_absent is not True or evidence is None or type(reason) is not str or not reason.strip() or wave["claims"]:
        raise WorkflowError("Require explicit administrator/operator proof of NON-submission and a reason; existing claims prove an allocation existed")
    same(evidence["wave_id"], wave_id, "non-submission evidence wave")
    same(evidence["scheduler_comment"], wave["scheduler_comment"], "non-submission evidence token")
    same(evidence["all_allocations_not_submitted_confirmed"], True, "non-submission confirmation")
    for index in range(len(wave["items"])):
        lock = relative_file(workflow.root, f"allocations/{wave_id}/{index}.lock")
        if lock.exists() or _is_link(lock):
            raise WorkflowError("Residual allocation ownership contradicts non-submission resolution; inspect and recover it explicitly")
    ref = immutable_json(workflow.root, f"resolutions/{uuid.uuid4().hex}.json", {"kind": "operator_confirmed_not_submitted",
        "evidence": evidence, "reason": reason, "recorded_at_utc": utc_now()})
    wave.update(status="confirmed_not_submitted", resolution=ref)
    workflow.commit({"kind": "non_submission_resolved", "wave_id": wave_id, "resolution": ref})
    return {"outcome": "non_submission_explicitly_resolved", "wave_id": wave_id}


def resolve_terminal(workflow, wave_id, *, confirmed=False, evidence=None, reason=None):
    """Explicit terminal proof when accounting is delayed/unavailable; never a timeout guess."""
    wave = workflow.wave(wave_id)
    if confirmed is not True or evidence is None or type(reason) is not str or not reason.strip() or wave["scheduler"] is None:
        raise WorkflowError("Require exact scheduler identity, actual authoritative terminal/inactive evidence for ALL allocations, explicit confirmation and reason")
    workflow.policy.require_recovery_ready(scheduler=True)
    observation = SlurmClient(workflow.policy).observe_wave(wave)
    if any(row.get("queue_present") for row in observation["allocations"]):
        raise WorkflowError("Scheduler still reports a managed allocation; terminal confirmation cannot override it")
    same(evidence["workflow_id"], workflow_identity(workflow.manifest), "terminal evidence workflow")
    same(evidence["wave_id"], wave_id, "terminal evidence wave")
    same(evidence["scheduler_comment"], wave["scheduler_comment"], "terminal evidence token")
    expected = [f"{wave['scheduler']['job_id']}_{i}" for i in range(len(wave["items"]))]
    same([row["scheduler_id"] for row in evidence["allocations"]], expected, "complete exact terminal allocation evidence")
    for index, row in enumerate(evidence["allocations"]):
        same(row["terminal_and_owners_inactive_confirmed"], True, "actual terminal and all owning processes inactive")
        if type(row["authoritative_evidence"]) is not str or not row["authoritative_evidence"].strip():
            raise WorkflowError("Each allocation needs actual administrator/accounting evidence, not an empty queue/PID/age inference")
        lock = relative_file(workflow.root, f"allocations/{wave_id}/{index}.lock")
        if lock.exists() or _is_link(lock):
            raise WorkflowError("Recover residual allocation locks explicitly before terminal resolution")
    ref = immutable_json(workflow.root, f"resolutions/{uuid.uuid4().hex}.json", {"kind": "operator_confirmed_terminal",
        "wave_id": wave_id, "scheduler": wave["scheduler"], "evidence": evidence, "reason": reason, "recorded_at_utc": utc_now()})
    wave.update(status="terminal", resolution=ref, scheduler_observation=observation)
    workflow.commit({"kind": "terminal_allocation_explicitly_resolved", "wave_id": wave_id, "resolution": ref})
    return {"outcome": "terminal_allocations_explicitly_resolved", "scientific_completion_asserted": False, "wave_id": wave_id}


def inspect_managed_locks(manifest, policy, *, scope="workflow", run_id=None):
    """Exact ownership evidence only; no inactive-owner inference or writes."""
    root = guarded_root(policy, recovery_only=True)
    head, state = _read_state(root, manifest)
    same(state["policy_fingerprint"], policy.fingerprint, "lock inspection original registry policy")
    paths, expected = [], {}
    if scope == "workflow":
        if run_id is not None:
            raise WorkflowError("Workflow locks do not take a scientific run selector")
        if root.exists():
            controller = root / ".controller.lock"
            paths.append(controller)
            expected[str(controller)] = {"workflow_id": workflow_identity(manifest)}
            if (root / "owner.json").exists() and (root / "HEAD.json").exists():
                for wave in state["waves"]:
                    for index in range(len(wave["items"])):
                        path = relative_file(root, f"allocations/{wave['wave_id']}/{index}.lock")
                        paths.append(path)
                        expected[str(path)] = {"workflow_id": workflow_identity(manifest), "wave_id": wave["wave_id"], "array_index": index}
    else:
        task = select_task(manifest, run_id=run_id)
        config = task_config(task, policy.recovery_storage_overrides())
        if scope == "training":
            from .recovery import inspect_locks
            rows = [e.__dict__ for e in inspect_locks(config, base_dir=policy.to_dict()["paths"]["base_dir"])]
            same(read_json(relative_file(root, "HEAD.json")), head, "stable registry policy during run-lock inspection")
            return {"lock_report_version": 1, "workflow_id": workflow_identity(manifest), "scope": scope, "run_id": run_id, "locks": rows}
        if scope != "evaluation":
            raise WorkflowError("Unknown lock recovery scope")
        from .evaluation_inputs import bind_evaluation_checkpoint
        from .evaluation_persistence import evaluation_directory
        from .evaluation_protocol import prescribed_slots
        if policy.to_dict()["paths"]["evaluation_root"] is None:
            raise WorkflowError("Evaluation lock inspection requires the original paths.evaluation_root")
        inputs = bind_evaluation_checkpoint(task, base_dir=policy.to_dict()["paths"]["base_dir"], overrides=policy.recovery_storage_overrides())
        namespace = evaluation_directory(inputs, evaluation_root=policy.to_dict()["paths"]["evaluation_root"])
        paths = [namespace / ".namespace.lock", namespace / ".sweep.lock",
                 *(namespace / "locks" / f"{s.slot_id}.lock" for s in prescribed_slots())]
        expected = {str(path): {"binding_sha256": content_identity(inputs.binding)} for path in paths}
        for slot in prescribed_slots():
            path = namespace / "locks" / f"{slot.slot_id}.lock"
            expected[str(path)].update(slot.to_dict())
    rows = []
    for path in paths:
        if not path.exists() and not _is_link(path):
            continue
        if any(_is_link(part) for part in (path, *path.parents)) or not path.is_file():
            raise WorkflowError("Linked/non-file lock evidence requires manual review")
        owner = read_json(path)
        validate_operational_lock(owner, expected[str(path)])
        rows.append({"path": str(path), "sha256": digest_file(path), "owner": owner})
    same(read_json(relative_file(root, "HEAD.json")), head, "stable registry policy during lock inspection")
    return {"lock_report_version": 1, "workflow_id": workflow_identity(manifest), "scope": scope, "run_id": run_id, "locks": rows}


def release_managed_locks(manifest, policy, evidence, *, confirmed=False, reason=None):
    if confirmed is not True or type(reason) is not str or not reason.strip():
        raise WorkflowError("Confirm ALL recorded owners/allocations inactive across nodes and supply an audit reason")
    scope, run_id = evidence["scope"], evidence["run_id"]
    actual = inspect_managed_locks(manifest, policy, scope=scope, run_id=run_id)
    same(actual, evidence, "complete unchanged lock ownership/path/digest evidence")
    if not actual["locks"]:
        raise WorkflowError("No verified lock evidence to recover")
    if scope == "training":
        from .recovery import LockEvidence, release_confirmed_locks
        task = select_task(manifest, run_id=run_id)
        return release_confirmed_locks(task_config(task, policy.recovery_storage_overrides()), base_dir=policy.to_dict()["paths"]["base_dir"],
            expected=tuple(LockEvidence(**row) for row in actual["locks"]), inactive_owners_confirmed=True, reason=reason)
    recovered = []
    for row in actual["locks"]:
        path = Path(row["path"])
        same(digest_file(path), row["sha256"], "lock bytes at explicit recovery")
        same(read_json(path), row["owner"], "lock owner at explicit recovery")
        archive = path.with_name(path.name + ".recovered-" + uuid.uuid4().hex)
        if archive.exists() or _is_link(archive):
            raise WorkflowError("Recovery archive exists; preserve evidence")
        atomic_write_json(archive.with_name(archive.name + ".audit.json"), {"workflow_id": workflow_identity(manifest),
            "scope": scope, "reason": reason, "inactive_owners_confirmed": True, "original": row, "archive": str(archive)})
        path.rename(archive)
        fsync_directory(path.parent)
        recovered.append(str(archive))
    return recovered


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "render", "submit", "status", "reconcile", "advance", "resolve-submission", "resolve-terminal", "adopt-policy", "inspect-locks", "release-locks"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--manifest", required=True)
        cmd.add_argument("--policy", required=True)
        if name in ("plan", "advance"):
            cmd.add_argument("--retry-run-id", nargs="+", default=[])
            cmd.add_argument("--retry-reason")
        if name == "render":
            cmd.add_argument("--plan", required=True)
            cmd.add_argument("--work-id", nargs="+")
        if name == "submit":
            cmd.add_argument("--bundle", required=True)
            cmd.add_argument("--execute", action="store_true")
        if name == "advance":
            cmd.add_argument("--submit", action="store_true")
        if name == "status":
            cmd.add_argument("--scheduler", action="store_true")
            cmd.add_argument("--workflow-only", action="store_true", help="Inspect historical registry authority without collecting scientific coverage or requiring a research environment")
        if name == "resolve-submission":
            cmd.add_argument("--wave-id", required=True)
            group = cmd.add_mutually_exclusive_group(required=True)
            group.add_argument("--job-id")
            group.add_argument("--confirmed-not-submitted", action="store_true")
            cmd.add_argument("--cluster")
            cmd.add_argument("--evidence")
            cmd.add_argument("--reason")
        if name == "adopt-policy":
            cmd.add_argument("--reason", required=True)
            cmd.add_argument("--previous-policy", required=True, help="Original sealed current policy; must match the stored fingerprint before corrected-policy adoption")
        if name == "resolve-terminal":
            cmd.add_argument("--wave-id", required=True)
            cmd.add_argument("--evidence", required=True)
            cmd.add_argument("--confirmed-terminal-inactive-owners", action="store_true")
            cmd.add_argument("--reason", required=True)
        if name == "inspect-locks":
            cmd.add_argument("--scope", choices=("workflow", "training", "evaluation"), default="workflow")
            cmd.add_argument("--run-id")
        if name == "release-locks":
            cmd.add_argument("--evidence", required=True)
            cmd.add_argument("--confirmed-inactive-owners", action="store_true")
            cmd.add_argument("--reason", required=True)
    restrictions = commands.add_parser("inspect-restrictions")
    restrictions.add_argument("--policy", required=True)
    args = parser.parse_args(argv)
    try:
        policy = load_policy(args.policy)
        if args.command == "inspect-restrictions":
            result = SlurmClient(policy).inspect_restrictions()
        else:
            recovery_commands = {"status", "reconcile", "resolve-submission", "resolve-terminal", "inspect-locks", "release-locks"}
            recovery_only = args.command in recovery_commands
            if recovery_only:
                scheduler_recovery = (args.command in ("reconcile", "resolve-terminal")
                    or (args.command == "status" and args.scheduler)
                    or (args.command == "resolve-submission" and args.job_id is not None))
                policy.require_recovery_ready(scheduler=scheduler_recovery)
            else:
                policy.require_ready()
            manifest = load_manifest(args.manifest)
            manifest_path = str(Path(args.manifest).expanduser().absolute())
            shared_path(manifest_path, "shared full manifest path")
            if args.command == "plan":
                result = build_work_plan(manifest, policy, manifest_path=manifest_path, state=read_workflow(manifest, policy),
                    retry_run_ids=args.retry_run_id, retry_reason=args.retry_reason)
            elif args.command == "status":
                historical_only = args.workflow_only or policy.to_dict()["paths"]["split_file"] is not None
                state = read_workflow(manifest, policy, recovery_only=historical_only)
                result = {"workflow": state, "scientific": None,
                          "scientific_coverage_collected": False, "scientific_coverage_diagnostic": None}
                if historical_only:
                    result["scientific_coverage_diagnostic"] = "Workflow-only historical inspection: scientific coverage was not collected; the original policy remains sealed and any global split override cannot authorize research execution"
                else:
                    from .evaluation_execution import collect_experiment_coverage
                    result["scientific"] = collect_experiment_coverage(manifest,
                        base_dir=policy.to_dict()["paths"]["base_dir"], overrides=policy.execution_overrides(), evaluation_root=policy.to_dict()["paths"]["evaluation_root"])
                    result["scientific_coverage_collected"] = True
                if args.scheduler:
                    result["scheduler"] = [{"wave_id": wave["wave_id"], **SlurmClient(policy).observe_wave(wave)} for wave in state["waves"] if wave["status"] != "confirmed_not_submitted"]
            elif args.command == "inspect-locks":
                result = inspect_managed_locks(manifest, policy, scope=args.scope, run_id=args.run_id)
            elif args.command == "release-locks":
                result = {"recovered_archives": release_managed_locks(manifest, policy, read_json(args.evidence),
                    confirmed=args.confirmed_inactive_owners, reason=args.reason)}
            else:
                with open_workflow(manifest, policy, initialize=args.command in ("render", "submit", "advance"),
                                   adopting_policy=args.command == "adopt-policy",
                                   previous_policy=load_policy(args.previous_policy) if args.command == "adopt-policy" else None,
                                   recovery_only=recovery_only) as workflow:
                    if args.command == "render":
                        plan = read_json(args.plan)
                        same(plan["manifest_path"], manifest_path, "render full manifest path")
                        if any(w["status"] not in ("terminal", "confirmed_not_submitted") for w in workflow.state["waves"]):
                            raise WorkflowError("Shared wave remains unresolved; no replacement rendering")
                        bundle, script = render_work(plan, manifest, policy, work_ids=args.work_id)
                        reference = write_bundle(workflow, bundle, script)
                        result = {"outcome": "rendered_only", "mapping": str(workflow.root / reference["relative_path"]), "reference": reference,
                                  "submission_arguments": submission_arguments(bundle, policy), "sbatch_invoked": False}
                    elif args.command == "submit":
                        bundle = read_json(args.bundle)
                        same(bundle["manifest_path"], manifest_path, "submission manifest path")
                        result = submit_bundle(workflow, bundle, execute=args.execute)
                    elif args.command == "reconcile":
                        result = workflow.reconcile(SlurmClient(policy))
                    elif args.command == "advance":
                        workflow.reconcile(SlurmClient(policy))
                        plan = build_work_plan(manifest, policy, manifest_path=manifest_path, state=workflow.state,
                            retry_run_ids=args.retry_run_id, retry_reason=args.retry_reason)
                        result = {"plan": plan, "submitted": False}
                        if args.submit and plan["items"] and not plan["unresolved_waves"]:
                            bundle, script = render_work(plan, manifest, policy)
                            write_bundle(workflow, bundle, script)
                            result.update(submission=submit_bundle(workflow, bundle, execute=True))
                            result["submitted"] = result["submission"]["outcome"] == "submitted"
                    elif args.command == "resolve-submission":
                        result = resolve_submission(workflow, args.wave_id, job_id=args.job_id, cluster=args.cluster,
                            confirmed_absent=args.confirmed_not_submitted, evidence=None if args.evidence is None else read_json(args.evidence), reason=args.reason)
                    elif args.command == "resolve-terminal":
                        result = resolve_terminal(workflow, args.wave_id, confirmed=args.confirmed_terminal_inactive_owners,
                            evidence=read_json(args.evidence), reason=args.reason)
                    else:  # adopt-policy requires all allocations conclusively terminal first.
                        old_policy = workflow.state["policy_fingerprint"]
                        for wave in workflow.state["waves"]:
                            if wave["status"] == "confirmed_not_submitted":
                                if wave["scheduler"] is not None or wave["claims"] or wave["resolution"] is None:
                                    raise WorkflowError("Non-submission must have its original proof and no scheduler acknowledgement/worker claims before policy adoption")
                                resolution = read_reference(workflow.root, wave["resolution"])
                                same(resolution["kind"], "operator_confirmed_not_submitted", "adoption non-submission resolution")
                                same(resolution["evidence"]["wave_id"], wave["wave_id"], "adoption non-submission wave")
                                same(resolution["evidence"]["scheduler_comment"], wave["scheduler_comment"], "adoption non-submission token")
                                same(resolution["evidence"]["all_allocations_not_submitted_confirmed"], True, "adoption original non-submission proof")
                            else:
                                if wave["scheduler"] is None:
                                    raise WorkflowError("Resolve the original uncertain submission before adopting a corrected policy")
                                resolution = None if wave["resolution"] is None else read_reference(workflow.root, wave["resolution"])
                                resolved_terminal = resolution is not None and resolution["kind"] == "operator_confirmed_terminal"
                                if resolved_terminal:
                                    same(resolution["wave_id"], wave["wave_id"], "adoption terminal-resolution wave")
                                    same(resolution["scheduler"], wave["scheduler"], "adoption terminal-resolution scheduler")
                                else:
                                    from .hpc_policy import OperationalPolicy
                                    original_wave_policy = OperationalPolicy(wave["policy"])
                                    original_wave_policy.require_recovery_ready(scheduler=True)
                                    if not SlurmClient(original_wave_policy).observe_wave(wave)["terminal"]:
                                        raise WorkflowError("Cannot adopt policy while original allocations/submission state are uncertain or active")
                            for index in range(len(wave["items"])):
                                path = relative_file(workflow.root, f"allocations/{wave['wave_id']}/{index}.lock")
                                if path.exists() or _is_link(path):
                                    raise WorkflowError("Cannot adopt policy with an unresolved allocation owner")
                            if wave["status"] != "confirmed_not_submitted":
                                wave["status"] = "terminal"
                        workflow.commit({"kind": "operator_policy_adopted", "previous_policy_fingerprint": old_policy, "reason": args.reason})
                        result = {"outcome": "policy_explicitly_adopted", "policy_fingerprint": policy.fingerprint}
        print(strict_json_text(result), end="")
        if isinstance(result, dict) and result.get("outcome") == "submission_uncertain":
            return 22
        if isinstance(result, dict) and result.get("submission", {}).get("outcome") == "submission_uncertain":
            return 22
        return 0
    except Exception as exc:
        print(strict_json_text({"workflow_error": f"{type(exc).__name__}: {exc}", "exit_code": 22,
                               "scientific_completion_asserted": False}), end="", file=sys.stderr)
        return 22


if __name__ == "__main__":
    raise SystemExit(main())
