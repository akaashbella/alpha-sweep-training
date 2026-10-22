"""Reviewable batch sources and immutable array mappings; never calls Slurm."""

from copy import deepcopy
import hashlib
from pathlib import Path
import shlex

from .data_contract import content_identity
from .hpc_policy import WorkflowError, exact, same, resource_budget, wave_capacity, array_argument
from .serialization import atomic_write_text
from .work_plans import seal, check_seal, validate_plan, validate_item, workflow_identity


def wave_identity(workflow_id, plan_digest, policy_fingerprint, items):
    return content_identity({"workflow_id": workflow_id, "plan_sha256": plan_digest,
        "policy_fingerprint": policy_fingerprint, "items": [item["document_sha256"] for item in items]})[7:]


def batch_source(policy, mapping_path, *, execution_pin):
    """Batch shell becomes owning Python PID via exec; B:USR1 reaches that PID only."""
    policy.require_ready()
    doc = policy.to_dict()
    env, r = doc["environment"], doc["resources"]
    from .execution_provenance import require_launch_inputs
    launch = require_launch_inputs(execution_pin, policy)
    quote = shlex.quote
    lines = ["#!/bin/bash", "set -euo pipefail",
             ': "${CUDA_VISIBLE_DEVICES:?Slurm must assign one visible GPU}"',
             'alpha_sweep_assigned_visibility="$CUDA_VISIBLE_DEVICES"']
    # Check before sourcing user code, with no project import or module probe.
    # GNU sha256sum/readlink are explicit future submit/compute-node prerequisites.
    lines.extend([
        'alpha_sweep_check_bytes() {',
        '  local checksum',
        '  if ! checksum=$(sha256sum < "$1"); then',
        '    echo "Cannot read pinned launch input: $1" >&2; exit 22',
        '  fi',
        '  if [[ "${checksum:0:64}" != "$2" ]]; then',
        '    echo "Pinned launch input changed: $1; replan before execution" >&2; exit 22',
        '  fi',
        '}',
    ])
    interpreter = launch["python"]
    lines.extend([
        'if [[ "$(readlink -f -- ' + quote(interpreter["configured_path"]) + ')" != ' + quote(interpreter["path"]) + ' ]]; then',
        '  echo "Configured interpreter target changed; replan before execution" >&2; exit 22',
        'fi',
        "alpha_sweep_check_bytes " + quote(interpreter["configured_path"]) + " " + quote(interpreter["sha256"]),
    ])
    if launch["activation_script"] is not None:
        activation = launch["activation_script"]
        lines.extend([
            "alpha_sweep_launch_parent=" + quote(activation["path"]),
            'while [[ "$alpha_sweep_launch_parent" != / ]]; do',
            '  if [[ -L "$alpha_sweep_launch_parent" ]]; then',
            '    echo "Linked activation path is rejected" >&2; exit 22',
            '  fi',
            '  alpha_sweep_launch_parent=$(dirname -- "$alpha_sweep_launch_parent")',
            'done',
            "alpha_sweep_check_bytes " + quote(activation["path"]) + " " + quote(activation["sha256"]),
        ])
    if env["activation_script"] is not None:
        lines.append("source " + quote(env["activation_script"]))
    if env["modules"]:
        lines.append("module load " + " ".join(quote(m) for m in env["modules"]))
    lines.extend(["cd " + quote(doc["paths"]["checkout"]),
        'export CUDA_VISIBLE_DEVICES="$alpha_sweep_assigned_visibility"',
        "export PYTHONPATH=" + quote(doc["paths"]["checkout"] + "/src"),
        "export PYTHONUNBUFFERED=1", "export PYTHONNOUSERSITE=1", "export SLURM_EXPORT_ENV=ALL"])
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        lines.append(f"export {key}={r['native_threads']}")
    for key, value in env["variables"].items():
        lines.append(f"export {key}=" + quote(value))
    lines.append("exec " + quote(env["python_executable"]) + " -m alpha_sweep_training.allocation_runner --mapping " + quote(str(mapping_path)))
    return "\n".join(lines) + "\n"


def render_work(plan, manifest, policy, *, work_ids=None):
    policy.require_ready()
    validate_plan(plan, manifest, policy)
    if plan["unresolved_waves"]:
        raise WorkflowError("An unresolved wave reserves capacity; render no replacement submission")
    selected, seen = [], set()
    if work_ids is not None:
        if type(work_ids) is not list or not work_ids or len(set(work_ids)) != len(work_ids):
            raise WorkflowError("Select unique eligible work unit keys")
        unknown = set(work_ids) - {item["unit_key"] for item in plan["items"]}
        if unknown:
            raise WorkflowError("Selected work is absent from the validated plan")
    for item in plan["items"]:
        if work_ids is not None and item["unit_key"] not in work_ids:
            continue
        if item["training_run_id"] in seen:
            if work_ids is not None:
                raise WorkflowError("A wave cannot train/finalize and evaluate the same training run concurrently")
            continue
        selected.append(deepcopy(item))
        seen.add(item["training_run_id"])
        if work_ids is None and len(selected) == wave_capacity(policy):
            break
    if not selected:
        raise WorkflowError("No eligible work to render; inspect complete/failure/phase/continuation coverage")
    budget = resource_budget(policy, len(selected))
    workflow_id = workflow_identity(manifest)
    wave_id = wave_identity(workflow_id, plan["document_sha256"], policy.fingerprint, selected)
    root = Path(policy.to_dict()["paths"]["workflow_root"])
    mapping = root / "bundles" / wave_id / "mapping.json"
    script = batch_source(policy, mapping, execution_pin=plan["execution_provenance"])
    bundle = seal({"bundle_version": policy.to_dict()["policy_version"], "kind": "allocation_wave", "wave_id": wave_id,
        "workflow_id": workflow_id, "plan_sha256": plan["document_sha256"], "manifest_path": plan["manifest_path"],
        "manifest_grid_id": manifest.grid_id, "manifest_document_sha256": manifest.document_sha256,
        "policy": policy.to_dict(), "policy_fingerprint": policy.fingerprint,
        "execution_provenance": plan["execution_provenance"], "items": selected,
        "array_mapping": [{"array_index": i, "work_item_sha256": item["document_sha256"],
                           "training_global_index": item["training_global_index"]} for i, item in enumerate(selected)],
        "scheduler_comment": f"alpha-sweep:{workflow_id[:12]}:{wave_id}", "resource_policy": budget,
        "script_sha256": hashlib.sha256(script.encode("utf-8")).hexdigest()})
    return bundle, script


def validate_bundle(bundle, manifest, policy):
    check_seal(bundle)
    exact(bundle, {"bundle_version", "kind", "wave_id", "workflow_id", "plan_sha256", "manifest_path",
        "manifest_grid_id", "manifest_document_sha256", "policy", "policy_fingerprint", "execution_provenance",
        "items", "array_mapping", "scheduler_comment", "resource_policy", "script_sha256", "document_sha256"}, "wave mapping")
    same(bundle["bundle_version"], policy.to_dict()["policy_version"], "bundle/policy version")
    same(bundle["kind"], "allocation_wave", "bundle lifecycle mapping")
    same(bundle["workflow_id"], workflow_identity(manifest), "bundle workflow")
    same(bundle["manifest_grid_id"], manifest.grid_id, "bundle manifest grid")
    same(bundle["manifest_document_sha256"], manifest.document_sha256, "bundle manifest document")
    same(bundle["policy"], policy.to_dict(), "bundle execution policy")
    same(bundle["policy_fingerprint"], policy.fingerprint, "bundle policy digest")
    if type(bundle["items"]) is not list or not bundle["items"]:
        raise WorkflowError("Allocation mapping is empty or malformed")
    for item in bundle["items"]:
        validate_item(item, manifest)
    if len({item["training_run_id"] for item in bundle["items"]}) != len(bundle["items"]):
        raise WorkflowError("Overlapping allocation ownership in wave")
    same(bundle["array_mapping"], [{"array_index": i, "work_item_sha256": item["document_sha256"],
        "training_global_index": item["training_global_index"]} for i, item in enumerate(bundle["items"])], "array index membership")
    wave_id = wave_identity(bundle["workflow_id"], bundle["plan_sha256"], policy.fingerprint, bundle["items"])
    same(bundle["wave_id"], wave_id, "immutable wave identity")
    same(bundle["scheduler_comment"], f"alpha-sweep:{bundle['workflow_id'][:12]}:{wave_id}", "exact submission identifier")
    same(bundle["resource_policy"], resource_budget(policy, len(bundle["items"])), "combined wave requests")
    return bundle


def write_bundle(workflow, bundle, script):
    from .checkpoint_io import relative_file, digest_file
    from .workflow_registry import immutable_json
    workflow.guard()
    validate_bundle(bundle, workflow.manifest, workflow.policy)
    root, wave_id = workflow.root, bundle["wave_id"]
    expected = batch_source(workflow.policy, root / "bundles" / wave_id / "mapping.json",
                            execution_pin=bundle["execution_provenance"])
    same(script, expected, "rendered allocation script")
    same(hashlib.sha256(script.encode("utf-8")).hexdigest(), bundle["script_sha256"], "batch script seal")
    target = relative_file(root, f"bundles/{wave_id}/batch.sh", create_parents=True)
    if target.exists():
        same(target.read_text(encoding="utf-8"), script, "immutable submission script reuse")
    else:
        atomic_write_text(target, script)
    same(digest_file(target), bundle["script_sha256"], "actual immutable batch source digest")
    return immutable_json(root, f"bundles/{wave_id}/mapping.json", bundle)


def submission_arguments(bundle, policy):
    policy.require_ready()
    r, paths = policy.to_dict()["resources"], policy.to_dict()["paths"]
    count = len(bundle["items"])
    args = ["--parsable", "--no-requeue", "--nodes=1", "--ntasks=1", "--gpus-per-node=1",
        "--partition=dgxh", f"--account={r['account']}", f"--cpus-per-task={r['cpus']}",
        f"--mem={r['memory_gb']}G", f"--time={r['wall_time']}",
        array_argument(policy, count), f"--signal=B:USR1@{policy.to_dict()['signals']['lead_seconds']}",
        "--export=NONE", f"--comment={bundle['scheduler_comment']}", f"--job-name=alpha-sweep-{bundle['wave_id'][:12]}",
        "--chdir=" + paths["checkout"], "--output=" + paths["slurm_log_root"] + "/%A_%a.out",
        "--error=" + paths["slurm_log_root"] + "/%A_%a.err"]
    if r["qos"] is not None:
        args.append("--qos=" + r["qos"])
    if r["constraint"] is not None:
        args.append("--constraint=" + r["constraint"])
    args.append(paths["workflow_root"] + f"/bundles/{bundle['wave_id']}/batch.sh")
    return args
