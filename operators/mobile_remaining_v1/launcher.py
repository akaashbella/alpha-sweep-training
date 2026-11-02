"""External, version-1 remaining-27 operator path; never edit the native package.

Preparation/inspection are CPU-only. Submission requires explicit --execute and
fresh operator evidence. Workers dispatch unchanged native Tasks, not seed aliases.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import uuid

VERSION = "mobile-remaining27-operator-v1"
COMMIT = "8010d07d9d1f0886ab1e180523f316623ef622a4"
SOURCE_SHA = "a5726bac3f449455d36e2f53840f3e0fd77768a7ab2b989d2884da20f920005c"
GRID = "2c0689e4aa3d0b6734323642a89dd7de809337caabcbe358ab600b7287a16167"
MANIFEST_SHA = "08555fb94e023849d37c0140833a21e0db12b02ce4ab382bc32d4202b76f6423"
PROFILE = "cifar10-mobile-eps0316-v1"
NATIVE = "/nfs/hpc/share/bellaak/research/alpha-sweep/alpha-sweep-mobile-eps0316"
STUDY = "/nfs/hpc/share/bellaak/research/alpha-sweep/mobile-eps0316-v1"
ENVIRONMENT = "/nfs/hpc/share/bellaak/research/alpha-sweep/alpha-sweep-training/.venv"
SOFTWARE = {"python": "3.11.15", "torch": "2.5.1+cu124", "torchvision": "0.20.1+cu124",
            "numpy": "1.26.4", "PyYAML": "6.0.3", "cuda": "12.4", "cudnn": "90100"}
ALPHAS = (0., .01, .02, .05, .1, .2, .3, .5, .75, 1.)
INDICES = tuple(range(3, 30))
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
BUNDLE_FILES = ("launcher.py", "report.py", "batch.sh", "test_operator.py", "run_cpu_checks.py", "HPC_HANDOFF.md")
HOME = Path(__file__).resolve().parent


def require(ok, message):
    if not ok:
        raise ValueError(message)


def same(actual, expected, label):
    require(canonical(actual) == canonical(expected), f"Incompatible {label}")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sealed(value):
    return {**value, "document_sha256": hashlib.sha256(canonical(value).encode()).hexdigest()}


def check(value):
    same(value["document_sha256"], sealed({k: v for k, v in value.items() if k != "document_sha256"})["document_sha256"], "operator seal")
    return value


def plain(path):
    path = Path(path).expanduser().absolute()
    require(not any(p.is_symlink() or getattr(p, "is_junction", lambda: False)() for p in (path, *path.parents)), f"Linked path: {path}")
    return path.resolve()


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    value = json.loads(plain(path).read_text(encoding="utf-8"), object_pairs_hook=pairs)
    canonical(value)  # Reject NaN/Infinity and overflowed numeric literals.
    return value


def digest(path):
    path = plain(path)
    with path.open("rb") as stream:
        value = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def fsync(path):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish(path, value):
    """Exclusive atomic publication; never replace existing evidence."""
    path = plain(path)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(canonical(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Atomic, exclusive visible name on shared storage.
        fsync(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def disjoint(root, protected):
    root = plain(root)
    for path in protected:
        path = plain(path)
        require(root != path and root not in path.parents and path not in root.parents, f"Namespace overlap: {root}, {path}")
    return root


def claim(root, protected, kind):
    root = disjoint(root, protected)
    require(not root.exists(), "New namespace required; preserve partial/completed evidence")
    require(root.parent.is_dir(), "Choose an existing persistent parent directory")
    root.mkdir()
    fsync(root.parent)
    owner = sealed({"version": VERSION, "kind": kind, "root": str(root), "token": uuid.uuid4().hex})
    publish(root / "OWNER.json", owner)
    return root, owner


def operator_pin():
    return {"home": str(HOME), "files": {name: digest(HOME / name) for name in BUNDLE_FILES}}


def mapping(index):
    require(type(index) is int and index in INDICES, "Select only global indices 3–29; pilot indices 0–2 are forbidden")
    return ALPHAS[index // 3], index % 3


def task_rows(manifest):
    require(len(manifest.tasks) == 30, "Require the original complete thirty-task manifest")
    rows = []
    for index in INDICES:
        task = manifest.tasks[index]
        alpha, seed = mapping(index)
        s, e = task.config.scientific, task.config.execution
        same([task.global_index, s["dataset"], s["architecture"], s["alpha"], s["training_seed"], s["training_seeds"]],
             [index, "cifar10", "mobilenetv3_large", alpha, seed, [0, 1, 2]], "index/alpha/seed membership")
        same([s["specification_version"], s["optimizer"]["epsilon"], e["num_workers"], e["device"]],
             [PROFILE, .0316, 4, "cuda:0"], "profile/runtime declaration")
        same(e["storage"]["output_root"], str(Path(STUDY) / "runs"), "native output root")
        for key in ("checkpoint_root", "result_root", "log_root"):
            same(e["storage"][key], None, "native nested artifact roots")
        rows.append({"global_index": index, "training_alpha": alpha, "training_seed": seed,
                     "run_id": task.run_id, "scientific_fingerprint": task.fingerprint})
    return rows


def native_inputs(*, require_pilots=True):
    """Read and pin actual HPC inputs. No imported source substitution/repair."""
    from alpha_sweep_training.mobile_candidate import load_pilot, _launch_inputs
    from alpha_sweep_training.execution_provenance import collect_provenance
    from alpha_sweep_training.artifacts import resolve_run_paths
    from alpha_sweep_training.coverage import inspect_task
    pilot_path = plain(Path(STUDY) / "pilot.json")
    pilot, manifest = load_pilot(pilot_path)
    same(pilot["source_root"], NATIVE, "frozen native checkout path")
    same([manifest.grid_id, manifest.document_sha256], [GRID, MANIFEST_SHA], "reviewed study identity")
    same(pilot["candidate_root"], STUDY, "study root")
    provenance = collect_provenance(NATIVE, attempt={"lifecycle": VERSION})
    same([provenance["code"]["commit"], provenance["code"]["source_sha256"], provenance["code"]["dirty"]],
         [COMMIT, SOURCE_SHA, False], "unchanged native source")
    same(provenance["software"], SOFTWARE, "native software")
    same(pilot["source_pin"]["source_sha256"], SOURCE_SHA, "pilot native source pin")
    same(pilot["software_pin"], SOFTWARE, "pilot software pin")
    same(str(Path(sys.executable).absolute()), ENVIRONMENT + "/bin/python", "configured existing interpreter")
    launch = _launch_inputs(ENVIRONMENT + "/bin/activate")
    same(launch, pilot["launch_inputs"], "unchanged pilot external inputs")
    rows = task_rows(manifest)
    # Complete original pilot journal authority, not the upload or scheduler, is
    # the execution gate. These observations acquire no writer leases.
    if require_pilots:
        observations = [inspect_task(t, base_dir=pilot["base_dir"]) for t in manifest.tasks[:3]]
        require(all(o.condition == "ok" and o.training_complete and o.completed_epochs == 200 for o in observations),
                "Verify all three existing full-budget pilot completions before freezing/dispatch")
    protected = [NATIVE, STUDY, HOME, *pilot["protected_roots"]]
    for task in manifest.tasks:
        storage = task.config.execution["storage"]
        protected.extend(p for k, p in storage.items() if p is not None and k != "split_file")
        if storage["split_file"]:
            protected.append(Path(storage["split_file"]).parent)
        protected.extend(resolve_run_paths(task.config, pilot["base_dir"]).directories())
    binding = {"native_commit": COMMIT, "source_sha256": SOURCE_SHA, "software": provenance["software"],
        "launch_inputs": launch, "manifest_path": str(Path(STUDY) / "study.json"),
        "manifest_file_sha256": digest(Path(STUDY) / "study.json"), "grid_id": GRID,
        "manifest_document_sha256": MANIFEST_SHA, "pilot_path": str(pilot_path),
        "pilot_file_sha256": digest(pilot_path), "base_dir": pilot["base_dir"], "selection": rows}
    return binding, manifest, protected


def inspection(path):
    value = read(path)
    keys = {"inspected_at_utc", "operator", "reason", "account", "qos", "node", "gres", "concurrency",
            "other_active_experiment_tasks", "all_prior_submissions_resolved", "evidence_files"}
    require(type(value) is dict and set(value) == keys, f"Inspection requires exactly {sorted(keys)}")
    for key in ("operator", "reason", "account", "node", "gres"):
        require(type(value[key]) is str and bool(value[key].strip()) and "\n" not in value[key], f"Explicit {key} required")
    require(all(re.fullmatch(r"[A-Za-z0-9_.-]+", value[k]) for k in ("account", "node")), "Single actual account and node names required")
    require(re.fullmatch(r"gpu(?::[A-Za-z0-9_.-]+)?:1", value["gres"]) is not None, "Require an inspected one-GPU GRES request")
    timestamp = datetime.fromisoformat(value["inspected_at_utc"].replace("Z", "+00:00"))
    require(timestamp.utcoffset() is not None and timestamp.utcoffset().total_seconds() == 0, "Inspection time must be UTC")
    require(timestamp <= datetime.now(timezone.utc), "Inspection cannot be dated in the future")
    require(value["qos"] is None or (type(value["qos"]) is str and value["qos"].strip()), "Explicit QOS or null required")
    require(value["qos"] is None or re.fullmatch(r"[A-Za-z0-9_.-]+", value["qos"]), "Single actual QOS name required")
    c, other = value["concurrency"], value["other_active_experiment_tasks"]
    require(type(c) is int and 1 <= c <= 4 and type(other) is int and 0 <= other and c + other <= 4,
            "Choose concurrency after inspection; old/new combined running allowance is at most four")
    require(value["all_prior_submissions_resolved"] is True, "Resolve all prior submission ambiguity before a new intent")
    require(type(value["evidence_files"]) is dict and set(value["evidence_files"]) == {"accounts", "nodes", "active_work"},
            "Attach actual current account/QOS, node/GPU and all active-work evidence files")
    return {"record": value, "record_path": str(plain(path)), "record_sha256": digest(path),
            "evidence": {key: {"path": str(plain(p)), "sha256": digest(p)} for key, p in value["evidence_files"].items()}}


def prepare(args):
    binding, _, protected = native_inputs()
    observed = inspection(args.inspection)
    root, owner = claim(args.output_dir, [*protected, *args.protected_root,
        args.inspection, *(v["path"] for v in observed["evidence"].values())], "remaining27_plan")
    for name in ("allocation-logs", "attempts"):
        (root / name).mkdir()
    fsync(root)
    plan = sealed({"version": VERSION, "root": str(root), "owner": owner,
        "operator_pin": operator_pin(), "native": binding, "inspection": observed,
        "extra_protected_roots": [str(plain(p)) for p in args.protected_root],
        "indices": list(INDICES), "operation": "auto", "automatic_requeue": False,
        "resources": {"cpus": 8, "memory_gb": 32, "wall_time": "12:00:00"},
        "retry_policy": "native auto; no numerical/execution retry override; no automatic resubmission"})
    publish(root / "plan.json", plan)
    return plan


def load_plan(path, *, require_pilots=True):
    path = plain(path)
    plan = check(read(path))
    same(plan["version"], VERSION, "operator version")
    same(str(path), str(Path(plan["root"]) / "plan.json"), "owned plan path")
    same(check(read(Path(plan["root"]) / "OWNER.json")), plan["owner"], "namespace owner")
    same([plan["owner"]["version"], plan["owner"]["kind"], plan["owner"]["root"]],
         [VERSION, "remaining27_plan", plan["root"]], "remaining-27 owner scope")
    same(plan["operator_pin"], operator_pin(), "external launcher/batch/report bytes")
    same(plan["indices"], list(INDICES), "remaining-27 selection")
    same([plan["operation"], plan["automatic_requeue"], plan["resources"]],
         ["auto", False, {"cpus": 8, "memory_gb": 32, "wall_time": "12:00:00"}], "operational policy")
    same(inspection(plan["inspection"]["record_path"]), plan["inspection"], "sealed inspection/evidence")
    actual, manifest, protected = native_inputs(require_pilots=require_pilots)
    same(plan["native"], actual, "frozen actual native/source/software/manifest/launch bindings")
    disjoint(plan["root"], [*protected, *plan["extra_protected_roots"]])
    return plan, manifest


def sbatch_argv(plan):
    r = plan["inspection"]["record"]
    require(not any(c in str(HOME) + plan["root"] for c in (",", "\n")), "Slurm export paths cannot contain commas/newlines")
    argv = ["sbatch", "--parsable", "--partition=dgxh", f"--account={r['account']}",
            f"--nodelist={r['node']}", f"--gres={r['gres']}", "--nodes=1", "--ntasks=1", "--cpus-per-task=8",
            "--mem=32G", "--time=12:00:00", f"--array=3-29%{r['concurrency']}", "--signal=B:USR1@300", "--no-requeue",
            f"--output={plan['root']}/allocation-logs/%A_%a.out", f"--error={plan['root']}/allocation-logs/%A_%a.err",
            f"--export=ALL,OPERATOR_HOME={HOME},REMAINING_PLAN={plan['root']}/plan.json"]
    if r["qos"] is not None:
        argv.append(f"--qos={r['qos']}")
    return [*argv, str(HOME / "batch.sh")]


def submit(args):
    require(args.execute and args.confirmed_current_inspection, "Submission requires explicit execution and current operator inspection confirmation")
    plan, _ = load_plan(args.plan)
    fresh = inspection(args.inspection)
    same(fresh["record"], plan["inspection"]["record"], "fresh approved resource/concurrency inspection; refreeze if it changed")
    same(fresh, plan["inspection"], "inspection evidence bytes")
    argv = sbatch_argv(plan)
    root = Path(plan["root"])
    intent = sealed({"version": VERSION, "plan_sha256": plan["document_sha256"], "argv": argv,
        "inspection": fresh, "recorded_at_utc": datetime.now(timezone.utc).isoformat()})
    publish(root / "submission-intent.json", intent)  # Exists => no blind retry.
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError):
        raise ValueError("Submission acknowledgement ambiguous; preserve intent and inspect scheduler/accounting before any new plan")
    publish(root / "submission-response.json", sealed({"intent_sha256": intent["document_sha256"],
        "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}))
    require(result.returncode == 0 and re.fullmatch(r"[1-9]\d*(?:;[A-Za-z0-9_.-]+)?\s*", result.stdout),
            "Submission not acknowledged unambiguously; do not repeat or infer that capacity is free")
    return acknowledge(plan, result.stdout.strip().split(";")[0], "sbatch --parsable response")


def acknowledge(plan, job_id, reason):
    require(re.fullmatch(r"[1-9]\d*", job_id or "") is not None and bool(reason.strip()), "Exact actual array ID and evidence reason required")
    root = Path(plan["root"])
    intent = check(read(root / "submission-intent.json"))
    same(intent["plan_sha256"], plan["document_sha256"], "submission intent/plan")
    if (root / "actual-array.json").exists():
        observed = check(read(root / "actual-array.json"))
        same([observed["array_job_id"], observed["plan_sha256"]], [job_id, plan["document_sha256"]], "observed worker array ID")
    record = sealed({"plan_sha256": plan["document_sha256"], "intent_sha256": intent["document_sha256"],
                     "array_job_id": job_id, "reason": reason})
    publish(root / "submission-job.json", record)
    return record


def allocation(plan, index):
    mapping(index)
    r = plan["inspection"]["record"]
    for key, value in {"SLURM_ARRAY_TASK_ID": str(index), "SLURM_JOB_NUM_NODES": "1", "SLURM_NTASKS": "1",
            "SLURM_CPUS_PER_TASK": "8", "SLURM_GPUS_ON_NODE": "1", "SLURM_JOB_PARTITION": "dgxh",
            "SLURM_JOB_ACCOUNT": r["account"], "SLURMD_NODENAME": r["node"]}.items():
        same(os.environ.get(key), value, f"actual allocation {key}")
    if r["qos"] is not None:
        same(os.environ.get("SLURM_JOB_QOS"), r["qos"], "actual QOS")
    require(os.environ.get("SLURM_RESTART_COUNT", "0") == "0", "No automatic Slurm requeue")
    job = os.environ.get("SLURM_ARRAY_JOB_ID", "")
    require(re.fullmatch(r"[1-9]\d*", job) is not None and re.fullmatch(r"[1-9]\d*", os.environ.get("SLURM_JOB_ID", "")) is not None,
            "Actual array/job identity required")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    require(bool(visible) and "," not in visible, "One visible assigned GPU required")
    same(os.environ.get("CUBLAS_WORKSPACE_CONFIG"), ":4096:8", "CUBLAS policy")
    require(all(os.environ.get(k) is None for k in THREAD_ENV), "Preserve unset thread environment")
    root = Path(plan["root"])
    intent = check(read(root / "submission-intent.json"))
    same([intent["plan_sha256"], intent["argv"]], [plan["document_sha256"], sbatch_argv(plan)], "exact submission intent")
    if (root / "submission-job.json").exists():
        ack = check(read(root / "submission-job.json"))
        same([ack["array_job_id"], ack["plan_sha256"]], [job, plan["document_sha256"]], "acknowledged array")
    record = sealed({"plan_sha256": plan["document_sha256"], "array_job_id": job})
    try:
        publish(root / "actual-array.json", record)
    except FileExistsError:
        same(check(read(root / "actual-array.json")), record, "single observed array identity")
    publish(root / "attempts" / f"{index}.claim.json", sealed({"plan_sha256": plan["document_sha256"],
        "global_index": index, "array_job_id": job, "job_id": os.environ["SLURM_JOB_ID"], "pid": os.getpid(),
        "actual_allocation": {key: os.environ.get(key) for key in ("SLURM_JOB_ACCOUNT", "SLURM_JOB_QOS",
            "SLURM_JOB_PARTITION", "SLURMD_NODENAME", "SLURM_CPUS_PER_TASK", "SLURM_GPUS_ON_NODE",
            "SLURM_MEM_PER_NODE", "CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG")}}))


def run(args):
    mapping(args.global_index)  # Reject pilot/out-of-range before imports/signals.
    from alpha_sweep_training.allocation_signals import allocation_signal_handlers
    from alpha_sweep_training.interruption import StopRequest
    stop = StopRequest()
    with allocation_signal_handlers(stop):
        plan, manifest = load_plan(args.plan)
        allocation(plan, args.global_index)
        stop.check(at_boundary=True, phase="remaining27_before_gpu")
        import torch
        torch.set_num_threads(8)
        torch.set_num_interop_threads(112)
        require(torch.cuda.device_count() == 1, "Require exactly one assigned device")
        p = torch.cuda.get_device_properties(0)
        same([p.name, p.major, p.minor, p.total_memory], ["NVIDIA H100 80GB HBM3", 9, 0, 85009367040], "audited full H100, not MIG")
        stop.check(at_boundary=True, phase="remaining27_before_dispatch")
        from alpha_sweep_training.task_execution import execute_task
        outcome = execute_task(manifest, manifest.tasks[args.global_index], base_dir=plan["native"]["base_dir"],
                               source_root=NATIVE, operation="auto", stop=stop)
        result = outcome.to_dict()
        publish(Path(plan["root"]) / "attempts" / f"{args.global_index}.outcome.json", sealed({"result": result,
                "plan_sha256": plan["document_sha256"], "stop": {"policy": stop.policy, "reason": stop.reason}}))
        return result, int(outcome.exit_code)


def status(plan, manifest):
    from alpha_sweep_training.coverage import collect_coverage
    return collect_coverage(manifest, base_dir=plan["native"]["base_dir"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--inspection", required=True)
    p.add_argument("--protected-root", action="append", required=True)
    for name in ("inspect", "status", "submit", "record-job", "run"):
        p = commands.add_parser(name)
        p.add_argument("--plan", required=True)
        if name == "submit":
            p.add_argument("--inspection", required=True)
            p.add_argument("--execute", action="store_true")
            p.add_argument("--confirmed-current-inspection", action="store_true")
        if name == "record-job":
            p.add_argument("--job-id", required=True)
            p.add_argument("--reason", required=True)
        if name == "run":
            p.add_argument("--global-index", type=int, choices=INDICES, required=True)
    args = parser.parse_args(argv)
    try:
        code = 0
        if args.command == "prepare":
            result = prepare(args)
        elif args.command == "run":
            result, code = run(args)
        elif args.command == "submit":
            result = submit(args)
        else:
            plan, manifest = load_plan(args.plan, require_pilots=args.command != "status")
            if args.command == "record-job":
                result = acknowledge(plan, args.job_id, args.reason)
            elif args.command == "status":
                result = status(plan, manifest)
            else:
                result = {"plan": plan, "sbatch_argv": sbatch_argv(plan), "shell_preview": shlex.join(sbatch_argv(plan))}
        print(canonical(result))
        return code
    except Exception as exc:
        from alpha_sweep_training.interruption import TrainingInterrupted
        code = 10 if isinstance(exc, TrainingInterrupted) else 22
        print(canonical({"exit_code": code, "error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return code


if __name__ == "__main__":
    raise SystemExit(main())
