"""External all-30 operator. Native research code and historical operators are read-only.

No command in this module has been run during source preparation. Linux/shared
storage and the frozen research interpreter are required for operational use.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import uuid

VERSION = "mobile-all30-evaluation-operator-v1"
COMMIT = "8010d07d9d1f0886ab1e180523f316623ef622a4"
SOURCE_SHA = "a5726bac3f449455d36e2f53840f3e0fd77768a7ab2b989d2884da20f920005c"
GRID = "2c0689e4aa3d0b6734323642a89dd7de809337caabcbe358ab600b7287a16167"
MANIFEST_SHA = "08555fb94e023849d37c0140833a21e0db12b02ce4ab382bc32d4202b76f6423"
PROFILE = "cifar10-mobile-eps0316-v1"
BASE = Path("/nfs/hpc/share/bellaak/research/alpha-sweep")
NATIVE = BASE / "alpha-sweep-mobile-eps0316"
STUDY = BASE / "mobile-eps0316-v1"
PYTHON = BASE / "alpha-sweep-training/.venv/bin/python"
EVALUATIONS = BASE / "mobile-eps0316-evaluations-v1"
OPS = BASE / "mobile-eps0316-evaluation-ops-v1"
OLD_OPERATOR = BASE / "mobile-remaining27-launch-v1"
OLD_PLAN = BASE / "mobile-eps0316-remaining27-ops-v1"
EVIDENCE = BASE / "mobile-all30-eval-evidence-v1"
ALPHAS = (0., .01, .02, .05, .1, .2, .3, .5, .75, 1.)
SOFTWARE = {"python": "3.11.15", "torch": "2.5.1+cu124", "torchvision": "0.20.1+cu124",
            "numpy": "1.26.4", "PyYAML": "6.0.3", "cuda": "12.4", "cudnn": "90100"}
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
BUNDLE_FILES = ("launcher.py", "report.py", "figures.py", "batch.sh", "HPC_HANDOFF.md", "REVIEW.md")
HOME = Path(__file__).absolute().parent


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def same(actual, expected, label):
    require(canonical(actual) == canonical(expected), f"Incompatible {label}")


def seal(value):
    require("document_sha256" not in value, "Cannot reseal an existing document")
    return {**value, "document_sha256": hashlib.sha256(canonical(value).encode()).hexdigest()}


def check(value):
    same(value["document_sha256"], seal({k: v for k, v in value.items()
                                       if k != "document_sha256"})["document_sha256"], "document seal")
    return value


def plain(value):
    path = Path(value).expanduser().absolute()
    require(not any(p.is_symlink() or getattr(p, "is_junction", lambda: False)()
                    for p in (path, *path.parents)), f"Linked path: {path}")
    require(".." not in path.parts, "Parent traversal is not an operational path")
    return path.resolve()


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    value = json.loads(plain(path).read_bytes().decode("utf-8"), object_pairs_hook=pairs)
    canonical(value)  # Reject nonfinite values; Python retains arbitrary-size integers.
    return value


def digest(path):
    value = hashlib.sha256()
    with plain(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def fsync(path):
    require(os.name == "posix", "Operational publication requires POSIX directory fsync")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish(path, value):
    """Atomic exclusive publication. Existing and ambiguous evidence survives."""
    path = plain(path)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
        fsync(path.parent)
    except BaseException:
        # Preserve the complete staging file on unknown publication outcome.
        raise
    else:
        temporary.unlink()
        fsync(path.parent)


def disjoint(root, protected):
    root = plain(root)
    for item in protected:
        other = plain(item)
        require(root != other and root not in other.parents and other not in root.parents,
                f"Namespace overlap: {root}, {other}")
    return root


def claim(root, protected, kind):
    root = disjoint(root, protected)
    require(root.parent.is_dir(), "Choose an existing persistent parent")
    root.mkdir()  # Existing incomplete/completed namespaces are never adopted.
    fsync(root.parent)
    owner = seal({"version": VERSION, "kind": kind, "root": str(root), "token": uuid.uuid4().hex})
    publish(root / "OWNER.json", owner)
    return root, owner


@contextmanager
def controller_lock(root):
    path = plain(root) / ".controller.lock"
    owner = seal({"token": uuid.uuid4().hex, "host": socket.gethostname(), "pid": os.getpid()})
    publish(path, owner)
    try:
        yield
    finally:
        same(read(path), owner, "controller lock ownership")
        path.unlink()
        fsync(path.parent)


def pin():
    # Every external import lives here. Never import the old launcher's helpers.
    owner = check(read(HOME / "BUNDLE_OWNER.json"))
    same([owner["version"], owner["kind"], owner["root"]], [VERSION, "external_evaluation_bundle", str(plain(HOME))],
         "exclusive external bundle owner")
    same(sorted(p.name for p in plain(HOME).iterdir()), sorted([*BUNDLE_FILES, "BUNDLE_OWNER.json"]),
         "exact external bundle inventory")
    return {"home": str(plain(HOME)), "owner": owner, "files": {name: digest(HOME / name) for name in BUNDLE_FILES}}


def own_bundle():
    same(sorted(p.name for p in plain(HOME).iterdir()), sorted(BUNDLE_FILES), "new unowned six-file bundle")
    disjoint(HOME, [NATIVE, STUDY, BASE / "main-v1", OLD_OPERATOR, OLD_PLAN, OPS, EVALUATIONS, EVIDENCE])
    owner = seal({"version": VERSION, "kind": "external_evaluation_bundle", "root": str(plain(HOME)),
                  "token": uuid.uuid4().hex})
    publish(HOME / "BUNDLE_OWNER.json", owner)
    return pin()


def mapping(index):
    require(type(index) is int and 0 <= index < 30, "Require an integer global index 0–29")
    return ALPHAS[index // 3], index % 3


def selection(manifest):
    same([len(manifest.tasks), list(manifest.seeds)], [30, [0, 1, 2]], "all-30 grid")
    result = []
    for index, task in enumerate(manifest.tasks):
        alpha, seed = mapping(index)
        s = task.config.scientific
        same([task.global_index, s["dataset"], s["architecture"], s["alpha"], s["training_seed"],
              s["specification_version"], s["optimizer"]["epsilon"]],
             [index, "cifar10", "mobilenetv3_large", alpha, seed, PROFILE, .0316], "task membership")
        result.append({"global_index": index, "run_id": task.run_id, "scientific_fingerprint": task.fingerprint,
                       "training_alpha": alpha, "training_seed": seed})
    return result


def native_inputs():
    """Read actual CPU checkpoint authority; no model construction or CUDA probe."""
    from alpha_sweep_training.mobile_candidate import load_pilot, _launch_inputs
    from alpha_sweep_training.execution_provenance import collect_provenance
    from alpha_sweep_training.evaluation_inputs import bind_evaluation_checkpoint
    from alpha_sweep_training.coverage import inspect_task
    from alpha_sweep_training.evaluation_protocol import prescribed_slots
    pilot, manifest = load_pilot(STUDY / "pilot.json")
    same([manifest.grid_id, manifest.document_sha256], [GRID, MANIFEST_SHA], "manifest/grid pins")
    same([pilot["source_root"], pilot["candidate_root"]], [str(NATIVE), str(STUDY)], "native paths")
    same(str(Path(sys.executable).absolute()), str(PYTHON), "configured existing interpreter")
    actual = collect_provenance(NATIVE, attempt={"lifecycle": VERSION})
    same([actual["code"]["commit"], actual["code"]["source_sha256"], actual["code"]["dirty"]],
         [COMMIT, SOURCE_SHA, False], "clean frozen native source")
    same(actual["code"], pilot["source_pin"], "full original source inventory")
    same(actual["software"], SOFTWARE, "actual software")
    same(pilot["software_pin"], SOFTWARE, "pilot software")
    launch = _launch_inputs(pilot["launch_inputs"]["activation_script"])
    same(launch, pilot["launch_inputs"], "interpreter/activation/modules/declaration bytes and paths")
    rows, bindings = selection(manifest), []
    expected_slots = [(a, r) for a in ALPHAS for r in ((0,) if a == 0 else (1, 2, 3))]
    same([(s.evaluation_alpha, s.evaluation_replicate) for s in prescribed_slots()], expected_slots,
         "all 28 native slots")
    protected = [NATIVE, STUDY, HOME, OLD_OPERATOR, OLD_PLAN, EVIDENCE, *pilot["protected_roots"]]
    for task in manifest.tasks:
        observation = inspect_task(task, base_dir=pilot["base_dir"])
        require(observation.condition == "ok" and observation.training_complete
                and observation.completed_epochs == 200, f"Actual training incomplete/unstable: {task.run_id}")
        inputs = bind_evaluation_checkpoint(task, base_dir=pilot["base_dir"])
        same(inputs.training_transaction, observation.transaction, "training observation/binding boundary")
        same(inputs.binding["training_provenance"]["code"], pilot["source_pin"], "training source")
        same(inputs.binding["training_provenance"]["software"], SOFTWARE, "training software")
        same([inputs.protocol.to_dict()["protocol_version"], inputs.protocol.to_dict()["randomness_version"]],
             ["weight_robustness_eval_v1", "sha256-best-checkpoint-realization-v1"], "native protocol")
        bindings.append(inputs.binding)
        protected.extend(inputs.paths.directories())
        storage = inputs.config.execution["storage"]
        protected.extend(p for k, p in storage.items() if p and k != "split_file")
        if storage["split_file"]:
            protected.append(Path(storage["split_file"]).parent)
    native = {"base_dir": str(plain(pilot["base_dir"])), "source_root": str(plain(NATIVE)),
              "source": actual["code"], "software": actual["software"], "launch_inputs": launch,
              "manifest_path": str(STUDY / "study.json"), "manifest_file_sha256": digest(STUDY / "study.json"),
              "manifest_document_sha256": MANIFEST_SHA, "grid_id": GRID,
              "pilot_file_sha256": digest(STUDY / "pilot.json"), "selection": rows, "bindings": bindings,
              "thread_policy": pilot["thread_policy"]}
    return native, manifest, protected


def compare_export(native, path):
    rows = read(path)
    require(type(rows) is list and len(rows) == 30, "Require all 30 exported bindings")
    mismatches = []
    for i, (expected, binding) in enumerate(zip(rows, native["bindings"])):
        row = native["selection"][i]
        for key in row:
            if canonical(expected.get(key)) != canonical(row[key]):
                mismatches.append({"global_index": i, "field": key})
        if canonical(expected.get("evaluation_binding")) != canonical(binding):
            mismatches.append({"global_index": i, "field": "full_evaluation_binding"})
        if expected.get("selected_epoch") != binding["best_validation"]["epoch_index"] + 1:
            mismatches.append({"global_index": i, "field": "selected_epoch"})
        if expected.get("best_checkpoint_identity") != binding["best_checkpoint_identity"]:
            mismatches.append({"global_index": i, "field": "best_checkpoint_identity"})
    return mismatches


def command(argv):
    return subprocess.run(argv, check=True, capture_output=True, text=True, timeout=45).stdout


def queue(account):
    """Expand pending array indices; conservatively count every account task."""
    output = command(["squeue", "--array", "--noheader", f"--account={account}",
                      "--format=%i|%F|%K|%T|%u|%a|%j"])
    rows = []
    for line in output.splitlines():
        fields = line.strip().split("|")
        require(len(fields) == 7 and all(fields), "Unparseable account queue; capacity is unknown")
        rows.append(dict(zip(("job", "array", "index", "state", "user", "account", "name"), fields)))
    return rows


def inspection(path, *, fresh):
    value = read(path)
    keys = {"inspected_at_utc", "operator", "reason", "account", "qos", "partition", "node", "gres",
            "concurrency", "other_project_reserved_tasks", "all_prior_submissions_resolved",
            "exclusive_project_window", "wall_time", "evidence_files"}
    require(type(value) is dict and set(value) == keys, f"Inspection needs exactly {sorted(keys)}")
    for key in ("operator", "reason"):
        require(type(value[key]) is str and value[key].strip(), f"Explicit {key} required")
    for key in ("account", "qos", "partition", "node"):
        require(type(value[key]) is str and re.fullmatch(r"[A-Za-z0-9_.-]+", value[key]), f"Actual {key} required")
    same(value["partition"], "dgxh", "target partition")
    require(re.fullmatch(r"gpu(?::[A-Za-z0-9_.-]+)?:1", value["gres"]) is not None, "One GPU GRES required")
    timestamp = datetime.fromisoformat(value["inspected_at_utc"].replace("Z", "+00:00"))
    require(timestamp.utcoffset() is not None and timestamp.utcoffset().total_seconds() == 0, "UTC inspection required")
    age = (datetime.now(timezone.utc) - timestamp).total_seconds()
    require(age >= 0 and (not fresh or age <= 900), "Submission inspection must be within 15 minutes")
    c, other = value["concurrency"], value["other_project_reserved_tasks"]
    require(type(c) is int and 1 <= c <= 4 and type(other) is int and 0 <= other and c + other <= 4,
            "All project reservations combined must be at most four")
    require(value["all_prior_submissions_resolved"] is True and value["exclusive_project_window"] is True,
            "Resolve ambiguous work and reserve project window against other stage launches")
    require(re.fullmatch(r"\d{2}:\d{2}:\d{2}", value["wall_time"]) is not None, "Explicit HH:MM:SS wall time")
    h, m, s = map(int, value["wall_time"].split(":"))
    require(m < 60 and s < 60 and h * 3600 + m * 60 + s > 300, "Time must exceed the 300-second warning")
    evidence = value["evidence_files"]
    require(type(evidence) is dict and set(evidence) == {"accounts", "nodes", "active_work", "storage", "walltime"},
            "Attach current account/QOS, node, whole-project active work, storage and wall-time evidence")
    return {"record": value, "path": str(plain(path)), "sha256": digest(path),
            "evidence": {k: {"path": str(plain(p)), "sha256": digest(p)} for k, p in evidence.items()}}


def prepare(args):
    native, _, protected = native_inputs()
    mismatches = compare_export(native, args.export_bindings)
    require(not mismatches, "Export/HPC mismatch; resolve before freezing: " + canonical(mismatches))
    observed = inspection(args.inspection, fresh=True)
    external_pin = pin()
    protected.extend([args.export_bindings, args.inspection, *(x["path"] for x in observed["evidence"].values())])
    disjoint(HOME, [p for p in protected if plain(p) != plain(HOME)])
    disjoint(EVALUATIONS, protected)
    root, owner = claim(OPS, [*protected, EVALUATIONS], "all30_evaluation_plan")
    (root / "waves").mkdir()
    fsync(root)
    plan = seal({"version": VERSION, "root": str(root), "owner": owner, "operator_pin": external_pin,
                 "native": native, "evaluation_root": str(EVALUATIONS), "indices": list(range(30)),
                 "protected_roots": sorted({str(plain(p)) for p in protected}),
                 "export": {"path": str(plain(args.export_bindings)), "sha256": digest(args.export_bindings)},
                 "inspection": observed, "resources": {"gpus": 1, "cpus": 8, "memory_gb": 32},
                 "policy": {"max_project_concurrent": 4, "automatic_requeue": False,
                            "reuse_legacy_clean": False, "all_native_slots": True}})
    publish(root / "plan.json", plan)
    return plan


def load_plan(path, *, bind=True):
    path = plain(path)
    same(str(path), str(OPS / "plan.json"), "dedicated fixed operational plan path")
    plan = check(read(path))
    same([plan["version"], plan["root"], plan["indices"], plan["evaluation_root"]],
         [VERSION, str(OPS), list(range(30)), str(EVALUATIONS)], "all-30 operational scope")
    same(check(read(OPS / "OWNER.json")), plan["owner"], "exclusive operational ownership")
    same([plan["owner"]["version"], plan["owner"]["kind"], plan["owner"]["root"]],
         [VERSION, "all30_evaluation_plan", str(OPS)], "owner scope")
    same(plan["operator_pin"], pin(), "every external file/path")
    same(plan["resources"], {"gpus": 1, "cpus": 8, "memory_gb": 32}, "resource declaration")
    same(plan["policy"], {"max_project_concurrent": 4, "automatic_requeue": False,
                          "reuse_legacy_clean": False, "all_native_slots": True}, "evaluation policy")
    same(digest(plan["export"]["path"]), plan["export"]["sha256"], "export bytes")
    if not bind:
        # Reporting must retain unbound rows instead of stopping on one failed
        # checkpoint read. It still validates the native manifest and external pin.
        from alpha_sweep_training.manifests import load_manifest
        from alpha_sweep_training.execution_provenance import collect_provenance
        from alpha_sweep_training.mobile_candidate import _launch_inputs
        actual = collect_provenance(NATIVE, attempt={"lifecycle": "historical_evaluation_report"})
        same(actual["code"], plan["native"]["source"], "pinned historical reader source")
        same(actual["software"], SOFTWARE, "historical reader software")
        same(_launch_inputs(plan["native"]["launch_inputs"]["activation_script"]),
             plan["native"]["launch_inputs"], "historical reader launch inputs")
        manifest = load_manifest(STUDY / "study.json")
        same([manifest.grid_id, manifest.document_sha256, digest(STUDY / "study.json")],
             [GRID, MANIFEST_SHA, plan["native"]["manifest_file_sha256"]], "report manifest")
        selection(manifest)
        return plan, manifest
    native, manifest, protected = native_inputs()
    same(native, plan["native"], "actual full source/software/launch/manifest/checkpoint bindings")
    require(not compare_export(native, plan["export"]["path"]), "Export changed or actual bindings differ")
    disjoint(OPS, [*protected, EVALUATIONS])
    disjoint(EVALUATIONS, protected)
    return plan, manifest


def wave_path(name):
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,47}", name or "") is not None, "Plain wave name required")
    return plain(OPS / "waves" / name)


def indices(text):
    require(type(text) is str and re.fullmatch(r"\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*", text), "Indices: 0-29 or comma-separated ranges")
    result = []
    for part in text.split(","):
        ends = list(map(int, part.split("-")))
        for endpoint in ends:
            mapping(endpoint)
        require(len(ends) == 1 or ends[0] <= ends[1], "Reversed index range")
        result.extend([ends[0]] if len(ends) == 1 else range(ends[0], ends[1] + 1))
    require(result and len(set(result)) == len(result), "Unique nonempty indices required")
    for index in result:
        mapping(index)
    return sorted(result)


def argv_for(plan, wave, observed, selected):
    r = observed["record"]
    require(not any(c in str(HOME) + str(wave) for c in (",", "\n")), "Unsafe Slurm export path")
    return ["sbatch", "--parsable", f"--account={r['account']}", f"--qos={r['qos']}",
            f"--partition={r['partition']}", f"--nodelist={r['node']}", f"--gres={r['gres']}",
            "--nodes=1", "--ntasks=1", "--cpus-per-task=8", "--mem=32G", f"--time={r['wall_time']}",
            "--array=" + ",".join(map(str, selected)) + f"%{r['concurrency']}",
            "--signal=B:USR1@300", "--no-requeue", f"--job-name=mobile-eval-{wave.name}",
            f"--comment={VERSION}:{plan['document_sha256']}:{wave.name}",
            f"--output={wave}/logs/%A_%a.out", f"--error={wave}/logs/%A_%a.err",
            f"--export=ALL,EVAL_OPERATOR_HOME={HOME},EVAL_PLAN={OPS}/plan.json,EVAL_WAVE={wave.name}",
            str(HOME / "batch.sh")]


def continuation_gate(plan, manifest, selected, retry_failed, reason):
    from alpha_sweep_training.evaluation_inputs import bind_evaluation_checkpoint
    from alpha_sweep_training.evaluation_coverage import collect_evaluation_coverage
    for i in selected:
        inputs = bind_evaluation_checkpoint(manifest.tasks[i], base_dir=plan["native"]["base_dir"])
        same(inputs.binding, plan["native"]["bindings"][i], "pre-submission binding")
        coverage = collect_evaluation_coverage(inputs, evaluation_root=EVALUATIONS)
        require(not any(coverage["counts"][k] for k in ("locked", "partial", "incompatible", "unstable", "inspection_error")),
                f"Unsafe evaluation authority at index {i}")
        require(not coverage["completion_inspection_error"] and not coverage["completion_inspection_deferred"],
                f"Unresolved sweep authority at index {i}")
        if coverage["counts"]["failed"] or coverage["counts"]["unacknowledged_failures"]:
            require(retry_failed and reason.strip(), f"Failed slots require explicit audited retry: {i}")


def preview_or_submit(args):
    plan, manifest = load_plan(args.plan)
    selected = indices(args.indices)
    observed = inspection(args.inspection, fresh=True)
    wave = wave_path(args.wave)
    continuation_gate(plan, manifest, selected, args.retry_failed, args.reason)
    argv = argv_for(plan, wave, observed, selected)
    if not args.execute:
        return {"submission_performed": False, "argv": argv, "shell_preview": shlex.join(argv),
                "retry_failed": args.retry_failed, "reason": args.reason,
                "note": "Submission repeats fresh pins, authority, queue and exclusive ownership checks"}
    require(args.confirmed_project_window and args.reason.strip(), "Explicit project reservation confirmation and audit reason required")
    with controller_lock(OPS):
        # At most one unresolved wave; even a lost sbatch acknowledgment reserves capacity.
        for previous in plain(OPS / "waves").iterdir():
            previous = plain(previous)
            require(previous.is_dir(), "Unknown wave evidence; inspect manually")
            require((previous / "settled.json").is_file(), "Previous wave unresolved: " + str(previous))
            validate_settlement(previous, plan)
        require(not wave.exists(), "Never reuse a wave, including incomplete submissions")
        # Revalidate after waiting for the controller lock, before any intent write.
        load_plan(args.plan)
        continuation_gate(plan, manifest, selected, args.retry_failed, args.reason)
        same(inspection(args.inspection, fresh=True), observed, "unchanged fresh inspection/evidence")
        current = queue(observed["record"]["account"])
        require(len(current) <= observed["record"]["other_project_reserved_tasks"],
                "Account work exceeds reserved other-project tasks; refresh whole-project evidence")
        wave.mkdir()
        fsync(wave.parent)
        for name in ("logs", "claims", "outcomes"):
            (wave / name).mkdir()
        fsync(wave)
        intent = seal({"version": VERSION, "plan_sha256": plan["document_sha256"], "wave": args.wave,
                       "argv": argv, "inspection": observed, "indices": selected,
                       "retry_failed": args.retry_failed, "reason": args.reason,
                       "queue": current, "created_at_utc": datetime.now(timezone.utc).isoformat()})
        publish(wave / "intent.json", intent)
        try:
            response = subprocess.run(argv, capture_output=True, text=True, timeout=45, check=False)
        except (OSError, subprocess.SubprocessError) as exc:
            raise ValueError("Acknowledgment unknown; preserve wave and reconcile Slurm before any submission") from exc
        publish(wave / "response.json", seal({"intent_sha256": intent["document_sha256"],
                "returncode": response.returncode, "stdout": response.stdout, "stderr": response.stderr}))
        require(response.returncode == 0 and re.fullmatch(r"[1-9]\d*(?:;[A-Za-z0-9_.-]+)?\s*", response.stdout),
                "Submission unacknowledged; do not repeat it or free reserved capacity")
        return acknowledge(plan, wave, response.stdout.strip().split(";")[0], "sbatch --parsable response", None)


def load_intent(plan, wave):
    intent = check(read(wave / "intent.json"))
    same([intent["version"], intent["plan_sha256"], intent["wave"]],
         [VERSION, plan["document_sha256"], wave.name], "intent identity")
    same(indices(",".join(map(str, intent["indices"]))), intent["indices"], "intent mapping")
    require(type(intent["retry_failed"]) is bool and intent["reason"].strip(), "Audited retry intent required")
    same(intent["argv"], argv_for(plan, wave, intent["inspection"], intent["indices"]), "exact sealed sbatch intent")
    return intent


def acknowledge(plan, wave, job, reason, evidence):
    require(re.fullmatch(r"[1-9]\d*", job or "") is not None and reason.strip(), "Exact array ID and reason required")
    intent = load_intent(plan, wave)
    reconciliation = None
    if evidence is not None:
        # Lost acknowledgment: obtain independent scheduler identity rather than
        # trusting an arbitrary job number or treating no queue rows as absence.
        reconciliation = command(["sacct", "-X", "--noheader", "--parsable2", "--jobs", job,
                                   "--format=JobID%80,JobName%80,Account%80,Partition%80,Comment%200"])
        expected_comment = f"{VERSION}:{plan['document_sha256']}:{wave.name}"
        found = set()
        for line in reconciliation.splitlines():
            fields = line.split("|")
            if len(fields) < 5:
                continue
            match = re.fullmatch(re.escape(job) + r"_(\d+)", fields[0])
            if match:
                found.add(int(match[1]))
                same(fields[1:5], [f"mobile-eval-{wave.name}", intent["inspection"]["record"]["account"],
                                   intent["inspection"]["record"]["partition"], expected_comment],
                     "independent Slurm submission identity")
        same(sorted(found), intent["indices"], "reconciled exact array mapping (wait for complete accounting)")
    if (wave / "observed-array.json").exists():
        same(read(wave / "observed-array.json")["array_job_id"], job, "observed array ID")
    record = seal({"intent_sha256": intent["document_sha256"], "array_job_id": job, "reason": reason,
                   "scheduler_reconciliation": reconciliation,
                   "evidence": None if evidence is None else {"path": str(plain(evidence)), "sha256": digest(evidence)}})
    publish(wave / "ack.json", record)
    return record


def allocation(plan, wave, index):
    mapping(index)
    intent = load_intent(plan, wave)
    require(index in intent["indices"], "Task absent from sealed submission")
    r = intent["inspection"]["record"]
    for key, value in {"SLURM_ARRAY_TASK_ID": str(index), "SLURM_JOB_NUM_NODES": "1", "SLURM_NTASKS": "1",
                       "SLURM_CPUS_PER_TASK": "8", "SLURM_GPUS_ON_NODE": "1", "SLURM_JOB_PARTITION": r["partition"],
                       "SLURM_JOB_ACCOUNT": r["account"], "SLURM_JOB_QOS": r["qos"], "SLURMD_NODENAME": r["node"],
                       "SLURM_MEM_PER_NODE": "32768", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}.items():
        same(os.environ.get(key), value, "actual allocation " + key)
    same(os.environ.get("SLURM_RESTART_COUNT", "0"), "0", "no Slurm requeue")
    job = os.environ.get("SLURM_ARRAY_JOB_ID", "")
    require(re.fullmatch(r"[1-9]\d*", job) is not None
            and re.fullmatch(r"[1-9]\d*", os.environ.get("SLURM_JOB_ID", "")), "Actual array and allocation IDs required")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    require(visible and "," not in visible and not visible.startswith("MIG-"), "One full assigned GPU required")
    require(all(os.environ.get(k) is None for k in THREAD_ENV), "Preserve recorded unset thread environment")
    if (wave / "ack.json").exists():
        ack = check(read(wave / "ack.json"))
        same([ack["array_job_id"], ack["intent_sha256"]], [job, intent["document_sha256"]], "acknowledged array")
    observed = seal({"intent_sha256": intent["document_sha256"], "array_job_id": job})
    try:
        publish(wave / "observed-array.json", observed)
    except FileExistsError:
        same(check(read(wave / "observed-array.json")), observed, "one array per intent")
    publish(wave / "claims" / f"{index}.json", seal({"global_index": index, "array_job_id": job,
            "job_id": os.environ["SLURM_JOB_ID"], "intent_sha256": intent["document_sha256"],
            "pid": os.getpid(), "host": socket.gethostname()}))
    return intent


def run(args):
    mapping(args.global_index)
    from alpha_sweep_training.allocation_signals import allocation_signal_handlers
    from alpha_sweep_training.interruption import StopRequest
    stop = StopRequest()
    with allocation_signal_handlers(stop):
        plan, manifest = load_plan(args.plan)
        wave = wave_path(args.wave)
        intent = allocation(plan, wave, args.global_index)
        stop.check(at_boundary=True, phase="external_evaluation_before_runtime")
        import torch
        policy = plan["native"]["thread_policy"]
        torch.set_num_threads(policy["torch_intraop"])
        torch.set_num_interop_threads(policy["torch_interop"])
        require(torch.cuda.device_count() == 1, "Exactly one visible device required")
        props = torch.cuda.get_device_properties(0)
        same([props.name, props.major, props.minor, props.total_memory],
             ["NVIDIA H100 80GB HBM3", 9, 0, 85009367040], "recorded full H100; reject MIG/different device")
        from alpha_sweep_training.evaluation_inputs import bind_evaluation_checkpoint
        from alpha_sweep_training.runtime import build_runtime
        task = manifest.tasks[args.global_index]
        inputs = bind_evaluation_checkpoint(task, base_dir=plan["native"]["base_dir"])
        same(inputs.binding, plan["native"]["bindings"][args.global_index], "allocation full binding")
        runtime = build_runtime(inputs.config, evaluation_only=True)
        expected = inputs.binding["training_provenance"]["allocation"]["runtime"]
        same(runtime.metadata(), expected, "recorded native effective runtime")
        stop.check(at_boundary=True, phase="external_evaluation_before_dispatch")
        from alpha_sweep_training.evaluation_execution import execute_evaluation
        outcome = execute_evaluation(manifest, task, base_dir=plan["native"]["base_dir"], source_root=NATIVE,
                evaluation_root=plan["evaluation_root"], evaluation_alphas=None, evaluation_replicates=None,
                retry_failed=intent["retry_failed"], reuse_legacy_clean=False, stop=stop,
                expected_binding=plan["native"]["bindings"][args.global_index])
        # Operational diagnostics only; native slot/sweep HEADs remain authority.
        try:
            publish(wave / "outcomes" / f"{args.global_index}.json", seal({"result": outcome.to_dict(),
                    "intent_sha256": intent["document_sha256"], "stop": {"policy": stop.policy, "reason": stop.reason}}))
        except Exception as exc:
            print(f"Outcome diagnostic publication failed: {exc}; inspect native coverage", file=sys.stderr)
            if outcome.exit_code == 0:
                return outcome.to_dict(), 22
        return outcome.to_dict(), int(outcome.exit_code)


TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}


def accounting(job, selected):
    output = command(["sacct", "-X", "--noheader", "--parsable2", "--jobs", job,
                      "--format=JobID%80,State%80,ExitCode"])
    rows = {}
    for line in output.splitlines():
        parts = line.split("|")
        if len(parts) < 3:
            continue
        match = re.fullmatch(re.escape(job) + r"_(\d+)", parts[0])
        if match:
            index = int(match[1])
            require(index not in rows, "Duplicate accounting task evidence")
            state = parts[1].split()[0].rstrip("+")
            rows[index] = {"state": state, "exit_code": parts[2]}
    same(sorted(rows), selected, "complete exact per-index accounting")
    require(all(row["state"] in TERMINAL for row in rows.values()), "Accounting is not terminal; keep reservations")
    return output, rows


def validate_settlement(wave, plan):
    intent = load_intent(plan, wave)
    ack = check(read(wave / "ack.json"))
    record = check(read(wave / "settled.json"))
    same([ack["intent_sha256"], record["intent_sha256"], record["array_job_id"]],
         [intent["document_sha256"], intent["document_sha256"], ack["array_job_id"]], "settled intent/array")
    same(record["inactive_across_nodes_confirmed"], True, "settlement inactivity evidence")
    same(sorted(map(int, record["rows"])), intent["indices"], "settled accounting mapping")
    require(all(r["state"] in TERMINAL for r in record["rows"].values()), "Unsettled accounting evidence")
    return record


def settle(args):
    plan, _ = load_plan(args.plan)
    wave = wave_path(args.wave)
    with controller_lock(OPS):
        intent = load_intent(plan, wave)
        ack = check(read(wave / "ack.json"))
        same(ack["intent_sha256"], intent["document_sha256"], "settlement acknowledgment")
        require(args.reason.strip() and args.confirmed_inactive_across_nodes,
                "Explicit all-owner/all-allocation inactivity confirmation and reason required")
        require(not any(row["array"] == ack["array_job_id"] for row in queue(intent["inspection"]["record"]["account"])),
                "Array remains queued or active")
        output, rows = accounting(ack["array_job_id"], intent["indices"])
        # Settlement releases scheduling reservations only, never scientific completion.
        record = seal({"intent_sha256": intent["document_sha256"], "array_job_id": ack["array_job_id"],
                       "reason": args.reason, "inactive_across_nodes_confirmed": True, "accounting": output,
                       "rows": {str(i): row for i, row in rows.items()}, "scientific_completion_authority": False,
                       "recorded_at_utc": datetime.now(timezone.utc).isoformat()})
        publish(wave / "settled.json", record)
        return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("own-bundle")
    p = commands.add_parser("inspect-inputs")
    p.add_argument("--export-bindings", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--export-bindings", required=True)
    p.add_argument("--inspection", required=True)
    for name in ("validate", "submit", "record-job", "settle", "run"):
        p = commands.add_parser(name)
        p.add_argument("--plan", required=True)
        if name != "validate":
            p.add_argument("--wave", required=True)
        if name == "submit":
            p.add_argument("--inspection", required=True)
            p.add_argument("--indices", default="0-29")
            p.add_argument("--retry-failed", action="store_true")
            p.add_argument("--reason", default="")
            p.add_argument("--execute", action="store_true")
            p.add_argument("--confirmed-project-window", action="store_true")
        elif name == "record-job":
            p.add_argument("--job-id", required=True)
            p.add_argument("--reason", required=True)
            p.add_argument("--evidence", required=True)
        elif name == "settle":
            p.add_argument("--reason", required=True)
            p.add_argument("--confirmed-inactive-across-nodes", action="store_true")
        elif name == "run":
            p.add_argument("--global-index", type=int, choices=range(30), required=True)
    args = parser.parse_args(argv)
    try:
        code = 0
        if args.command == "own-bundle":
            result = own_bundle()
        elif args.command == "inspect-inputs":
            native, _, _ = native_inputs()
            mismatches = compare_export(native, args.export_bindings)
            result = {"native": native, "export_mismatches": mismatches, "plan_frozen": False}
            code = 22 if mismatches else 0
        elif args.command == "prepare":
            result = prepare(args)
        elif args.command == "submit":
            result = preview_or_submit(args)
        elif args.command == "run":
            result, code = run(args)
        elif args.command == "settle":
            result = settle(args)
        else:
            plan, _ = load_plan(args.plan)
            if args.command == "record-job":
                with controller_lock(OPS):
                    result = acknowledge(plan, wave_path(args.wave), args.job_id, args.reason, args.evidence)
            else:
                result = {"validated": True, "plan_sha256": plan["document_sha256"], "submission_performed": False}
        print(canonical(result))
        return code
    except Exception as exc:
        # Do not import native code merely to format a pre-import rejection.
        code = 10 if isinstance(exc, InterruptedError) else 22
        print(canonical({"exit_code": code, "error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return code


if __name__ == "__main__":
    raise SystemExit(main())
