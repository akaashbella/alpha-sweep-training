"""External all-30 operator. Native research code and historical operators are read-only.

No command in this module has been run during source preparation. Linux/shared
storage and the frozen research interpreter are required for operational use.
"""

import argparse
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
import uuid

VERSION = "mobile-all30-evaluation-operator-v2"
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
OPS = BASE / "mobile-eps0316-evaluation-ops-v2"
V1_OPERATOR = BASE / "mobile-all30-eval-launch-v1"
V1_OPS = BASE / "mobile-eps0316-evaluation-ops-v1"
FAILED_INSPECTION = BASE / "mobile-all30-eval-inspection.ZUxVLZ"
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
RESOURCE_KEYS = ("account", "qos", "partition", "node", "gres", "wall_time")


def utc_now():
    return datetime.now(timezone.utc)


@contextmanager
def phase(label):
    started = time.monotonic()
    print(f"[mobile-eval] {label}: start", file=sys.stderr, flush=True)
    try:
        yield
    except BaseException:
        print(f"[mobile-eval] {label}: failed after {time.monotonic() - started:.1f}s", file=sys.stderr, flush=True)
        raise
    else:
        print(f"[mobile-eval] {label}: done in {time.monotonic() - started:.1f}s", file=sys.stderr, flush=True)


def legacy_quiescence():
    require(not plain(V1_OPS).exists(),
            "Historical v1 operations evidence exists; preserve it and reconcile cross-version reservations before new work")


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
    disjoint(HOME, [NATIVE, STUDY, BASE / "main-v1", OLD_OPERATOR, OLD_PLAN, OPS, EVALUATIONS, EVIDENCE,
                    V1_OPERATOR, V1_OPS, FAILED_INSPECTION])
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
    protected = [NATIVE, STUDY, HOME, OLD_OPERATOR, OLD_PLAN, EVIDENCE, V1_OPERATOR, V1_OPS,
                 FAILED_INSPECTION, *pilot["protected_roots"]]
    for task in manifest.tasks:
        checkpoint_started = time.monotonic()
        print(f"[mobile-eval] checkpoint {task.global_index + 1}/30: verifying {task.run_id}", file=sys.stderr, flush=True)
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
        print(f"[mobile-eval] checkpoint {task.global_index + 1}/30: verified in "
              f"{time.monotonic() - checkpoint_started:.1f}s", file=sys.stderr, flush=True)
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


def os_identity():
    """Resolve the executing POSIX identity from the kernel/account database."""
    require(hasattr(os, "getuid") and hasattr(os, "geteuid"), "POSIX OS identity is required")
    import pwd
    uid, effective = os.getuid(), os.geteuid()
    require(type(uid) is int and uid >= 0 and type(effective) is int and effective == uid,
            "Real/effective OS identities must match; do not launch through an identity-changing wrapper")
    entry = pwd.getpwuid(uid)
    require(type(entry.pw_name) is str and re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", entry.pw_name),
            "OS identity has an unsupported username")
    same(entry.pw_uid, uid, "OS UID/name lookup")
    reverse = pwd.getpwnam(entry.pw_name)
    same([reverse.pw_uid, reverse.pw_name], [uid, entry.pw_name], "OS username/UID round trip")
    return {"identity_version": 1, "uid": uid, "username": entry.pw_name}


def submitter_identity(value, *, verify_current=True):
    require(type(value) is dict and set(value) == {"identity_version", "uid", "username"},
            "Operational plan needs an OS submitter pin; preserve older plans without rewriting them")
    same(value["identity_version"], 1, "submitter identity version")
    require(type(value["uid"]) is int and value["uid"] >= 0 and type(value["username"]) is str
            and re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", value["username"]), "Invalid OS submitter pin")
    if verify_current:
        same(os_identity(), value, "current OS submitter identity")
    return value


def scheduler_user(username, uid, submitter):
    require(re.fullmatch(r"\d+", uid or "") is not None, "Unparseable scheduler UID")
    same([username, int(uid)], [submitter["username"], submitter["uid"]], "scheduler submitter identity")


def scheduler_environment(prefix):
    # Inherited SQUEUE_ACCOUNT/PARTITION/USERS, etc. must not narrow discovery.
    # Use the configured local cluster, without an inherited cluster selector.
    return {k: v for k, v in os.environ.items() if not k.startswith(prefix) and k != "SLURM_CLUSTERS"}


def command(argv, *, env=None):
    return subprocess.run(argv, check=True, capture_output=True, text=True, timeout=45, env=env).stdout


def queue(submitter):
    """The pinned user's expanded tasks across accounts on the local cluster."""
    submitter_identity(submitter)
    output = command(["squeue", "--local", "--all", "--array", "--noheader", "--states=all",
                      f"--user={submitter['uid']}", "--format=%i|%T|%u|%U|%a|%j"],
                     env=scheduler_environment("SQUEUE_"))
    rows, seen = [], set()
    for line in output.splitlines():
        fields = line.strip().split("|")
        require(len(fields) == 6 and all(fields), "Unparseable user queue; capacity is unknown")
        row = dict(zip(("job", "state", "user", "uid", "account", "name"), fields))
        scheduler_user(row["user"], row["uid"], submitter)
        match = re.fullmatch(r"([1-9]\d*)(?:_(\d+))?", row["job"])
        require(match is not None, "Queue must contain separate scalar jobs or expanded array elements")
        require(row["job"] not in seen, "Duplicate queue job; capacity is unknown")
        seen.add(row["job"])
        require(row["state"] in ACTIVE_STATES | TERMINAL, "Unknown queue state; capacity is unknown")
        if row["state"] in TERMINAL:
            continue  # Recently finished rows may remain visible in squeue.
        row.update(array=match[1] if match[2] is not None else None,
                   index=int(match[2]) if match[2] is not None else None, uid=int(row["uid"]))
        rows.append(row)
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
    age = (utc_now() - timestamp).total_seconds()
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
    result = {"record": value, "path": str(plain(path)), "sha256": digest(path),
              "evidence": {k: {"path": str(plain(p)), "sha256": digest(p)} for k, p in evidence.items()}}
    if fresh:
        root = plain(path).parent
        collection = check(read(root / "COLLECTION.json"))
        same(collection["version"], VERSION, "resource collection version")
        same(collection["started_at_utc"], value["inspected_at_utc"], "earliest resource collection time")
        same(collection["inspection_sha256"], result["sha256"], "collected inspection bytes")
        same(collection["evidence"], result["evidence"], "newly collected evidence bytes")
        same(check(read(root / "OWNER.json")), collection["owner"], "resource snapshot owner")
        same([collection["owner"]["version"], collection["owner"]["kind"], collection["owner"]["root"]],
             [VERSION, "resource_evidence", str(root)], "resource snapshot ownership scope")
        same({k: v["path"] for k, v in result["evidence"].items()},
             {k: str(root / (k + ".json")) for k in evidence}, "resource evidence belongs to its new snapshot")
        submitter_identity(collection["submitter"])
        start = datetime.fromisoformat(collection["started_at_utc"])
        end = datetime.fromisoformat(collection["finished_at_utc"])
        require(start <= end <= utc_now() and (end - start).total_seconds() <= 900,
                "Resource evidence collection itself exceeded freshness or has invalid timestamps")
        require(type(collection["elapsed_seconds"]) in (int, float) and 0 <= collection["elapsed_seconds"] <= 900,
                "Resource collection exceeded the 15-minute monotonic budget")
        receipts = read(result["evidence"]["accounts"]["path"])
        require(set(receipts) == {"associations", "qos", "partition", "node"}, "Incomplete collected resource receipts")
        for receipt in receipts.values():
            require(start <= datetime.fromisoformat(receipt["started_at_utc"])
                    <= datetime.fromisoformat(receipt["finished_at_utc"]) <= end,
                    "Resource receipt predates collection; never retimestamp old evidence")
        require(start <= datetime.fromisoformat(read(result["evidence"]["active_work"]["path"])["observed_at_utc"]) <= end,
                "Queue receipt predates collection")
        resource_eligibility(value, *(receipts[key]["stdout"] for key in ("associations", "qos", "partition", "node")))
        result["collection"] = {"path": str(root / "COLLECTION.json"), "sha256": digest(root / "COLLECTION.json")}
        require(0 <= (utc_now() - timestamp).total_seconds() <= 900,
                "Resource evidence expired while validating its bytes; collect again")
    return result


def seconds(value):
    if value in ("UNLIMITED", "NONE", "N/A", ""):
        return None
    match = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d+):(\d+)", value)
    require(match is not None, "Unparseable resource time limit: " + value)
    days, hours, minutes, secs = (int(v or 0) for v in match.groups())
    require(minutes < 60 and secs < 60, "Invalid resource time limit")
    return days * 86400 + hours * 3600 + minutes * 60 + secs


def control_fields(output):
    require(len(output.strip().splitlines()) == 1, "Expect exactly one scontrol resource record")
    pairs = re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=([^\s]*)", output.strip())
    require(len({key for key, _ in pairs}) == len(pairs), "Duplicate scontrol resource fields")
    return dict(pairs)


def partition_permits(fields, allow_key, deny_key, choice):
    """Decode explicit filter evidence: an allow list supersedes the deny list.

    scontrol may render only the active deny filter, with no Allow* field.
    Plain partition QoS is unrelated to these access filters. Account-name
    matching here is direct; account ancestry is not inferred from spelling.
    """
    def field(name):
        matches = [value for key, value in fields.items() if key.casefold() == name.casefold()]
        require(len(matches) <= 1, "Ambiguous partition filter field: " + name)
        if not matches:
            return False, set()
        value = matches[0]
        require(type(value) is str and bool(value), "Empty or invalid partition filter: " + name)
        names = value.split(",")
        require(all(re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", item) for item in names),
                "Malformed partition filter list: " + name)
        if any(item.casefold() == "all" for item in names):
            require(name == allow_key and len(names) == 1, "Unsupported ALL partition filter: " + name)
            return True, None
        return True, set(names)

    allow_present, allowed = field(allow_key)
    deny_present, denied = field(deny_key)
    require(allow_present or deny_present, "Missing partition access-filter evidence: " + allow_key + "/" + deny_key)
    if allow_present:
        return allowed is None or choice in allowed
    return choice not in denied


def resource_eligibility(request, accounts, qos, partition, node):
    """Validate chosen eligibility; never select replacements or change requests."""
    associations = [line.split("|") for line in accounts.splitlines()]
    require(associations and all(len(row) == 5 for row in associations), "Unparseable association evidence")
    require(any(row[0] == request["account"] and row[1] == os_identity()["username"]
                and row[2] in ("", request["partition"]) and request["qos"] in row[3].split(",")
                for row in associations), "Chosen account/QOS/partition is not currently eligible")
    qos_rows = [line.split("|") for line in qos.splitlines()]
    require(len(qos_rows) == 1 and len(qos_rows[0]) == 2 and qos_rows[0][0] == request["qos"],
            "Chosen QOS is absent or ambiguous")
    part, host = control_fields(partition), control_fields(node)
    same([part.get("PartitionName"), part.get("State"), host.get("NodeName")],
         [request["partition"], "UP", request["node"]], "current partition/node eligibility")
    require(request["partition"] in host.get("Partitions", "").split(","), "Chosen node left the partition")
    require(host.get("State", "").split("+")[0] in ("IDLE", "MIXED", "ALLOCATED")
            and not any(flag in host["State"] for flag in ("DRAIN", "FAIL", "DOWN", "MAINT")),
            "Chosen node is not currently eligible")
    for key, denied, choice in (("AllowAccounts", "DenyAccounts", request["account"]),
                                 ("AllowQos", "DenyQos", request["qos"])):
        require(partition_permits(part, key, denied, choice), "Partition no longer permits chosen " + choice)
    for limit in (qos_rows[0][1], part.get("MaxTime", "")):
        maximum = seconds(limit)
        require(maximum is None or seconds(request["wall_time"]) <= maximum, "Chosen wall time exceeds current limit")
    require(int(host.get("CPUTot", "0")) >= 8 and int(host.get("RealMemory", "0")) >= 32768,
            "Chosen node cannot satisfy the CPU/memory request")
    selected_type = request["gres"].split(":")[1] if request["gres"].count(":") == 2 else None
    gpu_counts = re.findall(r"(?:^|,)gpu(?::([A-Za-z0-9_.-]+))?:(\d+)(?:\([^)]*\))?", host.get("Gres", ""))
    require(sum(int(count) for kind, count in gpu_counts if selected_type is None or kind == selected_type) >= 1,
            "Chosen node cannot satisfy the declared GPU GRES")


def collect_resources(requested, submitter, protected, *, expected=None):
    """Collect new bytes after lengthy checks. Old timestamps are declarations only."""
    r = requested["record"]
    if expected is not None:
        same({k: r[k] for k in RESOURCE_KEYS}, {k: expected[k] for k in RESOURCE_KEYS}, "explicit frozen resource choices")
    submitter_identity(submitter)
    root, owner = claim(BASE / ("mobile-all30-eval-resources-v2-" + uuid.uuid4().hex),
                        [*protected, OPS, EVALUATIONS, V1_OPERATOR, V1_OPS, FAILED_INSPECTION], "resource_evidence")
    started = utc_now()
    started_monotonic = time.monotonic()
    receipts = {}
    def query(label, argv):
        with phase("fresh resource " + label):
            begin = utc_now()
            output = command(argv, env=scheduler_environment("SACCT_"))
            receipts[label] = {"argv": argv, "started_at_utc": begin.isoformat(),
                               "finished_at_utc": utc_now().isoformat(), "stdout": output}
            publish(root / (label + ".receipt.json"), receipts[label])
            return output
    with phase("collect current scheduler/resources"):
        accounts = query("associations", ["sacctmgr", "-nP", "show", "assoc", "where",
            "user=" + submitter["username"], "account=" + r["account"], "format=Account,User,Partition,QOS,DefaultQOS"])
        qos = query("qos", ["sacctmgr", "-nP", "show", "qos", "where", "name=" + r["qos"], "format=Name,MaxWall"])
        part = query("partition", ["scontrol", "--oneliner", "show", "partition", r["partition"]])
        host = query("node", ["scontrol", "--oneliner", "show", "node", r["node"]])
        resource_eligibility(r, accounts, qos, part, host)
        usage = shutil.disk_usage(BASE)
        require(usage.free > 0, "Shared storage has no available capacity; inspect quota/storage")
        with phase("fresh user queue and reservations"):
            current = queue(submitter)
        require(len(current) <= r["other_project_reserved_tasks"],
                "Submitting user's work exceeds reserved other-project tasks; refresh whole-project evidence")
        submitter_identity(submitter)
        data = {"accounts": receipts, "nodes": receipts["node"],
                "active_work": {"submitter": submitter, "queue": current, "observed_at_utc": utc_now().isoformat()},
                "storage": {"path": str(BASE), "total": usage.total, "used": usage.used, "free": usage.free},
                "walltime": {"declared_wall_time": r["wall_time"], "operator_reason": r["reason"],
                             "measurement_claim": False}}
        for key, value in data.items():
            publish(root / (key + ".json"), value)
        record = {**r, "inspected_at_utc": started.isoformat(),
                  "evidence_files": {key: str(root / (key + ".json")) for key in data}}
        publish(root / "inspection.json", record)
        observed = inspection(root / "inspection.json", fresh=False)
        publish(root / "COLLECTION.json", seal({"version": VERSION, "owner": owner, "submitter": submitter,
            "request": requested, "started_at_utc": started.isoformat(), "finished_at_utc": utc_now().isoformat(),
            "elapsed_seconds": time.monotonic() - started_monotonic,
            "inspection_sha256": observed["sha256"], "evidence": observed["evidence"]}))
        return inspection(root / "inspection.json", fresh=True)


def current_resources(observed):
    same(inspection(observed["path"], fresh=True), observed, "fresh collected resource evidence")


def static_native_inputs(native):
    """Revalidate current source/software/launch/document bytes, without tensor reads."""
    from alpha_sweep_training.manifests import load_manifest
    from alpha_sweep_training.execution_provenance import collect_provenance
    from alpha_sweep_training.mobile_candidate import _launch_inputs
    actual = collect_provenance(NATIVE, attempt={"lifecycle": "external_static_preflight"})
    same(actual["code"], native["source"], "pinned current reader source")
    same(actual["software"], SOFTWARE, "current reader software")
    same(_launch_inputs(native["launch_inputs"]["activation_script"]), native["launch_inputs"], "current launch inputs")
    manifest = load_manifest(STUDY / "study.json")
    same([manifest.grid_id, manifest.document_sha256, digest(STUDY / "study.json"), digest(STUDY / "pilot.json")],
         [GRID, MANIFEST_SHA, native["manifest_file_sha256"], native["pilot_file_sha256"]], "current manifest/pilot bytes")
    selection(manifest)
    return manifest


def prepare(args):
    submitter = os_identity()
    legacy_quiescence()
    requested = inspection(args.inspection, fresh=False)
    with phase("prepare full native/checkpoint verification"):
        native, _, protected = native_inputs()
        mismatches = compare_export(native, args.export_bindings)
        require(not mismatches, "Export/HPC mismatch; resolve before freezing: " + canonical(mismatches))
    external_pin = pin()
    protected.extend([args.export_bindings, args.inspection, *(x["path"] for x in requested["evidence"].values())])
    disjoint(HOME, [p for p in protected if plain(p) != plain(HOME)])
    disjoint(EVALUATIONS, protected)
    same(inspection(args.inspection, fresh=False), requested, "unchanged approved request")
    observed = collect_resources(requested, submitter, protected)
    protected.append(plain(observed["path"]).parent)
    static_native_inputs(native)
    same(pin(), external_pin, "external bytes after resource collection")
    submitter_identity(submitter)
    current_resources(observed)
    root, owner = claim(OPS, [*protected, EVALUATIONS], "all30_evaluation_plan")
    (root / "waves").mkdir()
    fsync(root)
    plan = seal({"version": VERSION, "root": str(root), "owner": owner, "operator_pin": external_pin,
                 "submitter": submitter,
                 "native": native, "evaluation_root": str(EVALUATIONS), "indices": list(range(30)),
                 "protected_roots": sorted({str(plain(p)) for p in protected}),
                 "export": {"path": str(plain(args.export_bindings)), "sha256": digest(args.export_bindings)},
                 "inspection": observed, "resources": {"gpus": 1, "cpus": 8, "memory_gb": 32},
                 "policy": {"max_project_concurrent": 4, "automatic_requeue": False,
                            "reuse_legacy_clean": False, "all_native_slots": True}})
    current_resources(observed)
    publish(root / "plan.json", plan)
    return plan


def load_plan(path, *, bind=True):
    path = plain(path)
    same(str(path), str(OPS / "plan.json"), "dedicated fixed operational plan path")
    plan = check(read(path))
    submitter_identity(plan.get("submitter"), verify_current=bind)
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
        return plan, static_native_inputs(plan["native"])
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
        started = time.monotonic()
        print(f"[mobile-eval] slot coverage checkpoint {i + 1}/30: start", file=sys.stderr, flush=True)
        inputs = bind_evaluation_checkpoint(manifest.tasks[i], base_dir=plan["native"]["base_dir"])
        same(inputs.binding, plan["native"]["bindings"][i], "pre-submission binding")
        coverage = collect_evaluation_coverage(inputs, evaluation_root=EVALUATIONS)
        require(not any(coverage["counts"][k] for k in ("locked", "partial", "incompatible", "unstable", "inspection_error")),
                f"Unsafe evaluation authority at index {i}")
        require(not coverage["completion_inspection_error"] and not coverage["completion_inspection_deferred"],
                f"Unresolved sweep authority at index {i}")
        if coverage["counts"]["failed"] or coverage["counts"]["unacknowledged_failures"]:
            require(retry_failed and reason.strip(), f"Failed slots require explicit audited retry: {i}")
        print(f"[mobile-eval] slot coverage checkpoint {i + 1}/30: done in {time.monotonic() - started:.1f}s",
              file=sys.stderr, flush=True)


def preview_or_submit(args):
    # This first read validates documents/source only, never checkpoints. Full
    # current binding verification occurs once, under the submission lock.
    initial, _ = load_plan(args.plan, bind=False)
    submitter_identity(initial["submitter"])
    legacy_quiescence()
    selected = indices(args.indices)
    requested = inspection(args.inspection, fresh=False)
    wave = wave_path(args.wave)
    if args.execute:
        require(args.confirmed_project_window and args.reason.strip(), "Explicit project reservation confirmation and audit reason required")
    with controller_lock(OPS) if args.execute else nullcontext():
        # At most one unresolved wave; even a lost sbatch acknowledgment reserves capacity.
        for previous in plain(OPS / "waves").iterdir():
            previous = plain(previous)
            require(previous.is_dir(), "Unknown wave evidence; inspect manually")
            require((previous / "settled.json").is_file(), "Previous wave unresolved: " + str(previous))
            validate_settlement(previous, initial)
        require(not wave.exists(), "Never reuse a wave, including incomplete submissions")
        with phase("submit full current source/checkpoint verification"):
            plan, manifest = load_plan(args.plan)
            same(plan, initial, "unchanged plan across controller acquisition")
            continuation_gate(plan, manifest, selected, args.retry_failed, args.reason)
        same(inspection(args.inspection, fresh=False), requested, "unchanged approved resource request")
        observed = collect_resources(requested, plan["submitter"], plan["protected_roots"],
                                     expected=plan["inspection"]["record"])
        same(load_plan(args.plan, bind=False)[0], plan, "current source/launch/plan bytes after resource collection")
        current_resources(observed)
        current = read(observed["evidence"]["active_work"]["path"])["queue"]
        argv = argv_for(plan, wave, observed, selected)
        if not args.execute:
            return {"submission_performed": False, "argv": argv, "shell_preview": shlex.join(argv),
                    "inspection": observed, "retry_failed": args.retry_failed, "reason": args.reason,
                    "note": "Preview collected new resource evidence; explicit submission repeats full checks then recollects"}
        submitter_identity(plan["submitter"])
        wave.mkdir()
        fsync(wave.parent)
        for name in ("logs", "claims", "outcomes"):
            (wave / name).mkdir()
        fsync(wave)
        intent = seal({"version": VERSION, "plan_sha256": plan["document_sha256"], "wave": args.wave,
                       "submitter": plan["submitter"],
                       "argv": argv, "inspection": observed, "indices": selected,
                       "retry_failed": args.retry_failed, "reason": args.reason,
                       "queue": current, "created_at_utc": datetime.now(timezone.utc).isoformat()})
        publish(wave / "intent.json", intent)
        # No expensive checkpoint operation follows collection. Delay, identity
        # or byte changes here leave the durable intent reserved, never submitted.
        submitter_identity(plan["submitter"])
        same(pin(), plan["operator_pin"], "external code immediately before sbatch")
        current_resources(observed)
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
    same(intent.get("submitter"), plan["submitter"], "intent OS submitter pin")
    same(indices(",".join(map(str, intent["indices"]))), intent["indices"], "intent mapping")
    require(type(intent["retry_failed"]) is bool and intent["reason"].strip(), "Audited retry intent required")
    same(intent["argv"], argv_for(plan, wave, intent["inspection"], intent["indices"]), "exact sealed sbatch intent")
    return intent


def acknowledge(plan, wave, job, reason, evidence):
    submitter_identity(plan["submitter"])
    require(re.fullmatch(r"[1-9]\d*", job or "") is not None and reason.strip(), "Exact array ID and reason required")
    intent = load_intent(plan, wave)
    reconciliation = None
    if evidence is not None:
        # Lost acknowledgment: obtain independent scheduler identity rather than
        # trusting an arbitrary job number or treating no queue rows as absence.
        reconciliation = command(["sacct", "-X", "--noheader", "--parsable2", "--jobs", job,
                                   f"--user={plan['submitter']['uid']}",
                                   "--format=JobID%80,JobName%80,Account%80,Partition%80,Comment%200,User%80,UID"],
                                  env=scheduler_environment("SACCT_"))
        expected_comment = f"{VERSION}:{plan['document_sha256']}:{wave.name}"
        found = set()
        for line in reconciliation.splitlines():
            fields = line.split("|")
            require(len(fields) == 7 and all(fields), "Unparseable reconciliation accounting")
            scheduler_user(fields[5], fields[6], plan["submitter"])
            match = re.fullmatch(re.escape(job) + r"_(\d+)", fields[0])
            require(match is not None or fields[0] == job, "Unexpected reconciliation job identity")
            same(fields[1:5], [f"mobile-eval-{wave.name}", intent["inspection"]["record"]["account"],
                               intent["inspection"]["record"]["partition"], expected_comment],
                 "independent Slurm submission identity")
            if match:
                require(int(match[1]) not in found, "Duplicate reconciliation task evidence")
                found.add(int(match[1]))
        same(sorted(found), intent["indices"], "reconciled exact array mapping (wait for complete accounting)")
    if (wave / "observed-array.json").exists():
        same(read(wave / "observed-array.json")["array_job_id"], job, "observed array ID")
    record = seal({"intent_sha256": intent["document_sha256"], "array_job_id": job, "reason": reason,
                   "submitter": plan["submitter"],
                   "scheduler_reconciliation": reconciliation,
                   "evidence": None if evidence is None else {"path": str(plain(evidence)), "sha256": digest(evidence)}})
    publish(wave / "ack.json", record)
    return record


def allocation(plan, wave, index):
    mapping(index)
    submitter_identity(plan["submitter"])
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
        same(ack.get("submitter"), plan["submitter"], "acknowledged OS submitter")
    observed = seal({"intent_sha256": intent["document_sha256"], "array_job_id": job, "submitter": plan["submitter"]})
    try:
        publish(wave / "observed-array.json", observed)
    except FileExistsError:
        same(check(read(wave / "observed-array.json")), observed, "one array per intent")
    publish(wave / "claims" / f"{index}.json", seal({"global_index": index, "array_job_id": job,
            "job_id": os.environ["SLURM_JOB_ID"], "intent_sha256": intent["document_sha256"],
            "submitter": plan["submitter"],
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
ACTIVE_STATES = {"PENDING", "RUNNING", "COMPLETING", "CONFIGURING", "SUSPENDED", "STOPPED",
                 "RESIZING", "RESV_DEL_HOLD", "REQUEUE_FED", "REQUEUE_HOLD", "REQUEUED",
                 "SIGNALING", "SPECIAL_EXIT", "STAGE_OUT"}


def accounting(job, selected, *, submitter):
    submitter_identity(submitter)
    output = command(["sacct", "-X", "--noheader", "--parsable2", "--jobs", job,
                      f"--user={submitter['uid']}", "--format=JobID%80,State%80,ExitCode,User%80,UID"],
                     env=scheduler_environment("SACCT_"))
    rows = {}
    for line in output.splitlines():
        parts = line.split("|")
        require(len(parts) == 5 and all(parts), "Unparseable settlement accounting")
        scheduler_user(parts[3], parts[4], submitter)
        match = re.fullmatch(re.escape(job) + r"_(\d+)", parts[0])
        require(match is not None or parts[0] == job, "Unexpected settlement job identity")
        state = parts[1].split()[0].rstrip("+")
        require(re.fullmatch(r"\d+:\d+", parts[2]) is not None, "Unparseable accounting exit code")
        require(state in TERMINAL, "Accounting is not terminal; keep reservations")
        if match:
            index = int(match[1])
            require(index not in rows, "Duplicate accounting task evidence")
            rows[index] = {"state": state, "exit_code": parts[2], "user": parts[3], "uid": int(parts[4])}
    same(sorted(rows), selected, "complete exact per-index accounting")
    require(all(row["state"] in TERMINAL for row in rows.values()), "Accounting is not terminal; keep reservations")
    return output, rows


def validate_settlement(wave, plan):
    intent = load_intent(plan, wave)
    ack = check(read(wave / "ack.json"))
    record = check(read(wave / "settled.json"))
    same([ack.get("submitter"), record.get("submitter")], [plan["submitter"], plan["submitter"]],
         "settled OS submitter identity")
    same([ack["intent_sha256"], record["intent_sha256"], record["array_job_id"]],
         [intent["document_sha256"], intent["document_sha256"], ack["array_job_id"]], "settled intent/array")
    same(record["inactive_across_nodes_confirmed"], True, "settlement inactivity evidence")
    same(sorted(map(int, record["rows"])), intent["indices"], "settled accounting mapping")
    require(all(r["state"] in TERMINAL for r in record["rows"].values()), "Unsettled accounting evidence")
    for row in record["rows"].values():
        scheduler_user(row["user"], str(row["uid"]), plan["submitter"])
    return record


def settle(args):
    plan, _ = load_plan(args.plan)
    submitter_identity(plan["submitter"])
    wave = wave_path(args.wave)
    with controller_lock(OPS):
        intent = load_intent(plan, wave)
        ack = check(read(wave / "ack.json"))
        same(ack["intent_sha256"], intent["document_sha256"], "settlement acknowledgment")
        same(ack.get("submitter"), plan["submitter"], "settlement OS submitter")
        require(args.reason.strip() and args.confirmed_inactive_across_nodes,
                "Explicit all-owner/all-allocation inactivity confirmation and reason required")
        current = queue(plan["submitter"])
        require(not any(row["array"] == ack["array_job_id"] or row["job"] == ack["array_job_id"] for row in current),
                "Array remains queued or active")
        output, rows = accounting(ack["array_job_id"], intent["indices"], submitter=plan["submitter"])
        # Settlement releases scheduling reservations only, never scientific completion.
        record = seal({"intent_sha256": intent["document_sha256"], "array_job_id": ack["array_job_id"],
                       "submitter": plan["submitter"], "queue": current,
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
    p.add_argument("--resource-request", "--inspection", dest="inspection", required=True,
                   help="Read-only reviewed choices/evidence; fresh resources are collected after full verification")
    for name in ("validate", "submit", "record-job", "settle", "run"):
        p = commands.add_parser(name)
        p.add_argument("--plan", required=True)
        if name != "validate":
            p.add_argument("--wave", required=True)
        if name == "submit":
            p.add_argument("--resource-request", "--inspection", dest="inspection", required=True,
                           help="Explicit reviewed choices; this command collects fresh resources after binding checks")
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
