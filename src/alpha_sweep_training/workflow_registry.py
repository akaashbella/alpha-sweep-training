"""One shared cooperating submission authority; immutable snapshots, no lock stealing."""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import os
import re
import socket
import time
import uuid

from .artifacts import _is_link
from .checkpoint_io import digest_file, relative_file
from .evaluation_persistence import _exclusive_lock
from .hpc_policy import OperationalPolicy, WorkflowError, exact, same, resource_budget
from .serialization import atomic_write_json, durable_mkdir, fsync_directory, read_json
from .slurm_client import utc_now
from .work_plans import check_seal, seal, validate_item, workflow_identity


class WorkflowControllerBusy(WorkflowError):
    pass


def controller_contention(path, owner):
    if _is_link(path) or (path.exists() and not path.is_file()):
        raise WorkflowError("Unsafe workflow controller lock; preserve it for operator review")
    try:
        recorded = read_json(path)
    except FileNotFoundError:
        if _is_link(path):
            raise WorkflowError("Controller lock became linked; manual review required")
        raise WorkflowControllerBusy("Controller lock changed during acquisition; bounded wait only")
    validate_operational_lock(recorded, owner)
    raise WorkflowControllerBusy("A compatible workflow controller owns the lock; no PID/age inference")


def validate_operational_lock(recorded, owner):
    exact(recorded, set(owner) | {"token", "hostname", "pid", "created_at_utc"}, "operational lock owner")
    for key, value in owner.items():
        same(recorded[key], value, "compatible controller owner")
    if type(recorded["token"]) is not str or not re.fullmatch(r"[0-9a-f]{32}", recorded["token"]) or type(recorded["pid"]) is not int or recorded["pid"] < 1 or type(recorded["hostname"]) is not str or not recorded["hostname"]:
        raise WorkflowError("Malformed controller ownership evidence")
    if type(recorded["created_at_utc"]) is not str:
        raise WorkflowError("Malformed lock creation timestamp")
    try:
        created = datetime.fromisoformat(recorded["created_at_utc"])
    except ValueError as exc:
        raise WorkflowError("Malformed lock creation timestamp") from exc
    if created.utcoffset() != timezone.utc.utcoffset(created):
        raise WorkflowError("Lock creation evidence must carry UTC timezone")


@contextmanager
def controller_lock(root, manifest, policy, *, wait=False, stop=None):
    deadline = time.monotonic() + policy.to_dict()["continuation"]["controller_wait_seconds"]
    while True:
        manager = _exclusive_lock(root / ".controller.lock", {"workflow_id": workflow_identity(manifest)},
                                  atomic_owner=True, contention=controller_contention)
        try:
            manager.__enter__()
        except WorkflowControllerBusy:
            if not wait or time.monotonic() >= deadline:
                raise
            if stop is not None:
                stop.check(at_boundary=True, phase="workflow_controller_wait")
            time.sleep(0.1)  # Future worker's bounded wait; never inside a signal handler.
            continue
        break
    try:
        yield read_json(root / ".controller.lock")
    finally:
        manager.__exit__(None, None, None)


def guarded_root(policy):
    policy.require_ready()
    root = Path(policy.to_dict()["paths"]["workflow_root"])
    for part in (root, *root.parents):
        if _is_link(part):
            raise WorkflowError(f"Linked workflow storage requires operator review: {part}")
    return root


def owner_document(manifest):
    return {"workflow_registry_version": 1, "workflow_id": workflow_identity(manifest),
            "manifest_grid_id": manifest.grid_id, "manifest_document_sha256": manifest.document_sha256}


def immutable_json(root, relative, document):
    target = relative_file(root, relative, create_parents=True)
    if target.exists():
        same(read_json(target), document, "immutable workflow file reuse")
    else:
        atomic_write_json(target, document)
    return {"relative_path": relative, "sha256": digest_file(target)}


def read_reference(root, reference):
    exact(reference, {"relative_path", "sha256"}, "workflow reference")
    target = relative_file(root, reference["relative_path"])
    same(digest_file(target), reference["sha256"], "workflow reference content")
    return read_json(target)


def validate_state(state, manifest):
    check_seal(state)
    exact(state, {"registry_state_version", "workflow_id", "revision", "policy", "policy_fingerprint",
                  "waves", "event", "recorded_at_utc", "document_sha256"}, "registry state")
    same(state["registry_state_version"], 1, "registry state version")
    same(state["workflow_id"], workflow_identity(manifest), "registry workflow")
    policy = OperationalPolicy(state["policy"])
    same(state["policy_fingerprint"], policy.fingerprint, "registry policy")
    if type(state["revision"]) is not int or state["revision"] < 0 or type(state["waves"]) is not list:
        raise WorkflowError("Malformed registry revision/waves")
    seen, ordinals = set(), {}
    for wave in state["waves"]:
        exact(wave, {"wave_id", "bundle_reference", "policy", "policy_fingerprint", "items", "status", "scheduler",
            "scheduler_comment", "submission_arguments", "intent_at_utc", "submission_diagnostic", "claims",
            "outcomes", "diagnostics", "scheduler_observation", "resolution"}, "registered wave")
        if wave["wave_id"] in seen:
            raise WorkflowError("Duplicate registered wave")
        seen.add(wave["wave_id"])
        old_policy = OperationalPolicy(wave["policy"])
        same(old_policy.fingerprint, wave["policy_fingerprint"], "wave policy")
        if wave["status"] not in ("submission_uncertain", "submitted", "terminal", "confirmed_not_submitted"):
            raise WorkflowError("Unknown wave allocation state")
        resource_budget(old_policy, len(wave["items"]))
        if not wave["items"] or len({item["training_run_id"] for item in wave["items"]}) != len(wave["items"]):
            raise WorkflowError("Wave contains empty, duplicate or overlapping training/evaluation ownership")
        for item in wave["items"]:
            validate_item(item, manifest)
            ordinal = ordinals.get(item["unit_key"], 0) + 1
            same(item["allocation_ordinal"], ordinal, "monotonic bounded allocation intent ordinal")
            if ordinal > old_policy.to_dict()["continuation"]["max_allocations_per_work"]:
                raise WorkflowError("Wave exceeded its explicitly declared allocation-intent budget")
            ordinals[item["unit_key"]] = ordinal
        for key in ("claims", "outcomes", "diagnostics"):
            if type(wave[key]) is not dict or set(wave[key]) - {str(i) for i in range(len(wave["items"]))}:
                raise WorkflowError("Registry allocation mapping escaped its wave")
        if wave["status"] == "submitted" and wave["scheduler"] is None:
            raise WorkflowError("Submitted wave requires actual scheduler identity")
    # Reconciliation may revoke old terminal observations (accounting loss or an
    # outside requeue). Preserve ALL uncertainty rather than rejecting the new
    # snapshot. reserve() still forbids any new wave until every wave is terminal.
    return state


def _read_state(root, manifest):
    same(read_json(relative_file(root, "owner.json")), owner_document(manifest), "workflow owner")
    head = exact(read_json(relative_file(root, "HEAD.json")), {"registry_head_version", "workflow_id", "current", "previous"}, "workflow HEAD")
    same(head["registry_head_version"], 1, "registry HEAD version")
    same(head["workflow_id"], workflow_identity(manifest), "workflow HEAD identity")
    if not head["current"]["relative_path"].startswith("states/"):
        raise WorkflowError("Workflow HEAD must reference an immutable state snapshot")
    state = validate_state(read_reference(root, head["current"]), manifest)
    if head["previous"] is not None:
        if not head["previous"]["relative_path"].startswith("states/"):
            raise WorkflowError("Previous workflow authority escaped its snapshot namespace")
        previous = validate_state(read_reference(root, head["previous"]), manifest)
        same(state["revision"], previous["revision"] + 1, "workflow snapshot revision ancestry")
    elif state["revision"] != 0:
        raise WorkflowError("Only initialized workflow authority lacks prior revision")
    from .work_render import validate_bundle
    for wave in state["waves"]:
        relative = f"bundles/{wave['wave_id']}/mapping.json"
        same(wave["bundle_reference"]["relative_path"], relative, "reserved mapping namespace")
        bundle = read_reference(root, wave["bundle_reference"])
        validate_bundle(bundle, manifest, OperationalPolicy(wave["policy"]))
        same(bundle["items"], wave["items"], "registry/bundle array membership")
        same(bundle["scheduler_comment"], wave["scheduler_comment"], "registry/bundle submission token")
    return head, state


def read_workflow(manifest, policy):
    root = guarded_root(policy)
    if not root.exists():
        return {"waves": [], "revision": 0}
    lock = root / ".controller.lock"
    if lock.exists() or _is_link(lock):
        raise WorkflowError("Workflow controller lock is present; no PID/age inference or replacement planning")
    head, state = _read_state(root, manifest)
    same(state["policy_fingerprint"], policy.fingerprint, "current operational policy (use explicit adopt-policy after quiescence)")
    same(read_json(relative_file(root, "HEAD.json")), head, "stable workflow read")
    if lock.exists() or _is_link(lock):
        raise WorkflowError("Workflow changed during observation; reinspect after controller exits")
    return state


@dataclass
class WorkflowLease:
    root: Path
    manifest: object
    policy: OperationalPolicy
    head: dict | None
    state: dict
    controller_owner: dict | None = None
    active: bool = True

    def guard(self):
        if not self.active:
            raise WorkflowError("Hold the shared workflow controller lease throughout every mutation/submission")
        same(read_json(relative_file(self.root, ".controller.lock")), self.controller_owner, "active exact workflow controller lease")
        if self.controller_owner is None or self.controller_owner["pid"] != os.getpid() or self.controller_owner["hostname"] != socket.gethostname():
            raise WorkflowError("Workflow authority belongs to another process/node")
        same(read_json(relative_file(self.root, "owner.json")), owner_document(self.manifest), "workflow owner under lease")
        if self.head is not None:
            same(read_json(relative_file(self.root, "HEAD.json")), self.head, "current workflow HEAD")

    def commit(self, event):
        self.guard()
        state = seal({"registry_state_version": 1, "workflow_id": workflow_identity(self.manifest),
            "revision": self.state["revision"] + 1, "policy": self.policy.to_dict(),
            "policy_fingerprint": self.policy.fingerprint, "waves": deepcopy(self.state["waves"]),
            "event": deepcopy(event), "recorded_at_utc": utc_now()})
        validate_state(state, self.manifest)
        reference = immutable_json(self.root, f"states/{uuid.uuid4().hex}.json", state)
        head = {"registry_head_version": 1, "workflow_id": workflow_identity(self.manifest),
                "current": reference, "previous": None if self.head is None else self.head["current"]}
        self.guard()
        atomic_write_json(relative_file(self.root, "HEAD.json"), head)
        self.head, self.state = head, state

    def wave(self, wave_id):
        self.guard()
        matches = [wave for wave in self.state["waves"] if wave["wave_id"] == wave_id]
        if len(matches) != 1:
            raise WorkflowError("Unknown or ambiguous workflow wave")
        return matches[0]

    def reserve(self, bundle, bundle_reference, submission_arguments):
        self.guard()
        if any(w["status"] not in ("terminal", "confirmed_not_submitted") for w in self.state["waves"]):
            raise WorkflowError("Previous wave is not conclusively terminal; pending/running/unknown allocations reserve capacity")
        if any(w["wave_id"] == bundle["wave_id"] for w in self.state["waves"]):
            raise WorkflowError("Wave already has submission intent; reconcile it, never submit again")
        resource_budget(self.policy, len(bundle["items"]))
        wave = {"wave_id": bundle["wave_id"], "bundle_reference": bundle_reference, "policy": self.policy.to_dict(),
            "policy_fingerprint": self.policy.fingerprint, "items": deepcopy(bundle["items"]),
            "status": "submission_uncertain", "scheduler": None, "scheduler_comment": bundle["scheduler_comment"],
            "submission_arguments": list(submission_arguments), "intent_at_utc": utc_now(), "submission_diagnostic": None,
            "claims": {}, "outcomes": {}, "diagnostics": {}, "scheduler_observation": None, "resolution": None}
        self.state["waves"].append(wave)
        # Commit uncertainty BEFORE sbatch. A crash here cannot authorize replay.
        self.commit({"kind": "submission_intent", "wave_id": bundle["wave_id"]})
        return self.wave(bundle["wave_id"])

    def record_acknowledgement(self, wave_id, scheduler):
        wave = self.wave(wave_id)
        if wave["scheduler"] is not None:
            for key in ("job_id", "cluster"):
                same(wave["scheduler"][key], scheduler[key], f"submission/runner scheduler {key}")
        wave.update(scheduler=deepcopy(scheduler), status="submitted")
        self.commit({"kind": "submission_acknowledged", "wave_id": wave_id})

    def reconcile(self, client):
        self.guard()
        for wave in self.state["waves"]:
            if wave["status"] == "confirmed_not_submitted":
                continue
            observation = None
            if wave["resolution"] is not None:
                resolution = read_reference(self.root, wave["resolution"])
                if resolution["kind"] == "operator_confirmed_terminal":
                    same(resolution["wave_id"], wave["wave_id"], "terminal-resolution wave")
                    same(resolution["scheduler"], wave["scheduler"], "terminal-resolution scheduler identity")
                    observation = {"terminal": True, "condition": "operator_verified_terminal",
                        "resolution": wave["resolution"], "allocations": [], "diagnostic": "Actual terminal/inactive evidence explicitly supplied by operator; no automatic absence inference"}
            # Refresh even an old terminal record, protecting against external requeue.
            if observation is None:
                observation = client.observe_wave(wave)
            wave["scheduler_observation"] = observation
            if observation["terminal"]:
                wave["status"] = "terminal"
            else:
                wave["status"] = "submission_uncertain" if wave["scheduler"] is None else "submitted"
            for index, claim in wave["claims"].items():
                path = relative_file(self.root, claim["outcome_relative_path"])
                if path.exists():
                    result = read_json(path)
                    validate_allocation_outcome(result, wave, index, claim)
                    wave["outcomes"][index] = result
            for index in range(len(wave["items"])):
                directory = relative_file(self.root, f"diagnostics/{wave['wave_id']}/{index}/probe").parent
                records = []
                if directory.exists():
                    for path in sorted(directory.iterdir()):
                        if _is_link(path):
                            raise WorkflowError("Linked allocation diagnostic evidence")
                        if path.is_file() and path.suffix == ".json":
                            diagnostic = read_json(path)
                            check_seal(diagnostic)
                            same(diagnostic["wave_id"], wave["wave_id"], "diagnostic wave")
                            same(diagnostic["array_index"], index, "diagnostic allocation")
                            same(diagnostic["work_item_sha256"], wave["items"][index]["document_sha256"], "diagnostic work")
                            records.append(diagnostic)
                wave["diagnostics"][str(index)] = records
            # A residual owning process/lease prevents quiescence even if Slurm says terminal.
            for index in range(len(wave["items"])):
                lock = relative_file(self.root, f"allocations/{wave['wave_id']}/{index}.lock")
                if lock.exists() or _is_link(lock):
                    wave["status"] = "submitted" if wave["scheduler"] else "submission_uncertain"
                    observation["terminal"] = False
                    observation["diagnostic"] = "Residual allocation lease; explicit owner/Slurm recovery required"
        self.commit({"kind": "scheduler_reconciliation"})
        return deepcopy(self.state)


@contextmanager
def open_workflow(manifest, policy, *, initialize=False, adopting_policy=False, wait=False, stop=None):
    root = guarded_root(policy)
    from .artifacts import resolve_run_paths
    from .manifests import task_config
    operational_roots = [Path(policy.to_dict()["paths"][key]) for key in ("workflow_root", "slurm_log_root", "evaluation_root")]
    for task in manifest.tasks:
        run_paths = resolve_run_paths(task_config(task, policy.execution_overrides()), policy.to_dict()["paths"]["base_dir"])
        for managed in run_paths.directories():
            if any(path == managed or managed in path.parents or path in managed.parents for path in operational_roots):
                raise WorkflowError("Operational roots must not overlap any protected scientific run directory")
    if not root.exists() and not initialize:
        raise WorkflowError("Workflow registry is absent; render/submit/advance initialize it explicitly")
    if root.exists() and not root.is_dir():
        raise WorkflowError("Workflow root is not a directory")
    durable_mkdir(root)
    with controller_lock(root, manifest, policy, wait=wait, stop=stop) as controller_owner:
        owner = relative_file(root, "owner.json")
        if not owner.exists():
            if not initialize or any(p.name != ".controller.lock" for p in root.iterdir()):
                raise WorkflowError("Partial/unowned workflow storage must be preserved for explicit recovery")
            atomic_write_json(owner, owner_document(manifest))
            lease = WorkflowLease(root, manifest, policy, None, {"waves": [], "revision": -1}, controller_owner=deepcopy(controller_owner))
            lease.commit({"kind": "initialized"})
        else:
            head, state = _read_state(root, manifest)
            if not adopting_policy:
                same(state["policy_fingerprint"], policy.fingerprint, "current workflow execution policy")
            lease = WorkflowLease(root, manifest, policy, head, deepcopy(state), controller_owner=deepcopy(controller_owner))
        try:
            yield lease
        finally:
            lease.active = False


def validate_allocation_outcome(record, wave, index, claim):
    check_seal(record)
    exact(record, {"allocation_outcome_version", "wave_id", "array_index", "work_item_sha256", "claim_id",
        "scheduler_id", "exit_code", "task_outcome", "safe_interruption", "diagnostic", "stop_request",
        "allocation_provenance", "recorded_at_utc", "document_sha256"}, "allocation outcome")
    same(record["allocation_outcome_version"], 1, "allocation outcome version")
    same(record["wave_id"], wave["wave_id"], "allocation wave")
    same(record["array_index"], int(index), "allocation mapping")
    same(record["work_item_sha256"], wave["items"][int(index)]["document_sha256"], "allocation work")
    same(record["claim_id"], claim["claim_id"], "allocation owner")
    same(record["scheduler_id"], claim["scheduler_id"], "allocation scheduler identity")
    if type(record["exit_code"]) is not int or record["exit_code"] not in (0, 10, 20, 21, 22) or type(record["safe_interruption"]) is not bool:
        raise WorkflowError("Malformed allocation outcome classification")
    if record["safe_interruption"] and record["exit_code"] != 10:
        raise WorkflowError("Only explicit clean interruption can authorize automatic continuation")
    result = record["task_outcome"]
    if result is not None:
        if type(result) is not dict:
            raise WorkflowError("Allocation task outcome must be a finite structured record or unavailable")
        same(result["lifecycle"], wave["items"][int(index)]["lifecycle"], "allocation/scientific lifecycle")
        same(int(result["exit_code"]), record["exit_code"], "allocation/scientific exit classification")
        for key, item_key in (("run_id", "training_run_id"), ("training_run_id", "training_run_id"), ("scientific_fingerprint", "training_fingerprint")):
            if key in result:
                same(result[key], wave["items"][int(index)][item_key], "allocation/scientific task membership")
    if record["safe_interruption"] and (result is None or result["outcome"] not in ("interrupted", "evaluation_interrupted", "not_started_stop_requested")
                                        or record["stop_request"]["policy"] is None):
        raise WorkflowError("Automatic continuation requires explicit stop and compatible authority evidence")
    return record


@contextmanager
def claim_allocation(manifest, policy, bundle, index, scheduler, provenance, *, stop=None):
    root = guarded_root(policy)
    if type(index) is not int or not 0 <= index < len(bundle["items"]):
        raise WorkflowError("Unknown allocation selector")
    _, current = _read_state(root, manifest)
    matches = [wave for wave in current["waves"] if wave["wave_id"] == bundle["wave_id"]]
    if len(matches) != 1:
        raise WorkflowError("No durable submission intent authorizes this allocation")
    same(read_reference(root, matches[0]["bundle_reference"]), bundle, "registered allocation before lock writes")
    relative = f"allocations/{bundle['wave_id']}/{index}.lock"
    lock = relative_file(root, relative, create_parents=True)
    with _exclusive_lock(lock, {"workflow_id": workflow_identity(manifest), "wave_id": bundle["wave_id"], "array_index": index}, atomic_owner=True):
        with open_workflow(manifest, policy, wait=True, stop=stop) as workflow:
            wave = workflow.wave(bundle["wave_id"])
            same(read_reference(root, wave["bundle_reference"]), bundle, "allocation reserved immutable mapping")
            if wave["status"] in ("terminal", "confirmed_not_submitted") or str(index) in wave["claims"]:
                raise WorkflowError("Duplicate/requeued/terminal allocation must not start another research process")
            if wave["scheduler"] is None:
                wave["scheduler"] = {"job_id": scheduler["array_job_id"], "cluster": scheduler["cluster"],
                                     "acknowledgement": "validated_allocation_runner_observed"}
                wave["status"] = "submitted"
            else:
                same(wave["scheduler"]["job_id"], scheduler["array_job_id"], "allocated array root")
                if wave["scheduler"]["cluster"] is not None:
                    same(wave["scheduler"]["cluster"], scheduler["cluster"], "allocated cluster")
                elif scheduler["cluster"] is not None:
                    wave["scheduler"]["cluster"] = scheduler["cluster"]
            claim_id = uuid.uuid4().hex
            claim = {"claim_id": claim_id, "scheduler_id": scheduler["scheduler_id"],
                "allocation_provenance": provenance, "claimed_at_utc": utc_now(),
                "outcome_relative_path": f"outcomes/{bundle['wave_id']}/{index}/{claim_id}.json"}
            wave["claims"][str(index)] = claim
            workflow.commit({"kind": "allocation_claimed", "wave_id": bundle["wave_id"], "array_index": index})
            reserved_wave = deepcopy(workflow.wave(bundle["wave_id"]))
        # Only this per-allocation lease remains held during research; controller is free.
        yield claim, reserved_wave


def publish_allocation_outcome(root, wave, index, claim, record):
    validate_allocation_outcome(record, wave, str(index), claim)
    # Call inside claim_allocation's context. The independent immutable outcome
    # survives a lost controller acknowledgement and is joined by reconciliation.
    lock = relative_file(root, f"allocations/{wave['wave_id']}/{index}.lock")
    owner = read_json(lock)
    same(owner["wave_id"], wave["wave_id"], "active allocation lease wave")
    same(owner["array_index"], index, "active allocation lease index")
    same(owner["pid"], os.getpid(), "allocation writer process")
    same(owner["hostname"], socket.gethostname(), "allocation writer node")
    return immutable_json(root, claim["outcome_relative_path"], record)
