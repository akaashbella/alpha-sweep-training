"""Independent evaluation ownership, slot locks and durable result authority."""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import socket
import uuid

from .artifacts import _is_link
from .checkpoint_io import digest_file, exact, fail, same
from .checkpoint_records import classification
from .data_contract import content_identity
from .evaluation_protocol import EvaluationSlot, perturbation_seed
from .provenance import software_provenance
from .serialization import atomic_write_json, durable_mkdir, fsync_directory, read_json
from .spec import BATCH_SIZE, TEST_SIZE, EVALUATION_RANDOMNESS_VERSION

EVALUATION_STORAGE_VERSION = 1
STATUSES = {"evaluation_completed", "failed_numerical", "failed_execution", "interrupted"}
PRESERVATION = {"verified": True, "method": "exact_owned_baseline",
                "scope": "all_parameters_buffers_bindings_prior_modes"}


class SweepFinalizationDeferred(RuntimeError):
    """Expected contention, without publication failure or completion proof."""
    def __init__(self, path):
        self.lock_path = str(path)
        super().__init__(f"Sweep finalization deferred after concurrent lock contention at {path}; "
                         "reinspect completion or invoke evaluation again after the owner exits; no lock was removed")


def inspect_sweep_lock(path, binding_sha256):
    """Read-only compatible-owner check; unsafe locks and I/O errors stay errors.

    No PID/age liveness inference. A valid recorded owner may require operator
    review if it has stopped; neither inspection nor contention removes it.
    A lock removed by its owner during inspection is no longer an obstruction.
    """
    if _is_link(path):
        fail("Linked sweep lock requires manual review")
    try:
        if not path.is_file():
            if _is_link(path):
                fail("Sweep lock became linked during inspection")
            if path.exists():
                fail("Non-file sweep lock requires manual review")
            return None
        value = read_json(path)
    except FileNotFoundError:
        if _is_link(path):
            fail("Sweep lock became linked during inspection")
        return None
    if _is_link(path):
        fail("Sweep lock became linked during inspection")
    exact(value, {"binding_sha256", "token", "hostname", "pid", "created_at_utc"}, "sweep lock owner")
    same(value["binding_sha256"], binding_sha256, "sweep lock checkpoint/protocol binding")
    token = value["token"]
    if type(token) is not str or len(token) != 32 or any(c not in "0123456789abcdef" for c in token):
        fail("Malformed sweep lock ownership token; manual review required")
    if type(value["hostname"]) is not str or not value["hostname"] or type(value["pid"]) is not int or value["pid"] < 1:
        fail("Malformed sweep lock host/PID evidence; manual review required")
    timestamp = value["created_at_utc"]
    if type(timestamp) is not str:
        fail("Malformed sweep lock creation time; manual review required")
    try:
        created = datetime.fromisoformat(timestamp)
    except ValueError:
        fail("Malformed sweep lock creation time; manual review required")
    if created.utcoffset() != timezone.utc.utcoffset(created):
        fail("Sweep lock creation time must carry UTC timezone evidence")
    return value


def evaluation_directory(inputs, *, evaluation_root=None) -> Path:
    storage = inputs.config.execution["storage"]
    base = Path(evaluation_root).expanduser().absolute() if evaluation_root is not None else Path(
        storage["result_root"] or storage["output_root"]) / "evaluations"
    identity = inputs.binding["training_identity"]
    path = base / identity["run_id"] / identity["scientific_fingerprint"] / inputs.binding[
        "best_checkpoint_identity"]["sha256"] / inputs.protocol.fingerprint
    for part in (path, *path.parents):
        if _is_link(part):
            fail(f"Linked evaluation storage is rejected: {part}")
    path = path.resolve()
    for training in inputs.paths.directories():
        if path == training or training in path.parents or path in training.parents:
            fail("Evaluation namespace must be outside training managed directories")
    return path


def _file(root, relative):
    # Reuse strict relative-path protection, including linked parents/targets.
    from .checkpoint_io import relative_file
    return relative_file(root, relative)


def _reference(path, root):
    return {"relative_path": path.relative_to(root).as_posix(), "sha256": digest_file(path)}


def _read_reference(root, value):
    exact(value, {"relative_path", "sha256"}, "evaluation reference")
    path = _file(root, value["relative_path"])
    if digest_file(path) != value["sha256"]:
        fail("Evaluation reference content digest mismatch")
    return read_json(path)


def record_identity(inputs, slot):
    binding = inputs.binding
    identity = binding["training_identity"]
    return {"evaluation_result_version": 1, "kind": "evaluation_realization",
        "dataset": binding["dataset"], "architecture": binding["architecture"],
        "training_run_id": identity["run_id"], "training_scientific_fingerprint": identity["scientific_fingerprint"],
        "training_alpha": binding["training_alpha"], "canonical_training_alpha": binding["canonical_training_alpha"],
        "training_seed": binding["training_seed"], "evaluation_protocol_id": binding["evaluation_protocol_id"],
        "evaluation_protocol_version": binding["evaluation_protocol_id"],
        "evaluation_protocol_fingerprint": binding["evaluation_protocol_fingerprint"], **slot.to_dict(),
        "perturbation_seed": perturbation_seed(binding["best_checkpoint_identity"], slot),
        "randomness_version": EVALUATION_RANDOMNESS_VERSION, "best_checkpoint": deepcopy(binding["best_checkpoint"]),
        "best_checkpoint_identity": deepcopy(binding["best_checkpoint_identity"]),
        "best_validation": deepcopy(binding["best_validation"]), "data_identity": deepcopy(binding["data_identity"]),
        "training_provenance": deepcopy(binding["training_provenance"])}


def validate_result(record, inputs, slot, *, evaluator_provenance=None):
    expected = record_identity(inputs, slot)
    exact(record, set(expected) | {"status", "test_metrics", "clean_state_preservation", "runtime",
        "evaluation_provenance", "attempt_id", "duration_seconds", "recorded_at_utc", "failure", "origin", "previous_record"}, "evaluation raw result")
    for key, value in expected.items():
        same(record[key], value, f"evaluation identity {key}")
    if record["status"] not in STATUSES:
        fail("Unknown evaluation result status")
    provenance = record["evaluation_provenance"]
    software_provenance(code=provenance["code"], software=provenance["software"], allocation=provenance.get("allocation"))
    if evaluator_provenance is not None:
        same(provenance["code"]["source_sha256"], evaluator_provenance["code"]["source_sha256"], "evaluation source for compatible slot reuse")
        same(provenance["software"], evaluator_provenance["software"], "evaluation software for compatible slot reuse")
    for key in ("attempt_id", "recorded_at_utc"):
        if type(record[key]) is not str or not record[key]:
            fail(f"Evaluation {key} must be actual and nonempty")
    duration = record["duration_seconds"]
    import math
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0:
        fail("Evaluation duration must be finite and nonnegative")
    origin = record["origin"]
    if type(origin) is not dict or origin.get("kind") not in ("executed", "legacy_import"):
        fail("Evaluation origin must be explicit")
    if origin["kind"] == "executed":
        exact(origin, {"kind"}, "executed evaluation origin")
    else:
        exact(origin, {"kind", "test_reference", "summary_reference", "original_test_provenance",
                       "original_duration_seconds", "original_recorded_at_utc", "original_runtime"}, "legacy evaluation origin")
        if slot.evaluation_alpha != 0 or inputs.legacy_clean is None:
            fail("Only verified legacy clean evidence can be imported into replicate 0")
        for key in ("test_reference", "summary_reference"):
            same(origin[key], inputs.legacy_clean[key], f"legacy origin {key}")
        same(inputs.legacy_clean["summary"].get("evaluation_state_preservation"), PRESERVATION, "immutable legacy state proof")
        same(origin["original_runtime"], inputs.legacy_clean["summary"].get("evaluation_runtime"), "original actual legacy test runtime")
        legacy_record = inputs.legacy_clean["test"]
        same(origin["original_test_provenance"], legacy_record["provenance"], "original legacy test provenance")
        same(origin["original_duration_seconds"], legacy_record["payload"]["duration_seconds"], "original legacy test duration")
        same(origin["original_recorded_at_utc"], legacy_record["recorded_at_utc"], "original legacy test timestamp")
    if record["previous_record"] is not None:
        exact(record["previous_record"], {"relative_path", "sha256"}, "evaluation record ancestry")
        if not record["previous_record"]["relative_path"].startswith(f"slots/{slot.slot_id}/records/"):
            fail("Evaluation ancestry escaped its slot")
    if record["status"] == "evaluation_completed":
        metrics = classification(record["test_metrics"])
        if metrics.examples != TEST_SIZE or metrics.batches != (TEST_SIZE + BATCH_SIZE - 1) // BATCH_SIZE:
            fail("A realization must cover all 10,000 official test images")
        same(record["clean_state_preservation"], PRESERVATION, "entire clean-state preservation")
        same(record["failure"], None, "successful realization failure field")
        runtime = record["runtime"]
        exact(runtime, {"engine_version", "device", "device_properties", "precision", "effective_backend",
                        "master_dtype", "evaluation_precision", "grad_scaler"}, "actual evaluation runtime metadata")
        same(runtime["precision"], inputs.protocol.to_dict()["precision"], "evaluation precision")
        same(runtime["master_dtype"], "float32", "evaluation clean dtype")
        same(runtime["grad_scaler"], None, "independent evaluation has no scaler")
        effective = runtime["effective_backend"]
        policy = inputs.protocol.to_dict()["precision"]
        for key in ("float32_matmul_precision", "deterministic_algorithms", "cudnn_deterministic", "cudnn_benchmark"):
            same(effective[key], policy[key], f"evaluation effective backend {key}")
        for key in ("allow_tf32_matmul", "allow_tf32_cudnn"):
            same(effective[key], policy["allow_tf32"], f"evaluation {key}")
        same(effective["deterministic_warn_only"], False, "evaluation deterministic enforcement")
        if policy["deterministic_algorithms"] and effective["cublas_workspace_config"] not in (":4096:8", ":16:8"):
            fail("Successful deterministic evaluation requires its actual supported cuBLAS workspace policy")
    else:
        same(record["test_metrics"], None, "unavailable partial/failed metrics")
        same(record["clean_state_preservation"], None, "failed realization preservation is not asserted")
        failure = exact(record["failure"], {"phase", "reason", "exception_type", "batch_index", "numerical_stage",
                       "affected_tensors", "partial_metrics_discarded"}, "evaluation failure")
        if any(type(failure[k]) is not str or not failure[k] for k in ("phase", "reason", "exception_type")):
            fail("Failed/interrupted evaluation requires diagnostics")
        same(failure["partial_metrics_discarded"], True, "partial pass aggregates discarded")
    return record


def read_slot(root, inputs, slot, *, evaluator_provenance=None):
    """Authority only; caller handles writer locks and unstable observations."""
    same(read_json(_file(root, "owner.json")), inputs.binding, "evaluation namespace owner")
    head_path = _file(root, f"slots/{slot.slot_id}/HEAD.json")
    if not head_path.exists():
        return None
    head = exact(read_json(head_path), {"evaluation_storage_version", "slot", "current", "previous"}, "evaluation HEAD")
    same(head["evaluation_storage_version"], EVALUATION_STORAGE_VERSION, "evaluation storage version")
    same(head["slot"], slot.to_dict(), "evaluation HEAD slot")
    if not head["current"]["relative_path"].startswith(f"slots/{slot.slot_id}/records/"):
        fail("Evaluation HEAD must refer to its own immutable slot record")
    record = validate_result(_read_reference(root, head["current"]), inputs, slot, evaluator_provenance=evaluator_provenance)
    same(head["previous"], record["previous_record"], "evaluation HEAD ancestry")
    if head["previous"] is not None:
        if not head["previous"]["relative_path"].startswith(f"slots/{slot.slot_id}/records/"):
            fail("Evaluation previous reference escaped its slot")
        validate_result(_read_reference(root, head["previous"]), inputs, slot)
    return head, record


@dataclass(frozen=True)
class EvaluationCommitReceipt:
    slot_id: str
    status: str
    reference: dict


@dataclass
class EvaluationLease:
    root: Path
    inputs: object
    slot: EvaluationSlot
    attempt_id: str
    provenance: dict
    active: bool = True

    def require_active(self):
        if not self.active:
            fail("Hold the independent slot lease throughout every evaluation write")

    def current(self):
        self.require_active()
        return read_slot(self.root, self.inputs, self.slot, evaluator_provenance=self.provenance)

    def publish(self, *, status, metrics, runtime, duration_seconds, failure=None,
                preservation=None, origin=None):
        self.require_active()
        prior = self.current()
        if prior is not None and prior[1]["status"] == "evaluation_completed":
            return EvaluationCommitReceipt(self.slot.slot_id, "evaluation_completed", prior[0]["current"])
        record = {**record_identity(self.inputs, self.slot), "status": status,
            "test_metrics": None if metrics is None else metrics.to_dict(), "runtime": deepcopy(runtime),
            "evaluation_provenance": deepcopy(self.provenance), "attempt_id": self.attempt_id,
            "duration_seconds": duration_seconds, "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "failure": deepcopy(failure), "clean_state_preservation": deepcopy(preservation),
            "origin": deepcopy(origin or {"kind": "executed"}),
            "previous_record": None if prior is None else prior[0]["current"]}
        validate_result(record, self.inputs, self.slot, evaluator_provenance=self.provenance)
        directory = _file(self.root, f"slots/{self.slot.slot_id}/records/probe").parent
        durable_mkdir(directory)
        path = _file(self.root, f"slots/{self.slot.slot_id}/records/{uuid.uuid4().hex}.json")
        if path.exists():
            fail("Immutable evaluation record already exists")
        atomic_write_json(path, record)
        reference = _reference(path, self.root)
        validate_result(_read_reference(self.root, reference), self.inputs, self.slot, evaluator_provenance=self.provenance)
        self.require_active()
        actual = self.current()
        if (None if actual is None else actual[0]) != (None if prior is None else prior[0]):
            fail("Evaluation authority changed before publication")
        head = {"evaluation_storage_version": EVALUATION_STORAGE_VERSION, "slot": self.slot.to_dict(),
                "current": reference, "previous": None if prior is None else prior[0]["current"]}
        atomic_write_json(_file(self.root, f"slots/{self.slot.slot_id}/HEAD.json"), head)
        return EvaluationCommitReceipt(self.slot.slot_id, status, reference)


@contextmanager
def _exclusive_lock(path, owner, *, defer_sweep_contention=False):
    if _is_link(path):
        fail("Linked evaluation lock requires manual review")
    token = uuid.uuid4().hex
    document = {**owner, "token": token, "hostname": socket.gethostname(),
                "pid": os.getpid(), "created_at_utc": datetime.now(timezone.utc).isoformat()}
    acquired, staging = False, None
    try:
        if defer_sweep_contention:
            # Publish complete sweep ownership atomically: another finalizer
            # must never observe our just-created empty/partially written lock.
            candidate = path.with_name(f"{path.name}.{token}.acquire.json")
            if candidate.exists() or _is_link(candidate):
                fail("Sweep lock staging path already exists; preserve it")
            atomic_write_json(candidate, document)
            staging = candidate
            try:
                os.link(staging, path)  # Exclusive, same-directory atomic creation.
            except FileExistsError as exc:
                inspect_sweep_lock(path, owner["binding_sha256"])
                # A released-lock race also defers; no stealing/acquisition retry.
                raise SweepFinalizationDeferred(path) from exc
            acquired = True
        else:
            try:
                descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError as exc:
                fail(f"Evaluation lock exists: {path}; never steal by PID/age ({exc})")
            acquired = True
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                from .serialization import strict_json_text
                stream.write(strict_json_text(document))
                stream.flush()
                os.fsync(stream.fileno())
        fsync_directory(path.parent)
        yield
    finally:
        if acquired and path.is_file() and not _is_link(path):
            try:
                recorded = read_json(path)
            except (OSError, ValueError):
                recorded = None
            if type(recorded) is dict and recorded.get("token") == token:
                path.unlink()
                fsync_directory(path.parent)
        if staging is not None and staging.is_file() and not _is_link(staging):
            same(read_json(staging), document, "owned sweep lock staging")
            staging.unlink()
            fsync_directory(staging.parent)


@contextmanager
def open_evaluation_slot(inputs, slot, *, evaluation_root=None, provenance, retry_failed=False):
    """Brief namespace initialization lock, then exactly one independent slot lock.

    No training writer lease is held. Existing orphan files are preserved; only
    an owned namespace can restart its unfinished realization. No overwrite,
    automatic repair, guessed checkpoint or lock-release fallback is provided.
    """
    if not isinstance(slot, EvaluationSlot) or type(retry_failed) is not bool:
        fail("Slot publication requires a prescribed EvaluationSlot and explicit bool retry choice")
    root = evaluation_directory(inputs, evaluation_root=evaluation_root)
    software_provenance(code=provenance["code"], software=provenance["software"], allocation=provenance.get("allocation"))
    durable_mkdir(root)
    owner_path = _file(root, "owner.json")
    if not owner_path.exists():
        with _exclusive_lock(root / ".namespace.lock", {"binding_sha256": content_identity(inputs.binding)}):
            if owner_path.exists():
                same(read_json(owner_path), inputs.binding, "evaluation owner")
            else:
                if any(p.name != ".namespace.lock" for p in root.iterdir()):
                    fail("Partial evaluation namespace has no owner; preserve evidence and inspect manually")
                atomic_write_json(owner_path, inputs.binding)
    same(read_json(owner_path), inputs.binding, "evaluation owner")
    locks = root / "locks"
    if _is_link(locks):
        fail("Linked evaluation lock directory")
    durable_mkdir(locks)
    with _exclusive_lock(_file(root, f"locks/{slot.slot_id}.lock"), {
            **slot.to_dict(), "binding_sha256": content_identity(inputs.binding)}):
        same(read_json(owner_path), inputs.binding, "evaluation owner under slot lock")
        current = read_slot(root, inputs, slot, evaluator_provenance=provenance)
        if not retry_failed and (current is None or current[1]["status"] != "evaluation_completed"):
            referenced = set()
            reference = None if current is None else current[0]["current"]
            while reference is not None:
                if reference["relative_path"] in referenced:
                    fail("Cycle in evaluation ancestry")
                referenced.add(reference["relative_path"])
                parent_record = validate_result(_read_reference(root, reference), inputs, slot)
                reference = parent_record["previous_record"]
            directory = _file(root, f"slots/{slot.slot_id}/records/probe").parent
            if current is not None and current[1]["status"] in ("failed_numerical", "failed_execution"):
                fail("Failed evaluation slot requires explicit retry under its lease")
            if directory.exists():
                for path in directory.iterdir():
                    if path.is_file() and path.suffix == ".json" and path.relative_to(root).as_posix() not in referenced:
                        candidate = validate_result(read_json(_file(root, path.relative_to(root).as_posix())),
                                                    inputs, slot, evaluator_provenance=provenance)
                        if candidate["status"] in ("failed_numerical", "failed_execution"):
                            fail("Unacknowledged failed evaluation evidence now requires explicit retry")
        directory = root / "slots" / slot.slot_id
        if any(_is_link(p) for p in (directory, directory.parent)):
            fail("Linked evaluation slot directory")
        durable_mkdir(directory)
        attempts = directory / "attempts"
        if _is_link(attempts):
            fail("Linked evaluation attempt directory")
        durable_mkdir(attempts)
        attempt_id = uuid.uuid4().hex
        attempt = _file(root, f"slots/{slot.slot_id}/attempts/{attempt_id}.json")
        atomic_write_json(attempt, {"evaluation_binding_sha256": content_identity(inputs.binding),
            **slot.to_dict(), "evaluation_provenance": provenance, "attempt_id": attempt_id})
        lease = EvaluationLease(root, inputs, slot, attempt_id, deepcopy(provenance))
        try:
            yield lease
        finally:
            lease.active = False


def publish_robustness_completion(inputs, *, evaluation_root=None, provenance):
    """Durable all-success authority; compatible sweep contention is typed deferral."""
    from .evaluation_protocol import prescribed_slots
    from .evaluation_coverage import inspect_robustness_completion
    root = evaluation_directory(inputs, evaluation_root=evaluation_root)
    if not root.is_dir():
        return None
    with _exclusive_lock(root / ".sweep.lock", {"binding_sha256": content_identity(inputs.binding)},
                         defer_sweep_contention=True):
        same(read_json(_file(root, "owner.json")), inputs.binding, "sweep owner")
        existing = inspect_robustness_completion(root, inputs)
        if existing is not None:
            return existing
        entries = []
        for slot in prescribed_slots():
            lock = root / "locks" / f"{slot.slot_id}.lock"
            if lock.exists() or _is_link(lock):
                return None
            current = read_slot(root, inputs, slot, evaluator_provenance=provenance)
            if current is None or current[1]["status"] != "evaluation_completed":
                return None
            entries.append({"slot": slot.to_dict(), "reference": current[0]["current"]})
        document = {"evaluation_completion_version": 1, "status": "robustness_completed",
            "binding_sha256": content_identity(inputs.binding), "slots": entries,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(), "publication_provenance": deepcopy(provenance)}
        directory = root / "sweep_completions"
        if _is_link(directory):
            fail("Linked sweep completion directory")
        durable_mkdir(directory)
        path = _file(root, f"sweep_completions/{uuid.uuid4().hex}.json")
        atomic_write_json(path, document)
        reference = _reference(path, root)
        same(_read_reference(root, reference), document, "staged sweep completion")
        atomic_write_json(_file(root, "SWEEP_HEAD.json"), {"evaluation_storage_version": 1, "current": reference})
        return inspect_robustness_completion(root, inputs)
