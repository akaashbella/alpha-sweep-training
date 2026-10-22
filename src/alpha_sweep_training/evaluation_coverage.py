"""Read-only 28-slot coverage and explicit robustness-completion inspection."""

from .artifacts import _is_link
from .checkpoint_io import digest_file, exact, fail, same
from .data_contract import content_identity
from .evaluation_persistence import (evaluation_directory, read_slot, validate_result,
                                     inspect_sweep_lock, _file, _read_reference)
from .evaluation_protocol import prescribed_slots
from .serialization import read_json


def _token(path):
    if _is_link(path):
        return "linked"
    return digest_file(path) if path.is_file() else "missing"


def observe_evaluation_slot(inputs, slot, *, evaluation_root=None, evaluator_provenance=None):
    row = {**slot.to_dict(), "condition": "pending", "stored_status": None, "reference": None,
           "failure": None, "diagnostic": None, "orphan_failures": []}
    root = evaluation_directory(inputs, evaluation_root=evaluation_root)
    lock = root / "locks" / f"{slot.slot_id}.lock"
    head = root / "slots" / slot.slot_id / "HEAD.json"
    owner = root / "owner.json"
    before = None
    try:
        before = (_token(lock), _token(head), _token(owner), _token(root / ".namespace.lock"))
        if lock.exists() or _is_link(lock) or (root / ".namespace.lock").exists():
            row.update(condition="locked", diagnostic="Independent evaluation writer owns this namespace/slot")
            return row
        if not root.exists():
            return row
        if not owner.is_file():
            row.update(condition="partial", diagnostic="Evaluation namespace has no valid owner; preserve evidence")
        else:
            same(read_json(_file(root, "owner.json")), inputs.binding, "evaluation owner")
            current = read_slot(root, inputs, slot, evaluator_provenance=evaluator_provenance)
            committed_paths = set()
            if current is not None:
                _, record = current
                row.update(condition="ok", stored_status=record["status"], reference=current[0]["current"],
                           failure=record["failure"])
                reference = current[0]["current"]
                while reference is not None:
                    if reference["relative_path"] in committed_paths:
                        fail("Cycle in evaluation result ancestry")
                    committed_paths.add(reference["relative_path"])
                    ancestor = validate_result(_read_reference(root, reference), inputs, slot)
                    reference = ancestor["previous_record"]
            # Valid failure evidence before HEAD is still visible and requires
            # explicit failed-slot retry. Never promote an uncommitted success.
            directory = _file(root, f"slots/{slot.slot_id}/records/probe").parent
            if directory.exists() and row["stored_status"] != "evaluation_completed":
                for path in sorted(directory.iterdir()):
                    if _is_link(path):
                        fail("Linked evaluation staging evidence")
                    if not path.is_file() or path.suffix != ".json" or path.relative_to(root).as_posix() in committed_paths:
                        continue
                    record = validate_result(read_json(path), inputs, slot, evaluator_provenance=evaluator_provenance)
                    if record["status"] in ("failed_numerical", "failed_execution"):
                        row["orphan_failures"].append({"relative_path": path.relative_to(root).as_posix(),
                            "sha256": digest_file(path), "status": record["status"], "failure": record["failure"]})
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        # Validation rejections are inspection conditions, never scientific
        # failures. Compatibility with a requested evaluator is explicit.
        incompatible = "Incompatible" in str(exc) or "compatible slot reuse" in str(exc)
        row.update(condition="incompatible" if incompatible else "inspection_error", diagnostic=message,
                   stored_status=None, reference=None, failure=None, orphan_failures=[])
    try:
        after = (_token(lock), _token(head), _token(owner), _token(root / ".namespace.lock"))
        if before != after:
            row.update(condition="unstable", diagnostic="Evaluation authority/locks changed during inspection",
                       stored_status=None, reference=None, failure=None, orphan_failures=[])
    except Exception as exc:
        row.update(condition="inspection_error", diagnostic=str(exc), stored_status=None,
                   reference=None, failure=None, orphan_failures=[])
    return row


def inspect_robustness_completion(root, inputs):
    head_path = root / "SWEEP_HEAD.json"
    if not head_path.exists():
        return None
    head = exact(read_json(_file(root, "SWEEP_HEAD.json")), {"evaluation_storage_version", "current"}, "sweep HEAD")
    same(head["evaluation_storage_version"], 1, "sweep storage version")
    if not head["current"]["relative_path"].startswith("sweep_completions/"):
        fail("Sweep completion must reference its immutable namespace document")
    document = exact(_read_reference(root, head["current"]), {"evaluation_completion_version", "status", "binding_sha256",
        "slots", "completed_at_utc", "publication_provenance"}, "robustness completion")
    same(document["evaluation_completion_version"], 1, "robustness completion version")
    same(document["status"], "robustness_completed", "robustness completion status")
    same(document["binding_sha256"], content_identity(inputs.binding), "robustness completion owner")
    if type(document["completed_at_utc"]) is not str or not document["completed_at_utc"]:
        fail("Robustness completion requires an actual completion timestamp")
    slots = prescribed_slots()
    if type(document["slots"]) is not list or len(document["slots"]) != len(slots):
        fail("Robustness completion requires all 28 prescribed individual slots")
    for slot, entry in zip(slots, document["slots"]):
        exact(entry, {"slot", "reference"}, "robustness completion slot")
        same(entry["slot"], slot.to_dict(), "robustness completion slot order")
        current = read_slot(root, inputs, slot)
        if current is None or current[1]["status"] != "evaluation_completed":
            fail("Robustness completion cannot include failed/missing slots")
        same(current[0]["current"], entry["reference"], "robustness completion raw result")
    from .provenance import software_provenance
    provenance = document["publication_provenance"]
    software_provenance(code=provenance["code"], software=provenance["software"], allocation=provenance.get("allocation"))
    return {"status": "robustness_completed", "reference": head["current"], "slots": len(slots)}


def collect_evaluation_coverage(inputs, *, evaluation_root=None, evaluator_provenance=None):
    from .planning_inspection import active_inspection, compatibility_key
    from .serialization import strict_json_text
    inspection = active_inspection()
    if inspection is None:
        return _collect_evaluation_coverage(inputs, evaluation_root=evaluation_root,
                                          evaluator_provenance=evaluator_provenance)
    root = evaluation_directory(inputs, evaluation_root=evaluation_root)
    key = ("evaluation_coverage", str(root), strict_json_text(inputs.binding),
           compatibility_key(evaluator_provenance))
    return inspection.reuse(key, (root,), lambda: _collect_evaluation_coverage(
        inputs, evaluation_root=evaluation_root, evaluator_provenance=evaluator_provenance))


def _collect_evaluation_coverage(inputs, *, evaluation_root=None, evaluator_provenance=None):
    root = evaluation_directory(inputs, evaluation_root=evaluation_root)
    rows = [observe_evaluation_slot(inputs, slot, evaluation_root=evaluation_root,
                                   evaluator_provenance=evaluator_provenance) for slot in prescribed_slots()]
    success = sum(row["condition"] == "ok" and row["stored_status"] == "evaluation_completed" for row in rows)
    counts = {"expected": len(rows), "successful": success,
        "failed": sum(row["condition"] == "ok" and row["stored_status"] in ("failed_numerical", "failed_execution") for row in rows),
        "interrupted": sum(row["condition"] == "ok" and row["stored_status"] == "interrupted" for row in rows),
        "unacknowledged_failures": sum(bool(row["orphan_failures"]) for row in rows)}
    counts["pending"] = counts["expected"] - counts["successful"] - counts["failed"] - counts["interrupted"]
    counts.update({condition: sum(row["condition"] == condition for row in rows) for condition in
                   ("locked", "partial", "incompatible", "unstable", "inspection_error")})
    completion, completion_error = None, None
    completion_deferred, deferred_reason = False, None
    try:
        sweep_owner = inspect_sweep_lock(root / ".sweep.lock", content_identity(inputs.binding))
        if sweep_owner is not None:
            completion_deferred, deferred_reason = True, "Compatible sweep lock owner; finalization/authority inspection deferred"
        elif any(row["condition"] in ("locked", "unstable") for row in rows):
            completion_deferred, deferred_reason = True, "Concurrent slot writer or unstable slot read; sweep authority inspection deferred"
        elif root.exists():
            before = _token(root / "SWEEP_HEAD.json")
            completion = inspect_robustness_completion(root, inputs)
            sweep_owner = inspect_sweep_lock(root / ".sweep.lock", content_identity(inputs.binding))
            if sweep_owner is not None or before != _token(root / "SWEEP_HEAD.json"):
                completion = None
                completion_deferred, deferred_reason = True, "Sweep authority/owner changed during inspection; reinspection deferred"
    except Exception as exc:
        completion = None
        completion_error = f"{type(exc).__name__}: {exc}"
    return {"evaluation_coverage_version": 1, "lifecycle": "evaluation", "binding": inputs.binding,
        "namespace": str(root), "counts": counts, "slots": rows,
        "clean_point_successful": rows[0]["condition"] == "ok" and rows[0]["stored_status"] == "evaluation_completed",
        "positive_realizations_successful": success - int(rows[0]["condition"] == "ok" and rows[0]["stored_status"] == "evaluation_completed"),
        "all_slots_successful": success == len(rows), "robustness_completed": completion is not None and success == len(rows),
        "historical_robustness_completion": completion,
        "settled_with_failures": success + counts["failed"] == len(rows) and counts["failed"] > 0,
        "completion": completion, "completion_inspection_error": completion_error,
        "completion_inspection_deferred": completion_deferred, "completion_deferred_reason": deferred_reason,
        "completion_publication_pending": success == len(rows) and completion is None}
