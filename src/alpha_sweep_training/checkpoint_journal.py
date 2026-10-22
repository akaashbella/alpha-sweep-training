"""Read-only authoritative journal validation, independent of derived CSV/status."""

from dataclasses import asdict, dataclass

from .artifacts import RunPaths
from .checkpoint_io import TRANSACTION_VERSION, ArtifactReference, exact, fail, relative_file, read_document, same
from .checkpoint_records import parse_epoch, validate_context, validate_epoch
from .checkpoints import compatible_provenance, read_checkpoint
from .identity import identity_metadata
from .metrics import BestSelection, Counters
from .spec import BATCH_SIZE, EPOCHS, TEST_SIZE
from .serialization import read_json
from .training_contracts import CheckpointReference, CommitReceipt

HEAD_PATH = "state/HEAD.json"
_STATUSES = {"initialized", "training", "interrupted", "trained_pending_test", "evaluating", "completed",
             "training_pending_completion", "training_completed", "failed_numerical", "failed_execution"}


def journal_version(value):
    if type(value) is not int or value not in (1, TRANSACTION_VERSION):
        fail("Unsupported training journal version; supported: legacy 1 and lifecycle 2")


def receipt_from_dict(value: dict) -> CommitReceipt:
    exact(value, {"identity", "commit_id", "completed_epochs", "latest", "best", "final"}, "commit receipt")
    references = {role: None if value[role] is None else CheckpointReference(**exact(value[role],
        {"role", "relative_path", "sha256", "completed_epochs"}, "checkpoint reference")) for role in ("latest", "best", "final")}
    return CommitReceipt(value["identity"], value["commit_id"], value["completed_epochs"], **references)


def transaction(paths: RunPaths, ref: ArtifactReference, config, provenance) -> dict:
    if ref.root != "metadata" or not ref.relative_path.startswith("state/transactions/"):
        fail("Transaction must reside in the metadata journal")
    value = exact(read_document(paths, ref), {"format_version", "identity", "transaction_id", "parent", "kind", "status",
        "receipt", "counters", "best", "records", "artifacts", "context", "provenance", "created_at_utc"}, "transaction")
    journal_version(value["format_version"])
    same(value["identity"], identity_metadata(config), "transaction identity")
    if ref.relative_path != f"state/transactions/{value['transaction_id']}.json":
        fail("Transaction filename/ID mismatch")
    if type(value["transaction_id"]) is not str or len(value["transaction_id"]) != 32 or any(c not in "0123456789abcdef" for c in value["transaction_id"]):
        fail("Invalid immutable transaction ID")
    if value["kind"] not in ("initial", "epoch", "evaluation", "completion", "training_completion", "failure") or value["status"] not in _STATUSES:
        fail("Unknown transaction kind/status")
    if value["format_version"] == 1 and (value["kind"] == "training_completion" or value["status"] in ("training_completed", "training_pending_completion")):
        fail("Legacy journal cannot claim the new training-only lifecycle")
    if value["format_version"] == 2 and (value["kind"] in ("evaluation", "completion") or value["status"] in ("completed", "evaluating", "trained_pending_test")):
        fail("Journal v2 separates training from independent evaluation")
    compatible_provenance(value["provenance"], provenance)
    validate_context(value["context"], config, provenance)
    counters = Counters(**exact(value["counters"], set(asdict(Counters())), "transaction counters"))
    best = None if value["best"] is None else BestSelection(**value["best"])
    receipt_from_dict(value["receipt"]).validate(config, counters, best)
    if value["kind"] == "initial":
        if counters != Counters() or value["parent"] is not None or value["status"] != "initialized":
            fail("Invalid initial transaction")
    elif value["parent"] is None:
        fail("Only initial transactions may lack committed ancestry")
    if value["kind"] in ("initial", "epoch"):
        same(value["receipt"]["commit_id"], value["transaction_id"], "training transaction ID")
    if value["kind"] == "epoch":
        pending = "trained_pending_test" if value["format_version"] == 1 else "training_pending_completion"
        same(value["status"], pending if counters.completed_epochs == EPOCHS else "training", "epoch transaction status")
        if counters.completed_epochs == 0:
            fail("An epoch transaction cannot be epoch zero")
    if value["kind"] == "failure" and value["status"] not in ("interrupted", "failed_numerical", "failed_execution"):
        fail("Invalid failure transaction status")
    if value["kind"] == "evaluation" and (value["status"] != "evaluating" or counters.completed_epochs != EPOCHS):
        fail("Evaluation transaction requires full committed training and evaluating status")
    return value


@dataclass(frozen=True)
class CommittedState:
    head: dict
    reference: ArtifactReference
    document: dict
    receipt: CommitReceipt
    latest: dict
    records: tuple
    best_metadata: dict | None = None

    @property
    def status(self) -> str:
        return self.document["status"]


def read_committed(paths: RunPaths, config, provenance, *, verify_previous: bool = True,
                   metadata_only: bool = False) -> CommittedState:
    """Validate full authority; planning may retain only compact, tensor-free evidence.

    Runtime/persistence callers always take the uncached full-state path. Planning
    reuse is operation-local and invalidates on any storage/ownership mutation.
    """
    def validate():
        result = _read_committed(paths, config, provenance, verify_previous=verify_previous,
                                 reuse_checkpoints=inspection is not None)
        if not metadata_only:
            return result
        return CommittedState(result.head, result.reference, result.document, result.receipt,
                              {"metadata": result.latest["metadata"]}, (), result.best_metadata)

    from .planning_inspection import active_inspection, training_paths
    from .serialization import strict_json_text
    inspection = active_inspection() if metadata_only else None
    if inspection is None:
        return validate()
    key = ("committed", strict_json_text(config.to_dict()), strict_json_text(provenance), verify_previous)
    return inspection.reuse(key, training_paths(paths), validate, track_dependencies=True)


def _read_committed(paths: RunPaths, config, provenance, *, verify_previous: bool = True,
                    reuse_checkpoints: bool = False) -> CommittedState:
    # Current and previous transactions often reference identical immutable
    # checkpoints. Reuse within this one validated read, release tensors before
    # returning compact evidence, and never enable this for runtime writers.
    from .planning_inspection import PlanningInspection
    checkpoints = PlanningInspection() if reuse_checkpoints else None
    path = relative_file(paths.metadata, HEAD_PATH)
    if not path.is_file():
        fail("No durable initial boundary: preserve this partially initialized run; explicit recovery/archived overwrite is required")
    head = exact(read_json(path), {"format_version", "identity", "current", "previous"}, "HEAD")
    journal_version(head["format_version"])
    same(head["identity"], identity_metadata(config), "HEAD identity")
    ref = ArtifactReference.from_dict(head["current"])
    document = transaction(paths, ref, config, provenance)
    same(head["format_version"], document["format_version"], "HEAD/current transaction version")
    same(document["parent"], head["previous"], "HEAD previous generation")
    result = validate_committed_document(paths, head, ref, document, config, provenance, checkpoint_cache=checkpoints)
    if verify_previous and head["previous"] is not None:
        previous_ref = ArtifactReference.from_dict(head["previous"])
        previous = transaction(paths, previous_ref, config, provenance)
        validate_committed_document(paths, {"previous": previous["parent"]}, previous_ref, previous, config, provenance,
                                    checkpoint_cache=checkpoints)
        if document["kind"] == "epoch":
            same(document["counters"]["completed_epochs"], previous["counters"]["completed_epochs"] + 1, "committed epoch ancestry")
            same(document["records"][:-1], previous["records"], "epoch record ancestry")
            same(result.latest["continuation"]["artifacts"]["previous_commit_id"], previous["receipt"]["commit_id"], "latest previous training boundary")
        else:
            for key in ("receipt", "counters", "best", "records"):
                same(document[key], previous[key], f"non-training transaction {key}")
    return result


def validate_committed_document(paths, head, ref, document, config, provenance, *, checkpoint_cache=None) -> CommittedState:
    counters = Counters(**exact(document["counters"], set(asdict(Counters())), "committed counters"))
    best = None if document["best"] is None else BestSelection(**exact(document["best"], {"epoch_index", "correct", "examples"}, "committed best"))
    receipt = receipt_from_dict(document["receipt"])
    receipt.validate(config, counters, best)
    payloads = {}
    for role in ("latest", "best", "final"):
        checkpoint = getattr(receipt, role)
        if checkpoint is not None:
            if checkpoint_cache is None:
                payloads[role] = read_checkpoint(paths, checkpoint, config, provenance)
            else:
                from .planning_inspection import compatibility_key
                from .serialization import strict_json_text
                path = relative_file(paths.checkpoints, checkpoint.relative_path)
                key = ("checkpoint", checkpoint, strict_json_text(config.to_dict()), compatibility_key(provenance))
                payloads[role] = checkpoint_cache.reuse(key, (path,),
                    lambda: read_checkpoint(paths, checkpoint, config, provenance), track_dependencies=True)
            if checkpoint.relative_path != f"generations/{payloads[role]['metadata']['generation']}.{role}.pt":
                fail("Role checkpoint reference is not its immutable generation")
    latest = payloads["latest"]
    same(latest["metadata"]["generation"], receipt.commit_id, "latest generation")
    same(latest["metadata"]["counters"], asdict(counters), "committed latest counters")
    same(latest["metadata"]["best"], document["best"], "committed latest selection")
    if "best" in payloads:
        same(payloads["best"]["metadata"]["best"], document["best"], "persisted best selection")
    if "final" in payloads:
        same(payloads["final"]["metadata"]["generation"], receipt.commit_id, "final generation")
    if type(document["records"]) is not list or len(document["records"]) != counters.completed_epochs:
        fail("Committed epoch records do not cover the completed boundary")
    same(latest["continuation"]["artifacts"]["records"], document["records"], "latest committed record references")
    records, previous, previous_best = [], Counters(), None
    for record_ref in document["records"]:
        reference = ArtifactReference.from_dict(record_ref)
        if reference.root != "results" or not reference.relative_path.startswith("records/"):
            fail("Epoch records must reside beneath results/records")
        parsed = parse_epoch(read_document(paths, reference), config, provenance)
        _, record, statistics = parsed
        validate_epoch(record, statistics, previous=previous, previous_best=previous_best, config=config,
                       components=latest["metadata"]["component_metadata"])
        previous, previous_best = record.totals, record.best
        records.append(parsed)
    same(asdict(previous), asdict(counters), "committed record counters")
    same(None if previous_best is None else asdict(previous_best), document["best"], "validation-selected best")
    artifacts = exact(document["artifacts"], {"completion", "failure"}, "transaction artifacts")
    if document["status"] == "training_completed":
        if counters.completed_epochs != EPOCHS or document["kind"] != "training_completion" or artifacts["completion"] is None:
            fail("Invalid training-only completion transaction")
        validate_training_completion(paths, document, receipt, config, provenance, tuple(records))
    elif document["status"] == "completed":
        if counters.completed_epochs != EPOCHS or document["kind"] != "completion" or artifacts["completion"] is None:
            fail("Invalid completion transaction")
        validate_completion(paths, document, receipt, config, provenance, tuple(records))
    elif artifacts["completion"] is not None or document["kind"] in ("completion", "training_completion"):
        fail("Completion artifact/status mismatch")
    if artifacts["failure"] is not None:
        failure = read_document(paths, ArtifactReference.from_dict(artifacts["failure"]))
        same(failure["kind"], "failure", "failure record kind")
        context = {key: value for key, value in failure.items() if key not in ("kind", "recorded_at_utc", "payload")}
        validate_context(context, config, provenance)
        if failure["payload"]["status"] != document["status"]:
            fail("Failure status disagrees with its transaction")
        validate_failure_payload(failure["payload"], counters, receipt)
    elif document["kind"] == "failure":
        fail("Missing durable failure record")
    return CommittedState(head, ref, document, receipt, latest, tuple(records),
                          None if "best" not in payloads else payloads["best"]["metadata"])


def validate_training_completion(paths, document, receipt, config, provenance, records):
    """No test fields: validate all immutable training artifacts and selection."""
    refs = exact(document["artifacts"]["completion"], {"summary"}, "training completion references")
    summary_ref = ArtifactReference.from_dict(refs["summary"])
    if summary_ref.root != "results" or not summary_ref.relative_path.startswith("training_completions/"):
        fail("Training completion summary must be immutable in training_completions")
    summary = read_document(paths, summary_ref)
    expected_keys = set(document["context"]) | {"training_completion_version", "status", "completed_epochs",
        "scientific", "checkpoints", "best_validation", "best_validation_accuracy", "accuracy_unit",
        "metric_artifacts", "epoch_records", "completion_reason", "completed_at_utc", "attempt_history",
        "transaction_id", "final_commit_id", "training_alpha", "canonical_training_alpha", "completion_provenance"}
    exact(summary, expected_keys, "training-only summary")
    for key, value in document["context"].items():
        same(summary[key], value, f"training completion context {key}")
    same(summary["training_completion_version"], 1, "training completion format")
    same(summary["status"], "training_completed", "training completion status")
    same(summary["completed_epochs"], EPOCHS, "training completion budget")
    same(summary["scientific"], config.scientific, "training science")
    same(summary["transaction_id"], document["transaction_id"], "training transaction")
    same(summary["final_commit_id"], receipt.commit_id, "training final commit")
    same(summary["checkpoints"], {role: asdict(getattr(receipt, role)) for role in ("latest", "best", "final")}, "training checkpoint roles")
    same(summary["epoch_records"], document["records"], "training records")
    same(summary["best_validation"], document["best"], "training validation selection")
    same(summary["best_validation_accuracy"], document["best"]["correct"] / document["best"]["examples"], "training validation accuracy")
    from .identity import canonical_alpha
    same(summary["training_alpha"], config.scientific["alpha"], "explicit training alpha")
    same(summary["canonical_training_alpha"], canonical_alpha(config.scientific["alpha"]), "explicit canonical training alpha")
    same(summary["accuracy_unit"], "fraction", "training accuracy units")
    same(summary["completion_reason"], "full_original_training_budget", "training completion reason")
    from .provenance import software_provenance
    writer = summary["completion_provenance"]
    software_provenance(code=writer["code"], software=writer["software"], allocation=writer.get("allocation"))
    if type(summary["completed_at_utc"]) is not str or not summary["completed_at_utc"]:
        fail("Training completion requires creation time")
    from .persistence import csv_views
    from .planning_inspection import active_inspection
    inspection = active_inspection()
    if inspection is not None:
        inspection.progress("Verifying immutable training-completion CSV bytes against validated epoch/layer records")
    epochs, layers = csv_views(records)
    exact(summary["metric_artifacts"], {"epochs_csv", "layer_statistics_csv"}, "training CSV references")
    for name, expected in (("epochs_csv", epochs), ("layer_statistics_csv", layers)):
        ref = ArtifactReference.from_dict(summary["metric_artifacts"][name])
        if ref.root != "results" or not ref.relative_path.startswith("training_completions/"):
            fail("Training completion CSV must be immutable in training_completions")
        same(ref.verify(paths).read_bytes().decode("utf-8"), expected, f"training {name}")


def validate_failure_payload(payload, counters, receipt) -> None:
    from dataclasses import fields
    from .metrics import FailureRecord
    exact(payload, {field.name for field in fields(FailureRecord)} | {"authoritative_completed_epochs", "authoritative_commit_id", "replay_required"}, "failure payload")
    if payload["status"] not in ("interrupted", "failed_execution", "failed_numerical"):
        fail("Invalid terminal/interruption record status")
    observed = Counters(**payload["observed_counters"])
    observed.validate(complete_boundary=False)
    same(payload["authoritative_completed_epochs"], counters.completed_epochs, "failure authoritative count")
    same(payload["authoritative_commit_id"], receipt.commit_id, "failure authoritative receipt")
    if type(payload["durable_completed_epochs"]) is not int or not 0 <= payload["durable_completed_epochs"] <= counters.completed_epochs:
        fail("Failure acknowledged progress exceeds authoritative progress")
    for key in ("training_loss", "validation_loss", "test_loss", "training_accuracy", "validation_accuracy", "test_accuracy"):
        same(payload[key], None, f"unavailable failure {key}")


def numerical_failure_evidence(paths, config, provenance) -> tuple[ArtifactReference, ...]:
    """Even an unacknowledged numerical failure is not a wall-time retry.

    Scan only small reserved failure records, preserving them all. A valid HEAD
    establishes checkpoint progress; this evidence establishes a terminal stop.
    Explicit recovery reconciles a selected orphan before further publication.
    """
    return _failure_evidence(paths, config, provenance, status="failed_numerical")


def execution_failure_evidence(paths, config, provenance, *, training_commit_id: str) -> tuple[ArtifactReference, ...]:
    """Unacknowledged execution failures at this latest boundary need a choice.

    Historical failures before subsequent durable training progress remain audit
    records; they do not permanently block later allocations. Completion wins.
    """
    return _failure_evidence(paths, config, provenance, status="failed_execution",
                             training_commit_id=training_commit_id)


def _failure_evidence(paths, config, provenance, *, status, training_commit_id=None):
    from dataclasses import fields
    from .metrics import FailureRecord
    from .checkpoint_io import digest_file
    from .artifacts import _is_link
    directory = relative_file(paths.results, "failures/probe").parent
    if not directory.exists():
        return ()
    result = []
    for path in sorted(directory.iterdir()):
        if _is_link(path):
            fail("Linked failure evidence requires explicit recovery")
        if not path.is_file() or path.suffix != ".json":
            continue
        record = read_json(path)
        context = {key: value for key, value in record.items() if key not in ("kind", "recorded_at_utc", "payload")}
        validate_context(context, config, provenance)
        same(record["kind"], "failure", "failure evidence kind")
        payload = exact(record["payload"], {field.name for field in fields(FailureRecord)} |
            {"authoritative_completed_epochs", "authoritative_commit_id", "replay_required"}, "failure evidence payload")
        Counters(**payload["observed_counters"]).validate(complete_boundary=False)
        if payload["status"] not in ("failed_numerical", "failed_execution", "interrupted"):
            fail("Unknown failure evidence status")
        count = payload["authoritative_completed_epochs"]
        if type(count) is not int or not 0 <= count <= EPOCHS or type(payload["replay_required"]) is not bool:
            fail("Invalid failure evidence authority")
        commit_id = payload["authoritative_commit_id"]
        if commit_id is None:
            if count != 0:
                fail("Uninitialized failure evidence cannot claim committed epochs")
        elif (type(commit_id) is not str or len(commit_id) != 32
                or any(c not in "0123456789abcdef" for c in commit_id)):
            fail("Invalid failure evidence training commit ID")
        for key in ("phase", "reason", "exception_type"):
            if type(payload[key]) is not str:
                fail(f"Invalid failure evidence {key}")
        if (type(payload["durable_completed_epochs"]) is not int
                or not 0 <= payload["durable_completed_epochs"] <= count):
            fail("Failure evidence acknowledged progress exceeds authority")
        if (payload["status"] == status and (training_commit_id is None
                or payload["authoritative_commit_id"] == training_commit_id)):
            result.append(ArtifactReference("results", path.relative_to(paths.results).as_posix(), digest_file(path)))
    return tuple(result)


def validate_completion(paths, document, receipt, config, provenance, records) -> None:
    from .checkpoint_records import classification
    from dataclasses import fields
    from .metrics import TestRecord
    import math
    references = exact(document["artifacts"]["completion"], {"test", "summary"}, "completion references")
    test = read_document(paths, ArtifactReference.from_dict(references["test"]))
    summary = read_document(paths, ArtifactReference.from_dict(references["summary"]))
    for value, kind in ((test, "test"), (summary, "summary")):
        if kind == "test":
            context = {key: child for key, child in value.items() if key not in ("kind", "recorded_at_utc", "payload")}
            validate_context(context, config, provenance)
            same(value["kind"], "test", "test record kind")
        else:
            for key, child in identity_metadata(config).items():
                same(value[key], child, f"summary {key}")
    payload = test["payload"]
    exact(payload, {field.name for field in fields(TestRecord)} | {"clean_weights", "checkpoint_role", "completion_pending_publication"}, "test payload")
    if type(payload["duration_seconds"]) not in (int, float) or not math.isfinite(payload["duration_seconds"]) or payload["duration_seconds"] < 0:
        fail("Invalid test duration")
    metrics = classification(payload["metrics"])
    if metrics.examples != TEST_SIZE or metrics.batches != (TEST_SIZE + BATCH_SIZE - 1) // BATCH_SIZE:
        fail("Completion requires all 10,000 official test examples")
    same(payload["clean_weights"], True, "clean test marker")
    same(payload["checkpoint_role"], "best", "test role")
    same(payload["best_checkpoint"], asdict(receipt.best), "test best reference")
    same(payload["best"], document["best"], "test selection")
    same(payload["final_commit_id"], receipt.commit_id, "test final commit")
    same(payload["completed_epochs"], EPOCHS, "test budget")
    same(payload["status"], "evaluating", "test record status")
    same(payload["completion_pending_publication"], True, "pre-publication test marker")
    same(summary["status"], "completed", "summary status")
    same(summary["completed_epochs"], EPOCHS, "summary budget")
    same(summary["scientific"], config.scientific, "summary scientific configuration")
    same(summary["transaction_id"], document["transaction_id"], "summary transaction")
    same(summary["final_commit_id"], receipt.commit_id, "summary final commit")
    compatible_provenance(summary["provenance"], provenance)
    same(summary["checkpoints"], {role: asdict(getattr(receipt, role)) for role in ("latest", "best", "final")}, "summary checkpoint references")
    same(summary["clean_test"], payload, "summary clean test")
    same(summary["epoch_records"], document["records"], "summary committed records")
    same(summary["best_validation"], document["best"], "summary validation selection")
    same(summary["best_validation_accuracy"], document["best"]["correct"] / document["best"]["examples"], "summary validation accuracy")
    same(summary["test_record"], references["test"], "summary test reference")
    from .persistence import csv_views
    epochs, layers = csv_views(records)
    exact(summary["metric_artifacts"], {"epochs_csv", "layer_statistics_csv"}, "completion CSV references")
    for name, expected in (("epochs_csv", epochs), ("layer_statistics_csv", layers)):
        reference = ArtifactReference.from_dict(summary["metric_artifacts"][name])
        if reference.root != "results" or not reference.relative_path.startswith("completions/"):
            fail("Completion CSV must be immutable content in results/completions")
        same(reference.verify(paths).read_bytes().decode("utf-8"), expected, f"committed {name} content")
