"""Durable training service: immutable generations, one HEAD commit, derived views."""

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import uuid
import re

from .artifacts import RunHandle
from .checkpoint_io import (
    TRANSACTION_VERSION, ArtifactReference, fail, fsync_directory, read_document,
    relative_file, same, write_document, write_text,
)
from .checkpoint_journal import (
    HEAD_PATH, numerical_failure_evidence, read_committed, transaction,
    validate_committed_document, validate_failure_payload,
)
from .checkpoint_records import classification, validate_context, validate_epoch
from .checkpoints import copy_clean_model, read_checkpoint, write_checkpoint
from .identity import identity_metadata
from .metrics import Counters, RecordContext, epoch_csv, layer_statistics_csv
from .metrics import BestSelection, FailureRecord, TestRecord
from .training_contracts import CheckpointReference
from torch import nn
from .serialization import atomic_write_json, atomic_write_text, read_json
from .spec import BATCH_SIZE, EPOCHS, TEST_SIZE
from .training_contracts import (
    BestLoadReceipt, CommitReceipt, CompletionReceipt, EpochBoundary, InitialBoundary,
)


def csv_views(records: tuple) -> tuple[str, str]:
    epochs, layers = [], []
    for index, (context, record, statistics) in enumerate(records):
        epochs.append(epoch_csv(context, record, include_header=index == 0))
        layers.append(layer_statistics_csv(context, record, statistics, include_header=index == 0))
    return "".join(epochs), "".join(layers)


class DurableTrainingPersistence:
    """One bound active run; callbacks are synchronous and never consume RNGs.

    HEAD is authoritative across all storage roots. A failure after replacing
    HEAD has an unknown acknowledgement outcome, resolved by reading HEAD, never
    by counting CSV rows. Immutable audit records outlive retired tensors.
    """
    def __init__(self, run: RunHandle):
        run.require_active()
        self.run = run
        self._loaded_best = None

    def _guard(self, run: RunHandle) -> None:
        if run is not self.run:
            fail("Persistence service is bound to a different run lease")
        run.require_active()

    def _current(self):
        self._guard(self.run)
        return read_committed(self.run.paths, self.run.config, self.run.provenance)

    def _document(self, generation, parent, kind, status, receipt, counters, best, records, artifacts, context):
        validate_context(context.identity, self.run.config, self.run.provenance)
        return {"format_version": TRANSACTION_VERSION, "identity": identity_metadata(self.run.config),
                "transaction_id": generation, "parent": None if parent is None else parent.reference.to_dict(),
                "kind": kind, "status": status, "receipt": asdict(receipt), "counters": asdict(counters),
                "best": None if best is None else asdict(best), "records": deepcopy(records), "artifacts": artifacts,
                "context": deepcopy(context.identity), "provenance": deepcopy(self.run.provenance),
                "created_at_utc": datetime.now(timezone.utc).isoformat()}

    def _commit(self, document, parent):
        self._guard(self.run)
        # Recheck the exact parent immediately before commit. Run locks exclude
        # cooperating writers, but an escaped/outdated service must still fail.
        if parent is None:
            if relative_file(self.run.paths.metadata, HEAD_PATH).exists():
                fail("Initial boundary already exists; use explicit continuation")
        else:
            same(read_json(relative_file(self.run.paths.metadata, HEAD_PATH)), parent.head, "transaction parent HEAD")
        ref = write_document(self.run, "metadata", f"state/transactions/{document['transaction_id']}.json", document)
        head = {"format_version": TRANSACTION_VERSION, "identity": identity_metadata(self.run.config),
                "current": ref.to_dict(), "previous": document["parent"]}
        committed = validate_committed_document(self.run.paths, head, ref, document, self.run.config, self.run.provenance)
        self._guard(self.run)
        # This same-filesystem replacement is the SINGLE authoritative point.
        atomic_write_json(relative_file(self.run.paths.metadata, HEAD_PATH, create_parents=True), head)
        self._publish_views(committed)
        self._prune(committed)
        return committed

    def commit_initial(self, run: RunHandle, boundary: InitialBoundary) -> CommitReceipt:
        self._guard(run)
        if run.operation not in ("new", "overwrite"):
            fail("Initial commit is only for a fresh new/archived-overwrite run")
        if relative_file(run.paths.metadata, HEAD_PATH).exists():
            fail("Initial boundary already exists; never replace it with fresh initialization")
        snapshot = boundary.snapshot
        same(asdict(snapshot.counters), asdict(Counters()), "fresh initial counters")
        if snapshot.best is not None:
            fail("Initial state has no best selection")
        generation = uuid.uuid4().hex
        latest = write_checkpoint(run, snapshot, "latest", generation, {"records": [], "previous_commit_id": None})
        receipt = CommitReceipt(identity_metadata(run.config), generation, 0, latest, None, None)
        document = self._document(generation, None, "initial", "initialized", receipt, snapshot.counters, None,
                                  [], {"completion": None, "failure": None}, boundary.context)
        return self._commit(document, None).receipt

    def commit_epoch(self, run: RunHandle, boundary: EpochBoundary) -> CommitReceipt:
        self._guard(run)
        current = self._current()
        if current.status in ("completed", "failed_numerical"):
            fail("Completed/numerically failed state cannot receive more training commits")
        same(boundary.previous_commit_id, current.receipt.commit_id, "previous training commit")
        counters = Counters(**current.document["counters"])
        from .metrics import BestSelection
        best = None if current.document["best"] is None else BestSelection(**current.document["best"])
        validate_epoch(boundary.record, boundary.statistics, previous=counters, previous_best=best,
                       config=run.config, components=boundary.snapshot.component_metadata)
        same(asdict(boundary.snapshot.counters), asdict(boundary.record.totals), "snapshot/record counters")
        same(asdict(boundary.snapshot.best), asdict(boundary.record.best), "snapshot/record selection")
        same(boundary.snapshot.component_metadata["data"], {**current.latest["metadata"]["component_metadata"]["data"],
             **{key: value for key, value in boundary.snapshot.component_metadata["data"].items()
                if key in ("num_workers", "pin_memory", "persistent_workers", "prefetch_factor")}}, "epoch loader metadata")
        generation = uuid.uuid4().hex
        record_ref = write_document(run, "results", f"records/{generation}.json", {
            "context": boundary.context.identity, "record": boundary.record.to_dict(),
            "statistics": [asdict(value) for value in boundary.statistics]})
        records = current.document["records"] + [record_ref.to_dict()]
        latest = write_checkpoint(run, boundary.snapshot, "latest", generation,
                                  {"records": records, "previous_commit_id": current.receipt.commit_id})
        selected = write_checkpoint(run, boundary.snapshot, "best", generation, {}) if boundary.save_best else current.receipt.best
        final = write_checkpoint(run, boundary.snapshot, "final", generation, {}) if boundary.save_final else None
        receipt = CommitReceipt(identity_metadata(run.config), generation, boundary.record.completed_epochs, latest, selected, final)
        document = self._document(generation, current, "epoch", boundary.record.status, receipt,
                                  boundary.snapshot.counters, boundary.snapshot.best, records,
                                  {"completion": None, "failure": None}, boundary.context)
        return self._commit(document, current).receipt

    def load_best(self, run: RunHandle, model: nn.Module, checkpoint: CheckpointReference,
                  selection: BestSelection) -> BestLoadReceipt:
        self._guard(run)
        current = self._current()
        if current.receipt.completed_epochs != EPOCHS or current.status in ("completed", "failed_numerical"):
            fail("Best test loading requires full committed training with evaluation pending")
        same(asdict(checkpoint), asdict(current.receipt.best), "selected persisted best reference")
        same(asdict(selection), current.document["best"], "selected persisted best metadata")
        payload = read_checkpoint(run.paths, checkpoint, run.config, run.provenance)
        copy_clean_model(model, payload)
        context = RecordContext({**current.document["context"], "attempt_id": run.attempt_id,
                                 "provenance": deepcopy(run.provenance)})
        document = self._document(uuid.uuid4().hex, current, "evaluation", "evaluating", current.receipt,
            Counters(**current.document["counters"]), selection, current.document["records"],
            {"completion": None, "failure": None}, context)
        self._commit(document, current)
        self._loaded_best = checkpoint
        return BestLoadReceipt(identity_metadata(run.config), checkpoint)

    def publish_completion(self, run: RunHandle, context: RecordContext, test: TestRecord,
                           final_commit: CommitReceipt) -> CompletionReceipt:
        self._guard(run)
        current = self._current()
        if current.status == "completed":
            # Idempotent reconciliation after lost acknowledgement. Never rerun
            # evaluation just to recreate a previously committed completion.
            same(asdict(final_commit), asdict(current.receipt), "already completed final boundary")
            self._publish_views(current)
            ref = current.document["artifacts"]["completion"]["summary"]
            return CompletionReceipt(identity_metadata(run.config), final_commit.commit_id, ref["sha256"])
        if current.status == "failed_numerical":
            fail("Numerical failure is terminal; completion is not an automatic retry")
        if self._loaded_best != current.receipt.best:
            fail("Completion must follow this service's verified persisted-best load for clean test")
        same(asdict(final_commit), asdict(current.receipt), "final completion boundary")
        if current.receipt.completed_epochs != EPOCHS or len(current.records) != EPOCHS:
            fail("Completion requires all original epochs and committed statistics")
        payload = test.to_dict()
        if (test.metrics.examples != TEST_SIZE or test.metrics.batches != (TEST_SIZE + BATCH_SIZE - 1) // BATCH_SIZE
                or test.completed_epochs != EPOCHS or test.status != "evaluating"
                or type(test.duration_seconds) not in (int, float) or test.duration_seconds < 0):
            fail("Invalid full-budget clean test")
        classification(payload["metrics"])
        same(payload["best_checkpoint"], asdict(current.receipt.best), "clean test checkpoint")
        same(asdict(test.best), current.document["best"], "clean test selection")
        same(test.final_commit_id, current.receipt.commit_id, "clean test final commit")
        generation = uuid.uuid4().hex
        test_ref = write_document(run, "results", f"completions/{generation}.test.json", context.envelope("test", payload))
        epochs, layers = csv_views(current.records)
        metric_refs = {"epochs_csv": write_text(run, "results", f"completions/{generation}.epochs.csv", epochs).to_dict(),
                       "layer_statistics_csv": write_text(run, "results", f"completions/{generation}.layers.csv", layers).to_dict()}
        best = current.document["best"]
        summary = {**deepcopy(context.identity), "status": "completed", "completed_epochs": EPOCHS,
                   "scientific": run.config.scientific, "checkpoints": {role: asdict(getattr(current.receipt, role))
                       for role in ("latest", "best", "final")}, "best_validation": best,
                   "best_validation_accuracy": best["correct"] / best["examples"], "accuracy_unit": "fraction",
                   "clean_test": payload, "test_record": test_ref.to_dict(), "metric_artifacts": metric_refs,
                   "epoch_records": current.document["records"], "completion_reason": "full_original_budget_and_clean_best_test",
                   "evaluated_at_utc": datetime.now(timezone.utc).isoformat(), "attempt_history": "attempts",
                   "transaction_id": generation, "final_commit_id": current.receipt.commit_id}
        summary_ref = write_document(run, "results", f"completions/{generation}.summary.json", summary)
        from .metrics import BestSelection
        document = self._document(generation, current, "completion", "completed", current.receipt,
            Counters(**current.document["counters"]), BestSelection(**best), current.document["records"],
            {"completion": {"test": test_ref.to_dict(), "summary": summary_ref.to_dict()}, "failure": None}, context)
        self._commit(document, current)
        return CompletionReceipt(identity_metadata(run.config), current.receipt.commit_id, summary_ref.sha256)

    def publish_failure(self, run: RunHandle, context: RecordContext, failure: FailureRecord) -> None:
        self._guard(run)
        if failure.status not in ("interrupted", "failed_numerical", "failed_execution"):
            fail("Unknown failure/interruption status")
        failure.observed_counters.validate(complete_boundary=False)
        # Read authoritative outcome FIRST. A completed transaction dominates
        # a caller that did not receive its receipt or whose diagnostics failed.
        path = relative_file(run.paths.metadata, HEAD_PATH)
        current = self._current() if path.exists() else None
        if current is not None and current.status == "completed":
            self._publish_views(current)
            return
        if current is not None and current.status == "failed_numerical" and failure.status != "failed_numerical":
            fail("An execution/interruption record cannot downgrade terminal numerical failure")
        generation = uuid.uuid4().hex
        record_ref = write_document(run, "results", f"failures/{generation}.json", context.envelope("failure", {
            **asdict(failure), "authoritative_completed_epochs": 0 if current is None else current.receipt.completed_epochs,
            "authoritative_commit_id": None if current is None else current.receipt.commit_id,
            "replay_required": current is not None and failure.observed_counters != Counters(**current.document["counters"])}))
        if current is None:
            # Preserve failed epoch-zero staging evidence; no fake latest/HEAD.
            atomic_write_json(relative_file(run.paths.metadata, "status.json"), context.envelope("status", {
                "status": failure.status, "durable_initial_boundary": False, "failure": record_ref.to_dict()}))
            return
        from .metrics import BestSelection
        best = None if current.document["best"] is None else BestSelection(**current.document["best"])
        document = self._document(generation, current, "failure", failure.status, current.receipt,
            Counters(**current.document["counters"]), best, current.document["records"],
            {"completion": None, "failure": record_ref.to_dict()}, context)
        self._commit(document, current)

    def repair_publication(self) -> None:
        """Explicit recovery of committed views; never reconstruct a missing HEAD."""
        if self.run.operation != "recover":
            fail("Publication repair requires explicitly selected operation='recover'")
        current = self._current()
        if current.status not in ("completed", "failed_numerical") and numerical_failure_evidence(
                self.run.paths, self.run.config, self.run.provenance):
            fail("Reconcile terminal numerical failure evidence before repairing nonterminal publication")
        self._publish_views(current)
        self._prune(current)

    def reconcile_numerical_failure(self, reference: ArtifactReference) -> None:
        """Explicitly select and commit terminal failure evidence after lost ack."""
        if self.run.operation != "recover":
            fail("Numerical evidence reconciliation requires explicitly selected recovery")
        current = self._current()
        if current.status == "completed":
            self._publish_views(current)
            return
        if reference not in numerical_failure_evidence(self.run.paths, self.run.config, self.run.provenance):
            fail("Select an actual digest-validated numerical failure evidence reference")
        record = read_document(self.run.paths, reference)
        counters = Counters(**current.document["counters"])
        validate_failure_payload(record["payload"], counters, current.receipt)
        from .metrics import BestSelection
        best = None if current.document["best"] is None else BestSelection(**current.document["best"])
        context = RecordContext({key: value for key, value in record.items() if key not in ("kind", "recorded_at_utc", "payload")})
        document = self._document(uuid.uuid4().hex, current, "failure", "failed_numerical", current.receipt,
                                  counters, best, current.document["records"],
                                  {"completion": None, "failure": reference.to_dict()}, context)
        self._commit(document, current)

    def _publish_views(self, current) -> None:
        self._guard(self.run)
        epochs, layers = csv_views(current.records)
        atomic_write_text(relative_file(self.run.paths.results, "epochs.csv"), epochs)
        atomic_write_text(relative_file(self.run.paths.results, "layer_statistics.csv"), layers)
        document = current.document
        context = RecordContext(document["context"])
        atomic_write_json(relative_file(self.run.paths.metadata, "status.json"), context.envelope("status", {
            "status": document["status"], "completed_epochs": current.receipt.completed_epochs,
            "commit_id": current.receipt.commit_id, "transaction": current.reference.to_dict(), "artifacts": document["artifacts"]}))
        metadata_path = relative_file(self.run.paths.metadata, "run_metadata.json")
        metadata = read_json(metadata_path)
        for key, value in identity_metadata(self.run.config).items():
            same(metadata[key], value, f"metadata {key}")
        atomic_write_json(metadata_path, {**metadata, "status": document["status"],
            "completed_epochs": current.receipt.completed_epochs, "commit_id": current.receipt.commit_id,
            "transaction": current.reference.to_dict()})
        if document["artifacts"]["completion"] is not None:
            refs = document["artifacts"]["completion"]
            atomic_write_json(relative_file(self.run.paths.results, "test.json"), read_document(self.run.paths, ArtifactReference.from_dict(refs["test"])))
            atomic_write_json(relative_file(self.run.paths.results, "summary.json"), read_document(self.run.paths, ArtifactReference.from_dict(refs["summary"])))
        elif document["artifacts"]["failure"] is not None:
            record = read_document(self.run.paths, ArtifactReference.from_dict(document["artifacts"]["failure"]))
            atomic_write_json(relative_file(self.run.paths.results, "failure.json"), record)
            atomic_write_json(relative_file(self.run.paths.results, "summary.json"), {**deepcopy(context.identity),
                "status": document["status"], "completed_epochs": current.receipt.completed_epochs,
                "checkpoints": asdict(current.receipt), "clean_test": None, "failure": record["payload"]})
        else:
            atomic_write_json(relative_file(self.run.paths.results, "summary.json"), {**deepcopy(context.identity),
                "status": document["status"], "completed_epochs": current.receipt.completed_epochs,
                "checkpoints": asdict(current.receipt), "clean_test": None, "transaction": current.reference.to_dict()})

    def _prune(self, current) -> None:
        """Only proven committed ancestors, never orphans/staging/unrelated paths."""
        self._guard(self.run)
        protected = set()
        for reference in (current.reference.to_dict(), current.head["previous"]):
            if reference is None:
                continue
            doc = transaction(self.run.paths, ArtifactReference.from_dict(reference), self.run.config, self.run.provenance)
            for role in ("latest", "best", "final"):
                checkpoint = doc["receipt"][role]
                if checkpoint is not None:
                    protected.add(checkpoint["relative_path"])
        reference, seen = current.reference, set()
        while reference is not None:
            if reference.relative_path in seen:
                fail("Cycle in committed transaction ancestry")
            seen.add(reference.relative_path)
            doc = transaction(self.run.paths, reference, self.run.config, self.run.provenance)
            for role in ("latest", "best", "final"):
                value = doc["receipt"][role]
                if value is None or value["relative_path"] in protected:
                    continue
                from .training_contracts import CheckpointReference
                checkpoint = CheckpointReference(**value)
                checkpoint.validate(role, checkpoint.completed_epochs)
                if not re.fullmatch(r"generations/[0-9a-f]{32}\." + role + r"\.pt", checkpoint.relative_path):
                    fail("Refusing to prune a non-generation path")
                path = relative_file(self.run.paths.checkpoints, checkpoint.relative_path)
                if path.exists():
                    ArtifactReference("checkpoints", checkpoint.relative_path, checkpoint.sha256).verify(self.run.paths)
                    self._guard(self.run)
                    path.unlink()
                    fsync_directory(path.parent)
            reference = None if doc["parent"] is None else ArtifactReference.from_dict(doc["parent"])
