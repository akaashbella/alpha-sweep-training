"""Exclusive derived namespaces, finite exports and durable last-marker publication."""

from contextlib import contextmanager
import csv
import json
import os
from pathlib import Path
import socket
import uuid

from .artifacts import _is_link
from .checkpoint_io import digest_file, relative_file
from .reporting_contract import (REPORTING_VERSION, require, sealed, verify_seal,
    reporting_provenance, utc_now, validate_collection)
from .serialization import atomic_write_json, durable_mkdir, fsync_directory, read_json, strict_json_text


def _plain(path):
    require(not any(_is_link(p) for p in (path, *path.parents)), f"Linked reporting storage/input rejected: {path}")


def _required_outputs(kind, files):
    required = {"collection": {"collection.json", "training.csv", "evaluation.csv", "epochs.csv", "layer_statistics.csv", "coverage.json", "diagnostics.json"},
        "summaries": {"summaries.json", "validation_selections.json", "coverage.json", "training_groups.csv", "checkpoint_robustness.csv",
            "robustness_groups.csv", "clean_seed_pairs.csv", "noise_realization_pairs.csv", "paired_training_groups.csv"},
        "figures": {"figure_index.json", "coverage.json", "coverage_training.csv", "coverage_evaluation.csv"}}
    require(kind in required, "Unknown derived publication kind")
    require(type(files) is list and len({ref["relative_path"] for ref in files}) == len(files), "Duplicate/malformed derived output references")
    require(required[kind] <= {ref["relative_path"] for ref in files}, "Derived publication is missing required outputs")
    if kind == "figures":
        require(any(ref["role"] == "scientific_figure" and ref["relative_path"].endswith(".png") for ref in files) and
                any(ref["role"] == "scientific_figure" and ref["relative_path"].endswith((".pdf", ".svg")) for ref in files),
                "Figure publication needs raster and vector outputs")


class ReportWriter:
    def __init__(self, root, owner):
        self.root, self.owner, self.files, self.active = root, owner, [], True

    def guard(self):
        require(self.active, "Keep the derived-output ownership context active")
        _plain(self.root)
        require(read_json(relative_file(self.root, "OWNER.json")) == self.owner, "Reporting ownership changed")

    def target(self, relative):
        self.guard()
        require(relative not in ("OWNER.json", "COMPLETE.json", "PUBLICATION_FAILURE.json", "UNCONFIRMED_COMPLETE.json"), "Reserved reporting ownership/publication filename")
        path = relative_file(self.root, relative, create_parents=True)
        require(not path.exists(), f"Derived output collision; no overwrite: {path}")
        return path

    def acknowledge(self, path, role):
        self.guard()
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
        fsync_directory(path.parent)
        self.files.append({"relative_path": path.relative_to(self.root).as_posix(), "sha256": digest_file(path), "role": role})

    def json(self, relative, value, role):
        path = self.target(relative)
        atomic_write_json(path, value)
        self.acknowledge(path, role)

    def csv(self, relative, rows, fieldnames, role):
        path = self.target(relative)
        with path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            for row in rows:
                strict_json_text(row)
                writer.writerow({k: "" if v is None else json.dumps(v, sort_keys=True, allow_nan=False, ensure_ascii=False)
                                 if isinstance(v, (dict, list, tuple)) else v for k, v in row.items()})
            stream.flush()
            os.fsync(stream.fileno())
        self.acknowledge(path, role)


@contextmanager
def open_report(collection, *, output_dir, source_root, kind, options=None, extra_protected=()):
    validate_collection(collection)
    require(collection.get("workflow_root") is not None, "Declare workflow_root before derived publication, including an unexecuted workflow")
    root = Path(output_dir).expanduser().absolute()
    _plain(root)
    root = root.resolve()  # Normalize '..' before any overlap/ownership decision.
    protected = [Path(p).expanduser().absolute() for p in (*collection["protected_roots"], source_root, *extra_protected)]
    for path in protected:
        _plain(path)
        path = path.resolve()
        require(root != path and root not in path.parents and path not in root.parents,
                f"Derived output overlaps protected scientific/workflow/source storage: {path}")
    require(not root.exists(), f"Derived directory already exists; preserve it and select a new output path: {root}")
    durable_mkdir(root.parent)
    root.mkdir()  # Exclusive ownership across concurrent report publishers.
    fsync_directory(root.parent)
    owner = sealed({"reporting_version": REPORTING_VERSION, "namespace": "derived-reporting-v1", "kind": kind,
        "grid_id": collection["grid_id"], "manifest_document_sha256": collection["manifest_document_sha256"],
        "collection_sha256": collection["document_sha256"], "options": options or {},
        "publisher_provenance": reporting_provenance(source_root, kind), "created_at_utc": utc_now(),
        "hostname": socket.gethostname(), "pid": os.getpid(), "token": uuid.uuid4().hex,
        "scientific_completion_authority": False})
    atomic_write_json(root / "OWNER.json", owner)
    writer = ReportWriter(root, owner)
    try:
        yield writer
        writer.guard()
        _required_outputs(kind, writer.files)
        for ref in writer.files:
            require(digest_file(relative_file(root, ref["relative_path"])) == ref["sha256"], "Derived file changed before publication")
        require(reporting_provenance(source_root, kind)["code"] == owner["publisher_provenance"]["code"], "Reporting source changed during publication")
        atomic_write_json(root / "COMPLETE.json", sealed({"reporting_version": REPORTING_VERSION,
            "status": "derived_publication_complete", "owner_sha256": owner["document_sha256"],
            "files": writer.files, "published_at_utc": utc_now(), "scientific_completion_authority": False}))
    except Exception as exc:
        try:
            writer.guard()
            marker = root / "COMPLETE.json"
            if marker.exists():
                marker.rename(root / "UNCONFIRMED_COMPLETE.json")
                fsync_directory(root)
            atomic_write_json(root / "PUBLICATION_FAILURE.json", {"reason": f"{type(exc).__name__}: {exc}",
                "recorded_at_utc": utc_now(), "scientific_completion_authority": False})
        except Exception:
            pass  # Retain partial files; never invent a complete marker.
        raise
    finally:
        writer.active = False


def _epoch_exports(collection, *, layers=False):
    for run in collection["training"]:
        if run["snapshot"] is None:
            continue
        identity = {k: run[k] for k in ("training_run_id", "dataset", "architecture", "canonical_training_alpha", "training_seed")}
        for epoch in run["snapshot"]["epochs"]:
            record = epoch["document"]["record"]
            shared = {**identity, "epoch_label": record["epoch_index"] + 1, "reference": epoch["reference"]}
            if layers:
                for entry in epoch["document"]["statistics"]:
                    yield {**shared, **entry, "statistic_unit": "clean_weight_population_std_correction_0"}
            else:
                yield {**shared, "stored_status": record["status"], "training_kind": record["training_kind"],
                    "training_accuracy": record["training"]["accuracy"], "validation_accuracy": record["validation"]["accuracy"],
                    "training_loss": record["training"]["loss"], "validation_loss": record["validation"]["loss"],
                    "accuracy_unit": "fraction", "loss_unit": "mean_cross_entropy_nats_per_example", "record": record}


IDENTITY_COLUMNS = ("training_run_id", "training_scientific_fingerprint", "training_global_index", "phase", "phase_index",
                    "dataset", "architecture", "training_alpha", "canonical_training_alpha", "training_seed")


def _training_export(row):
    snapshot, best = row["snapshot"], row["best_validation_metrics"]
    return {**{k: row[k] for k in IDENTITY_COLUMNS},
        **{k: row[k] for k in ("condition", "stored_status", "completed_epochs", "training_complete", "full_budget_committed",
            "evaluation_eligible", "training_finalization_pending", "terminal_numerical", "execution_failed", "diagnostic", "locks")},
        "best_validation_accuracy": None if best is None else best["accuracy"], "best_validation_loss": None if best is None else best["loss"],
        "best_selection": None if snapshot is None else snapshot["transaction"]["best"],
        "checkpoint_receipt": None if snapshot is None else snapshot["receipt"], "transaction": row["transaction"],
        "provenance": None if snapshot is None else snapshot["provenance"], "failure": row["failure"],
        "orphan_failure_evidence": row["evidence_failures"], "legacy_clean": row["legacy_clean"],
        "accuracy_unit": "fraction", "loss_unit": "mean_cross_entropy_nats_per_example"}


def _evaluation_export(row):
    raw = row["raw_record"]
    metrics = None if raw is None else raw["test_metrics"]
    return {**{k: row[k] for k in IDENTITY_COLUMNS}, "logical_index": row["logical_index"], **row["slot"],
        **{k: row[k] for k in ("condition", "stored_status", "successful", "robustness_completed", "perturbation_seed",
            "best_checkpoint_identity", "namespace", "raw_reference", "failure", "orphan_failures", "diagnostic")},
        "requested_protocol_fingerprint": row["requested_protocol_fingerprint"], "binding": row["binding"],
        **{name: None if metrics is None else metrics[name] for name in ("accuracy", "loss", "correct", "examples", "batches")},
        **{name: None if raw is None else raw[name] for name in ("evaluation_protocol_id", "evaluation_protocol_fingerprint",
            "randomness_version", "clean_state_preservation", "runtime", "evaluation_provenance", "training_provenance",
            "origin", "attempt_id", "duration_seconds", "recorded_at_utc", "previous_record")},
        "accuracy_unit": "fraction", "loss_unit": "mean_cross_entropy_nats_per_example"}


def publish_collection(collection, *, output_dir, source_root):
    from .aggregation import summarize_results
    coverage = summarize_results(collection)["coverage"]
    with open_report(collection, output_dir=output_dir, source_root=source_root, kind="collection") as writer:
        writer.json("collection.json", collection, "validated_collection")
        for key, export in (("training", _training_export), ("evaluation", _evaluation_export)):
            writer.csv(f"{key}.csv", (export(row) for row in collection[key]), list(export(collection[key][0])), f"individual_{key}")
        shared = ["training_run_id", "dataset", "architecture", "canonical_training_alpha", "training_seed", "epoch_label", "reference"]
        writer.csv("epochs.csv", _epoch_exports(collection), shared + ["stored_status", "training_kind", "training_accuracy",
            "validation_accuracy", "training_loss", "validation_loss", "accuracy_unit", "loss_unit", "record"], "committed_epoch_metrics")
        writer.csv("layer_statistics.csv", _epoch_exports(collection, layers=True), shared + ["name", "module_type", "shape",
            "numel", "population_std", "statistic_unit"], "committed_clean_layer_statistics")
        writer.json("coverage.json", coverage, "coverage")
        writer.json("diagnostics.json", collection["diagnostics"], "inspection_diagnostics")
    return inspect_report(output_dir)


def publish_summaries(collection, *, output_dir, source_root, collection_dir=None):
    from .aggregation import summarize_results
    aggregate = summarize_results(collection)
    extra = () if collection_dir is None else (collection_dir,)
    with open_report(collection, output_dir=output_dir, source_root=source_root, kind="summaries", extra_protected=extra) as writer:
        writer.json("summaries.json", aggregate, "derived_summaries")
        writer.json("validation_selections.json", aggregate["validation_selections"], "validation_only_selection")
        writer.json("coverage.json", aggregate["coverage"], "coverage")
        for name in ("training_groups", "checkpoint_robustness", "robustness_groups", "clean_seed_pairs", "noise_realization_pairs", "paired_training_groups"):
            writer.csv(f"{name}.csv", aggregate[name], list(aggregate[name][0]), name)
    return inspect_report(output_dir)


def inspect_report(report_dir):
    root = Path(report_dir).expanduser().absolute()
    _plain(root)
    root = root.resolve()
    require(root.is_dir(), "Reporting directory is missing")
    owner = read_json(relative_file(root, "OWNER.json"))
    verify_seal(owner)
    require(type(owner["reporting_version"]) is int and owner["reporting_version"] == REPORTING_VERSION and owner["namespace"] == "derived-reporting-v1" and
            owner["scientific_completion_authority"] is False, "Unsupported derived namespace")
    complete = root / "COMPLETE.json"
    failure = root / "PUBLICATION_FAILURE.json"
    unconfirmed = root / "UNCONFIRMED_COMPLETE.json"
    _plain(complete)
    _plain(failure)
    _plain(unconfirmed)
    if not complete.exists() or failure.exists() or unconfirmed.exists():
        return {"derived_publication_complete": False, "owner": owner,
            "failure": read_json(failure) if failure.is_file() else None, "diagnostic": "Partial/incomplete derived publication; no scientific conclusion"}
    document = read_json(relative_file(root, "COMPLETE.json"))
    verify_seal(document)
    require(type(document["reporting_version"]) is int and document["reporting_version"] == REPORTING_VERSION and document["owner_sha256"] == owner["document_sha256"] and
            document["scientific_completion_authority"] is False and document["status"] == "derived_publication_complete", "Derived completion owner/version mismatch")
    _required_outputs(owner["kind"], document["files"])
    for ref in document["files"]:
        require(digest_file(relative_file(root, ref["relative_path"])) == ref["sha256"], "Derived output reference digest mismatch")
    coverage = next((ref for ref in document["files"] if ref["role"] == "coverage"), None)
    return {"derived_publication_complete": True, "owner": owner, "publication": document,
            "coverage": None if coverage is None else read_json(relative_file(root, coverage["relative_path"]))}


def load_collection(report_dir):
    report = inspect_report(report_dir)
    require(report["derived_publication_complete"] and report["owner"]["kind"] == "collection", "Require a completed collection publication")
    require(any(r["relative_path"] == "collection.json" and r["role"] == "validated_collection" for r in report["publication"]["files"]),
            "Publication does not bind its collection file")
    collection = read_json(relative_file(Path(report_dir).expanduser().absolute(), "collection.json"))
    validate_collection(collection)
    require(collection["document_sha256"] == report["owner"]["collection_sha256"], "Owner/collection digest mismatch")
    return collection
