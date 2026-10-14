"""Versioned derived reporting contracts; never a scientific completion authority."""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib import metadata
from pathlib import Path
import platform

from .data_contract import content_identity
from .identity import canonical_alpha
from .manifests import manifest_from_dict
from .serialization import strict_json_text
from .spec import EPOCHS, TOTAL_RUNS, TOTAL_EVALUATION_SLOTS

REPORTING_VERSION = 1
CONVENTIONS = {"accuracy_unit": "fraction", "difference_unit": "percentage_points",
    "standard_deviation": "sample_ddof_1; null_for_n_less_than_2",
    "official_summary": "all_prescribed_observations_required; no_partial_summary",
    "robustness_summary": "realizations_within_checkpoint_then_means_across_training_seeds",
    "selection": "all_ten_complete_three_seed_validation_candidates; all_exact_ties_in_grid_order",
    "checkpoint_policy": "clean_validation_selected_best; earliest_epoch_on_exact_tie",
    "comparison": "paired_descriptive_only; no_significance_or_confidence_intervals",
    "rounding": "display_only; never_before_comparison_or_aggregation"}
STABILITY = "Independent stable task/checkpoint reads; not an atomic whole-grid snapshot; no writer locks or repair"


class ReportingError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ReportingError(message)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sealed(value):
    strict_json_text(value)
    return {**deepcopy(value), "document_sha256": content_identity(value)[7:]}


def verify_seal(value):
    require(type(value) is dict and "document_sha256" in value, "Missing reporting document seal")
    require(value["document_sha256"] == content_identity({k: v for k, v in value.items() if k != "document_sha256"})[7:],
            "Reporting document digest mismatch")


def reporting_provenance(source_root, role):
    # Actual imported checkout, using the existing inventory. Package metadata
    # does not initialize CUDA/models or compare current source with old research.
    from .execution_provenance import source_inventory, SOURCE_INVENTORY_VERSION
    inventory = source_inventory(source_root)
    software = {"python": platform.python_version()}
    for name in ("torch", "torchvision", "numpy", "PyYAML", "matplotlib"):
        try:
            software[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            software[name] = None
    require(source_inventory(source_root) == inventory, "Reporting source changed during inventory")
    return {"role": role, "code": {"inventory_version": SOURCE_INVENTORY_VERSION, "inventory": inventory,
        "source_sha256": hashlib.sha256(strict_json_text({"version": SOURCE_INVENTORY_VERSION,
                                                        "files": inventory}).encode("utf-8")).hexdigest()},
        "software": software}


def task_identity(task):
    science = task.config.scientific
    return {"training_run_id": task.run_id, "training_scientific_fingerprint": task.fingerprint,
        "training_global_index": task.global_index, "phase": task.phase, "phase_index": task.phase_index,
        "dataset": science["dataset"], "architecture": science["architecture"],
        "training_alpha": science["alpha"], "canonical_training_alpha": canonical_alpha(science["alpha"]),
        "training_seed": science["training_seed"]}


def validate_collection(collection):
    """Validate a frozen exported observation, without rereading mutable live stores.

    Existing raw/epoch validators are reused. Original checkpoint/HEAD verification
    was performed by collect_results; this snapshot never authorizes computation.
    """
    verify_seal(collection)
    require(type(collection.get("reporting_version")) is int and collection.get("reporting_version") == REPORTING_VERSION and collection.get("kind") == "result_collection",
            "Unsupported reporting collection version/kind; no implicit migration")
    require(collection.get("conventions") == CONVENTIONS and collection.get("stability") == STABILITY,
            "Unknown reporting conventions/stability contract")
    manifest = manifest_from_dict(collection["manifest"])
    require(collection["grid_id"] == manifest.grid_id and collection["manifest_document_sha256"] == manifest.document_sha256,
            "Collection/full-manifest linkage mismatch")
    training, evaluation = collection["training"], collection["evaluation"]
    require(type(training) is list and type(evaluation) is list, "Training/evaluation observations must be lists")
    require(len(training) == TOTAL_RUNS and len(evaluation) == TOTAL_EVALUATION_SLOTS, "Collection must retain 240/6720 rows")
    from .evaluation_protocol import prescribed_slots, evaluation_protocol, validate_protocol, perturbation_seed
    from .evaluation_persistence import validate_result
    from .checkpoint_records import parse_epoch, validate_epoch, validate_context
    from .checkpoint_journal import receipt_from_dict
    from .metrics import Counters
    from types import SimpleNamespace
    for task, row in zip(manifest.tasks, training):
        identity = task_identity(task)
        require(all(type(row[k]) is int for k in ("training_seed", "training_global_index", "phase", "phase_index")) and
                type(row["training_alpha"]) in (int, float), "Invalid typed training identity")
        require(all(row.get(k) == v for k, v in identity.items()), "Training row membership/order mismatch")
        require(row["scientific_settings"] == task.config.scientific and row["prepared_data"] == task.prepared_data,
                "Reporting science/prepared settings must match manifest")
        require(all(type(row[k]) is bool for k in ("training_complete", "full_budget_committed", "evaluation_eligible",
            "training_finalization_pending", "terminal_numerical", "execution_failed")), "Invalid training observation flags")
        snapshot = row["snapshot"]
        require(snapshot is not None or row["best_validation_metrics"] is None, "Unavailable snapshot cannot carry accepted validation metrics")
        require(row["condition"] == "ok" or snapshot is None, "Unsafe observation cannot carry accepted training metrics")
        if snapshot is not None:
            require(snapshot["transaction_reference"] == row["transaction"], "Training snapshot reference mismatch")
            document = snapshot["transaction"]
            validate_context(document["context"], task.config, snapshot["provenance"])
            require(document["status"] == row["stored_status"] and document["counters"]["completed_epochs"] == row["completed_epochs"],
                    "Training observation/progress disagreement")
            require(row["training_complete"] == (document["status"] in ("completed", "training_completed")), "Training completion disagreement")
            require(len(snapshot["epochs"]) == row["completed_epochs"], "Snapshot must contain all committed epochs")
            previous, best = Counters(), None
            for index, epoch in enumerate(snapshot["epochs"]):
                require(epoch["reference"] == document["records"][index], "Committed epoch reference mismatch")
                _, record, stats = parse_epoch(epoch["document"], task.config, snapshot["provenance"])
                validate_epoch(record, stats, previous=previous, previous_best=best, config=task.config,
                               components=snapshot["component_metadata"])
                previous, best = record.totals, record.best
            require(document["best"] == (None if best is None else {"epoch_index": best.epoch_index, "correct": best.correct, "examples": best.examples}),
                    "Reported selection must come from committed validation history")
            require(snapshot["receipt"] == document["receipt"], "Training receipt snapshot mismatch")
            receipt_from_dict(snapshot["receipt"]).validate(task.config, previous, best)
            require(document["counters"] == {k: getattr(previous, k) for k in ("completed_epochs", "attempted_batches", "processed_examples", "successful_updates", "skipped_updates")},
                    "Committed counters disagree with epoch coverage")
            if row["training_complete"]:
                require(document["artifacts"]["completion"] is not None and "summary" in snapshot["completion_documents"] and
                        snapshot["completion_documents"]["summary"]["reference"] == document["artifacts"]["completion"]["summary"],
                        "Training completion must retain the accepted immutable completion reference")
            expected_best = None if best is None else snapshot["epochs"][best.epoch_index]["document"]["record"]["validation"]
            require(row["best_validation_metrics"] == expected_best, "Best-validation metrics disagree with committed history")
        require(not row["training_complete"] or (snapshot is not None and row["completed_epochs"] == EPOCHS),
                "Completed reporting row requires verified full budget")
        if snapshot is not None:
            require(row["full_budget_committed"] == (row["completed_epochs"] == EPOCHS) and
                    row["evaluation_eligible"] == row["full_budget_committed"], "Reporting eligibility/budget mismatch")
        protocol = evaluation_protocol(task.config)
        for offset, slot in enumerate(prescribed_slots()):
            value = evaluation[task.global_index * 28 + offset]
            require(type(value["successful"]) is bool and type(value["robustness_completed"]) is bool,
                    "Evaluation reporting flags must be boolean")
            require(all(value.get(k) == v for k, v in identity.items()) and value["slot"] == slot.to_dict(),
                    "Evaluation logical membership/order mismatch")
            require(value["requested_protocol"] == protocol.to_dict(), "Requested evaluation protocol mismatch")
            require(value["requested_protocol_fingerprint"] == protocol.fingerprint and value["logical_index"] == task.global_index * 28 + offset,
                    "Expected protocol/logical index mismatch")
            binding, raw = value["binding"], value["raw_record"]
            if binding is None:
                require(raw is None and value["perturbation_seed"] is None and value["best_checkpoint_identity"] is None,
                        "Unbound slots cannot invent checkpoint/seed/raw evidence")
            else:
                validate_protocol(binding["evaluation_protocol"], task.config)
                require(snapshot is not None and row["completed_epochs"] == EPOCHS, "Bound evaluation requires accepted full-budget training authority")
                require(binding["best_checkpoint"] == snapshot["receipt"]["best"] and binding["best_validation"] == snapshot["transaction"]["best"] and
                        binding["best_checkpoint_identity"]["sha256"] == binding["best_checkpoint"]["sha256"] and
                        binding["best_checkpoint"]["relative_path"] == f"generations/{binding['best_checkpoint_identity']['generation']}.best.pt",
                        "Evaluation binding disagrees with verified training best selection/reference")
                require(binding["evaluation_protocol_fingerprint"] == protocol.fingerprint and
                        binding["evaluation_protocol_id"] == protocol.to_dict()["protocol_version"], "Bound protocol identity mismatch")
                require(all(binding[k] == identity[k] for k in ("dataset", "architecture", "training_alpha", "canonical_training_alpha", "training_seed")),
                        "Bound scientific identity mismatch")
                require(binding["training_identity"]["run_id"] == task.run_id and
                        binding["training_identity"]["scientific_fingerprint"] == task.fingerprint,
                        "Evaluation binding belongs to another training task")
                require(all(binding["data_identity"].get(k) == v for k, v in task.prepared_data.items()), "Bound prepared-data identities disagree with manifest")
                require(binding["best_checkpoint_identity"] == value["best_checkpoint_identity"] and
                        perturbation_seed(binding["best_checkpoint_identity"], slot) == value["perturbation_seed"], "Bound checkpoint/seed mismatch")
                if raw is not None:
                    require(value["condition"] == "ok" and value["raw_reference"] is not None, "Unsafe/unreferenced raw result")
                    validate_result(raw, SimpleNamespace(binding=binding, protocol=protocol, legacy_clean=row["legacy_clean"]), slot)
                    require(raw["status"] == value["stored_status"], "Raw/stored evaluation status mismatch")
            require(value["successful"] == (value["condition"] == "ok" and raw is not None and raw["status"] == "evaluation_completed"),
                    "Success must come from verified raw slot authority")
        coverage = row["evaluation_coverage"]
        if coverage is not None:
            own_slots = evaluation[task.global_index * 28:(task.global_index + 1) * 28]
            require(all(s["robustness_completed"] == coverage["robustness_completed"] for s in own_slots), "Sweep/slot observation disagreement")
            if coverage["robustness_completed"]:
                completion = coverage["completion_document"]
                require(completion is not None and all(s["successful"] for s in own_slots), "Robustness completion needs all 28 verified slots and sweep evidence")
                require(completion["status"] == "robustness_completed" and completion["binding_sha256"] == content_identity(coverage["binding"]) and
                        completion["slots"] == [{"slot": s["slot"], "reference": s["raw_reference"]} for s in own_slots],
                        "Frozen sweep authority does not bind all collected raw references")
    expected_epochs = [{"training_run_id": run["training_run_id"], "reference": epoch["reference"],
                        "epoch_label": epoch["document"]["record"]["epoch_index"] + 1}
                       for run in training if run["snapshot"] is not None for epoch in run["snapshot"]["epochs"]]
    require(collection["epochs"] == expected_epochs and collection["layer_statistics"] == expected_epochs,
            "Epoch/layer indices must reference the single accepted embedded documents")
    roots = collection["protected_roots"]
    require(type(roots) is list and roots and all(type(p) is str and Path(p).is_absolute() for p in roots),
            "Declare absolute protected storage roots")
    for row in training:
        require(set(row["paths"]) == {"metadata", "checkpoints", "results", "logs"} and all(Path(p).is_absolute() for p in row["paths"].values()),
                "Training storage linkage must retain its absolute managed paths")
        require(all(any(Path(root) == Path(p) or Path(root) in Path(p).parents for root in roots) for p in row["paths"].values()),
                "Protected roots omit declared scientific storage")
    require(collection["workflow_root"] is None or collection["workflow_root"] in roots, "Protected roots omit workflow storage")
    strict_json_text(collection)
    return manifest
