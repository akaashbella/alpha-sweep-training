"""Descriptive analysis of report CSVs when collection.json is unavailable/truncated.

Export consistency/checksums are not checkpoint or scientific completion authority.
Never read collection.json here, and never tune from evaluation/test scores.
"""

from collections import OrderedDict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from .artifacts import _is_link
from .data_contract import file_digest
from .diagnostic_protocol import DiagnosticError, CORE_FILES
from .identity import canonical_alpha
from .optimization_health import analyze_run, validate_health_policy
from .spec import ALPHAS, ARCHITECTURES, TRAIN_SIZE, VALIDATION_SIZE, BATCH_SIZE


def _json(text):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise DiagnosticError(f"Duplicate exported JSON key: {key}")
            result[key] = value
        return result
    value = json.loads(text, object_pairs_hook=unique_object,
        parse_constant=lambda token: (_ for _ in ()).throw(DiagnosticError(f"Nonfinite exported JSON: {token}")))
    from .serialization import strict_json_text
    strict_json_text(value)
    return value


def _rows(path):
    if any(_is_link(p) for p in (path, *path.parents)) or not path.is_file():
        raise DiagnosticError(f"Missing or linked CSV: {path}")
    limit = csv.field_size_limit(32 * 1024 * 1024)
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            if reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise DiagnosticError("Missing/duplicate CSV headings")
            for row in reader:
                if None in row or any(v is None for v in row.values()):
                    raise DiagnosticError("Malformed/truncated CSV row")
                yield row
    finally:
        csv.field_size_limit(limit)


def report_csv_health(root, policy, *, source_root=None):
    policy = validate_health_policy(policy)
    root = Path(root).expanduser().absolute()
    collection = root / "collection"
    paths = [collection / "training.csv", collection / "epochs.csv"]
    layers = collection / "layer_statistics.csv"
    if layers.is_file():
        paths.append(layers)
    if any(_is_link(part) for path in paths for part in (path, *path.parents)):
        raise DiagnosticError("Linked CSV inputs are forbidden")
    before = {str(p): file_digest(p) for p in paths}
    actual_core = None
    if source_root is not None:
        from .execution_provenance import source_inventory
        inventory = source_inventory(source_root)
        actual_core = {path: inventory.get(path) for path in CORE_FILES}
    runs = OrderedDict()
    for row in _rows(paths[0]):
        if row["dataset"] != "cifar10":
            continue
        run_id = row["training_run_id"]
        if run_id in runs or row["accuracy_unit"] != "fraction":
            raise DiagnosticError("Duplicate run or unknown accuracy units")
        alpha, seed = float(row["training_alpha"]), int(row["training_seed"])
        if alpha not in ALPHAS or row["architecture"] not in ARCHITECTURES or row["canonical_training_alpha"] != canonical_alpha(alpha):
            raise DiagnosticError("CSV identity is outside the prescribed grid")
        if row["training_complete"] not in ("True", "False"):
            raise DiagnosticError("Unrecognized exported completion boolean")
        best = None if not row["best_selection"] else _json(row["best_selection"])
        if row["condition"] != "ok" and best is not None:
            raise DiagnosticError("Unsafe export observations cannot carry accepted best metrics")
        runs[run_id] = {"training_run_id": run_id, "dataset": "cifar10", "architecture": row["architecture"],
            "training_alpha": alpha, "training_seed": seed, "condition": row["condition"], "stored_status": row["stored_status"],
            "training_complete": row["training_complete"] == "True", "evaluation_coverage": None,
            "scientific_settings": {"model": {"num_classes": 10}},
            "snapshot": None if best is None else {"transaction": {"best": best}, "epochs": []},
            "reported_provenance": _json(row["provenance"]) if row.get("provenance") else None,
            "reported_checkpoint_receipt": _json(row["checkpoint_receipt"]) if row.get("checkpoint_receipt") else None,
            "exported_best_validation_accuracy": None if not row["best_validation_accuracy"] else float(row["best_validation_accuracy"]),
            "exported_completed_epochs": None if not row["completed_epochs"] else int(row["completed_epochs"])}
    seeds = sorted({row["training_seed"] for row in runs.values()})
    grid = {(a, alpha, seed) for a in ARCHITECTURES for alpha in ALPHAS for seed in seeds}
    identities = {(r["architecture"], r["training_alpha"], r["training_seed"]) for r in runs.values()}
    if len(seeds) != 3 or len(runs) != 120 or identities != grid:
        raise DiagnosticError("Require the full 120-row CIFAR-10 training CSV; missing identities cannot be invented without the original manifest")
    epochs = {}
    for row in _rows(paths[1]):
        if row["dataset"] != "cifar10":
            continue
        run = runs[row["training_run_id"]]
        _same_identity(row, run)
        label = int(row["epoch_label"])
        record = _json(row["record"])
        key = (run["training_run_id"], label)
        if key in epochs or not 1 <= label <= 200 or record["epoch_index"] + 1 != label or record["completed_epochs"] != label:
            raise DiagnosticError("Duplicate/inconsistent exported epoch")
        if row["accuracy_unit"] != "fraction" or row["loss_unit"] != "mean_cross_entropy_nats_per_example":
            raise DiagnosticError("Unknown epoch metric units")
        for split, prefix in (("training", "training"), ("validation", "validation")):
            metric = record[split]
            allowed = (VALIDATION_SIZE,) if split == "validation" else (TRAIN_SIZE, TRAIN_SIZE // BATCH_SIZE * BATCH_SIZE)
            if (type(metric["examples"]) is not int or metric["examples"] <= 0 or type(metric["correct"]) is not int
                    or metric["examples"] not in allowed
                    or not 0 <= metric["correct"] <= metric["examples"] or not math.isfinite(metric["loss"]) or metric["loss"] < 0
                    or float(row[f"{prefix}_accuracy"]) != metric["correct"] / metric["examples"]
                    or float(row[f"{prefix}_loss"]) != metric["loss"]):
                raise DiagnosticError("Flat and embedded epoch metrics disagree")
        epochs[key] = {"record": record, "exported_median_population_std": None,
                       "exported_reference": _json(row["reference"]) if row.get("reference") else None}
    if layers.is_file():
        previous, values, names, seen = None, [], set(), set()
        def finish():
            if previous is not None:
                epochs[previous]["exported_median_population_std"] = statistics.median(values)
        for row in _rows(layers):
            if row["dataset"] != "cifar10":
                continue
            _same_identity(row, runs[row["training_run_id"]])
            key = (row["training_run_id"], int(row["epoch_label"]))
            if key not in epochs:
                raise DiagnosticError("Layer statistic has no exported epoch")
            if key != previous:
                finish()
                if key in seen:
                    raise DiagnosticError("Layer groups must be contiguous, as emitted by the report exporter")
                seen.add(key)
                previous, values, names = key, [], set()
            value = float(row["population_std"])
            if (not math.isfinite(value) or value < 0 or row["name"] in names
                    or row["statistic_unit"] != "clean_weight_population_std_correction_0"):
                raise DiagnosticError("Invalid/duplicate clean layer statistic")
            names.add(row["name"])
            values.append(value)
        finish()
    observations = []
    for run_id, run in runs.items():
        entries = [epochs[key] for key in sorted(epochs) if key[0] == run_id]
        if entries and run["snapshot"] is None:
            raise DiagnosticError("Epoch data cannot be silently discarded when exported best selection is absent")
        if run["snapshot"] is not None:
            if len(entries) != (run["exported_completed_epochs"] or 0) or any(
                    v["record"]["epoch_index"] != i for i, v in enumerate(entries)):
                raise DiagnosticError("Epoch coverage disagrees with training CSV")
            run["snapshot"]["epochs"] = [{"document": v} for v in entries]
            best = run["snapshot"]["transaction"]["best"]
            if entries and (best["epoch_index"] >= len(entries) or best != entries[-1]["record"]["best"]):
                raise DiagnosticError("Exported selected-best metadata disagrees with epoch history")
            if entries:
                winner = max(range(len(entries)), key=lambda i: entries[i]["record"]["validation"]["correct"])
                selected_metric = entries[winner]["record"]["validation"]
                if best != {"epoch_index": winner, "correct": selected_metric["correct"], "examples": selected_metric["examples"]}:
                    raise DiagnosticError("Exported best selection violates strict improvement/earliest exact tie")
            if best["correct"] / best["examples"] != run["exported_best_validation_accuracy"]:
                raise DiagnosticError("Exported best accuracy disagrees with selection")
        observed = analyze_run(run, policy)
        observed["reported_training_complete"] = observed.pop("training_complete")
        observed["training_completion_verified"] = None
        observed["evaluation_completion_verified"] = None
        observed["completion_authority"] = "unverified_export; original journal/checkpoints unavailable"
        observed["reported_provenance"] = run["reported_provenance"]
        observed["reported_checkpoint_receipt"] = run["reported_checkpoint_receipt"]
        recorded = (run["reported_provenance"] or {}).get("code", {}).get("inventory")
        observed["reported_core_source_differences"] = (None if actual_core is None or type(recorded) is not dict else [
            {"path": path, "reported": recorded.get(path), "actual": actual_core[path]}
            for path in CORE_FILES if recorded.get(path) != actual_core[path]])
        observations.append(observed)
    summaries = {}
    for name in ("validation_selections.json", "summaries.json", "coverage.json"):
        path = root / "summaries" / name
        if path.is_file():
            try:
                if any(_is_link(part) for part in (path, *path.parents)):
                    raise DiagnosticError("Linked summary evidence")
                content = path.read_bytes()
                summaries[name] = {"sha256": hashlib.sha256(content).hexdigest(), "document": _json(content.decode("utf-8"))}
            except (ValueError, OSError) as exc:
                summaries[name] = {"error": f"{type(exc).__name__}: {exc}"}
    after = {str(p): file_digest(p) for p in paths}
    if before != after:
        raise DiagnosticError("CSV inputs changed during analysis")
    return {"health_version": 1, "kind": "export_only_optimization_health", "research_authority": False,
        "collection_json": "not opened; reported truncated", "input_sha256": after, "health_policy": policy,
        "actual_core_file_sha256": actual_core,
        "layer_statistics_coverage": "Exported rows only; owned-inventory completeness unverified without the original journal/component metadata",
        "prescribed_manifest_available": False, "exported_seeds": seeds, "individual_runs": observations,
        "seed_groups": _export_seed_groups(observations, policy), "summary_evidence": summaries,
        "training_alpha_selection": "not performed; no test metrics used", "score_filtering": False}


def _same_identity(row, run):
    if (row["architecture"] != run["architecture"] or int(row["training_seed"]) != run["training_seed"]
            or float(row["canonical_training_alpha"]) != run["training_alpha"]):
        raise DiagnosticError("Conflicting identities across exported CSVs")


def _export_seed_groups(rows, policy):
    groups = OrderedDict()
    for row in rows:
        groups.setdefault((row["architecture"], row["training_alpha"]), []).append(row)
    result = []
    for (architecture, alpha), members in groups.items():
        values = [m["best_validation_accuracy"] for m in members if m["best_validation_accuracy"] is not None]
        result.append({"architecture": architecture, "training_alpha": alpha,
            "individuals": [{"seed": m["training_seed"], "accuracy": m["best_validation_accuracy"]} for m in members],
            "official_three_seed_summary": False, "exported_three_seed_values_available": len(values) == 3,
            "descriptive_accuracy_range": max(values) - min(values) if len(values) == 3 else None,
            "severe_seed_variation": max(values) - min(values) >= policy["seed_accuracy_range"] if len(values) == 3 else None})
    return result
