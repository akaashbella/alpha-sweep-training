"""Read completed summaries only: no models, checkpoints or live store inspection."""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path, PurePosixPath
import re
import stat
import statistics
from typing import Any

from alpha_sweep_training.reporting_contract import CONVENTIONS, verify_seal
from alpha_sweep_training.serialization import read_json

from .contracts import ALPHAS, ARCHITECTURES, DATASETS, EVALUATION_ALPHAS, AnalysisError, canonical_alpha, require

REQUIRED_FILES = {
    "summaries.json", "validation_selections.json", "coverage.json", "training_groups.csv",
    "checkpoint_robustness.csv", "robustness_groups.csv", "clean_seed_pairs.csv",
    "noise_realization_pairs.csv", "paired_training_groups.csv",
}
HEADER = ("dataset", "architecture", "training_alpha", "canonical_training_alpha")


def plain_path(value: str | Path) -> Path:
    path = Path(value).expanduser().absolute()
    for item in (path, *path.parents):
        require(not item.is_symlink() and not (hasattr(item, "is_junction") and item.is_junction()),
                f"Linked path rejected: {item}")
        if item.exists():
            require(not getattr(item.lstat(), "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0),
                    f"Reparse point rejected: {item}")
    return path.resolve()


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _digest_stream(stream)


def _digest_stream(stream) -> str:
    result = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        result.update(block)
    return result.hexdigest()


def _sha(value: Any, label: str, prefix: str = "") -> None:
    require(type(value) is str and re.fullmatch(re.escape(prefix) + "[0-9a-f]{64}", value) is not None,
            f"Invalid {label} digest")


def _number(value: Any, label: str, *, accuracy: bool = False, nonnegative: bool = False) -> float:
    require(type(value) in (int, float) and math.isfinite(value), f"{label}: expected finite number")
    require(not accuracy or 0 <= value <= 1, f"{label}: accuracy must be a fraction")
    require(not nonnegative or value >= 0, f"{label}: expected nonnegative value")
    return float(value)


def _checkpoint(value: Any) -> None:
    if value is None:
        return
    require(type(value) is dict and set(value) == {"generation", "sha256"}, "Invalid checkpoint identity")
    require(type(value["generation"]) is str and bool(value["generation"]), "Missing checkpoint generation")
    _sha(value["sha256"], "checkpoint")


def validate_stat(value: dict, expected: int, *, accuracy: bool = False, nonnegative: bool = False) -> None:
    """Verify the producer's complete-only summary, retaining its eligibility decisions."""
    require(type(value) is dict and set(value) == {"expected_count", "eligible_count", "complete", "individuals", "exclusions", "official"},
            "Malformed summary statistic")
    require(type(value["expected_count"]) is int and value["expected_count"] == expected, "Wrong prescribed observation count")
    require(type(value["individuals"]) is list and len(value["individuals"]) == expected, "Missing prescribed individual observations")
    values, exclusions = [], []
    for row in value["individuals"]:
        require(type(row) is dict and "value" in row and "reason" in row, "Malformed summary individual")
        if row["value"] is None:
            require(type(row["reason"]) is str and bool(row["reason"]), "Missing exclusion reason")
            exclusions.append(row)
        else:
            values.append(_number(row["value"], "individual", accuracy=accuracy, nonnegative=nonnegative))
            require(row["reason"] is None, "Eligible individual carries an exclusion")
            require("observed_value" not in row or row["observed_value"] == row["value"], "Accepted/observed value mismatch")
        if "observed_value" in row and row["observed_value"] is not None:
            _number(row["observed_value"], "observed value", accuracy=accuracy, nonnegative=nonnegative)
    require(type(value["eligible_count"]) is int and value["eligible_count"] == len(values), "Eligible count mismatch")
    require(type(value["complete"]) is bool and value["complete"] == (len(values) == expected), "Summary completeness mismatch")
    require(value["exclusions"] == exclusions, "Summary exclusion mismatch")
    official = value["official"]
    if len(values) < expected:
        require(official is None, "Incomplete realization/seed set has an official mean")
        return
    require(type(official) is dict and set(official) == {"mean", "sample_sd", "count"}, "Malformed official summary")
    require(type(official["count"]) is int and official["count"] == expected, "Official count mismatch")
    actual = _number(official["mean"], "mean", accuracy=accuracy, nonnegative=nonnegative)
    require(math.isclose(actual, math.fsum(v / expected for v in values), rel_tol=1e-12, abs_tol=1e-14), "Official mean disagrees with individuals")
    if expected == 1:
        require(official["sample_sd"] is None, "Single observation must have null sample SD")
    else:
        sd = _number(official["sample_sd"], "sample SD", nonnegative=True)
        require(math.isclose(sd, statistics.stdev(values), rel_tol=1e-12, abs_tol=1e-14), "Official sample SD mismatch")


def _group_key(row: dict) -> tuple:
    require(row["dataset"] in DATASETS and row["architecture"] in ARCHITECTURES, "Unknown dataset/architecture")
    alpha = _number(row["training_alpha"], "training alpha")
    require(alpha in ALPHAS and row["canonical_training_alpha"] == canonical_alpha(alpha), "Training alpha identity mismatch")
    return row["dataset"], row["architecture"], alpha


def _eval_key(row: dict) -> tuple:
    alpha = _number(row["evaluation_alpha"], "evaluation alpha")
    require(alpha in EVALUATION_ALPHAS and row["canonical_evaluation_alpha"] == canonical_alpha(alpha), "Evaluation alpha identity mismatch")
    return (*_group_key(row), alpha)


def _indexed(rows: Any, key, label: str) -> dict:
    require(type(rows) is list, f"{label}: expected list")
    result = {}
    for row in rows:
        require(type(row) is dict, f"{label}: expected object")
        index = key(row)
        require(index not in result, f"Duplicate {label}: {index}")
        result[index] = row
    return result


def validate_summaries(document: dict) -> tuple[int, ...]:
    """Validate summary-v1 membership and all statistics used by Stage 1.

    This validates frozen producer decisions, not the underlying live experiment.
    No partial or incompatible value is promoted into an eligible observation.
    """
    verify_seal(document)
    require(type(document.get("reporting_version")) is int and document["reporting_version"] == 1
            and document.get("kind") == "result_summaries", "Unsupported summary version/kind; regenerate with existing reporting CLI")
    require(document.get("conventions") == CONVENTIONS, "Unsupported reporting conventions")
    for name in ("manifest_document_sha256", "collection_sha256"):
        _sha(document[name], name)
    require(type(document["grid_id"]) is str and bool(document["grid_id"]), "Missing grid identity")
    groups = _indexed(document["training_groups"], _group_key, "training group")
    expected_groups = {(d, a, t) for d in DATASETS for a in ARCHITECTURES for t in ALPHAS}
    require(set(groups) == expected_groups, "Summaries must preserve all 80 training groups")
    runs, seeds = {}, None
    for group in groups.values():
        individuals = group["validation_accuracy"]["individuals"]
        validate_stat(group["validation_accuracy"], 3, accuracy=True)
        candidate = tuple(i["training_seed"] for i in individuals)
        require(len(set(candidate)) == 3 and all(type(s) is int and 0 <= s < 2**32 for s in candidate), "Invalid three-seed plan")
        if seeds is None:
            seeds = candidate
        require(candidate == seeds, "Training seed plan/order changed across configurations")
        require(group["training_run_ids"] == [r["training_run_id"] for r in individuals], "Training run membership mismatch")
        for row in individuals:
            run = row["training_run_id"]
            prefix = f"{group['dataset']}__{group['architecture']}__alpha-{group['canonical_training_alpha']}__seed-{row['training_seed']}"
            allowed = {prefix}
            if group["dataset"] == "cifar100" and group["architecture"] == "mobilenetv3_large":
                allowed.add(prefix + "__recipe-mobile-eps0316-v1")
            require(run in allowed and run not in runs, "Incorrect or duplicate training run identity")
            runs[run] = {**{k: group[k] for k in HEADER}, "training_seed": row["training_seed"]}
        for metric in ("clean_loss", "clean_accuracy"):
            stat = group[metric]
            validate_stat(stat, 3, accuracy=metric == "clean_accuracy", nonnegative=metric == "clean_loss")
            require([(r["training_run_id"], r["training_seed"]) for r in stat["individuals"]] ==
                    [(r["training_run_id"], r["training_seed"]) for r in individuals], "Clean group identity mismatch")
    checkpoints = _indexed(document["checkpoint_robustness"], lambda r: (r["training_run_id"], _eval_key(r)[-1]), "checkpoint/evaluation-alpha")
    require(set(checkpoints) == {(r, e) for r in runs for e in EVALUATION_ALPHAS}, "Summaries must preserve all 2,400 checkpoint/alpha records")
    ordered_runs = [run for d in DATASETS for a in ARCHITECTURES for t in ALPHAS
                    for run in groups[(d, a, t)]["training_run_ids"]]
    global_indices = {run: i for i, run in enumerate(ordered_runs)}
    offsets = {e: (0 if e == 0 else 1 + 3*(i-1)) for i, e in enumerate(EVALUATION_ALPHAS)}
    run_checkpoints = {}
    for (run, alpha), row in checkpoints.items():
        require(all(row[k] == v and type(row[k]) is type(v) for k, v in runs[run].items()), "Checkpoint/training identity mismatch")
        _checkpoint(row["best_checkpoint_identity"])
        known = run_checkpoints.setdefault(run, [])
        if row["best_checkpoint_identity"] is not None:
            known.append(row["best_checkpoint_identity"])
            require(all(c == known[0] for c in known), "Mismatched checkpoint across evaluation alphas")
        count = 1 if alpha == 0 else 3
        for metric in ("loss", "accuracy"):
            validate_stat(row[metric], count, accuracy=metric == "accuracy", nonnegative=metric == "loss")
            individuals = row[metric]["individuals"]
            require(all(i["training_run_id"] == run and type(i["training_seed"]) is int and i["training_seed"] == row["training_seed"] for i in individuals), "Realization/checkpoint identity mismatch")
            indices = [i["logical_index"] for i in individuals]
            require(all(type(i) is int and 0 <= i < 6720 for i in indices) and len(set(indices)) == count, "Duplicate/invalid logical realization indices")
            require(indices == list(range(indices[0], indices[0] + count)), "Realization indices out of order")
            start = global_indices[run]*28 + offsets[alpha]
            require(indices == list(range(start, start+count)), "Logical slot identity does not match prescribed grid")
            require(all(i["logical_index"] // 28 == checkpoints[(run, 0)][metric]["individuals"][0]["logical_index"] // 28 for i in individuals), "Realizations belong to another checkpoint")
        require([i["logical_index"] for i in row["loss"]["individuals"]] == [i["logical_index"] for i in row["accuracy"]["individuals"]], "Loss/accuracy realization linkage mismatch")
        require(row["loss"]["complete"] == row["accuracy"]["complete"], "Loss/accuracy completeness mismatch")
        if row["loss"]["complete"]:
            require(row["best_checkpoint_identity"] is not None, "Complete observation has no checkpoint")
            _sha(row["compatibility_signature"], "compatibility", "sha256:")
        if alpha == 0:
            require(row["noise_change_pp"] is None, "Clean observation has perturbation pairs")
        else:
            validate_stat(row["noise_change_pp"], 3)
            pairs = row["noise_change_pp"]["individuals"]
            require([i["evaluation_replicate"] for i in pairs] == [1, 2, 3] and
                    all(i["training_run_id"] == run and i["training_seed"] == row["training_seed"] and
                        i["canonical_evaluation_alpha"] == row["canonical_evaluation_alpha"] for i in pairs), "Paired realization identity mismatch")
            if row["noise_change_pp"]["complete"]:
                clean = checkpoints[(run, 0)]
                require(clean["accuracy"]["complete"] and row["accuracy"]["complete"] and
                        row["compatibility_signature"] == clean["compatibility_signature"], "Paired mean lacks compatible clean baseline")
                change = 100*(row["accuracy"]["official"]["mean"]-clean["accuracy"]["official"]["mean"])
                require(math.isclose(change, row["noise_change_pp"]["official"]["mean"], rel_tol=1e-12, abs_tol=1e-12), "Paired accuracy change mismatch")
    # Cross-seed compatibility is supplied by the producer, independently of each
    # checkpoint's own complete realization set. Preserve it as a separate gate.
    robust = _indexed(document["robustness_groups"], _eval_key, "robustness group")
    require(set(robust) == {(*g, e) for g in expected_groups for e in EVALUATION_ALPHAS}, "Missing expected cross-seed robustness groups")
    for key, group in robust.items():
        run_ids = groups[key[:3]]["training_run_ids"]
        signatures = {checkpoints[(run, key[-1])]["compatibility_signature"] for run in run_ids
                      if checkpoints[(run, key[-1])]["accuracy"]["complete"]}
        for metric in ("loss", "accuracy", "noise_change_pp"):
            if metric == "noise_change_pp" and key[-1] == 0:
                require(group[metric] is None, "Clean cross-seed group has noise changes")
                continue
            stat = group[metric]
            validate_stat(stat, 3, accuracy=metric == "accuracy", nonnegative=metric == "loss")
            require(len(signatures) <= 1 or stat["eligible_count"] == 0, "Incompatible checkpoint signatures used across seeds")
            require([(r["training_run_id"], r["training_seed"]) for r in stat["individuals"]] == list(zip(run_ids, seeds)), "Cross-seed group identity mismatch")
            for individual in stat["individuals"]:
                if individual["value"] is not None:
                    source = checkpoints[(individual["training_run_id"], key[-1])][metric]["official"]
                    require(source is not None and math.isclose(source["mean"], individual["value"], rel_tol=1e-12, abs_tol=1e-14), "Cross-seed value disagrees with checkpoint mean")
    require(document["coverage"]["expected_training"] == 240 and document["coverage"]["expected_evaluation_slots"] == 6720, "Coverage identity mismatch")
    pairs = _indexed(document["clean_seed_pairs"], lambda r: r["training_run_id"], "clean training pair")
    require(set(pairs) == set(runs), "Missing paired-training configuration identities")
    for run, pair in pairs.items():
        require(all(pair[k] == v for k, v in runs[run].items()), "Paired-training identity mismatch")
        base_group = groups[(pair["dataset"], pair["architecture"], 0)]
        baseline = base_group["training_run_ids"][seeds.index(pair["training_seed"])]
        require(pair["baseline_run_id"] == baseline, "Paired-training baseline identity mismatch")
        if pair["value"] is not None:
            _number(pair["value"], "paired accuracy change")
            require(pair["reason"] is None, "Eligible pair carries an exclusion")
        else:
            require(type(pair["reason"]) is str and bool(pair["reason"]), "Missing paired-training exclusion")
    return seeds


@dataclass(frozen=True)
class SummaryInput:
    root: Path
    document: dict
    seeds: tuple[int, ...]
    provenance: dict
    file_digests: dict[str, str]

    def assert_unchanged(self) -> None:
        require(not (self.root / "PUBLICATION_FAILURE.json").exists() and not (self.root / "UNCONFIRMED_COMPLETE.json").exists(),
                "Input publication changed during analysis")
        for name, expected in self.file_digests.items():
            path = plain_path(self.root / name)
            require(digest(path) == expected, f"Input reporting artifact changed: {name}")


def load_summaries(report_dir: str | Path) -> SummaryInput:
    """Reuse existing seal/strict-JSON validators without importing torch-bound I/O.

    Equivalent publication file checks to inspect_report, with summary-specific
    role and identity checks. Nothing here follows live scientific references.
    """
    try:
        root = plain_path(report_dir)
        require(root.is_dir(), "Missing summaries publication; see README reporting commands")
        for name in ("PUBLICATION_FAILURE.json", "UNCONFIRMED_COMPLETE.json"):
            require(not (root / name).exists(), f"Incomplete/corrupt publication: {name}")
        marker_digests = {name: digest(plain_path(root / name)) for name in ("OWNER.json", "COMPLETE.json")}
        owner = read_json(plain_path(root / "OWNER.json"))
        complete = read_json(plain_path(root / "COMPLETE.json"))
        verify_seal(owner)
        verify_seal(complete)
        require(type(owner["reporting_version"]) is int and owner["reporting_version"] == 1 and
                owner["namespace"] == "derived-reporting-v1" and owner["kind"] == "summaries" and
                owner["scientific_completion_authority"] is False, "Incompatible summaries publication owner")
        require(type(complete["reporting_version"]) is int and complete["reporting_version"] == 1 and
                complete["owner_sha256"] == owner["document_sha256"] and
                complete["status"] == "derived_publication_complete" and complete["scientific_completion_authority"] is False,
                "Incomplete/incompatible publication metadata")
        files = complete["files"]
        require(type(files) is list, "Malformed publication file references")
        paths, roles = {}, {}
        for ref in files:
            name = ref["relative_path"]
            require(type(name) is str and name not in paths, "Duplicate publication file reference")
            parsed = PurePosixPath(name)
            require(not parsed.is_absolute() and ".." not in parsed.parts and ":" not in name and "\\" not in name and str(parsed) == name and name != ".", "Unsafe publication reference")
            path = plain_path(root / name)
            require(root in path.parents and path.is_file(), "Missing publication file")
            _sha(ref["sha256"], "publication file")
            require(digest(path) == ref["sha256"], f"Corrupt publication file: {name}")
            paths[name], roles[name] = ref["sha256"], ref["role"]
        require(REQUIRED_FILES <= set(paths) and roles["summaries.json"] == "derived_summaries" and roles["coverage.json"] == "coverage", "Missing required summaries publication files/roles")
        document = read_json(root / "summaries.json")
        seeds = validate_summaries(document)
        for key in ("grid_id", "manifest_document_sha256", "collection_sha256"):
            require(owner[key] == document[key], f"Owner/summary {key} mismatch")
        require(read_json(root / "coverage.json") == document["coverage"], "Coverage document mismatch")
        paths.update(marker_digests)
        result = SummaryInput(root, document, seeds, deepcopy({
            "report_dir": str(root), "owner": owner, "completion": complete,
            "summary_document_sha256": document["document_sha256"], "summary_file_sha256": paths["summaries.json"],
            "grid_id": document["grid_id"], "manifest_document_sha256": document["manifest_document_sha256"],
            "collection_sha256": document["collection_sha256"],
            "validation_scope": "frozen published summary; no live checkpoint/raw authority reread",
        }), paths)
        result.assert_unchanged()
        return result
    except AnalysisError:
        raise
    except (OSError, ValueError, KeyError, TypeError, IndexError, OverflowError) as exc:
        raise AnalysisError(f"Invalid summaries publication: {exc}") from exc
