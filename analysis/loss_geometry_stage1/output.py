"""Exclusive, separate outputs with provenance and an end-of-write checksum marker."""

import csv
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import platform

from alpha_sweep_training.serialization import strict_json_text

from . import __version__
from .contracts import ALPHAS, ARCHITECTURES, DATASETS, LIMITATIONS, WINDOWS, require
from .loading import SummaryInput, digest, plain_path


def validate_output_path(output: str | Path, report: Path, protected: list[str | Path]) -> Path:
    require(bool(protected), "Declare all scientific/workflow/original-report storage with --protected-root")
    target = plain_path(output)
    repository = Path(__file__).resolve().parents[2]
    for value in (report, repository, *protected):
        root = plain_path(value)
        require(target != root and root not in target.parents and target not in root.parents,
                f"Output overlaps protected input/research/source storage: {root}")
    require(not target.exists(), f"Output directory already exists; choose a new path: {target}")
    return target


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(strict_json_text(value))


def write_csv(path: Path, rows: list[dict]) -> None:
    require(bool(rows), f"No schema/rows for CSV {path.name}")
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if value is None else json.dumps(value, sort_keys=True, allow_nan=False, ensure_ascii=False)
                             if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()})


def analysis_metadata(source: SummaryInput, options: dict) -> dict:
    versions = {"python": platform.python_version(), "analysis": __version__}
    for name in ("alpha-sweep-training", "numpy", "matplotlib"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    package = Path(__file__).resolve().parent
    repository = package.parents[1]
    code = {p.relative_to(repository).as_posix(): digest(p) for p in sorted(package.glob("*.py"))}
    for name in ("aggregation.py", "reporting_contract.py", "serialization.py", "spec.py", "identity.py", "data_contract.py", "config.py", "manifests.py"):
        path = repository / "src" / "alpha_sweep_training" / name
        code[path.relative_to(repository).as_posix()] = digest(path)
    code["pyproject.toml"] = digest(repository / "pyproject.toml")
    return dict(analysis_version=__version__, created_at_utc=datetime.now(timezone.utc).isoformat(),
        input_provenance=source.provenance, options=options, software_versions=versions, source_file_sha256=code,
        training_seeds=list(source.seeds), fitting_windows=WINDOWS,
        equations=dict(delta_loss="mean_positive_loss - same_checkpoint_clean_loss",
            delta_accuracy_pp="100 * (perturbed_accuracy_fraction - same_checkpoint_clean_accuracy_fraction)",
            coefficient_c="sum(alpha^2 * delta_loss) / sum(alpha^4)",
            empirical_curvature_proxy="2*c; not a directly measured Hessian trace",
            rmse="sqrt(RSS/n), n is the number of fitting points; no degrees-of-freedom correction",
            r_squared="1 - RSS / sum((observed_delta_loss - mean(observed_delta_loss))^2); centered, unclamped; null if denominator is zero"),
        aggregation="Complete within-checkpoint realization means supplied by reporting-v1; individual-seed fits first; complete compatible three-seed mean and sample SD ddof=1 afterward",
        fit_policy="All prescribed fitting alphas required; unweighted origin OLS; coefficients and loss differences may be negative",
        breakdown_policy="Residuals only; no formal breakdown threshold defined",
        limitations=list(LIMITATIONS), scientific_completion_authority=False)


def findings(result: dict, figure_index: list[dict]) -> str:
    coverage = result["coverage"]
    lines = ["# Stage 1: loss geometry findings", "",
        f"Accepted loss responses: {coverage['eligible_loss_responses']}/2400 (including clean anchors). "
        f"Eligible individual-checkpoint fits: {coverage['eligible_fits']}/480. "
        "Publication completeness and scientific coverage are separate; see coverage.json for every expected configuration and realization.", "",
        "Differences below are perturbation training minus clean training, paired by training seed. "
        "Negative proxy differences indicate lower fitted loss sensitivity. Every training alpha is retained; "
        "no test result selects a preferred alpha. Only complete compatible sets produce means and sample SDs.", ""]
    def fmt(value):
        return "unavailable" if value is None else f"{value:.6g}"
    for dataset in DATASETS:
        for architecture in ARCHITECTURES:
            lines += [f"## {dataset} / {architecture}", "",
                "| Training alpha | Clean accuracy (%) | Primary 2c mean +/- SD | Paired primary 2c difference | Secondary 2c mean +/- SD | Seeds (primary/secondary) |",
                "| --- | --- | --- | --- | --- | --- |"]
            for alpha in ALPHAS:
                def summary(metric, window=None):
                    return next(s for s in result["summaries"] if (s["dataset"], s["architecture"], s["training_alpha"], s["metric"], s["fit_window"]) ==
                                (dataset, architecture, alpha, metric, window) and (window is not None or s["evaluation_alpha"] == 0))
                primary = summary("empirical_curvature_proxy_2c", "primary")
                secondary = summary("empirical_curvature_proxy_2c", "secondary")
                clean = summary("clean_accuracy")
                pair = next((s for s in result["paired_summaries"] if (s["dataset"], s["architecture"], s["training_alpha"], s["fit_window"]) ==
                             (dataset, architecture, alpha, "primary")), None)
                lines.append(f"| {alpha:g} | {fmt(None if clean['mean'] is None else 100*clean['mean'])} | {fmt(primary['mean'])} +/- {fmt(primary['sample_sd'])} | "
                             f"{'reference' if alpha == 0 else fmt(pair['mean'])} | {fmt(secondary['mean'])} +/- {fmt(secondary['sample_sd'])} | {primary['eligible_count']}/3; {secondary['eligible_count']}/3 |")
            own = [r for r in result["loss_sensitivity"] if r["dataset"] == dataset and r["architecture"] == architecture and r["eligible"]]
            acc = [100*r["perturbed_accuracy"] for r in own]
            lines += ["", f"Accepted absolute perturbed accuracy spans {fmt(min(acc) if acc else None)} to {fmt(max(acc) if acc else None)} percent across checkpoints and evaluation alphas. "
                      "Inspect individual accuracies before interpreting small loss responses as useful classification performance.", "",
                      "| Evaluation alpha | Compatibility signature (prefix) | Pearson r: Delta loss vs accuracy degradation | Pairs / expected in scope |",
                      "| --- | --- | --- | --- |"]
            for r in result["relationships"]:
                if (r["dataset"], r["architecture"], r["training_alpha"], r["x_metric"]) == (dataset, architecture, None, "delta_loss"):
                    lines.append(f"| {r['evaluation_alpha']:g} | {str(r['compatibility_signature'])[:19]} | {fmt(r['pearson_r'])} ({r['status']}) | {r['count']}/30 |")
            lines += ["", "These correlations condition on evaluation alpha but vary training alpha and seed; they describe association, not a causal or independent-replicate test. "
                      "Within-training-alpha and fitted-proxy associations are in relationships.csv.", ""]
            for window in WINDOWS:
                comparisons = [s for s in result["paired_summaries"] if s["dataset"] == dataset and s["architecture"] == architecture and
                               s["fit_window"] == window and s["complete"]]
                smaller = sum(s["mean"] < 0 for s in comparisons)
                larger = sum(s["mean"] > 0 for s in comparisons)
                ties = sum(s["mean"] == 0 for s in comparisons)
                lines.append(f"For {window} proxy comparisons, {len(comparisons)}/9 positive training alphas have complete paired summaries: "
                             f"{smaller} smaller, {larger} larger and {ties} equal to clean training in mean 2c. These counts are descriptive, not a significance test.")
            lines.append("")
            for window in WINDOWS:
                residuals = [p for p in result["predictions"] if p["dataset"] == dataset and p["architecture"] == architecture and
                             p["fit_window"] == window and p["point_role"] == "extrapolation" and p["residual"] is not None]
                if residuals:
                    largest = max(residuals, key=lambda p: abs(p["residual"]))
                    lines.append(f"For {window} fits, the largest available absolute extrapolation residual is {abs(largest['residual']):.6g} nats/example "
                                 f"at evaluation alpha {largest['evaluation_alpha']:g} ({largest['training_run_id']}). This is a residual diagnostic, not a breakdown threshold.")
                else:
                    lines.append(f"No eligible extrapolation residuals for {window} fits.")
            lines.append("")
    unavailable = [f for f in figure_index if f["status"] != "generated"]
    lines += ["## Figures and interpretation", "", f"Generated figure documents: {len(figure_index)-len(unavailable)}; unavailable: {len(unavailable)}. "
              "Each generated document has PNG and PDF versions. figure_index.json records missing-data reasons.", ""]
    lines += [f"- {item}" for item in LIMITATIONS]
    lines += ["", "No Hessian, training, checkpoint loading or neural-network evaluation was performed by this analysis.", ""]
    return "\n".join(lines)


def publish(source: SummaryInput, result: dict, output: Path, protected: list[str], *, dpi: int) -> dict:
    from .plotting import plot_analysis
    # Dependency availability and input validity are checked before claiming output.
    try:
        import matplotlib  # Availability only; rendering uses Figure/CanvasAgg.
    except ImportError as exc:
        from .contracts import AnalysisError
        raise AnalysisError("Matplotlib unavailable; use an environment with the existing reporting extra") from exc
    output = validate_output_path(output, source.root, protected)
    options = dict(dpi=dpi, formats=["png", "pdf"], output_dir=str(output), protected_roots=[str(plain_path(p)) for p in protected])
    metadata_doc = analysis_metadata(source, options)
    source.assert_unchanged()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()  # Exclusive claim; a concurrent publisher cannot reuse it.
    try:
        write_json(output / "analysis_metadata.json", metadata_doc)
        for name, key in (("loss_sensitivity", "loss_sensitivity"), ("quadratic_fits", "quadratic_fits"),
                          ("group_summaries", "summaries"), ("predictions_residuals", "predictions"),
                          ("paired_training_comparisons", "training_pairs"), ("paired_training_summaries", "paired_summaries"),
                          ("relationships", "relationships")):
            write_csv(output / f"{name}.csv", result[key])
        write_json(output / "coverage.json", result["coverage"])
        figure_index = plot_analysis(result, output, source.seeds, dpi)
        write_json(output / "figure_index.json", figure_index)
        with (output / "findings.md").open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(findings(result, figure_index))
        source.assert_unchanged()
        files = [{"relative_path": p.relative_to(output).as_posix(), "sha256": digest(p)} for p in sorted(output.rglob("*")) if p.is_file()]
        completion = dict(status="stage1_analysis_outputs_complete", analysis_version=__version__, files=files,
                          summary_document_sha256=source.document["document_sha256"], scientific_completion_authority=False)
        write_json(output / "COMPLETE.json", completion)
        return dict(output_dir=str(output), eligible_fits=result["coverage"]["eligible_fits"], figure_documents=sum(f["status"] == "generated" for f in figure_index))
    except Exception as exc:
        write_json(output / "ANALYSIS_FAILURE.json", dict(error=f"{type(exc).__name__}: {exc}", scientific_completion_authority=False))
        raise
