"""Compare immutable empirical responses with the independently estimated traces."""

import math
import statistics
from collections import defaultdict

from analysis.layer_sensitivity_stage2.records import run_root
from analysis.loss_geometry_stage1.statistics import quadratic_fit

from .protocol import corrected
from .provenance import call_reader, stage2_checkpoint, stage2_plan
from .storage import code_digest, plain, publication, require, seal, sha_file
from .summary import stats, write_publication


def comparison_metrics(observed, kappa, alpha, mc_se):
    factor = .5*alpha**2
    prediction = None if kappa is None else factor*kappa
    residual = None if observed is None or prediction is None else observed-prediction
    return {"observed_delta_loss": observed, "predicted_delta_loss": prediction, "residual": residual,
            "absolute_residual": None if residual is None else abs(residual),
            "prediction_mc_se": None if mc_se is None else factor*mc_se}


def settings_relation(trace, other, *, batch_size):
    runtime = trace["runtime"]
    if not runtime or not other:
        return {"status": "unavailable", "differences": [], "stage3": runtime, "earlier": other}
    differences, unknown = [], []
    precision = other["precision"]
    effective = other["effective_backend"]
    if precision.get("mode") != "float32":
        differences.append("forward precision/autocast")
    for key in ("allow_tf32_matmul", "allow_tf32_cudnn", "cudnn_enabled", "cudnn_benchmark", "cudnn_deterministic",
                "deterministic_algorithms", "deterministic_warn_only", "float32_matmul_precision", "cublas_workspace_config"):
        if key not in effective:
            unknown.append(key)
        elif effective[key] != runtime["effective"][key]:
            differences.append(key)
    if batch_size != 128:
        differences.append("batch size")
    device = other.get("device_properties", {})
    for name, current in (("torch", runtime["software"]["torch"]), ("torchvision", runtime["software"]["torchvision"]),
                          ("cuda", runtime["software"]["cuda"]), ("cudnn", runtime["software"]["cudnn"])):
        historical = other.get(name, device.get(name))
        if historical is None:
            unknown.append(name)
        elif str(historical) != str(current):
            differences.append(name)
    if device.get("name") is None:
        unknown.append("device")
    elif runtime["gpu"]["name"] != device["name"]:
        differences.append("device")
    return {"status": "different" if differences else "partially_recorded" if unknown else "matching_recorded_settings",
            "differences": differences, "unknown": unknown, "stage3": runtime, "earlier": other}


def dimensions(trace):
    return {k: trace[k] for k in ("dataset", "architecture", "training_seed", "training_alpha", "checkpoint_id",
                                  "checkpoint", "scientific_fingerprint", "target_id", "target_name", "scope")}


def fit_pair(trace, observations, context):
    rows = []
    for window in ("primary", "secondary"):
        fit = quadratic_fit(observations, window)
        proxy = fit["empirical_curvature_proxy_2c"]
        rows.append({**dimensions(trace), **context, **fit, "hessian_trace": trace["mean"],
                     "hessian_mc_se": trace["mc_se"], "probe_count": trace["eligible"],
                     "proxy_minus_trace": None if proxy is None or trace["mean"] is None else proxy-trace["mean"]})
    return rows


def stage2_comparison(plan, audit_row, stage2, raw, traces):
    core = plan["core"]
    clean = raw["clean", 0., 0]
    matched = core["split"] == "validation" and clean["data"] == audit_row["stage2"]["data"]
    traces_by_target = {r["target_id"]: r for r in traces}
    measurements, fits, interactions, ranks = [], [], [], []
    reps = stage2["realizations_per_positive_alpha"]
    for target in audit_row["stage2_row"]["targets"]:
        trace = traces_by_target[target["id"]]
        points = {}
        match = settings_relation(trace, clean["runtime"], batch_size=core["batch_size"])
        context = {"hessian_split": core["split"], "empirical_split": stage2["split"],
                   "split_match": matched, "precision": match, "empirical_source": "stage2",
                   "status": "matched_validation" if matched else "excluded_split_mismatch"}
        for alpha in stage2["evaluation_alphas"][1:]:
            items = [raw[target["id"], alpha, rep] for rep in range(1, reps+1)]
            observed = stats([item["changes"]["delta_loss"] for item in items], reps)
            accuracy = stats([item["changes"]["accuracy_change_pp"] for item in items], reps)
            points[alpha] = observed["mean"] if matched else None
            fields = comparison_metrics(observed["mean"] if matched else None, trace["mean"], alpha, trace["mc_se"])
            measurements.append({**dimensions(trace), **context, **fields, "evaluation_alpha": alpha,
                                 "stage2_loss_mean": observed["mean"], "stage2_realizations": observed["eligible"],
                                 "stage2_sample_sd": observed["sample_sd"], "stage2_perturbation_se": observed["mc_se"],
                                 "stage3_probes": trace["eligible"], "stage3_expected_probes": trace["expected"],
                                 "stage3_trace": trace["mean"], "stage3_sample_sd": trace["sample_sd"], "stage3_mc_se": trace["mc_se"],
                                 "stage2_clean_loss": clean["metrics"]["loss"], "stage2_clean_accuracy": clean["metrics"]["accuracy"],
                                 "stage3_clean_loss": trace["clean_loss"], "stage3_clean_accuracy": trace["clean_accuracy"],
                                 "stage2_accuracy_change_pp": accuracy["mean"], "stage2_accuracy_change_se": accuracy["mc_se"]})
        fits.extend(fit_pair(trace, points, context))
    network = traces_by_target["network:all"]
    stages = ["stage:"+name for name in audit_row["stage2_row"]["layout"]["stage_order"]]
    targets = {target["id"] for target in audit_row["stage2_row"]["targets"]}
    for alpha in stage2["evaluation_alphas"][1:]:
        if not set(stages) <= targets:
            continue
        within_alpha = [r for r in measurements if r["evaluation_alpha"] == alpha and r["scope"] == "stage"]
        for entry in within_alpha:
            # Competition ranks retain exact ties. Only rank the full audited partition.
            if matched and len(within_alpha) == len(stages) and all(r["predicted_delta_loss"] is not None for r in within_alpha):
                ranks.append({**dimensions(entry), "evaluation_alpha": alpha,
                              "observed_rank": 1 + sum(r["observed_delta_loss"] > entry["observed_delta_loss"] for r in within_alpha),
                              "predicted_rank": 1 + sum(r["predicted_delta_loss"] > entry["predicted_delta_loss"] for r in within_alpha)})
        residuals = [raw["network:all", alpha, rep]["changes"]["delta_loss"] -
                     math.fsum(raw[stage, alpha, rep]["changes"]["delta_loss"] for stage in stages)
                     for rep in range(1, reps+1)]
        local = None
        if matched and network["mean"] is not None:
            local = .5*alpha**2*(network["mean"] - math.fsum(traces_by_target[t]["mean"] for t in stages))
        interactions.append({**dimensions(network), "evaluation_alpha": alpha, "empirical_split": stage2["split"],
                             "hessian_split": core["split"], "split_match": matched,
                             "stage2_paired_residuals": residuals, **stats(residuals, reps), "local_algebraic_residual": local})
    baseline = {**dimensions(network), "hessian_split": core["split"], "empirical_split": stage2["split"],
                "split_match": matched, "precision": settings_relation(network, clean["runtime"], batch_size=core["batch_size"]),
                "stage2_clean_loss": clean["metrics"]["loss"], "stage3_clean_loss": network["clean_loss"],
                "stage2_clean_accuracy": clean["metrics"]["accuracy"], "stage3_clean_accuracy": network["clean_accuracy"],
                "loss_difference": None if not matched or network["clean_loss"] is None else network["clean_loss"]-clean["metrics"]["loss"],
                "accuracy_difference_pp": None if not matched or network["clean_accuracy"] is None else
                                          100*(network["clean_accuracy"]-clean["metrics"]["accuracy"])}
    return measurements, fits, interactions, baseline, ranks


def load_stage1(report, collection):
    from alpha_sweep_training.report_publication import load_collection
    from analysis.loss_geometry_stage1.loading import load_summaries
    from analysis.loss_geometry_stage1.analysis import analyze

    source = load_summaries(report)
    frozen = load_collection(collection)
    require(frozen["document_sha256"] == source.document["collection_sha256"] and
            frozen["manifest_document_sha256"] == source.document["manifest_document_sha256"],
            "Stage 1 summary and frozen original collection do not match")
    analysis = analyze(source.document, source.seeds)
    source.assert_unchanged()
    slots = defaultdict(list)
    for row in frozen["evaluation"]:
        slots[row["training_run_id"], row["slot"]["evaluation_alpha"]].append(row)
    return ({(r["training_run_id"], r["evaluation_alpha"]): r for r in analysis["loss_sensitivity"]}, slots,
            source.provenance, frozen["protected_roots"])


def frozen_test_evidence(source, row):
    """Require fingerprints, actual identities and complete original Stage 1 eligibility."""
    from alpha_sweep_training.aggregation import _signature

    response, slots, _, _ = source
    native = row["stage2_row"]
    run = native["run_id"]
    binding = native["evidence"]["binding"]
    best = binding["best_checkpoint_identity"]
    fingerprint = binding["training_identity"]["scientific_fingerprint"]
    data = native["evidence"]["prepared_data"]
    points, runtimes, refs = {}, [], []
    clean = slots.get((run, 0.), [])
    if not clean or not clean[0]["successful"]:
        return points, runtimes, refs
    clean_signature = _signature(clean[0])
    for (run_id, alpha), item in response.items():
        if run_id != run or alpha == 0 or not item["eligible"] or item["best_checkpoint_identity"] != best:
            continue
        raw = [*clean, *slots[run, alpha]]
        if len(raw) != 4 or not all(r["successful"] and r["best_checkpoint_identity"] == best and
                r["training_scientific_fingerprint"] == fingerprint and _signature(r) == clean_signature and
                all(r["raw_record"]["data_identity"].get(k) == v for k, v in data.items()) for r in raw):
            continue
        points[alpha] = item["delta_loss"]
        result = raw[0]["raw_record"]
        software = result["evaluation_provenance"]["software"]
        runtimes.append({**result["runtime"], "torch": software["torch"], "torchvision": software["torchvision"]})
        refs.extend(r["raw_reference"] for r in raw)
    return points, runtimes, refs


def compare(plan, summary_dir, output, *, stage1_report=None, stage1_collection=None, historical_evaluation_root=None):
    summary = publication(summary_dir, "summary")
    core = plan["core"]
    require(summary["plan_sha256"] == plan["document_sha256"], "Summary belongs to another plan")
    directory = plain(core["stage2_plan_dir"])
    stage2 = stage2_plan(directory)
    require(stage2["document_sha256"] == core["stage2_plan_sha256"] and
            sha_file(directory/"COMPLETE.json") == core["stage2_plan_marker_sha256"], "Stage 2 plan changed")
    root = run_root(stage2, directory, core["stage2_results"], create=False)
    protect = [*summary["protected_roots"], str(plain(summary_dir))]
    require(bool(stage1_report) == bool(stage1_collection), "Supply Stage 1 summaries and their matching frozen collection together")
    source = load_stage1(stage1_report, stage1_collection) if stage1_report else None
    if source:
        protect.extend([str(plain(stage1_report)), str(plain(stage1_collection)), *source[3]])
    if historical_evaluation_root:
        protect.append(str(plain(historical_evaluation_root)))
    empirical, fits, interaction, baselines, rankings, stage1 = [], [], [], [], [], []
    by_checkpoint = defaultdict(list)
    failures = []
    for trace in summary["traces"]:
        by_checkpoint[trace["checkpoint_id"]].append(trace)
    for row in core["checkpoints"]:
        corrected(row["stage2_row"])
        checkpoint_id = row["checkpoint_id"]
        checkpoint = next(r for r in stage2["checkpoints"] if r["checkpoint_id"] == checkpoint_id)
        require(checkpoint == row["stage2_row"], "Stage 2 audit row identity changed")
        raw, inv = stage2_checkpoint(stage2, checkpoint, root)
        require(sha_file(root/"checkpoints"/checkpoint_id/"COMPLETE.json") == row["stage2"]["complete_file_sha256"]
                and inv["document_sha256"] == row["stage2"]["inventory_sha256"]
                and raw["clean", 0., 0]["document_sha256"] == row["stage2"]["clean_sha256"], "Stage 2 result binding changed")
        traces = by_checkpoint[checkpoint_id]
        observations, local_fits, residual, baseline, ranks = stage2_comparison(plan, row, stage2, raw, traces)
        empirical.extend(observations)
        fits.extend(local_fits)
        interaction.extend(residual)
        baselines.append(baseline)
        rankings.extend(ranks)
        network = next(r for r in traces if r["scope"] == "network")
        points, runtimes, references = frozen_test_evidence(source, row) if source else ({}, [], [])
        empirical_source = "stage1_frozen_report"
        if not points and historical_evaluation_root and row["reader"]["historical"]:
            try:
                native = call_reader(row["reader"], "test_evidence", {"row": checkpoint, "evaluation_root": str(plain(historical_evaluation_root))})
                points = {float(k): v for k, v in native["points"].items()}
                runtimes, references = [native["runtime"]], native["references"]
                empirical_source = "corrected_native_test_results_stage1_method"
            except (ValueError, OSError, RuntimeError, KeyError) as exc:
                failures.append({"checkpoint_id": checkpoint_id, "reason": str(exc)})
        match = core["split"] == "test"
        context = {"hessian_split": core["split"], "empirical_split": "test", "split_match": match,
                   "status": ("split_matched_test" if match else "cross_split_exploratory") if points else "missing_exact_corrected_test_evidence",
                   "precision": settings_relation(network, runtimes[0] if runtimes else None, batch_size=core["batch_size"]),
                   "empirical_source": empirical_source, "raw_references": references}
        stage1.extend(fit_pair(network, points, context))
    metadata = {"summary_sha256": summary["document_sha256"], "plan_sha256": plan["document_sha256"],
                "stage2_plan_sha256": stage2["document_sha256"], "stage1_provenance": source[2] if source else None,
                "code_sha256": code_digest(), "protected_roots": protect, "historical_test_failures": failures}
    document = seal("comparison", {**metadata, "stage1": stage1, "stage2": empirical, "fits": fits,
                                    "interactions": interaction, "baselines": baselines, "rankings": rankings})
    available = [r for r in empirical if r["residual"] is not None]
    same = sum(r["precision"]["status"] == "matching_recorded_settings" for r in available)
    grouped = defaultdict(list)
    for r in available:
        grouped[r["scope"], r["evaluation_alpha"]].append(r["absolute_residual"])
    lines = ["# Comparison with empirical perturbations", "",
             f"There are {len(available)} split-matched Stage 2 residuals, with {same} also matching all checked numerical settings.",
             "The clean baseline table retains the observed precision-policy differences.", "",
             "| Target scope | Evaluation alpha | Matched rows | Median absolute residual (nats/example) |",
             "| --- | --- | --- | --- |"]
    lines.extend(f"| {scope} | {alpha:g} | {len(x)} | {statistics.median(x):.6g} |" for (scope, alpha), x in sorted(grouped.items()))
    lines.extend(["", "Stage 1 test responses and validation Hessians describe different objectives. The comparison retains that split label.",
                  "Trace Monte Carlo SE and empirical perturbation SE describe different measurements and remain separate.",
                  "The algebraic partition check is distinct from the Stage 2 finite-perturbation interaction residual.",
                  "Agreement and disagreement are descriptive and have no prespecified breakdown threshold.", ""])
    target = write_publication(output, protect, "comparison", document,
                               {"stage1_comparisons.csv": stage1, "stage2_comparisons.csv": empirical,
                                "empirical_quadratic_fits.csv": fits, "approximation_residuals.csv": empirical,
                                "stage2_interactions.csv": interaction, "baseline_comparisons.csv": baselines, "stage_rankings.csv": rankings},
                               {"analysis_metadata.json": seal("analysis_metadata", metadata), "findings.md": "\n".join(lines)})
    return {"output": str(target), "stage2_matched_residuals": len(available), "stage1_eligible_fits": sum(r["eligible"] for r in stage1)}
