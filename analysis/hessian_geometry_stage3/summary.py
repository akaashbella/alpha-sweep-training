"""Checkpoint-first statistics with explicit completeness and paired probe sums."""

from collections import defaultdict
import math
import statistics

from analysis.layer_sensitivity_stage2.report import write_csv

from .records import namespace, observe
from .storage import code_digest, disjoint, finish, identity, install, seal

COUNTS = (1, 2, 4, 8, 16, 32, 64)


def stats(values, expected):
    valid = [v for v in values if v is not None]
    complete = len(valid) == expected == len(values)
    sd = statistics.stdev(valid) if complete and expected > 1 else None
    return {"expected": expected, "eligible": len(valid), "complete": complete,
            "mean": math.fsum(v/expected for v in valid) if complete else None,
            "sample_sd": sd, "mc_se": None if sd is None else sd/math.sqrt(expected)}


def ratio(numerator, denominator, scale=1.):
    return None if numerator is None or denominator is None or abs(denominator) <= max(1e-12, 1e-9*abs(scale)) else numerator/denominator


def target_specs(layout):
    all_names = [t["name"] for t in layout["tensors"]]
    yield "network:all", "network", "all", all_names
    for name in layout["stage_order"]:
        yield f"stage:{name}", "stage", name, layout["stages"][name]
    for name in all_names:
        yield f"tensor:{name}", "tensor", name, [name]
    for name, names in sorted(layout["diagnostics"].items()):
        yield f"diagnostic:{name}", "diagnostic", name, names


def value_for(record, scope, name):
    c = record["measurement"]["contributions"]
    return c["whole"] if scope == "network" else c[{"stage": "stages", "tensor": "tensors", "diagnostic": "diagnostics"}[scope]][name]


def pairs(traces):
    base = {(r["dataset"], r["architecture"], r["training_seed"], r["target_id"]): r for r in traces if r["training_alpha"] == 0}
    rows = []
    for trace in traces:
        if trace["training_alpha"] == 0:
            continue
        key = tuple(trace[k] for k in ("dataset", "architecture", "training_seed", "target_id"))
        clean = base.get(key)
        reason = None
        if not clean or not clean["complete"] or not trace["complete"]:
            reason = "Missing complete checkpoint estimate"
        elif trace["cohort_sha256"] != clean["cohort_sha256"]:
            reason = "Incompatible recipe, data, inventory or numerical settings"
        rows.append({**{k: trace[k] for k in ("checkpoint_id", "dataset", "architecture", "training_seed", "training_alpha",
                    "target_id", "target_name", "scope", "split", "eligible", "expected")}, "checkpoint": trace["checkpoint"],
                    "clean_training_checkpoint": clean["checkpoint"] if clean else None,
                    "clean_training_checkpoint_id": clean["checkpoint_id"] if clean else None,
                    "clean_trace": None if clean is None else clean["mean"], "clean_mc_se": None if clean is None else clean["mc_se"],
                    "noise_trace": trace["mean"], "noise_mc_se": trace["mc_se"],
                    "noise_minus_clean": None if reason else trace["mean"]-clean["mean"], "reason": reason})
    return rows


def seed_aggregation(traces, expected_seeds):
    bins = defaultdict(dict)
    for trace in traces:
        key = tuple(trace[k] for k in ("dataset", "architecture", "training_alpha", "target_id", "split"))
        bins[key][trace["training_seed"]] = trace
    rows = []
    for key, entries in bins.items():
        observed = [entries.get(seed) for seed in expected_seeds]
        seed_values = [r["mean"] if r else None for r in observed]
        summary = stats(seed_values, len(expected_seeds))
        signatures = {r["cohort_sha256"] for r in observed if r and r["complete"]}
        compatible = len(signatures) <= 1
        if not compatible:
            summary.update(complete=False, mean=None, sample_sd=None)
        del summary["mc_se"]
        seed_sd = summary.pop("sample_sd")
        rows.append({**dict(zip(("dataset", "architecture", "training_alpha", "target_id", "split"), key)),
                     **summary, "training_seed_sample_sd": seed_sd, "compatible": compatible,
                     "individual_seed_means": dict(zip(map(str, expected_seeds), seed_values))})
    return rows


def collect(plan, plan_dir, results):
    root = namespace(plan, plan_dir, results)
    core, count = plan["core"], plan["count"]
    traces, normalized, convergence, coverage = [], [], [], []
    for row in core["checkpoints"]:
        stage2 = row["stage2_row"]
        layout = stage2["layout"]
        failure = None
        try:
            records, inv, status = observe(plan, row, root)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
            failure = f"{type(exc).__name__}: {exc}"
            records, inv = {}, None
            status = {"checkpoint_id": row["checkpoint_id"], **row["key"], "publication_complete": False,
                      "slots": [{"probe": p, "status": "rejected", "reason": failure} for p in range(1, count+1)]}
        status.update(eligible=len(records), expected=count, numerical_status="rejected" if failure else "validated_successes_only")
        coverage.append(status)
        first = next(iter(records.values()), None)
        binding = stage2["evidence"]["binding"]
        settings = first["runtime"] if first else None
        recipe = {k: v for k, v in stage2["evidence"]["scientific"].items() if k not in ("alpha", "training_seed")}
        common = {**row["key"], "checkpoint_id": row["checkpoint_id"], "checkpoint": binding["best_checkpoint_identity"],
                  "scientific_fingerprint": binding["training_identity"]["scientific_fingerprint"], "split": core["split"],
                  "cohort_sha256": identity({"recipe": recipe, "inventory": layout, "runtime": settings,
                                            "prepared": stage2["evidence"]["prepared_data"], "split": core["split"]}),
                  "runtime": settings, "clean_loss": first["measurement"]["clean_loss"] if first else None,
                  "clean_accuracy": first["measurement"]["clean_accuracy"] if first else None,
                  "numerical_status": status["numerical_status"], "stage2_clean_loss": row["stage2"]["metrics"]["loss"],
                  "stage2_clean_accuracy": row["stage2"]["metrics"]["accuracy"], "stage2_runtime": row["stage2"]["runtime"]}
        names = {t["name"]: t["numel"] for t in layout["tensors"]}
        totals = sum(names.values())
        q_full = [value_for(records[p], "network", "all") if p in records else None for p in range(1, count+1)]
        full = stats(q_full, count)["mean"]
        trace_scale = math.fsum(abs(stats([value_for(records[p], "tensor", name) for p in records], count)["mean"]) for name in names) if full is not None else 0.
        for target, scope, name, members in target_specs(layout):
            values = {str(p): value_for(record, scope, name) for p, record in records.items()}
            q = [values.get(str(p)) for p in range(1, count+1)]
            spec = {**common, "target_id": target, "scope": scope, "target_name": name}
            trace = {**spec, **stats(q, count), "probe_values": values, "missing_probes": [p for p in range(1, count+1) if p not in records]}
            traces.append(trace)
            n = sum(names[t] for t in members)
            energy = math.fsum(names[t]*inv["scales"][t]**2 for t in members) if inv else None
            normalized.append({**spec, "mean": trace["mean"], "eligible_elements": n, "parameter_fraction": n/totals,
                               "variance_energy": energy,
                               "trace_per_million": None if trace["mean"] is None else 1e6*trace["mean"]/n,
                               "trace_per_variance_energy": ratio(trace["mean"], energy),
                               "fraction_of_full_trace": ratio(trace["mean"], full, trace_scale)})
            for prefix in (p for p in COUNTS if p <= count):
                convergence.append({**spec, "probe_prefix": prefix, **stats(q[:prefix], prefix)})
    seeds = sorted({r["key"]["training_seed"] for r in core["checkpoints"]})
    return {"traces": traces, "normalized": normalized, "convergence": convergence, "pairs": pairs(traces),
            "seed_summaries": seed_aggregation(traces, seeds), "coverage": coverage}


def write_publication(output, protected, kind, document, tables, extras):
    root = disjoint(output, protected)
    root.mkdir(parents=True)
    names = []
    for name, rows in tables.items():
        write_csv(root/name, rows, ["checkpoint_id", "target_id", "status"])
        names.append(name)
    for name, value in extras.items():
        if isinstance(value, str):
            with (root/name).open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(value)
        else:
            install(root/name, value)
        names.append(name)
    install(root/f"{kind}.json", document)
    finish(root, kind, [*names, f"{kind}.json"])
    return root


def summarize(plan, plan_dir, results, output):
    core = plan["core"]
    result = collect(plan, plan_dir, results)
    metadata = {"plan_sha256": plan["document_sha256"], "work_sha256": plan["work_sha256"], "code_sha256": code_digest(),
                "stage2_plan_sha256": core["stage2_plan_sha256"], "split": core["split"], "numerics": core["numerics"],
                "randomness": core["randomness"], "distribution": core["distribution"], "workload": plan["workload"],
                "extensions": plan["extensions"],
                "protected_roots": [*core["protected_roots"], str(plan_dir), str(results)]}
    document = seal("summary", {**metadata, **result})
    tables = {
        "whole_network_traces.csv": [r for r in result["traces"] if r["scope"] == "network"],
        "tensor_traces.csv": [r for r in result["traces"] if r["scope"] == "tensor"],
        "stage_traces.csv": [r for r in result["traces"] if r["scope"] == "stage"],
        "diagnostic_traces.csv": [r for r in result["traces"] if r["scope"] == "diagnostic"],
        "normalized_curvature.csv": result["normalized"], "probe_convergence.csv": result["convergence"],
        "paired_training_comparisons.csv": result["pairs"], "training_seed_summaries.csv": result["seed_summaries"],
    }
    valid = sum(r["eligible"] for r in result["coverage"])
    complete = sum(r["publication_complete"] for r in result["coverage"])
    findings = ("# Stage 3 weighted Hessian traces\n\n"
                f"Validated {valid} of {plan['workload']['dataset_probes']} dataset probes and {complete} of "
                f"{len(result['coverage'])} checkpoint completion markers. The objective uses the {core['split']} split.\n\n"
                "The trace estimates are signed. Monte Carlo SE describes independent probe draws at a fixed checkpoint. "
                "This pilot uses one training seed. The trace depends on the parameterization and the perturbation covariance. "
                "Agreement with the earlier empirical responses is assessed in the comparison publication.\n")
    target = write_publication(output, metadata["protected_roots"], "summary", document, tables,
                               {"analysis_metadata.json": seal("analysis_metadata", metadata),
                                "coverage.json": seal("coverage", {"checkpoints": result["coverage"]}), "findings.md": findings})
    return {"output": str(target), "accepted_probes": valid, "completed_checkpoints": complete}
