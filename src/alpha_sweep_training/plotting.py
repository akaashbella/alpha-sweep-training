"""Lazy headless scientific figures from validated frozen reporting observations."""

from io import BytesIO

from .aggregation import summarize_results
from .identity import canonical_alpha
from .report_publication import open_report, inspect_report
from .reporting_contract import require
from .spec import ALPHAS, ARCHITECTURES, DATASETS, EVALUATION_ALPHAS, EPOCHS

COLORS = ("#0072B2", "#D55E00", "#009E73")
MARKERS = ("o", "s", "^")


def _edges(values):
    return [values[0] - (values[1] - values[0]) / 2,
            *[(a + b) / 2 for a, b in zip(values, values[1:])], values[-1] + (values[-1] - values[-2]) / 2]


def plot_results(collection, *, output_dir, source_root, collection_dir=None, formats=("png", "pdf", "svg"), dpi=200):
    """Render six required families. Missing data is masked, never a zero score.

    No live authority reread, research execution, scheduler call or publication
    repair. Collection validation and summaries are independent of Matplotlib.
    """
    aggregate = summarize_results(collection)
    require(type(dpi) is int and 72 <= dpi <= 1200, "Figure DPI must be an integer in 72..1200")
    require(formats and len(set(formats)) == len(formats) and set(formats) <= {"png", "pdf", "svg"} and
            "png" in formats and bool(set(formats) & {"pdf", "svg"}), "Export PNG and at least one vector format, without duplicates")
    extra = () if collection_dir is None else (collection_dir,)
    with open_report(collection, output_dir=output_dir, source_root=source_root, kind="figures",
                     options={"formats": list(formats), "dpi": dpi}, extra_protected=extra) as writer:
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib import colormaps
            from matplotlib.lines import Line2D
            import numpy as np
        except ImportError as exc:
            raise ValueError("Install the declared reporting extra in the future authorized environment before plotting") from exc
        manifest = collection["manifest"]
        seeds = manifest["seeds"]
        seed_style = {seed: (COLORS[i], MARKERS[i]) for i, seed in enumerate(seeds)}
        specifications = []

        def canvas(title, *, panels=1):
            figure = Figure(figsize=(10, 5 if panels == 1 else 7), layout="constrained")
            FigureCanvasAgg(figure)
            axes = figure.subplots(panels, 1, squeeze=False).ravel()
            figure.suptitle(title)
            return figure, axes

        def alpha_axis(axis, name, values=ALPHAS):
            axis.set_xlabel(name)
            axis.set_xticks(values, [canonical_alpha(a) for a in values], rotation=60)
            axis.set_xlim(_edges(values)[0], _edges(values)[-1])
            axis.grid(alpha=0.2)

        def save(figure, name, specification):
            for extension in formats:
                path = writer.target(f"figures/{name}.{extension}")
                with BytesIO() as buffer:
                    figure.savefig(buffer, format=extension, dpi=dpi)
                    with path.open("xb") as stream:
                        stream.write(buffer.getvalue())
                writer.acknowledge(path, "scientific_figure")
            figure.clear()
            specifications.append({"name": name, "collection_sha256": collection["document_sha256"],
                "checkpoint_policy": "clean_validation_selected_best", "uncertainty": "sample_SD_ddof_1; no_CI",
                "positions": "actual_numeric_alphas; overlapping_markers_use_deterministic_size/shape/color",
                **specification})

        # Individual points are larger than supporting mean markers. All are at
        # actual alpha positions; concentric sizes/outline shapes distinguish ties.
        for dataset in DATASETS:
            for architecture in ARCHITECTURES:
                groups = [g for g in aggregate["training_groups"] if g["dataset"] == dataset and g["architecture"] == architecture]
                comparisons = [g for g in aggregate["paired_training_groups"] if g["dataset"] == dataset and g["architecture"] == architecture]
                for family, summaries, field, ylabel, scale in (
                    ("validation", groups, "validation_accuracy", "Best clean validation accuracy (%)", 100),
                    ("clean_test", groups, "clean_accuracy", "Independent clean-test accuracy (%)", 100),
                    ("clean_difference", comparisons, "summary", "Clean-test difference from training alpha 0 (pp)", 1)):
                    figure, (axis,) = canvas(f"{dataset} / {architecture} — {family}\nBest checkpoint; individual training seeds; supporting mean ± sample SD")
                    counts = []
                    for group in summaries:
                        summary = group[field]
                        x = group["training_alpha"]
                        for individual in summary["individuals"]:
                            color, marker = seed_style[individual["training_seed"]]
                            observed = individual.get("observed_value", individual["value"])
                            if observed is not None:
                                i = seeds.index(individual["training_seed"])
                                axis.scatter([x], [scale * observed], color=color, marker=marker,
                                    s=95 - i * 20, facecolors="none", linewidths=1.8,
                                    label=f"training seed {individual['training_seed']}" if x == 0 else None, zorder=3)
                        if summary["official"] is not None:
                            mean = summary["official"]
                            axis.errorbar([x], [scale * mean["mean"]], yerr=None if mean["sample_sd"] is None else [scale * mean["sample_sd"]],
                                color="black", marker=".", markersize=4, capsize=3, label="complete 3-seed mean ± sample SD" if x == 0 else None)
                        axis.text(x, 0.02, f"{summary['eligible_count']}/3", transform=axis.get_xaxis_transform(), fontsize=7, ha="center")
                        counts.append({"canonical_training_alpha": group["canonical_training_alpha"], "eligible": summary["eligible_count"], "expected": 3})
                    if family == "clean_difference":
                        axis.axhline(0, color="gray", linewidth=0.8)
                    alpha_axis(axis, "Training alpha")
                    axis.set_ylabel(ylabel)
                    seed_handles = [Line2D([], [], color=seed_style[s][0], marker=seed_style[s][1], linestyle="none",
                                          markerfacecolor="none", label=f"training seed {s}") for s in seeds]
                    axis.legend(handles=seed_handles + [Line2D([], [], color="black", marker=".", linestyle="none",
                        label="complete 3-seed mean ± sample SD")], fontsize=8)
                    save(figure, f"{dataset}__{architecture}__{family}", {"family": family, "counts": counts,
                        "accuracy_display_unit": "percent" if scale == 100 else "percentage_points", "missing": "no_score_marker; contributing_counts_shown"})

                for alpha in ALPHAS:
                    key = canonical_alpha(alpha)
                    checkpoint_groups = [g for g in aggregate["checkpoint_robustness"] if g["dataset"] == dataset and
                                         g["architecture"] == architecture and g["canonical_training_alpha"] == key]
                    cross = [g for g in aggregate["robustness_groups"] if g["dataset"] == dataset and
                             g["architecture"] == architecture and g["canonical_training_alpha"] == key]
                    figure, (axis,) = canvas(f"{dataset} / {architecture} / training alpha {key}\nBest checkpoint: realizations, seed means, supporting cross-seed mean ± sample SD")
                    for checkpoint in checkpoint_groups:
                        color, marker = seed_style[checkpoint["training_seed"]]
                        for j, individual in enumerate(checkpoint["accuracy"]["individuals"]):
                            if individual["observed_value"] is not None:
                                axis.scatter([checkpoint["evaluation_alpha"]], [100 * individual["observed_value"]], color=color,
                                    marker=("o", "x", "+")[j], s=65 + 20 * j, linewidths=1.2, alpha=0.7,
                                    label=f"seed {checkpoint['training_seed']}: individual realizations" if checkpoint["evaluation_alpha"] == 0 else None)
                        official = checkpoint["accuracy"]["official"]
                        if official is not None:
                            axis.scatter([checkpoint["evaluation_alpha"]], [100 * official["mean"]], color=color,
                                marker=marker, s=110, facecolors="none", linewidths=2,
                                label=f"seed {checkpoint['training_seed']}: checkpoint mean" if checkpoint["evaluation_alpha"] == 0 else None)
                    for group in cross:
                        summary = group["accuracy"]
                        if summary["official"] is not None:
                            official = summary["official"]
                            axis.errorbar([group["evaluation_alpha"]], [100 * official["mean"]], yerr=[100 * official["sample_sd"]],
                                color="black", marker=".", capsize=3, label="3-seed mean ± cross-seed sample SD" if group["evaluation_alpha"] == 0 else None)
                        axis.text(group["evaluation_alpha"], 0.02, f"{summary['eligible_count']}/3 seeds", transform=axis.get_xaxis_transform(), fontsize=6, rotation=40)
                    alpha_axis(axis, "Evaluation alpha", EVALUATION_ALPHAS)
                    axis.set_ylabel("Full-test accuracy (%)")
                    axis.legend(handles=[Line2D([], [], color=seed_style[s][0], marker=seed_style[s][1], linestyle="none",
                        markerfacecolor="none", label=f"training seed {s}; hollow large = checkpoint mean") for s in seeds] +
                        [Line2D([], [], color="black", marker=".", linestyle="none", label="complete cross-seed mean ± sample SD")],
                        title="Realizations: clean r0=o; positive r1=o, r2=x, r3=+", fontsize=7, title_fontsize=7)
                    save(figure, f"{dataset}__{architecture}__train-{key}__robustness", {"family": "robustness",
                        "training_alpha": alpha, "counts": [{"evaluation_alpha": g["evaluation_alpha"], "eligible_seeds": g["accuracy"]["eligible_count"],
                            "within_checkpoint_counts": g["within_checkpoint_sample_sds"]} for g in cross],
                        "clean_replicates": 1, "positive_replicates": 3})

                # Actual numeric positions, nonuniform cell boundaries. No
                # interpolation; incomplete cells have a visible cross and n/3.
                for seed in (None, *seeds):
                    source = aggregate["robustness_groups"] if seed is None else aggregate["checkpoint_robustness"]
                    lookup = {(g["canonical_training_alpha"], g["canonical_evaluation_alpha"]): g for g in source
                              if g["dataset"] == dataset and g["architecture"] == architecture and (seed is None or g["training_seed"] == seed)}
                    values, mask, counts = [], [], []
                    for alpha in ALPHAS:
                        data_line, mask_line, count_line = [], [], []
                        for eval_alpha in EVALUATION_ALPHAS:
                            summary = lookup[(canonical_alpha(alpha), canonical_alpha(eval_alpha))]["accuracy"]
                            data_line.append(0 if summary["official"] is None else 100 * summary["official"]["mean"])
                            mask_line.append(summary["official"] is None)
                            count_line.append((summary["eligible_count"], summary["expected_count"]))
                        values.append(data_line); mask.append(mask_line); counts.append(count_line)
                    label = "complete two-stage cross-training-seed means" if seed is None else f"training seed {seed}: checkpoint realization means"
                    figure, (axis,) = canvas(f"{dataset} / {architecture} — accuracy heatmap\nBest checkpoint; {label}; masked incomplete cells")
                    cmap = colormaps["viridis"].copy()
                    cmap.set_bad("#e0e0e0")
                    mesh = axis.pcolormesh(_edges(EVALUATION_ALPHAS), _edges(ALPHAS), np.ma.array(values, mask=mask),
                                          cmap=cmap, vmin=0, vmax=100, shading="flat")
                    figure.colorbar(mesh, ax=axis, label="Full-test accuracy (%)")
                    alpha_axis(axis, "Evaluation alpha", EVALUATION_ALPHAS)
                    axis.set_yticks(ALPHAS, [canonical_alpha(a) for a in ALPHAS])
                    axis.set_ylabel("Training alpha")
                    for i, alpha in enumerate(ALPHAS):
                        for j, eval_alpha in enumerate(EVALUATION_ALPHAS):
                            count, expected = counts[i][j]
                            axis.text(eval_alpha, alpha, ("× " if mask[i][j] else "") + f"{count}/{expected}",
                                      fontsize=5, ha="center", va="center", color="black" if mask[i][j] else "white")
                    save(figure, f"{dataset}__{architecture}__heatmap__{'all-seeds' if seed is None else f'seed-{seed}'}",
                         {"family": "accuracy_heatmap", "training_seed": seed, "counts": counts, "missing": "masked_gray_cross; no_interpolation"})

        for run in collection["training"]:
            records = [] if run["snapshot"] is None else [e["document"]["record"] for e in run["snapshot"]["epochs"]]
            figure, axes = canvas(f"{run['dataset']} / {run['architecture']} / training alpha {run['canonical_training_alpha']} / seed {run['training_seed']}\n"
                                  f"Committed epochs {len(records)}/{EPOCHS}; {run['condition']}; {run['stored_status']}", panels=2)
            # Split contiguous segments explicitly, even though journal validation
            # currently requires a consecutive committed prefix.
            segments = []
            for record in records:
                if not segments or record["epoch_index"] != segments[-1][-1]["epoch_index"] + 1:
                    segments.append([])
                segments[-1].append(record)
            for metric, axis, scale in (("accuracy", axes[0], 100), ("loss", axes[1], 1)):
                for role, color in (("training", COLORS[0]), ("validation", COLORS[1])):
                    for index, segment in enumerate(segments):
                        axis.plot([r["epoch_index"] + 1 for r in segment], [scale * r[role][metric] for r in segment],
                                  color=color, label=f"{role}: {segment[0]['training_kind'] if role == 'training' else 'clean weights'}" if index == 0 else None)
                axis.set_xlim(1, EPOCHS)
                axis.set_xlabel("Human epoch (1–200); committed observations only")
                axis.set_ylabel("Accuracy (%)" if metric == "accuracy" else "Mean cross-entropy loss (nats/example)")
                axis.grid(alpha=0.2)
                if records:
                    axis.legend(fontsize=8)
                else:
                    axis.text(0.5, 0.5, "No accepted committed epochs", transform=axis.transAxes, ha="center")
            save(figure, f"learning/{run['training_run_id']}", {"family": "learning_curve", "training_run_id": run["training_run_id"],
                 "committed_epoch_count": len(records), "expected_epochs": EPOCHS, "incomplete": len(records) != EPOCHS,
                 "checkpoint_policy": "all_committed_training/clean_validation_epochs; no_padding_or_smoothing"})
        writer.json("figure_index.json", {"collection_sha256": collection["document_sha256"], "figures": specifications,
                                          "conventions": aggregate["conventions"]}, "figure_specifications")
        writer.json("coverage.json", aggregate["coverage"], "coverage")
        writer.csv("coverage_training.csv", ({k: r[k] for k in ("training_run_id", "dataset", "architecture", "canonical_training_alpha",
            "training_seed", "condition", "stored_status", "completed_epochs", "training_complete", "diagnostic")} for r in collection["training"]),
            ["training_run_id", "dataset", "architecture", "canonical_training_alpha", "training_seed", "condition", "stored_status", "completed_epochs", "training_complete", "diagnostic"], "coverage_table")
        writer.csv("coverage_evaluation.csv", ({k: r[k] for k in ("training_run_id", "slot", "condition", "stored_status", "successful", "robustness_completed", "diagnostic")} for r in collection["evaluation"]),
            ["training_run_id", "slot", "condition", "stored_status", "successful", "robustness_completed", "diagnostic"], "coverage_table")
    return inspect_report(output_dir)
