"""Eight reproducible Matplotlib figure families from a sealed summary."""

import math

from alpha_sweep_training.spec import ARCHITECTURES, DATASETS

from .storage import disjoint, finish, install, published, require, seal, source_digest


def plot(summary_dir, output, *, alpha, dpi, scope="stage", profile_targets=()):
    import numpy as np
    from matplotlib import colormaps, rc_context
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    summary = published(summary_dir, "summary", "summary.json")
    root = disjoint(output, [summary_dir, summary["plan_dir"], summary["results"], *summary["protected_roots"]])
    rows = summary["cross_seed_summaries"]
    declared_alphas = {r["evaluation_alpha"] for r in rows if r["evaluation_alpha"] > 0}
    require(not declared_alphas or alpha in declared_alphas, "Selected figure alpha is absent from summary")
    known = {r["target_name"] for r in rows if r["scope"] == scope}
    require(len(set(profile_targets)) == len(profile_targets) and set(profile_targets) <= known,
            "Choose unique profile target names declared within the selected scope")
    require(scope != "tensor" or 1 <= len(profile_targets) <= 12,
            "Individual tensor figures require one to twelve explicit --profile-target names")
    root.mkdir(parents=True)
    index = []

    def save(figure, dataset, architecture, family, count):
        base = f"{dataset}_{architecture}_{family}"
        paths = []
        if count:
            for ext in ("png", "pdf"):
                name = f"{base}.{ext}"
                with (root / name).open("xb") as stream:
                    figure.savefig(stream, format=ext, dpi=dpi,
                                   metadata={"Software": "layer_sensitivity_stage2"} if ext == "png" else
                                   {"Creator": "layer_sensitivity_stage2", "CreationDate": None, "ModDate": None})
                paths.append(name)
        index.append({"family": family, "dataset": dataset, "architecture": architecture,
                      "points": count, "files": paths,
                      "reason": None if count else "No eligible complete observations for these filters. See summary coverage.json."})
        figure.clear()

    def canvas(title, columns=1):
        fig = Figure(figsize=(8*columns, 4.9), layout="constrained")
        FigureCanvasAgg(fig)
        axes = fig.subplots(1, columns, squeeze=False).ravel()
        fig.suptitle(title)
        for axis in axes:
            axis.grid(alpha=.18)
        return fig, axes

    def value(row, field="mean"):
        return math.nan if row is None or row[field] is None else row[field]

    with rc_context({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42}):
        for dataset in DATASETS:
            for architecture in ARCHITECTURES:
                title = f"{dataset} / {architecture} / {summary['split']} / {scope}"
                stage = [r for r in rows if (r["dataset"], r["architecture"], r["scope"]) == (dataset, architecture, scope)
                         and (not profile_targets or r["target_name"] in profile_targets)]
                own = [r for r in summary["checkpoint_summaries"] if (r["dataset"], r["architecture"], r["scope"]) == (dataset, architecture, scope)
                       and (not profile_targets or r["target_name"] in profile_targets)]
                stages = list(dict.fromkeys(r["target_name"] for r in own))
                trains = sorted({r["training_alpha"] for r in stage})
                colors = {a: colormaps["tab10"](i % 10) for i, a in enumerate(trains)}
                lookup = {(r["training_alpha"], r["evaluation_alpha"], r["target_name"], r["metric"]): r for r in stage}

                def get(train, eval_alpha, group, metric):
                    return lookup.get((train, eval_alpha, group, metric))

                def profile(axis, metric, *, factor=1.):
                    count = 0
                    for train in trains:
                        group_rows = [get(train, alpha, group, metric) for group in stages]
                        values = [factor*value(r) for r in group_rows]
                        count += sum(math.isfinite(y) for y in values)
                        axis.plot(range(len(stages)), values, "o-", color=colors[train], ms=4, label=f"train {train:g}")
                        for i, r in enumerate(group_rows):
                            if r is not None and r["sample_sd"] is not None:
                                axis.errorbar(i, values[i], yerr=abs(factor)*r["sample_sd"], color=colors[train], capsize=2)
                    # Hollow points preserve seed-level checkpoint means.
                    for r in own:
                        if r["evaluation_alpha"] == alpha and r["metric"] == metric and r["complete"]:
                            axis.scatter(stages.index(r["target_name"]), factor*r["mean"], s=24,
                                         facecolors="none", edgecolors=[colors[r["training_alpha"]]], alpha=.7)
                            count += 1
                    axis.set_xticks(range(len(stages)), [s.replace("_", " ") for s in stages], rotation=33, ha="right")
                    axis.axhline(0, color=".55", lw=.7)
                    if count:
                        axis.legend(ncols=2 if len(trains) > 4 else 1, fontsize=8)
                    return count

                for family, metric, ylabel, factor in (
                    ("01_stage_loss", "delta_loss", "Delta CE (nats/example)", 1.),
                    ("02_accuracy_degradation", "accuracy_change_pp", "Accuracy degradation (pp)", -1.),
                    ("05_training_profiles", "delta_loss", "Delta CE (nats/example)", 1.),
                    ("08_prediction_flip_rate", "prediction_flip_rate", "Prediction flip rate (%)", 100.),
                ):
                    fig, (ax,) = canvas(f"{title} | evaluation alpha {alpha:g}\nHollow points: checkpoint means | bars: cross-seed sample SD")
                    count = profile(ax, metric, factor=factor)
                    ax.set_ylabel(ylabel)
                    if family == "05_training_profiles":
                        # Display paired training-alpha contrasts on the same stage order.
                        ax.clear()
                        paired = 0
                        contrasts = {(r["training_alpha"], r["training_seed"], r["target_name"]): r["noise_minus_clean_training"]
                                     for r in summary["training_pairs"] if (r["dataset"], r["architecture"], r["scope"]) ==
                                     (dataset, architecture, scope) and r["evaluation_alpha"] == alpha and r["metric"] == metric
                                     and r["noise_minus_clean_training"] is not None}
                        for train in (a for a in trains if a > 0):
                            for seed in sorted({r["training_seed"] for r in own}):
                                values = [contrasts.get((train, seed, group), math.nan) for group in stages]
                                if any(math.isfinite(v) for v in values):
                                    ax.plot(range(len(stages)), values, "o-", color=colors[train], label=f"train {train:g}, seed {seed}")
                                    paired += sum(math.isfinite(v) for v in values)
                        ax.axhline(0, color=".55", lw=.7)
                        ax.grid(alpha=.18)
                        ax.set_xticks(range(len(stages)), [s.replace("_", " ") for s in stages], rotation=33, ha="right")
                        ax.set_ylabel("Noise-trained minus clean-trained Delta CE (nats/example)")
                        ax.set_title("Same-seed pairs within compatible data and runtime settings")
                        fig.suptitle(f"{title} | evaluation alpha {alpha:g}")
                        if paired:
                            ax.legend(fontsize=7, ncols=3)
                        count = paired
                    save(fig, dataset, architecture, family, count)

                fig, axes = canvas(f"{title} | evaluation alpha {alpha:g}\nHollow points: checkpoint means | bars: cross-seed sample SD", 3)
                count = 0
                for ax, metric, label in zip(axes, ("delta_loss", "delta_loss_per_million_parameters", "delta_loss_per_expected_squared_norm"),
                                             ("Raw Delta CE", "Delta CE / million weights", "Delta CE / expected squared norm")):
                    count += profile(ax, metric)
                    ax.set_ylabel(label)
                save(fig, dataset, architecture, "03_raw_and_normalized", count)

                fig, (ax,) = canvas(f"{title} | evaluation alpha {alpha:g}\nComplete seed means | gray cells indicate unavailable means")
                grid = np.array([[value(get(t, alpha, g, "delta_loss")) for g in stages] for t in trains])
                count = int(np.isfinite(grid).sum())
                if count:
                    limit = max(float(np.nanmax(np.abs(grid))), 1e-12)
                    cmap = colormaps["RdBu_r"].copy()
                    cmap.set_bad(".8")
                    image = ax.imshow(np.ma.masked_invalid(grid), aspect="auto", cmap=cmap, vmin=-limit, vmax=limit)
                    fig.colorbar(image, ax=ax, label="Delta CE (nats/example)")
                    ax.set_xticks(range(len(stages)), stages, rotation=33, ha="right")
                    ax.set_yticks(range(len(trains)), [f"{t:g}" for t in trains])
                    ax.set_ylabel("Training alpha")
                save(fig, dataset, architecture, "04_training_alpha_heatmap", count)

                fig, (ax,) = canvas(f"{dataset} / {architecture} / {summary['split']} | paired stage realizations")
                residuals = [r for r in summary["residuals"] if (r["dataset"], r["architecture"]) == (dataset, architecture)
                             and r["interaction_residual"] is not None]
                if residuals:
                    for i, train in enumerate(sorted({r["training_alpha"] for r in residuals})):
                        subset = [r for r in residuals if r["training_alpha"] == train]
                        ax.scatter([r["sum_isolated"] for r in subset], [r["full_network"] for r in subset],
                                   c=[colormaps["tab10"](i % 10)], s=[16+60*r["evaluation_alpha"] for r in subset], alpha=.6, label=f"train {train:g}")
                    extent = [r[key] for r in residuals for key in ("sum_isolated", "full_network")] + [0.]
                    low, high = min(extent), max(extent)
                    ax.plot([low, high], [low, high], color=".25", lw=.8)
                    ax.set_aspect("equal", adjustable="datalim")
                    ax.legend(fontsize=8)
                ax.set(xlabel="Sum of isolated stages (nats/example)", ylabel="Whole-network Delta CE (nats/example)")
                save(fig, dataset, architecture, "06_isolated_sum_vs_network", len(residuals))

                for train in trains:
                    fig, axes = canvas(f"{title} | training alpha {train:g}\nComplete seed means | shaded band: +/- cross-seed sample SD", 2)
                    count = 0
                    for ax, limit in zip(axes, (1., .10)):
                        xs = sorted(a for a in declared_alphas if a <= limit)
                        for i, group in enumerate(stages):
                            ys = [value(get(train, a, group, "delta_loss")) for a in xs]
                            count += sum(math.isfinite(y) for y in ys)
                            color = colormaps["tab20"](i % 20)
                            if any(math.isfinite(y) for y in ys):
                                ax.plot([0., *xs], [0., *ys], "o-", color=color, ms=3, label=group)
                                sds = [value(get(train, a, group, "delta_loss"), "sample_sd") for a in xs]
                                ax.fill_between(xs, [y-s for y, s in zip(ys, sds)], [y+s for y, s in zip(ys, sds)],
                                                color=color, alpha=.12)
                        ax.set(xlabel="Evaluation alpha", ylabel="Delta CE (nats/example)")
                    if count:
                        axes[0].legend(fontsize=7, ncols=2)
                    save(fig, dataset, architecture, f"07_noise_response_train_{train:g}", count)
                if not trains:
                    fig, _ = canvas(title)
                    save(fig, dataset, architecture, "07_noise_response", 0)
    install(root / "figures.json", seal("figures", {"summary_sha256": summary["document_sha256"], "dpi": dpi,
                                                   "source_sha256": source_digest(), "profile_scope": scope,
                                                   "profile_targets": list(profile_targets),
                                                   "profile_evaluation_alpha": alpha, "index": index}))
    finish(root, "figures", ["figures.json", *(name for row in index for name in row["files"])])
    return {"output": str(root), "figures_generated": sum(bool(row["files"]) for row in index)}
