"""Ten scientific figure families from validated Stage 3 publications."""

from collections import defaultdict

from .protocol import ARCHITECTURES, DATASETS
from .storage import code_digest, disjoint, finish, install, plain, publication, require, seal

TRAIN_COLORS = {0.: "#0072B2", .10: "#D55E00", .30: "#009E73"}
ALPHA_MARKERS = {.01: "o", .02: "s", .05: "^", .10: "D", .20: "v"}


def plot(summary_dir, comparison_dir, output, dpi):
    from matplotlib import colormaps, rc_context
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    summary = publication(summary_dir, "summary")
    comparison = publication(comparison_dir, "comparison") if comparison_dir else None
    roots = [str(plain(summary_dir)), *summary["protected_roots"]]
    if comparison:
        require(comparison["summary_sha256"] == summary["document_sha256"], "Comparison belongs to another summary")
        roots.extend([str(plain(comparison_dir)), *comparison["protected_roots"]])
    root = disjoint(output, roots)
    root.mkdir(parents=True)
    index = []

    def canvas(title, panels=1):
        figure = Figure(figsize=(6.7*panels, 4.8), layout="constrained")
        FigureCanvasAgg(figure)
        axes = figure.subplots(1, panels, squeeze=False).ravel()
        figure.suptitle(title)
        for ax in axes:
            ax.grid(alpha=.18)
        return figure, axes

    def save(fig, family, dataset, architecture, available, expected=None, reason=None):
        stem = f"{dataset}_{architecture}_{family}"
        names = []
        if available:
            for extension in ("png", "pdf"):
                name = f"{stem}.{extension}"
                with (root/name).open("xb") as stream:
                    metadata = {"Software": "hessian_geometry_stage3"} if extension == "png" else {
                        "Creator": "hessian_geometry_stage3", "CreationDate": None, "ModDate": None}
                    fig.savefig(stream, format=extension, dpi=dpi, metadata=metadata)
                names.append(name)
        index.append({"dataset": dataset, "architecture": architecture, "family": family, "rendered_points": available,
                      "expected_points": expected, "missing_points": None if expected is None else expected-available,
                      "reason": None if names else reason or "No eligible complete observations. See coverage and comparison tables",
                      "files": names})
        fig.clear()

    def legend(ax, **kwargs):
        h, labels = ax.get_legend_handles_labels()
        if h:
            labels_map = dict(zip(labels, h))
            ax.legend(labels_map.values(), labels_map.keys(), fontsize=8, **kwargs)

    def zero(ax):
        ax.axhline(0, color=".55", lw=.6, zorder=0)

    def equality(ax, values):
        if values:
            low, high = min(values), max(values)
            pad = max(.04*(high-low), 1e-8)
            ax.plot([low-pad, high+pad], [low-pad, high+pad], color=".4", lw=.8)
            ax.set_xlim(low-pad, high+pad)
            ax.set_ylim(low-pad, high+pad)
            ax.set_aspect("equal", adjustable="box")

    def stage_axis(ax, names):
        ax.set_xticks(range(len(names)), [name.replace("_", " ") for name in names], rotation=35, ha="right")

    def used(rows, **match):
        return [row for row in rows if all(row[k] == v for k, v in match.items())]

    with rc_context({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42}):
        for dataset in DATASETS:
            for architecture in ARCHITECTURES:
                ks = {"dataset": dataset, "architecture": architecture}
                traces = used(summary["traces"], **ks)
                network = used(traces, scope="network")
                stages = used(traces, scope="stage")
                names = list(dict.fromkeys(r["target_name"] for r in stages))
                title = f"{dataset.upper()} / {architecture} / Hessian split: {summary['split']}"
                cap = summary["workload"]["probes_per_checkpoint"]
                mc = f"{cap} probes per complete estimate | bars: probe Monte Carlo SE"
                trace_units = "Weighted Hessian trace (nats/example)"
                ce = "Delta CE (nats/example)"
                stage2_caption = f"Stage 2 validation: 8 Gaussian realizations | Stage 3: {cap} probes"
                empirical = used(comparison["stage2"], **ks) if comparison else []
                valid = [r for r in empirical if r["residual"] is not None]
                pairs = used(summary["pairs"], scope="stage", **ks)

                fig, (ax,) = canvas(f"{title}\nWhole network | {mc}")
                count = 0
                for r in network:
                    train = r["training_alpha"]
                    if r["complete"]:
                        ax.errorbar(train, r["mean"], yerr=r["mc_se"], marker="o", color=TRAIN_COLORS[train], capsize=3)
                        count += 1
                ax.set_xticks(list(TRAIN_COLORS), [f"{a:g}" if any(r["complete"] and r["training_alpha"] == a for r in network)
                                                   else f"{a:g}\nmissing" for a in TRAIN_COLORS])
                ax.set(xlabel="Training alpha", ylabel=trace_units)
                zero(ax)
                save(fig, "01_network_trace", dataset, architecture, count, len(network))

                fig, (ax,) = canvas(f"{title}\nAll disjoint stages | {mc}")
                count = 0
                for r in stages:
                    if r["complete"]:
                        train = r["training_alpha"]
                        offset = .18*(list(TRAIN_COLORS).index(train)-1)
                        ax.errorbar(names.index(r["target_name"])+offset, r["mean"], yerr=r["mc_se"], marker="o",
                                    color=TRAIN_COLORS[train], capsize=2, label=f"train {train:g}")
                        count += 1
                stage_axis(ax, names)
                ax.set_ylabel(trace_units)
                zero(ax)
                legend(ax)
                save(fig, "02_stage_trace", dataset, architecture, count, len(stages))

                transitions = used(summary["convergence"], scope="network", **ks)
                fig, axes = canvas(f"{title}\nComplete probe prefixes | gaps indicate missing probes", 2)
                count = 0
                for train, color in TRAIN_COLORS.items():
                    rows = sorted(used(transitions, training_alpha=train), key=lambda r:r["probe_prefix"])
                    for ax, field in zip(axes, ("mean", "mc_se")):
                        ax.plot([r["probe_prefix"] for r in rows], [r[field] for r in rows], "o-", ms=3, color=color, label=f"train {train:g}")
                    count += sum(r["complete"] for r in rows)
                axes[0].set_ylabel(trace_units)
                axes[1].set_ylabel("Monte Carlo SE (nats/example)")
                for ax in axes:
                    ax.set_xlabel("Prespecified probe prefix size")
                zero(axes[0])
                if count:
                    legend(axes[0])
                save(fig, "03_probe_convergence", dataset, architecture, count, len(transitions))

                fig, (ax,) = canvas(f"{title}\n{stage2_caption} | x: prediction MC SE | y: perturbation SE")
                extent = []
                network_comparison = used(valid, scope="network")
                for r in network_comparison:
                    x, y = r["predicted_delta_loss"], r["observed_delta_loss"]
                    ax.errorbar(x, y, xerr=r["prediction_mc_se"], yerr=r["stage2_perturbation_se"],
                                color=TRAIN_COLORS[r["training_alpha"]], marker=ALPHA_MARKERS[r["evaluation_alpha"]], ms=5, alpha=.8)
                    extent.extend([x, y])
                equality(ax, extent)
                if extent:
                    handles = [Line2D([], [], color=c, marker="o", lw=0, label=f"train {a:g}") for a,c in TRAIN_COLORS.items()]
                    handles.extend(Line2D([], [], color=".4", marker=m, lw=0, label=f"eval {a:g}") for a,m in ALPHA_MARKERS.items())
                    ax.legend(handles=handles, fontsize=7, ncols=2)
                ax.set(xlabel=f"Predicted {ce}", ylabel=f"Observed {ce}")
                save(fig, "04_network_observed_vs_predicted", dataset, architecture, len(network_comparison), len(used(empirical, scope="network")))

                fig, (ax,) = canvas(f"{title}\n{stage2_caption} | solid: observed SE | dashed: prediction MC SE")
                ax.axvspan(.01, .05, color=".6", alpha=.12, label="primary fit range")
                ax.axvline(.10, lw=.7, color=".5", ls=":", label="secondary endpoint")
                for train, color in TRAIN_COLORS.items():
                    rows = sorted(used(network_comparison, training_alpha=train), key=lambda r:r["evaluation_alpha"])
                    for field, error, style, label in (("observed_delta_loss", "stage2_perturbation_se", "-", "observed"),
                                                      ("predicted_delta_loss", "prediction_mc_se", "--", "predicted")):
                        if rows:
                            ax.errorbar([r["evaluation_alpha"] for r in rows], [r[field] for r in rows],
                                        yerr=[r[error] for r in rows], linestyle=style, marker="o", ms=3,
                                        color=color, capsize=2, label=f"train {train:g} {label}")
                ax.set(xlabel="Evaluation alpha", ylabel=ce)
                zero(ax)
                legend(ax, ncols=2)
                save(fig, "05_network_alpha_response", dataset, architecture, len(network_comparison), len(used(empirical, scope="network")))

                fig, (ax,) = canvas(f"{title}\n{cap} probes | Hessian MC SE on x | empirical fits: 8 validation or 3 test draws per alpha")
                fits = used(comparison["fits"] + comparison["stage1"], scope="network", **ks) if comparison else []
                extent = []
                for r in fits:
                    x, y = r["hessian_trace"], r["empirical_curvature_proxy_2c"]
                    if x is None or y is None:
                        continue
                    mark = "o" if r["empirical_split"] == "validation" else "s" if r["split_match"] else "^"
                    color = TRAIN_COLORS[r["training_alpha"]]
                    ax.errorbar(x, y, xerr=r["hessian_mc_se"], fmt=mark, color=color,
                                markerfacecolor=color if r["fit_window"] == "primary" else "none",
                                label=f"train {r['training_alpha']:g}, {r['empirical_split']}" + (" cross-split" if not r["split_match"] else ""))
                    extent.extend([x, y])
                equality(ax, extent)
                ax.set(xlabel=trace_units, ylabel="Empirical 2c (nats/example)")
                ax.set_title("Filled: primary window | hollow: secondary window")
                legend(ax, ncols=2)
                save(fig, "06_empirical_proxy", dataset, architecture, len(extent)//2, len(fits))

                stage_valid = used(valid, scope="stage")
                fig, axes = canvas(f"{title}\n{stage2_caption} | x: prediction MC SE | y: perturbation SE", 3)
                palette = {name: colormaps["tab20"](i) for i, name in enumerate(names)}
                extent = [r[k] for r in stage_valid for k in ("observed_delta_loss", "predicted_delta_loss")]
                for ax, train in zip(axes, TRAIN_COLORS):
                    for r in used(stage_valid, training_alpha=train):
                        ax.errorbar(r["predicted_delta_loss"], r["observed_delta_loss"], xerr=r["prediction_mc_se"],
                                    yerr=r["stage2_perturbation_se"], fmt=ALPHA_MARKERS[r["evaluation_alpha"]],
                                    color=palette[r["target_name"]], ms=4, alpha=.8, label=r["target_name"])
                    equality(ax, extent)
                    ax.set(title=f"Training alpha {train:g}", xlabel=f"Predicted {ce}", ylabel=f"Observed {ce}")
                if extent:
                    handles = [Line2D([], [], lw=0, color=palette[n], marker="o", label=n) for n in names]
                    handles.extend(Line2D([], [], lw=0, color=".4", marker=m, label=f"eval {a:g}") for a,m in ALPHA_MARKERS.items())
                    axes[-1].legend(handles=handles, fontsize=7, ncols=2)
                save(fig, "07_stage_observed_vs_predicted", dataset, architecture, len(stage_valid), len(used(empirical, scope="stage")))

                fig, axes = canvas(f"{title}\nSame-seed checkpoints | {cap} probes each | bars: separate probe Monte Carlo SE", 2)
                count = 0
                for ax, train in zip(axes, (.10, .30)):
                    selected = [r for r in used(pairs, training_alpha=train) if r["noise_minus_clean"] is not None]
                    for key, se, color, offset, label in (("clean_trace", "clean_mc_se", TRAIN_COLORS[0.], -.13, "train 0"),
                                                         ("noise_trace", "noise_mc_se", TRAIN_COLORS[train], .13, f"train {train:g}")):
                        if selected:
                            ax.errorbar([names.index(r["target_name"])+offset for r in selected], [r[key] for r in selected],
                                        yerr=[r[se] for r in selected], fmt="o", ms=3, capsize=2, color=color, label=label)
                    count += len(selected)
                    stage_axis(ax, names)
                    ax.set_ylabel(trace_units)
                    zero(ax)
                    legend(ax)
                save(fig, "08_paired_training_profiles", dataset, architecture, count, len(pairs))

                fig, axes = canvas(f"{title}\n{stage2_caption} | residual = observed minus predicted | SEs retained separately in tables", 2)
                for ax, points, scope in zip(axes, (network_comparison, stage_valid), ("Whole network", "All disjoint stages")):
                    groups = defaultdict(list)
                    for r in points:
                        groups[r["training_alpha"], r["target_name"]].append(r)
                    for (train, name), rows in groups.items():
                        rows.sort(key=lambda r:r["evaluation_alpha"])
                        color = TRAIN_COLORS[train] if scope == "Whole network" else palette[name]
                        ax.plot([r["evaluation_alpha"] for r in rows], [r["residual"] for r in rows],
                                ["-", "--", ":"][list(TRAIN_COLORS).index(train)], marker=".", color=color,
                                label=f"{name}, train {train:g}" if scope == "Whole network" else name)
                    ax.set(title=scope, xlabel="Evaluation alpha", ylabel="Residual (nats/example)")
                    zero(ax)
                    legend(ax, ncols=2)
                    if scope != "Whole network":
                        ax.set_title("Stages | train 0 solid, 0.1 dashed, 0.3 dotted")
                save(fig, "09_approximation_residual", dataset, architecture, len(network_comparison)+len(stage_valid), len(empirical))

                fig, axes = canvas(f"{title}\nNetwork trace | {cap} probes | Stage 2 degradation: 8 validation realizations", 2)
                count = 0
                for r in network:
                    if r["complete"]:
                        axes[0].errorbar(r["mean"], 100*r["clean_accuracy"], xerr=r["mc_se"], fmt="o",
                                         color=TRAIN_COLORS[r["training_alpha"]], label=f"train {r['training_alpha']:g}")
                        count += 1
                for r in network_comparison:
                    axes[1].errorbar(r["stage3_trace"], -r["stage2_accuracy_change_pp"], xerr=r["stage3_mc_se"],
                                     yerr=r["stage2_accuracy_change_se"], fmt=ALPHA_MARKERS[r["evaluation_alpha"]],
                                     color=TRAIN_COLORS[r["training_alpha"]], label=f"eval {r['evaluation_alpha']:g}")
                axes[0].set(xlabel=trace_units, ylabel="Stage 3 clean accuracy (%)")
                axes[1].set(xlabel=trace_units, ylabel="Stage 2 accuracy degradation (pp)")
                for ax in axes:
                    legend(ax)
                save(fig, "10_trace_and_accuracy", dataset, architecture, count+len(network_comparison), len(network)+len(used(empirical, scope="network")))

    metadata = {"summary_sha256": summary["document_sha256"],
                "comparison_sha256": None if comparison is None else comparison["document_sha256"],
                "source_sha256": code_digest(), "dpi": dpi, "index": index}
    install(root/"figures.json", seal("figures", metadata))
    install(root/"figure_index.json", seal("figure_index", metadata))
    finish(root, "figures", ["figures.json", "figure_index.json", *(name for entry in index for name in entry["files"])])
    return {"output": str(root), "figure_documents": sum(bool(entry["files"]) for entry in index)}
