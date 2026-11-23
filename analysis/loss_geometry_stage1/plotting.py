"""Headless Matplotlib figures made exclusively from accepted analytical rows."""

import math
from pathlib import Path

from .contracts import ALPHAS, ARCHITECTURES, DATASETS, EVALUATION_ALPHAS, WINDOWS, AnalysisError


def plot_analysis(result: dict, output: Path, seeds: tuple[int, ...], dpi: int) -> list[dict]:
    try:
        from matplotlib import colormaps, rc_context
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.lines import Line2D
    except ImportError as exc:
        raise AnalysisError("Matplotlib is required; use an environment with the existing reporting extra installed") from exc

    figures = output / "figures"
    figures.mkdir()
    index = []
    colors = {a: colormaps["tab10"](i) for i, a in enumerate(ALPHAS)}
    markers = ("o", "s", "^")
    seed_key = "; ".join(f"{marker} = seed {seed}" for marker, seed in zip(markers, seeds))

    def canvas(title, rows=1, cols=1, *, sharex=False, sharey=False):
        figure = Figure(figsize=(6.4*cols if cols < 3 else 16, 4.5*rows), layout="constrained")
        FigureCanvasAgg(figure)
        axes = figure.subplots(rows, cols, squeeze=False, sharex=sharex, sharey=sharey).ravel()
        figure.suptitle(title, fontsize=13)
        for axis in axes:
            axis.grid(alpha=.2)
            axis.axhline(0, color=".7", lw=.6)
        return figure, axes

    def save(figure, family, dataset, architecture, count, *, variant="", explanation=""):
        name = f"figure{family}_{dataset}_{architecture}" + (f"_{variant}" if variant else "")
        entry = dict(name=name, family=family, dataset=dataset, architecture=architecture,
                     accepted_points=count, status="generated" if count else "unavailable_no_eligible_data",
                     explanation=explanation or ("No eligible measurements; consult coverage.json" if not count else ""),
                     training_seeds=list(seeds), seed_markers=list(markers),
                     uncertainty="Complete compatible three-seed sample SD ddof=1; no CI; individual seed data retained",
                     files=[])
        if count:
            for extension in ("png", "pdf"):
                relative = f"figures/{name}.{extension}"
                # Output root was exclusively claimed. All fixed names are unique.
                with (output / relative).open("xb") as stream:
                    figure.savefig(stream, format=extension, dpi=dpi,
                        metadata={"Software": "loss_geometry_stage1"} if extension == "png" else {"Creator": "loss_geometry_stage1", "CreationDate": None, "ModDate": None})
                entry["files"].append(relative)
        figure.clear()
        index.append(entry)

    def seed_points(axis, rows, xname, yname, color, *, connect=False):
        for i, seed in enumerate(seeds):
            own = sorted((r for r in rows if r["training_seed"] == seed), key=lambda r: r[xname])
            xs = [r[xname] for r in own]
            ys = [math.nan if r[yname] is None else r[yname] for r in own]
            axis.plot(xs, ys, marker=markers[i], color=color, ms=3.2, lw=.6 if connect else 0,
                      alpha=.6, markerfacecolor="none")

    def response_summary(axis, dataset, architecture, train, metric, color, *, squared=False, limit=1.):
        rows = [r for r in result["summaries"] if r["summary_kind"] == "loss_response" and
                (r["dataset"], r["architecture"], r["training_alpha"], r["metric"]) == (dataset, architecture, train, metric) and r["evaluation_alpha"] <= limit]
        rows.sort(key=lambda r: r["evaluation_alpha"])
        x = [r["evaluation_alpha"]**2 if squared else r["evaluation_alpha"] for r in rows]
        axis.errorbar(x, [r["mean"] if r["complete"] else math.nan for r in rows],
                      yerr=[r["sample_sd"] if r["complete"] else math.nan for r in rows],
                      color=color, lw=1.3, capsize=2, label=f"train alpha={train:g}")

    with rc_context({"font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 10,
                     "pdf.fonttype": 42, "savefig.facecolor": "white"}):
        for dataset in DATASETS:
            for architecture in ARCHITECTURES:
                identity = f"{dataset} / {architecture}"
                rows = [r for r in result["loss_sensitivity"] if r["dataset"] == dataset and r["architecture"] == architecture]
                fits = [r for r in result["quadratic_fits"] if r["dataset"] == dataset and r["architecture"] == architecture]
                pred = [r for r in result["predictions"] if r["dataset"] == dataset and r["architecture"] == architecture]
                count = sum(r["eligible"] for r in rows)
                fig, axes = canvas(f"1. Loss response | {identity}\n{seed_key}; complete three-seed mean +/- sample SD", cols=2)
                for axis, limit in zip(axes, (1., .1)):
                    for train in ALPHAS:
                        selected = [r for r in rows if r["training_alpha"] == train and r["evaluation_alpha"] <= limit]
                        seed_points(axis, selected, "evaluation_alpha", "delta_loss", colors[train], connect=True)
                        response_summary(axis, dataset, architecture, train, "delta_loss", colors[train], limit=limit)
                    axis.set(xlabel="Evaluation alpha", ylabel="Delta CE loss (nats/example)", xlim=(0, limit))
                axes[-1].legend(fontsize=8, ncols=2)
                save(fig, 1, dataset, architecture, count)

                # One panel per training alpha avoids hiding fit-window differences
                # beneath ten overlaid curves. Full and local views retain zero.
                for variant, limit in (("local", .1), ("full", 1.)):
                    fig, axes = canvas(f"2. Quadratic loss fits | {identity} | {variant}\n{seed_key}; primary solid, secondary dashed; outside fit range dotted/dash-dot",
                                       rows=2, cols=5, sharex=True, sharey=True)
                    for axis, train in zip(axes, ALPHAS):
                        selected = [{**r, "alpha_squared": r["evaluation_alpha"]**2} for r in rows if r["training_alpha"] == train and r["evaluation_alpha"] <= limit]
                        seed_points(axis, selected, "alpha_squared", "delta_loss", colors[train])
                        for fit in fits:
                            if fit["training_alpha"] != train or not fit["eligible"]:
                                continue
                            end = max(WINDOWS[fit["fit_window"]])**2
                            end = min(end, limit**2)
                            start = min(WINDOWS[fit["fit_window"]])**2
                            local_style = "-" if fit["fit_window"] == "primary" else "--"
                            extra_style = ":" if fit["fit_window"] == "primary" else "-."
                            c = fit["coefficient_c"]
                            axis.plot([0, start], [0, c*start], extra_style, color=colors[train], lw=.8, alpha=.6)
                            axis.plot([start, end], [c*start, c*end], local_style, color=colors[train], lw=.9)
                            if end < limit**2:
                                axis.plot([end, limit**2], [c*end, c*limit**2], extra_style, color=colors[train], lw=.8, alpha=.6)
                        axis.axvspan(.01**2, .05**2, color=".5", alpha=.08)
                        axis.set(title=f"Training alpha {train:g}", xlabel="Evaluation alpha squared", ylabel="Delta CE (nats/example)", xlim=(0, limit**2))
                    save(fig, 2, dataset, architecture, count, variant=variant,
                         explanation="Shading: primary fitting range; secondary additionally uses alpha=.10. Lines are individual seed fits; no predictive inference.")

                fig, axes = canvas(f"3. Empirical curvature proxy | {identity}\n{seed_key}; complete three-seed mean +/- sample SD", cols=2, sharey=True)
                for axis, window in zip(axes, WINDOWS):
                    selected = [f for f in fits if f["fit_window"] == window]
                    seed_points(axis, selected, "training_alpha", "empirical_curvature_proxy_2c", "#0072B2")
                    aggregate = [s for s in result["summaries"] if (s["dataset"], s["architecture"], s["fit_window"], s["metric"]) ==
                                 (dataset, architecture, window, "empirical_curvature_proxy_2c")]
                    aggregate.sort(key=lambda s: s["training_alpha"])
                    axis.errorbar([s["training_alpha"] for s in aggregate], [s["mean"] if s["complete"] else math.nan for s in aggregate],
                                  yerr=[s["sample_sd"] if s["complete"] else math.nan for s in aggregate], color="black", capsize=3, lw=1)
                    axis.set(title=window, xlabel="Training alpha", ylabel="Empirical proxy 2c (nats/example per alpha squared)")
                save(fig, 3, dataset, architecture, sum(f["eligible"] for f in fits))

                fig, axes = canvas(f"4. Observed versus predicted loss increase | {identity}\no: fitting points; x: larger alphas; baseline omitted", cols=2, sharex=True, sharey=True)
                bounds, used = [], 0
                for axis, window in zip(axes, WINDOWS):
                    for row in pred:
                        if row["fit_window"] != window or row["evaluation_alpha"] == 0 or row["observed_delta_loss"] is None or row["predicted_delta_loss"] is None:
                            continue
                        x, y = row["predicted_delta_loss"], row["observed_delta_loss"]
                        axis.scatter([x], [y], marker="o" if row["point_role"] == "fitting" else "x", s=15,
                                     color=colors[row["training_alpha"]], alpha=.65)
                        bounds.extend((x, y))
                        used += 1
                    axis.set(title=window, xlabel="Predicted Delta CE (nats/example)", ylabel="Observed Delta CE (nats/example)")
                if bounds:
                    low, high = min(0., min(bounds)), max(0., max(bounds))
                    pad = .04*(high-low) if high != low else .01
                    for axis in axes:
                        axis.plot([low-pad, high+pad], [low-pad, high+pad], color="black", lw=.8)
                        axis.set_xlim(low-pad, high+pad)
                        axis.set_ylim(low-pad, high+pad)
                        axis.set_aspect("equal", adjustable="box")
                fig.legend(handles=[Line2D([], [], marker="o", lw=0, color=colors[a], label=f"train {a:g}") for a in ALPHAS],
                           ncols=5, loc="outside lower center", fontsize=8)
                save(fig, 4, dataset, architecture, used)

                fig, axes = canvas(f"5. Accuracy and loss response | {identity}\nIndividual checkpoint means; size increases with evaluation alpha")
                axis = axes[0]
                for row in rows:
                    if row["eligible"] and row["evaluation_alpha"] > 0:
                        axis.scatter([row["delta_loss"]], [row["delta_accuracy_pp"]],
                            color=colors[row["training_alpha"]], s=12+45*row["evaluation_alpha"], alpha=.6)
                axis.set(xlabel="Delta CE loss (nats/example)", ylabel="Accuracy change (percentage points)")
                handles = [Line2D([], [], marker="o", lw=0, color=colors[a], label=f"train {a:g}") for a in ALPHAS]
                train_legend = axis.legend(handles=handles, ncols=2, fontsize=8, loc="upper right")
                axis.add_artist(train_legend)
                axis.legend(handles=[Line2D([], [], marker="o", lw=0, color=".4", markersize=math.sqrt(12+45*a),
                                           label=f"eval {a:g}") for a in (.01, .1, 1.)], loc="lower left", fontsize=8)
                save(fig, 5, dataset, architecture, sum(r["eligible"] and r["evaluation_alpha"] > 0 for r in rows),
                     explanation="Negative accuracy change means degradation. Absolute accuracy is retained in CSV and findings; low sensitivity can coexist with poor performance.")

                fig, axes = canvas(f"6. Paired loss response relative to clean training | {identity}\n{seed_key}; negative means smaller loss increase",
                                   rows=3, cols=3, sharex=True, sharey=True)
                pair_rows = [p for p in result["training_pairs"] if p["dataset"] == dataset and p["architecture"] == architecture and p["metric"] == "delta_loss"]
                for axis, train in zip(axes, ALPHAS[1:]):
                    selected = [p for p in pair_rows if p["training_alpha"] == train]
                    seed_points(axis, selected, "evaluation_alpha", "difference", colors[train], connect=True)
                    sums = [s for s in result["paired_summaries"] if s["dataset"] == dataset and s["architecture"] == architecture and s["training_alpha"] == train and s["metric"] == "delta_loss"]
                    sums.sort(key=lambda s: s["evaluation_alpha"])
                    axis.errorbar([s["evaluation_alpha"] for s in sums], [s["mean"] if s["complete"] else math.nan for s in sums],
                                  yerr=[s["sample_sd"] if s["complete"] else math.nan for s in sums], color="black", lw=1, capsize=2)
                    axis.set(title=f"Training alpha {train:g} minus 0", xlabel="Evaluation alpha",
                             ylabel="Paired Delta CE difference (nats/example)")
                save(fig, 6, dataset, architecture, sum(p["eligible"] for p in pair_rows))
    return index
