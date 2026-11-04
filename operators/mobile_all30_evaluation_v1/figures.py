"""Derived revised-MobileNet plots. Run only after separately authorized reporting."""

import argparse
import math
from pathlib import Path
import statistics

import launcher as op


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    report_path = op.plain(args.report)
    report = op.check(op.read(report_path))
    op.same(report["version"], "mobile-all30-evaluation-report-v1", "report version")
    op.require(len(report["training"]) == 30 and len(report["evaluation"]) == 840, "Full observation grid required")
    protected = [op.NATIVE, op.STUDY, op.HOME, op.OPS, op.EVALUATIONS, op.OLD_OPERATOR, op.OLD_PLAN,
                 report_path.parent, op.BASE / "main-v1"]
    root, _ = op.claim(args.output_dir, protected, "derived_evaluation_figures")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = plt.get_cmap("tab10")
    groups = report["summaries"]["across_training_seeds"]
    seeds = report["summaries"]["per_training_seed"]
    relative = report["summaries"]["paired_clean_relative"]
    files = []

    def save(fig, name):
        fig.suptitle("CIFAR-10 MobileNetV3-Large · RMSprop epsilon 0.0316\n" + name.replace("_", " "))
        fig.tight_layout(rect=(0, 0.03, 1, 0.91))
        for extension in ("png", "pdf"):
            target = root / (name + "." + extension)
            fig.savefig(target, dpi=180)
            files.append(target.name)
        plt.close(fig)

    for metric in ("accuracy", "loss"):
        fig, ax = plt.subplots(figsize=(9, 6))
        for j, alpha in enumerate(op.ALPHAS):
            selected = [g for g in groups if g["training_alpha"] == alpha]
            scale = 100 if metric == "accuracy" else 1
            values = [scale * g["mean_" + metric] if g["complete_three_seeds"] else math.nan for g in selected]
            sd = [scale * g["between_seed_sample_sd_" + metric] if g["complete_three_seeds"] else math.nan for g in selected]
            ax.errorbar(op.ALPHAS, values, yerr=sd, color=colors(j), marker="o", linewidth=1,
                        label=f"training alpha {alpha:g}")
        ax.set(xlabel="Evaluation alpha", ylabel="Test accuracy (%)" if metric == "accuracy" else "Test cross-entropy")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=.25)
        fig.text(.5, .01, "Within-checkpoint means, then three-seed mean ± sample SD; missing sets are gaps", ha="center", fontsize=8)
        save(fig, "test_" + metric)
        fig, axes = plt.subplots(2, 5, figsize=(18, 8), sharex=True)
        for j, (alpha, ax) in enumerate(zip(op.ALPHAS, axes.flat)):
            for seed in (0, 1, 2):
                selected = [g for g in seeds if g["training_alpha"] == alpha and g["training_seed"] == seed]
                scale = 100 if metric == "accuracy" else 1
                ax.plot(op.ALPHAS, [scale * g["mean_" + metric] if g["complete"] else math.nan for g in selected],
                        color=colors(seed), label=f"seed {seed}")
                for group in selected:
                    for item in group["individual"]:
                        if item["metrics"] is not None:
                            ax.scatter(group["evaluation_alpha"], scale * item["metrics"][metric], color=colors(seed), s=10, alpha=.6)
            ax.set(title=f"Training alpha {alpha:g}", xlabel="Evaluation alpha",
                   ylabel="Accuracy (%)" if metric == "accuracy" else "Cross-entropy")
            ax.grid(alpha=.25)
            ax.legend(fontsize=7)
        fig.text(.5, .01, "Points: individual prescribed passes. Lines: complete within-checkpoint realization means. No smoothing.", ha="center", fontsize=8)
        save(fig, "individual_" + metric)
    for key, label, baseline in (("accuracy_difference_from_own_clean_pp", "Change from own clean test (pp)", 0),
                                 ("accuracy_retention_from_own_clean", "Accuracy / own clean test accuracy", 1)):
        fig, ax = plt.subplots(figsize=(9, 6))
        for j, alpha in enumerate(op.ALPHAS):
            means, deviations = [], []
            for beta in op.ALPHAS:
                values = [r[key] for r in relative if r["training_alpha"] == alpha and r["evaluation_alpha"] == beta]
                complete = len(values) == 3 and all(v is not None for v in values)
                means.append(statistics.mean(values) if complete else math.nan)
                deviations.append(statistics.stdev(values) if complete else math.nan)
            ax.errorbar(op.ALPHAS, means, yerr=deviations, color=colors(j), marker="o", linewidth=1,
                        label=f"training alpha {alpha:g}")
        ax.axhline(baseline, color="black", linestyle="--", linewidth=.8)
        ax.set(xlabel="Evaluation alpha", ylabel=label)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=.25)
        fig.text(.5, .01, "Each seed uses its own checkpoint's clean baseline; paired three-seed mean ± sample SD", ha="center", fontsize=8)
        save(fig, key)
    op.publish(root / "FIGURES.json", op.seal({"report_sha256": op.digest(report_path),
               "scientific_completion_authority": False, "files": {name: op.digest(root / name) for name in files},
               "formulas": report["summaries"]["relative_formulas"],
               "missing_observations": sum(r["test_metrics"] is None for r in report["evaluation"]),
               "scope": "Revised MobileNet only; final presentation must supply each graph type for every model"}))
    print(op.canonical({"figures": str(root), "files": files}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
