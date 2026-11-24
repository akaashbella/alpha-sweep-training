"""Render all ten figure families from explicitly synthetic observations."""

from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from ..figures import plot
from ..storage import plain, publication, publish, seal
from ..summary import pairs
from .test_estimator import BEST


class FigureTests(unittest.TestCase):
    def test_all_families_render_from_sealed_fixture(self):
        with TemporaryDirectory(prefix="stage3-software-fixture-") as directory:
            root = plain(directory)
            common = {"dataset": "cifar10", "architecture": "resnet50", "training_seed": 7, "checkpoint": BEST,
                      "split": "validation", "mean": -1., "mc_se": .1, "complete": True, "eligible": 16, "expected": 16,
                      "cohort_sha256": "SYNTHETIC", "clean_accuracy": .6}
            traces = [{**common, "checkpoint_id": f"synthetic-{train}", "training_alpha": train,
                       "target_id": target, "scope": scope, "target_name": name, "mean": value*(1+train)}
                      for train in (0., .10, .30) for target, scope, name, value in
                      (("network:all", "network", "all", 1.), ("stage:stem", "stage", "stem", -1.),
                       ("stage:head", "stage", "head", 2.))]
            summary = seal("summary", {"split": "validation", "traces": traces, "pairs": pairs(traces), "protected_roots": [],
                           "workload": {"probes_per_checkpoint": 16},
                           "convergence": [{**trace, "probe_prefix": p} for trace in traces for p in (2, 4, 8, 16)]})
            summary_dir = publish(root/"summary", [], "summary", summary)
            stage2 = [{**trace, "evaluation_alpha": a, "predicted_delta_loss": .5*a*a*trace["mean"],
                       "observed_delta_loss": .6*a*a*trace["mean"], "residual": .1*a*a*trace["mean"],
                       "stage3_trace": trace["mean"], "prediction_mc_se": .5*a*a*.1, "stage3_mc_se": .1,
                       "stage2_perturbation_se": a*a*.05, "stage2_accuracy_change_pp": -a,
                       "stage2_accuracy_change_se": .02} for a in (.01, .02, .05, .10, .20) for trace in traces]
            fits = [{**trace, "empirical_split": split, "split_match": split == "validation",
                     "hessian_trace": trace["mean"], "empirical_curvature_proxy_2c": .8*trace["mean"], "hessian_mc_se": .1,
                     "fit_window": window} for trace in traces for split in ("validation", "test") for window in ("primary", "secondary")]
            compared = seal("comparison", {"summary_sha256": summary["document_sha256"], "protected_roots": [], "stage2": stage2,
                            "stage1": [r for r in fits if r["empirical_split"] == "test"],
                            "fits": [r for r in fits if r["empirical_split"] == "validation"]})
            compared_dir = publish(root/"compare", [summary_dir], "comparison", compared)
            with patch("analysis.hessian_geometry_stage3.figures.DATASETS", ("cifar10",)), patch(
                "analysis.hessian_geometry_stage3.figures.ARCHITECTURES", ("resnet50",)
            ):
                output = plot(summary_dir, compared_dir, root/"figures", 72)
            self.assertEqual(output["figure_documents"], 10)
            sealed = publication(root/"figures", "figures")
            self.assertEqual(len(sealed["index"]), 10)
            self.assertTrue(all(len(row["files"]) == 2 for row in sealed["index"]))
