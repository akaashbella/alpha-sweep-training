"""Checkpoint identity, completeness and statistical replication checks."""

from copy import deepcopy
import unittest

from ..analysis import analyze
from ..contracts import AnalysisError
from ..loading import validate_summaries
from .fixtures import SEEDS, fixture, reseal


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = fixture()
        validate_summaries(cls.document)
        cls.result = analyze(cls.document, SEEDS)

    def test_clean_pairing_loss_and_accuracy_units(self):
        for row in self.result["loss_sensitivity"]:
            self.assertAlmostEqual(row["clean_loss"], 1+row["training_alpha"]+row["training_seed"]/100)
            self.assertAlmostEqual(row["delta_loss"], (3/(1+row["training_alpha"])+row["training_seed"]/100)*row["evaluation_alpha"]**2)
            self.assertAlmostEqual(row["delta_accuracy_pp"], 100*(row["perturbed_accuracy"]-row["clean_accuracy"]))
            self.assertAlmostEqual(row["accuracy_degradation_pp"], -row["delta_accuracy_pp"])

    def test_full_expected_coverage_and_seed_first_fitting(self):
        self.assertEqual(len(self.result["loss_sensitivity"]), 2400)
        self.assertEqual(len(self.result["quadratic_fits"]), 480)
        self.assertEqual(len(self.result["predictions"]), 4800)
        self.assertEqual(len(self.result["coverage"]["realization_slots"]), 6720)
        fit = self.result["quadratic_fits"][0]
        self.assertAlmostEqual(fit["coefficient_c"], 3.11)
        group = next(s for s in self.result["summaries"] if s["summary_kind"] == "quadratic_fit" and s["fit_window"] == "primary")
        self.assertEqual(group["eligible_count"], 3)
        self.assertAlmostEqual(group["mean"], 3.22)
        self.assertAlmostEqual(group["sample_sd"], .11)

    def test_deterministic_and_no_input_mutation(self):
        original = deepcopy(self.document)
        again = analyze(self.document, SEEDS)
        self.assertEqual(again, self.result)
        self.assertEqual(original, self.document)

    def test_correlations_keep_incompatible_settings_in_separate_cohorts(self):
        # At the analysis boundary deliberately separate one checkpoint's accepted
        # settings. No malformed document is published as a scientific fixture.
        doc = deepcopy(self.document)
        doc["checkpoint_robustness"][0]["compatibility_signature"] = "sha256:" + "d"*64
        doc["checkpoint_robustness"][1]["compatibility_signature"] = "sha256:" + "d"*64
        result = analyze(doc, SEEDS)
        groups = [r for r in result["relationships"] if r["dataset"] == "cifar10" and r["architecture"] == "resnet50" and
                  r["training_alpha"] is None and r["evaluation_alpha"] == .01 and r["x_metric"] == "delta_loss"]
        self.assertEqual(len(groups), 2)
        self.assertEqual(sorted(r["count"] for r in groups), [1, 29])

    def test_incomplete_realizations_and_missing_clean_baseline(self):
        doc = fixture(missing={(0, .02, 2), (1, 0., 0)})
        validate_summaries(doc)
        result = analyze(doc, SEEDS)
        self.assertEqual(len(result["coverage"]["realization_slots"]), 6720)
        self.assertFalse(result["loss_sensitivity"][2]["eligible"])
        self.assertIsNone(result["loss_sensitivity"][2]["delta_loss"])
        self.assertEqual(result["quadratic_fits"][0]["valid_points"], 2)
        self.assertFalse(result["quadratic_fits"][0]["eligible"])
        self.assertTrue(all(r["delta_loss"] is None for r in result["loss_sensitivity"][10:20]))
        self.assertIn("Clean baseline", result["loss_sensitivity"][11]["exclusion_reason"])
        groups = [g for g in result["summaries"] if g["summary_kind"] == "quadratic_fit" and g["dataset"] == "cifar10" and g["architecture"] == "resnet50" and g["training_alpha"] == 0]
        self.assertTrue(all(g["mean"] is None for g in groups))

    def test_negative_loss_changes_are_preserved(self):
        doc = deepcopy(self.document)
        row = doc["checkpoint_robustness"][1]
        clean_loss = doc["checkpoint_robustness"][0]["loss"]["official"]["mean"]
        row["loss"]["official"]["mean"] = clean_loss-.01
        result = analyze(doc, SEEDS)  # Numerical analysis boundary; schema mutation is deliberately not published.
        self.assertAlmostEqual(result["loss_sensitivity"][1]["delta_loss"], -.01)

    def test_valid_individual_training_pairs_survive_another_missing_seed(self):
        doc = fixture(missing={(0, .02, 2)})
        validate_summaries(doc)
        result = analyze(doc, SEEDS)
        pairs = [p for p in result["training_pairs"] if p["dataset"] == "cifar10" and p["architecture"] == "resnet50" and
                 p["training_alpha"] == .01 and p["evaluation_alpha"] == .02]
        self.assertEqual(sum(p["eligible"] for p in pairs), 2)
        self.assertEqual(sum(p["difference"] is not None for p in pairs), 2)
        summary = next(p for p in result["paired_summaries"] if p["dataset"] == "cifar10" and p["architecture"] == "resnet50" and
                       p["training_alpha"] == .01 and p["evaluation_alpha"] == .02)
        self.assertIsNone(summary["mean"])
        self.assertEqual(summary["eligible_count"], 2)

    def test_existing_cross_seed_incompatibility_blocks_official_analysis_mean(self):
        doc = deepcopy(self.document)
        group = doc["robustness_groups"][1]
        for metric in ("loss", "accuracy", "noise_change_pp"):
            stat = group[metric]
            for row in stat["individuals"]:
                row.update(value=None, reason="incompatible cross-seed evaluation settings")
            stat.update(official=None, complete=False, eligible_count=0, exclusions=deepcopy(stat["individuals"]))
        validate_summaries(reseal(doc))
        result = analyze(doc, SEEDS)
        self.assertTrue(result["loss_sensitivity"][1]["eligible"])
        self.assertFalse(result["loss_sensitivity"][1]["cross_seed_eligible"])
        group = next(s for s in result["summaries"] if s["metric"] == "delta_loss" and s["evaluation_alpha"] == .01)
        self.assertIsNone(group["mean"])

    def test_duplicate_and_mismatched_identities_rejected(self):
        mutations = [lambda d: d["checkpoint_robustness"].append(deepcopy(d["checkpoint_robustness"][0])),
                     lambda d: d["checkpoint_robustness"][1].update(training_seed=999),
                     lambda d: d["checkpoint_robustness"][1].update(canonical_evaluation_alpha="0.02"),
                     lambda d: d["checkpoint_robustness"][1].update(best_checkpoint_identity=dict(generation="different", sha256="d"*64)),
                     lambda d: d["checkpoint_robustness"][1]["loss"]["individuals"][1].update(logical_index=1),
                     lambda d: d["clean_seed_pairs"][0].update(baseline_run_id="wrong"),
                     lambda d: d["checkpoint_robustness"].pop()]
        for mutate in mutations:
            doc = deepcopy(self.document)
            mutate(doc)
            with self.subTest(mutation=mutate), self.assertRaises(AnalysisError):
                validate_summaries(reseal(doc))
