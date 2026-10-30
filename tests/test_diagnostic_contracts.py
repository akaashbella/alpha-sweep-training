"""Deferred diagnostic policy/export checks. No execution during implementation."""

from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


@unittest.skipUnless(os.environ.get("ALPHA_SWEEP_DIAGNOSTIC_TESTS") == "1", "Explicit diagnostic test execution required")
class DiagnosticContracts(unittest.TestCase):
    def policy(self):
        return dict(near_chance_margin=.02, extreme_loss=1e6, early_best_epochs=5,
                    weak_final_accuracy=.12, seed_accuracy_range=.2)

    def test_one_factor_identity_and_no_input_mutation(self):
        from alpha_sweep_training.diagnostic_protocol import validate_protocol, diagnostic_identity, DiagnosticError
        science = {"dataset": "cifar10", "architecture": "mobilenetv3_large", "alpha": 0.,
            "optimizer": {"epsilon": .001, "learning_rate": .01},
            "precision": {"mode": "amp_bfloat16"}, "training": {"cosine_min_lr": 0.}}
        config = SimpleNamespace(scientific=science, require_research_ready=lambda: None)
        protocol = dict(protocol_version=1, kind="optimization_diagnostic_v1", base_scientific_fingerprint="unit",
            arm="baseline", epsilon=.001, learning_rate=.01, precision_mode="amp_bfloat16",
            epochs=20, max_batches_per_epoch=None, trace_every=1, activation_modules=[], initial_checkpoint=None, source_difference_review={})
        original = deepcopy(protocol)
        with patch("alpha_sweep_training.diagnostic_protocol.scientific_fingerprint", return_value="unit"):
            baseline = validate_protocol(protocol, config)
            candidate = {**protocol, "arm": "epsilon", "epsilon": .0316}
            validate_protocol(candidate, config)
            self.assertNotEqual(diagnostic_identity(baseline), diagnostic_identity(candidate))
            with self.assertRaises(DiagnosticError):
                validate_protocol({**candidate, "learning_rate": .001}, config)
            with self.assertRaises(DiagnosticError):
                validate_protocol({**protocol, "max_batches_per_epoch": 2}, config)
            with self.assertRaises(DiagnosticError):
                validate_protocol({**protocol, "base_scientific_fingerprint": "wrong"}, config)
        self.assertEqual(protocol, original)
        self.assertEqual(science["optimizer"]["epsilon"], .001)

    def test_weight_sd_ratio_names_post_training_epoch_boundaries(self):
        from alpha_sweep_training.optimization_health import analyze_run
        row = dict(training_run_id="unit", dataset="cifar10", architecture="mobilenetv3_large", training_alpha=0.,
            training_seed=101, condition="ok", stored_status="training", training_complete=False,
            scientific_settings={"model": {"num_classes": 10}})
        metric = dict(correct=500, examples=5000, loss=2.)
        row["snapshot"] = {"transaction": {"best": dict(epoch_index=0, correct=500, examples=5000)}, "epochs": [
            {"document": {"record": dict(epoch_index=i, training=metric, validation=metric),
                          "statistics": [{"population_std": sd}]}} for i, sd in ((0, 4.), (1, 1.))]}
        result = analyze_run(row, self.policy())
        self.assertNotIn("median_weight_sd_final_to_initial_ratio", result)
        self.assertEqual(result["median_weight_sd_last_to_first_recorded_epoch_ratio"], .25)
        self.assertEqual(result["median_weight_sd_ratio_numerator_epoch_label"], 2)
        self.assertEqual(result["median_weight_sd_ratio_denominator_epoch_label"], 1)
        row["snapshot"]["epochs"][0]["document"]["statistics"][0]["population_std"] = 0.
        self.assertIsNone(analyze_run(row, self.policy())["median_weight_sd_last_to_first_recorded_epoch_ratio"])

    def test_auxiliary_source_review_cannot_waive_core_or_software_changes(self):
        from alpha_sweep_training.diagnostic_protocol import require_replay_source, DiagnosticError
        changed = {"path": "pyproject.toml", "saved": "a" * 64, "actual": "b" * 64}
        report = {"source_comparison": {"core_file_differences": [], "software_equal": True,
                                         "recorded_file_differences": [changed]}}
        with self.assertRaises(DiagnosticError):
            require_replay_source(report, {})
        review = {changed["path"]: {"saved": changed["saved"], "actual": changed["actual"], "reason": "Reviewed added CLI entry point only"}}
        require_replay_source(report, review)
        report["source_comparison"]["core_file_differences"] = [{"path": "src/alpha_sweep_training/optimizers.py"}]
        with self.assertRaises(DiagnosticError):
            require_replay_source(report, review)

    def test_output_disjointness_and_exclusive_claim_preserve_original_bytes(self):
        from alpha_sweep_training.diagnostic_protocol import new_output, DiagnosticError
        with tempfile.TemporaryDirectory(prefix="alpha-diagnostic-guard-unit-") as temporary:
            root = Path(temporary)
            original = root / "original"
            original.mkdir()
            evidence = original / "raw.bin"
            evidence.write_bytes(b"unchanged")
            with self.assertRaises(DiagnosticError):
                new_output(original / "diagnostics", [original])
            with self.assertRaises(DiagnosticError):
                new_output(root, [original])
            target = new_output(root / "separate", [original])
            with self.assertRaises(DiagnosticError):
                new_output(target, [original])
            self.assertEqual(evidence.read_bytes(), b"unchanged")
            self.assertEqual(list(original.iterdir()), [evidence])
            (Path(target) / "diagnostic_owner.json").write_text('{}', encoding="utf-8")
            with self.assertRaises(DiagnosticError):
                target / "new-write.json"

    def test_csv_health_uses_csvs_and_not_truncated_collection_or_test_results(self):
        from alpha_sweep_training.diagnostic_csv import report_csv_health
        from alpha_sweep_training.spec import ALPHAS, ARCHITECTURES
        from alpha_sweep_training.identity import canonical_alpha
        with tempfile.TemporaryDirectory(prefix="alpha-health-csv-unit-") as temporary:
            root = Path(temporary)
            collection = root / "collection"
            collection.mkdir()
            (collection / "collection.json").write_bytes(b'{"truncated":')
            training_rows, epoch_rows = [], []
            for architecture in ARCHITECTURES:
                for alpha in ALPHAS:
                    for seed in (101, 102, 103):
                        name = f"unit-{architecture}-{alpha}-{seed}"
                        best = {"epoch_index": 0, "correct": 4000, "examples": 5000}
                        shared = {"training_run_id": name, "dataset": "cifar10", "architecture": architecture,
                                  "canonical_training_alpha": canonical_alpha(alpha), "training_seed": seed}
                        training_rows.append({**shared, "training_alpha": alpha, "condition": "ok", "stored_status": "training",
                            "completed_epochs": 2, "training_complete": "False", "accuracy_unit": "fraction",
                            "best_selection": json.dumps(best), "best_validation_accuracy": .8})
                        for epoch in (0, 1):
                            training = {"correct": 36000 if epoch == 0 else 4500, "examples": 45000, "loss": .5 if epoch == 0 else 1e28}
                            validation = {"correct": 4000 if epoch == 0 else 400, "examples": 5000, "loss": .5 if epoch == 0 else 1e28}
                            record = {"epoch_index": epoch, "completed_epochs": epoch + 1, "training": training,
                                      "validation": validation, "best": best}
                            epoch_rows.append({**shared, "epoch_label": epoch + 1, "record": json.dumps(record),
                                "accuracy_unit": "fraction", "loss_unit": "mean_cross_entropy_nats_per_example",
                                "training_accuracy": training["correct"] / training["examples"],
                                "validation_accuracy": validation["correct"] / validation["examples"],
                                "training_loss": training["loss"], "validation_loss": validation["loss"]})
            for filename, rows in (("training.csv", training_rows), ("epochs.csv", epoch_rows)):
                with (collection / filename).open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            report = report_csv_health(root, self.policy())
            self.assertEqual(len(report["individual_runs"]), 120)
            self.assertIn("not opened", report["collection_json"])
            self.assertFalse(report["research_authority"])
            for row in report["individual_runs"]:
                self.assertTrue(row["flags"]["extreme_finite_loss"])
                self.assertTrue(row["flags"]["near_chance_final_training"])
                self.assertTrue(row["flags"]["early_best_and_weak_final"])
                self.assertIsNone(row["training_completion_verified"])
                self.assertEqual(row["epochs"][-1]["clean_validation"]["loss"], 1e28)
            self.assertTrue(all(not row["official_three_seed_summary"] for row in report["seed_groups"]))
            self.assertEqual((collection / "collection.json").read_bytes(), b'{"truncated":')


if __name__ == "__main__":
    unittest.main()
