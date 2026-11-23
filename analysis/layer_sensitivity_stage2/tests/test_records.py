from contextlib import contextmanager, nullcontext
from copy import deepcopy
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from alpha_sweep_training.evaluation_state import CleanStateBaseline
from alpha_sweep_training.weights import discover_eligible_weights

from ..measurement import inventory
from ..planning import make_plan, require_corrected
from ..records import checkpoint_root, inspect_checkpoint, run_selected
from ..report import collect, compare_training, descriptive, stable_ratio, summarize
from ..storage import Stage2Error, disjoint, install, plain, publish_document, published, seal
from .test_measurement import CHECKPOINT, fixture


def stub_plan(workdir):
    model, images, labels = fixture()
    layout = {"tensors": [entry.to_dict() for entry in discover_eligible_weights(model)],
              "stages": {"stem": ["0.weight"], "classifier": ["3.weight"]}, "stage_order": ["stem", "classifier"],
              "diagnostics": {}}
    key = {"dataset": "cifar10", "architecture": "resnet50", "training_alpha": 0., "training_seed": 123}
    evidence = {"binding": {"best_checkpoint_identity": CHECKPOINT, "training_identity": {"scientific_fingerprint": "synthetic"}},
                "scientific": {"optimizer": {}, "model": {"num_classes": 3}}, "prepared_data": {"split_identity": "synthetic"}}
    row = {"key": key, "evidence": evidence, "layout": layout, "status": "eligible", "run_id": "synthetic"}
    excluded = {"key": dict(key, architecture="mobilenetv3_large"), "status": "excluded", "reason": "synthetic missing provenance"}
    audited = seal("audit", {"source_sha256": "synthetic-code", "selected_seeds": [123], "protected_roots": [str(workdir/"protected")],
                             "checkpoints": [row, excluded]})
    with patch("analysis.layer_sensitivity_stage2.planning.source_digest", return_value="synthetic-code"):
        plan = make_plan(audited, split="validation", alphas=(.05,), realizations=2, scopes=("stages",), max_passes=7)
        self_same = make_plan(audited, split="validation", alphas=(.05,), realizations=2, scopes=("stages",), max_passes=7)
    assert plan == self_same
    # Manufactured input has seven examples. Production make_plan always uses
    # 5,000/10,000 and run independently verifies the full prepared dataset.
    plan = seal("plan", {**{k: v for k, v in plan.items() if k != "document_sha256"}, "examples_per_pass": 7})
    return plan, model, images, labels


class RecordTests(unittest.TestCase):
    def test_corrected_mobilenet_requires_declared_and_effective_recipe(self):
        for dataset in ("cifar10", "cifar100"):
            science = {"architecture": "mobilenetv3_large", "dataset": dataset, "optimizer": {"name": "rmsprop", "epsilon": .001}}
            with self.assertRaisesRegex(Stage2Error, "epsilon=0.0316"):
                require_corrected(science)
            science["optimizer"]["epsilon"] = .0316
            with self.assertRaises(Stage2Error):
                require_corrected(science, {"declared_recipe": science["optimizer"], "effective_options": {}})
            with self.assertRaises(Stage2Error):
                require_corrected(science, {"declared_recipe": science["optimizer"], "effective_options": {"eps": .001}})
            require_corrected(science, {"declared_recipe": science["optimizer"], "effective_options": {"eps": .0316}})

    def test_exclusive_publication_and_protected_paths(self):
        with TemporaryDirectory(prefix="stage2-fixture-") as temp:
            root = plain(temp)
            first = root / "audit"
            data = seal("audit", {"test": [1, 2, 3]})
            publish_document(first, [root/"protected"], "audit", data)
            self.assertEqual(published(first, "audit", "audit.json"), data)
            with self.assertRaises(Stage2Error):
                publish_document(first, [root/"protected"], "audit", data)
            with self.assertRaises(FileExistsError):
                install(first/"audit.json", seal("audit", {"test": -1}))
            self.assertEqual(published(first, "audit", "audit.json"), data)
            for output in (first, first/"nested", root):
                with self.assertRaises(Stage2Error):
                    disjoint(output, [first])
            with (first/"audit.json").open("a", encoding="utf-8") as stream:
                stream.write(" ")
            with self.assertRaisesRegex(Stage2Error, "file mismatch"):
                published(first, "audit", "audit.json")

    def test_bounds_and_complete_statistics(self):
        with TemporaryDirectory(prefix="stage2-fixture-") as temp:
            plan, *_ = stub_plan(plain(temp))
            self.assertEqual(plan["estimates"]["passes"], 7)
            self.assertEqual(plan["estimates"]["checkpoints_requested"], 2)
        self.assertIsNone(descriptive([1.], 2)["mean"])
        self.assertEqual(descriptive([-1., 0.], 2)["mean"], -.5)
        self.assertIsNone(stable_ratio(-.1, 0., 1.))
        self.assertIsNone(stable_ratio(-.1, 1e-9, 1.))
        self.assertEqual(stable_ratio(-.1, -.2, 1.), .5)

    def test_training_contrasts_match_complete_compatible_seed(self):
        core = {"dataset": "cifar10", "architecture": "resnet50", "training_seed": 1, "target_id": "stage:stem",
                "target_name": "stem", "scope": "stage", "evaluation_alpha": .05, "metric": "delta_loss",
                "complete": True, "compatibility_signature": "fixed"}
        clean = {**core, "training_alpha": 0., "checkpoint_id": "clean", "mean": .25}
        noise = {**core, "training_alpha": .1, "checkpoint_id": "noise", "mean": .125}
        result = compare_training([clean, noise])[0]
        self.assertEqual(result["noise_minus_clean_training"], -.125)
        self.assertEqual(result["clean_training_checkpoint_id"], "clean")
        mismatch = {**noise, "compatibility_signature": "different"}
        self.assertIsNone(compare_training([clean, mismatch])[0]["noise_minus_clean_training"])
        self.assertIsNone(compare_training([clean, {**noise, "training_seed": 2}])[0]["noise_minus_clean_training"])

    def test_bounded_resume_failure_and_identity_validation(self):
        with TemporaryDirectory(prefix="stage2-fixture-") as temp:
            root = plain(temp)
            plan, model, images, labels = stub_plan(root)
            row = plan["checkpoints"][0]
            inv, scales = inventory(model, row["layout"])
            results = root/"results"
            runtime = SimpleNamespace(device=torch.device("cpu"), autocast=nullcontext,
                                      require_active=lambda: None, batch=lambda batch, classes: batch)
            data = {"split": "validation", "examples": 7, "ordered_indices_sha256": "b"*64,
                    "prepared": row["evidence"]["prepared_data"]}

            @contextmanager
            def worker(*args):
                loader = DataLoader(TensorDataset(images, labels), batch_size=128)
                yield model, loader, runtime, CleanStateBaseline(model), inv, scales, data, {"fixture": True}

            def run(**kwargs):
                return run_selected(plan, root/"plan", row, results, target_ids=None,
                                    **{"max_passes": 1, "retry_failed": False, **kwargs})

            with patch("analysis.layer_sensitivity_stage2.records.source_digest", return_value="synthetic-code"), patch(
                "analysis.layer_sensitivity_stage2.components.prepared_worker", worker):
                self.assertEqual(run()["accepted_passes"], 1)
                directory = checkpoint_root(results, row)
                clean_path = next((directory/"slots").glob("*.json"))
                original = clean_path.read_bytes()
                with patch("analysis.layer_sensitivity_stage2.measurement.measure", side_effect=RuntimeError("synthetic failure")):
                    with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                        run()
                with self.assertRaisesRegex(Stage2Error, "--retry-failed"):
                    run()
                outcome = run(max_passes=8, retry_failed=True)
                self.assertEqual(outcome["executed_passes"], 6)
                self.assertTrue(outcome["checkpoint_complete"])
                self.assertEqual(run()["executed_passes"], 0)
            self.assertEqual(clean_path.read_bytes(), original)
            records, coverage, _, completion, _ = inspect_checkpoint(plan, row, results)
            self.assertTrue(completion)
            self.assertEqual(len(records), 7)
            self.assertTrue(any(len(c["attempts"]) > 1 for c in coverage))
            report = collect(plan, root/"plan", results)
            self.assertEqual(len(report["raw"]), 7)
            self.assertEqual(len(report["residuals"]), 2)
            self.assertTrue(all(r["interaction_residual"] is not None for r in report["residuals"]))
            summary_dir = root/"summary"
            summarize(plan, root/"plan", results, summary_dir)
            self.assertTrue((summary_dir/"COMPLETE.json").is_file())
            from ..figures import plot
            figures_dir = root/"figures"
            with patch("analysis.layer_sensitivity_stage2.figures.DATASETS", ("cifar10",)), patch(
                "analysis.layer_sensitivity_stage2.figures.ARCHITECTURES", ("resnet50",)):
                plot(summary_dir, figures_dir, alpha=.05, dpi=72)
            figures = published(figures_dir, "figures", "figures.json")["index"]
            self.assertEqual({r["family"][:2] for r in figures}, set("0"+str(i) for i in range(1, 9)))
            self.assertEqual(sum(bool(r["files"]) for r in figures), 7)  # No noise-trained pair in this fixture.
            mismatch = deepcopy(row)
            mismatch["evidence"]["binding"]["best_checkpoint_identity"]["sha256"] = "c"*64
            with self.assertRaisesRegex(Stage2Error, "identity mismatch"):
                inspect_checkpoint(plan, mismatch, results)
