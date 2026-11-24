"""Synthetic provenance, immutable publications and comparison formulas."""

from contextlib import contextmanager
from copy import deepcopy
from itertools import product
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from alpha_sweep_training.manifests import DATA_KEYS
from analysis.layer_sensitivity_stage2.groups import targets

from ..comparison import comparison_metrics, fit_pair, settings_relation, stage2_comparison
from ..protocol import (ARCHITECTURES, DATASETS, HISTORICAL_COMMIT, HISTORICAL_SOURCE, PILOT_ALPHAS,
                        PRESERVATION, RANDOMNESS, corrected, decomposition, extend, load_plan, numerical_policy,
                        pilot_membership, plan_document, probe_seed)
from ..records import binding, execute, namespace, observe, probe_file, validate_inventory, validate_probe
from ..storage import Stage3Error, code_digest, disjoint, identity, install, plain, publish, seal
from ..summary import collect, ratio
from ..worker import validate_runtime
from .test_estimator import BEST, entry


def layout():
    return {"tensors": [entry("first"), entry("second")], "stages": {"stem": ["first.weight"], "head": ["second.weight"]},
            "stage_order": ["stem", "head"], "diagnostics": {"combined": ["first.weight", "second.weight"]}}


def row(dataset="cifar10", architecture="resnet50", alpha=0., historical=False):
    key = {"dataset": dataset, "architecture": architecture, "training_alpha": alpha, "training_seed": 3}
    run_id = f"fixture-{dataset}-{architecture}-{alpha}"
    recipe = {"name": "rmsprop", "epsilon": .0316}
    science = {"dataset": dataset, "architecture": architecture, "alpha": alpha, "training_seed": 3,
               "specification_version": "fixture", "optimizer": recipe}
    prepared = {name: "sha256:"+identity({"synthetic": name}) for name in sorted(DATA_KEYS)}
    native = {"checkpoint_id": identity(key)[:20], "key": key, "status": "eligible", "run_id": run_id, "layout": layout(),
              "source": {"manifest_sha256": "e"*64},
              "targets": [{"id": "network:all"}, {"id": "stage:stem"}, {"id": "stage:head"}],
              "evidence": {"scientific": science, "prepared_data": prepared, "format_version": 1,
                           "best_optimizer": {"declared_recipe": recipe, "effective_options": {"eps": .0316}},
                           "best_training_provenance": {"code": {"commit": HISTORICAL_COMMIT, "source_sha256": HISTORICAL_SOURCE}},
                           "binding": {**key, "training_identity": {"run_id": run_id, "specification_version": "fixture", "scientific_fingerprint": "c"*64},
                                       "best_checkpoint_identity": BEST, "best_checkpoint": {
                                           "role": "best", "sha256": BEST["sha256"], "completed_epochs": 10,
                                           "relative_path": f"generations/{BEST['generation']}.best.pt",
                                       }, "data_identity": {**prepared, "dataset": dataset, "training_seed": key["training_seed"]}}}}
    scales = [dict(t, population_std=1. if i else .5) for i,t in enumerate(layout()["tensors"])]
    earlier = {"precision": {"mode": "amp_bfloat16"}, "effective_backend": {"allow_tf32_cudnn": True}}
    return {"checkpoint_id": native["checkpoint_id"], "key": key, "status": "eligible", "stage2_row": native,
            "reader": {"historical": historical}, "stage2": {"inventory": {"tensors": scales},
                "data": {"prepared": prepared, "examples": 7, "split": "validation", "ordered_indices_sha256": "f"*64},
                "metrics": {"loss": 2., "accuracy": 3/7}, "runtime": earlier}}


def core():
    return {"checkpoints": [row(d,a,t) for d,a,t in product(DATASETS, ARCHITECTURES, PILOT_ALPHAS)],
            "split": "validation", "examples": 7, "batch_size": 3, "numerics": numerical_policy("cudnn"),
            "randomness": RANDOMNESS, "distribution": "Rademacher", "protected_roots": [],
            "code_sha256": code_digest(), "stage2_plan_sha256": "d"*64}


def measured_runtime(numerics):
    return {"numerics": numerics, "effective": {"allow_tf32_matmul": False, "allow_tf32_cudnn": False, "autocast": False,
             "cudnn_enabled": True, "cudnn_benchmark": False, "cudnn_deterministic": True,
             "deterministic_algorithms": True, "deterministic_warn_only": False, "float32_matmul_precision": "highest",
             "cublas_workspace_config": ":4096:8"}, "software": {"torch": "2.5.1+cu124", "torchvision": "0.20.1+cu124",
             "python": "3.11.1", "cuda": "12.4", "cudnn": 90100}, "gpu": {"name": "synthetic", "capability": [9,0], "total_memory_bytes": 1000}}


def measurement():
    return {"clean_loss": 2., "clean_accuracy": 3/7, "clean_correct": 3, "examples": 7, "batches": 3,
            "contributions": decomposition(layout(), {"first.weight": -1., "second.weight": .5})}


def candidate(plan, checkpoint):
    return seal("inventory", {"binding": binding(plan, checkpoint), "layout": layout(), "scales": {"first.weight": .5, "second.weight": 1.}})


def document(plan, checkpoint, probe):
    inv = candidate(plan, checkpoint)
    return seal("probe", {"probe": probe, "binding": binding(plan, checkpoint), "inventory_sha256": inv["document_sha256"],
                         "status": "success", "preservation": PRESERVATION, "randomness": RANDOMNESS, "distribution": "Rademacher",
                         "seeds": {t: probe_seed(BEST, probe, t) for t in inv["scales"]}, "measurement": measurement(),
                         "runtime": measured_runtime(plan["core"]["numerics"]), "data": checkpoint["stage2"]["data"],
                         "duration_seconds": 1., "peak_allocated_bytes": 100, "peak_reserved_bytes": 120})


class RecordsTests(unittest.TestCase):
    def test_pilot_exact_membership_partition_and_pass_budget(self):
        stage_counts = dict(zip(ARCHITECTURES, (6, 8, 10, 9)))
        checkpoints = []
        for dataset, arch, alpha in product(DATASETS, ARCHITECTURES, PILOT_ALPHAS):
            task = row(dataset, arch, alpha)["stage2_row"]
            names = [f"stage_{i}" for i in range(stage_counts[arch])]
            task["layout"] = {"tensors": [entry(name) for name in names], "stage_order": names,
                              "stages": {name: [f"{name}.weight"] for name in names}, "diagnostics": {}}
            task["targets"] = targets(task["layout"], ("stages",))
            task["passes"] = 1+40*len(task["targets"])
            checkpoints.append(task)
        plan = {"checkpoints": checkpoints, "split": "validation", "examples_per_pass": 5000,
                "evaluation_alphas": [0., .01, .02, .05, .10, .20], "realizations_per_positive_alpha": 8,
                "selected_seeds": [3], "scopes": ["stages"], "estimates": {"passes": 8904}}
        self.assertEqual(len(pilot_membership(plan)), 24)
        bad = deepcopy(plan)
        bad["checkpoints"][0]["targets"][0]["tensors"].pop()
        with self.assertRaisesRegex(Stage3Error, "targets"):
            pilot_membership(bad)
        bad = deepcopy(plan)
        bad["checkpoints"][0] = bad["checkpoints"][1]
        with self.assertRaisesRegex(Stage3Error, "membership"):
            pilot_membership(bad)
        work = plan_document({**core(), "examples": 5000, "batch_size": 32}, 16, 384, [])["workload"]
        self.assertEqual((work["dataset_probes"], work["examples"], work["batch_hvps"]), (384, 1920000, 60288))

    def test_corrected_recipe_and_provenance_join(self):
        checkpoint = row(architecture="mobilenetv3_large")["stage2_row"]
        corrected(checkpoint)
        old = deepcopy(checkpoint)
        old["evidence"]["best_optimizer"]["effective_options"]["eps"] = .001
        with self.assertRaisesRegex(Stage3Error, "epsilon=0.0316"):
            corrected(old)
        old = deepcopy(checkpoint)
        old["evidence"]["best_training_provenance"]["code"]["commit"] = "0"*40
        with self.assertRaisesRegex(Stage3Error, "historical"):
            corrected(old)
        old = deepcopy(checkpoint)
        old["evidence"]["binding"]["best_checkpoint"]["sha256"] = "d"*64
        with self.assertRaisesRegex(Stage3Error, "mismatch"):
            corrected(old)
        old = deepcopy(checkpoint)
        old["evidence"]["binding"]["best_checkpoint"]["relative_path"] = f"generations/{'e'*32}.best.pt"
        with self.assertRaisesRegex(Stage3Error, "reference mismatch"):
            corrected(old)
        for missing in ({}, {name: value for name, value in checkpoint["evidence"]["prepared_data"].items()
                             if name != "normalization_identity"}):
            old = deepcopy(checkpoint)
            old["evidence"]["prepared_data"] = missing
            with self.assertRaisesRegex(Stage3Error, "prepared-data identities"):
                corrected(old)
        old = deepcopy(checkpoint)
        old["evidence"]["binding"]["data_identity"]["preparation_identity"] = "sha256:" + "e"*64
        with self.assertRaisesRegex(Stage3Error, "prepared-data identities"):
            corrected(old)

    def test_runtime_data_and_incomplete_probes_reject(self):
        plan = plan_document(core(), 16, 384, [])
        checkpoint = plan["core"]["checkpoints"][0]
        inv = candidate(plan, checkpoint)
        validate_inventory(inv, plan, checkpoint)
        valid = document(plan, checkpoint, 1)
        validate_probe(valid, plan, checkpoint, 1, inv)
        for key, subkey, value in (("data", "split", "test"), ("measurement", "batches", 2),
                                    ("binding", "checkpoint_id", "other"), ("seeds", "first.weight", 0)):
            with self.subTest(key=key):
                bad = deepcopy(valid)
                bad[key][subkey] = value
                with self.assertRaises(Stage3Error):
                    validate_probe(bad, plan, checkpoint, 1, inv)
        actual = measured_runtime(plan["core"]["numerics"])
        for setting, value in (("allow_tf32_matmul", True), ("cudnn_enabled", False)):
            bad = deepcopy(actual)
            bad["effective"][setting] = value
            with self.assertRaisesRegex(Stage3Error, "backend"):
                validate_runtime(bad, plan["core"]["numerics"])

    def test_exclusive_checkpoint_resume_and_append_only_extension(self):
        with TemporaryDirectory(prefix="stage3-software-") as directory:
            root = plain(directory)
            frozen = core()
            plan = plan_document(frozen, 16, 384, [])
            plan_dir = publish(root/"plan", [], "plan", plan)
            extended = extend(plan, plan_dir, count=32, cap=768, justification="Prespecified precision sensitivity")
            next_dir = publish(root/"extended", [plan_dir], "plan", extended)
            self.assertEqual(load_plan(next_dir), extended)
            self.assertEqual(extended["work_sha256"], plan["work_sha256"])
            checkpoint = frozen["checkpoints"][0]
            results = root/"results"
            checkpoint_dir = results/"checkpoints"/checkpoint["checkpoint_id"]
            request = {"plan_dir": str(plan_dir), "plan_sha256": plan["document_sha256"], "results": str(results),
                       "checkpoint_id": checkpoint["checkpoint_id"], "device": "cuda:0", "probes": [1],
                       "max_probes": 1, "retry_failed": False}
            record_runtime = measured_runtime(frozen["numerics"])

            @contextmanager
            def fixture_worker(*_):
                yield {"runtime": record_runtime, "layout": layout(), "floats": candidate(plan, checkpoint)["scales"],
                       "device": "cuda:0", "data": checkpoint["stage2"]["data"], **dict.fromkeys(
                           ("model", "loader", "scales", "entries", "guard", "prepare", "classes"))}

            with (
                patch("analysis.hessian_geometry_stage3.worker.components", fixture_worker),
                patch("analysis.hessian_geometry_stage3.worker.effective_backend", return_value=record_runtime["effective"]),
                patch("analysis.hessian_geometry_stage3.estimator.measure_probe", return_value=measurement()) as measure,
                patch.object(torch.cuda, "synchronize"),
                patch.object(torch.cuda, "reset_peak_memory_stats"),
                patch.object(torch.cuda, "max_memory_allocated", return_value=20),
                patch.object(torch.cuda, "max_memory_reserved", return_value=22),
            ):
                self.assertEqual(execute(request, checkpoint["reader"])["executed"], [1])
                first = probe_file(checkpoint_dir, 1)
                original = first.read_bytes()
                self.assertEqual(execute(request, checkpoint["reader"])["executed"], [])
                ext = {**request, "plan_dir": str(next_dir), "plan_sha256": extended["document_sha256"], "probes": [17]}
                measure.side_effect = RuntimeError("synthetic derivative failure")
                with self.assertRaisesRegex(RuntimeError, "synthetic derivative"):
                    execute(ext, checkpoint["reader"])
                with self.assertRaisesRegex(Stage3Error, "retry-failed"):
                    execute(ext, checkpoint["reader"])
                measure.side_effect = None
                self.assertEqual(execute({**ext, "retry_failed": True}, checkpoint["reader"])["executed"], [17])
            self.assertEqual(original, first.read_bytes())
            records, _, coverage = observe(extended, checkpoint, results)
            self.assertEqual(set(records), {1, 17})
            self.assertFalse(coverage["publication_complete"])
            self.assertEqual([a["status"] for a in coverage["slots"][16]["attempts"]].count("failed"), 1)
            with self.assertRaises(FileExistsError):
                install(first, document(plan, checkpoint, 1))
            with self.assertRaises(ValueError):
                disjoint(plan_dir/"nested", [plan_dir])
            with self.assertRaises(ValueError):
                namespace(plan, plan_dir, plan_dir)
            first_summary = collect(extended, next_dir, results)
            self.assertEqual(first_summary, collect(extended, next_dir, results))
            self.assertIsNone(first_summary["traces"][0]["mean"])
            self.assertEqual(first_summary["convergence"][0]["mean"], -.5)

    def test_comparison_equations_and_split_compatibility(self):
        checkpoint = row()
        common = {**checkpoint["key"], "checkpoint_id": checkpoint["checkpoint_id"], "checkpoint": BEST,
                  "scientific_fingerprint": "c"*64, "mean": -2., "mc_se": .2, "sample_sd": .8,
                  "eligible": 16, "expected": 16, "clean_loss": 1.9, "clean_accuracy": 3/7,
                  "runtime": measured_runtime(numerical_policy("cudnn"))}
        traces = [{**common, "target_id": target, "scope": scope, "target_name": name, "mean": value}
                  for target, scope, name, value in (("network:all", "network", "all", -2.),
                                                     ("stage:stem", "stage", "stem", -1.),
                                                     ("stage:head", "stage", "head", -1.))]
        grid = [0., .01, .02, .05, .1, .2]
        stage2 = {"split": "validation", "evaluation_alphas": grid, "realizations_per_positive_alpha": 8}
        raw = {("clean", 0., 0): {"data": checkpoint["stage2"]["data"], "metrics": checkpoint["stage2"]["metrics"],
                                 "runtime": checkpoint["stage2"]["runtime"]}}
        for trace in traces:
            for alpha, rep in product(grid[1:], range(1, 9)):
                raw[trace["target_id"], alpha, rep] = {"changes": {"delta_loss": 3*alpha**2, "accuracy_change_pp": -1.}}
        plan = {"core": {"split": "validation", "batch_size": 32}}
        measurements, fits, interactions, baseline, ranks = stage2_comparison(plan, checkpoint, stage2, raw, traces)
        first = measurements[0]
        expected = comparison_metrics(3*.01**2, -2., .01, .2)
        self.assertAlmostEqual(first["residual"], expected["residual"])
        self.assertAlmostEqual(first["prediction_mc_se"], .5*.01**2*.2)
        self.assertEqual(first["stage2_perturbation_se"], 0.)
        self.assertAlmostEqual(fits[0]["empirical_curvature_proxy_2c"], 6.)
        self.assertEqual(interactions[0]["local_algebraic_residual"], 0.)
        self.assertAlmostEqual(baseline["loss_difference"], -.1)
        self.assertEqual(ranks[0]["observed_rank"], 1)
        self.assertEqual(settings_relation(traces[0], checkpoint["stage2"]["runtime"], batch_size=32)["status"], "different")
        cross = stage2_comparison({"core": {"split": "test", "batch_size": 32}}, checkpoint, stage2, raw, traces)
        self.assertIsNone(cross[0][0]["residual"])
        stage1 = fit_pair(traces[0], {alpha: 2*alpha**2 for alpha in grid[1:]},
                          {"hessian_split": "validation", "empirical_split": "test", "status": "cross_split_exploratory"})
        self.assertAlmostEqual(stage1[0]["proxy_minus_trace"], 6.)
        self.assertIsNone(ratio(-1., 1e-13))
