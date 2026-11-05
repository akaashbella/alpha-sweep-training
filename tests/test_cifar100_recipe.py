"""Focused CPU checks. Prepared readers are mocked; no research data are created.

Native model/optimizer checks run only with the pinned existing dependencies.
Fixture seeds, normalization and precision below are test values, never defaults.
"""

import importlib.metadata
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from alpha_sweep_training.config import ConfigError, RunConfig, from_declaration
from alpha_sweep_training.data_contract import PREPROCESSING, content_identity
from alpha_sweep_training.identity import run_identity, scientific_fingerprint
from alpha_sweep_training.manifests import (
    ManifestError, build_manifest, evaluation_plan, manifest_from_dict,
    select_task, task_config, validate_phase_document,
)
from alpha_sweep_training.spec import (
    ARCHITECTURES, CIFAR100_MOBILE_EPS0316_VERSION, DATASETS,
    SPECIFICATION_VERSION, default_document, optimizer_recipe,
)


def ready_document(dataset, architecture):
    document = default_document(dataset, architecture)
    science = document["scientific"]
    science.update(training_seed=0, training_seeds=[0, 1, 2])
    science["data"]["split"]["identity"] = content_identity(["fixture-split", dataset])
    science["data"]["normalization"] = dict(
        mean=[.1, .2, .3], std=[.9, .8, .7],
        source="normalization-v1:" + content_identity(["fixture-normalization", dataset]))
    science["training"].update(drop_last=False, warmup_start_factor=.1,
                              cosine_min_lr=0., scheduler_interval="epoch")
    science["precision"].update(mode="float32", float32_matmul_precision="highest",
        allow_tf32=False, deterministic_algorithms=True,
        cudnn_deterministic=True, cudnn_benchmark=False)
    document["execution"]["storage"]["split_file"] = f"fixture/{dataset}/split.json"
    return document


def revised(config):
    return from_declaration(config.to_dict(), overrides={
        "scientific.specification_version": CIFAR100_MOBILE_EPS0316_VERSION,
        "scientific.optimizer.epsilon": .0316})


def fixture_preparation_identity(dataset, training, test, split, normalization):
    # A test-only preparation boundary, so declaration tests need no NumPy.
    # This does not verify raw CIFAR files, split contents or normalization.
    return content_identity(dict(format_version=1, dataset=dataset,
        official_training_identity=training, official_test_identity=test,
        split_identity=split, normalization_identity=normalization,
        preprocessing_identity=content_identity(PREPROCESSING)))


def fixture_inputs():
    bases = {dataset: RunConfig(ready_document(dataset, "resnet50")) for dataset in DATASETS}
    models = {architecture: {"scientific": {"architecture": architecture,
        "optimizer": optimizer_recipe(architecture)}} for architecture in ARCHITECTURES}
    bundles = {}
    for dataset, config in bases.items():
        split = config.scientific["data"]["split"]["identity"]
        normalization = content_identity(["fixture-normalization", dataset])
        training, test = (content_identity([kind, dataset]) for kind in ("train", "test"))
        bundles[dataset] = SimpleNamespace(manifest={
            "identity": fixture_preparation_identity(dataset, training, test, split, normalization),
            "split_identity": split, "normalization_identity": normalization,
            "preprocessing_identity": content_identity(PREPROCESSING)},
            sources=SimpleNamespace(training_record={"identity": training},
                                    test_record={"identity": test}))
    return bases, models, bundles


class RecipeDeclarations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bases, cls.models, bundles = fixture_inputs()
        def prepared(config, **kwargs):
            assert kwargs["metadata_only"] is True
            return bundles[config.scientific["dataset"]]
        boundary = SimpleNamespace(load_prepared_dataset=prepared,
                                   preparation_identity=fixture_preparation_identity)
        preparation = patch.dict(sys.modules, {"alpha_sweep_training.prepared": boundary})
        preparation.start()
        cls.addClassCleanup(preparation.stop)
        cls.original = build_manifest(cls.bases, cls.models, seeds=[0, 1, 2], base_dir=".")
        cls.updated = build_manifest(cls.bases, cls.models, seeds=[0, 1, 2], base_dir=".",
                                     cifar100_mobile_eps0316=True)

    def test_historical_golden_fingerprints_and_ids(self):
        # Frozen hashes from pre-change main's schema-4 spec and identity function,
        # using ready_document's explicit fixture decisions (not research inputs).
        expected = HISTORICAL_FINGERPRINTS
        for task in self.original.tasks:
            science = task.config.scientific
            if science["alpha"] != 0 or science["training_seed"] != 0:
                continue
            key = (science["dataset"], science["architecture"])
            self.assertEqual(task.fingerprint, expected[key])
            self.assertEqual(task.run_id, f"{key[0]}__{key[1]}__alpha-0__seed-0")
            self.assertEqual(science["specification_version"], "main-v1")
            self.assertEqual(RunConfig(task.config.to_dict()).to_dict(), task.config.to_dict())

    def test_exactly_thirty_scientific_changes_and_distinct_storage(self):
        from alpha_sweep_training.artifacts import resolve_run_paths
        changed = []
        for old, new in zip(self.original.tasks, self.updated.tasks):
            science = old.config.scientific
            target = (science["dataset"], science["architecture"]) == ("cifar100", "mobilenetv3_large")
            if not target:
                self.assertEqual(old.config.to_dict(), new.config.to_dict())
                self.assertEqual((old.run_id, old.fingerprint), (new.run_id, new.fingerprint))
                continue
            changed.append(new.global_index)
            expected = old.config.to_dict()
            expected["scientific"]["specification_version"] = CIFAR100_MOBILE_EPS0316_VERSION
            expected["scientific"]["optimizer"]["epsilon"] = .0316
            self.assertEqual(new.config.to_dict(), expected)
            self.assertNotEqual(old.fingerprint, new.fingerprint)
            self.assertEqual(new.run_id, old.run_id + "__recipe-mobile-eps0316-v1")
            old_paths, new_paths = (resolve_run_paths(t.config, Path.cwd()) for t in (old, new))
            for a, b in zip(old_paths.directories(), new_paths.directories()):
                self.assertNotEqual(a, b)
        self.assertEqual(changed, list(range(150, 180)))
        self.assertNotEqual(self.original.grid_id, self.updated.grid_id)
        self.assertEqual([b.to_dict() for b in self.bases.values()],
                         [ready_document(d, "resnet50") for d in DATASETS])

    def test_scope_and_version_epsilon_mismatch_rejected(self):
        for dataset in DATASETS:
            for architecture in ARCHITECTURES:
                if (dataset, architecture) == ("cifar100", "mobilenetv3_large"):
                    continue
                with self.subTest(dataset=dataset, architecture=architecture), self.assertRaises(ConfigError):
                    revised(RunConfig(ready_document(dataset, architecture)))
        baseline = RunConfig(ready_document("cifar100", "mobilenetv3_large"))
        for overrides in (
            {"scientific.optimizer.epsilon": .0316},
            {"scientific.specification_version": CIFAR100_MOBILE_EPS0316_VERSION},
            {"scientific.specification_version": "unknown"},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ConfigError):
                from_declaration(baseline.to_dict(), overrides=overrides)
        candidate = revised(baseline)
        for key, value in (("epsilon", .001), ("learning_rate", .02), ("momentum", .8),
                           ("rmsprop_alpha", .8), ("weight_decay", .0001)):
            with self.subTest(key=key), self.assertRaises(ConfigError):
                from_declaration(candidate.to_dict(), overrides={f"scientific.optimizer.{key}": value})

    def test_partial_versioned_declaration_and_execution_overrides(self):
        config = from_declaration({"schema_version": 4, "scientific": {
            "dataset": "cifar100", "architecture": "mobilenetv3_large",
            "specification_version": CIFAR100_MOBILE_EPS0316_VERSION}})
        self.assertEqual(config.scientific["optimizer"]["epsilon"], .0316)
        self.assertIn("scientific.data.split.identity", config.unresolved_settings())
        task = self.updated.tasks[150]
        moved = task_config(task, {"execution.storage.output_root": "fixture-relocated"})
        self.assertEqual(scientific_fingerprint(moved), task.fingerprint)
        self.assertEqual(run_identity(moved), task.run_id)

    def test_full_coverage_linkage_and_baseline_mapping(self):
        loaded = manifest_from_dict(self.updated.to_dict())
        self.assertEqual(len(loaded.tasks), 240)
        for phase in (1, 2):
            tasks = validate_phase_document(loaded.phase_view(phase), loaded)
            self.assertEqual(len(tasks), 120)
            self.assertEqual([t.phase_index for t in tasks], list(range(120)))
            self.assertEqual(len({t.run_id for t in tasks}), 120)
        self.assertEqual([t.global_index for t in loaded.tasks], list(range(240)))
        baseline = [t for t in loaded.tasks if t.phase == 2 and t.config.training_alpha == 0]
        self.assertEqual([(t.global_index, t.phase_index) for t in baseline],
            [(120 + offset + seed, offset + seed) for offset in (0, 30, 60, 90) for seed in range(3)])
        for task in baseline:
            self.assertEqual(select_task(loaded, phase=2, phase_index=task.phase_index).run_id, task.run_id)
            self.assertEqual(task.config.scientific["training"]["epochs"], 200)
            self.assertEqual(task.config.scientific["training"]["batch_size"], 128)
        with self.assertRaises(ManifestError):
            validate_phase_document(self.original.phase_view(2), loaded)

    def test_robustness_counts_unchanged(self):
        plan = evaluation_plan(self.updated)
        phase = [row for row in plan["tasks"] if row["training_phase"] == 2]
        self.assertEqual(len(phase), 120)
        self.assertEqual(sum(len(row["slots"]) for row in phase), 3360)
        self.assertEqual(plan["logical_evaluation_slots"], 6720)
        for row in phase:
            self.assertEqual(len(row["slots"]), 28)

    def test_builder_flag_is_strict_and_cli_passes_it(self):
        from contextlib import redirect_stdout
        import io
        from alpha_sweep_training import controller
        with self.assertRaises(ManifestError):
            build_manifest(self.bases, self.models, seeds=[0, 1, 2], base_dir=".", cifar100_mobile_eps0316=1)
        with patch.object(controller, "_build", return_value={}) as build, redirect_stdout(io.StringIO()):
            self.assertEqual(controller.main(["build", "--cifar10-config", "fixture10.yaml",
                "--cifar100-config", "fixture100.yaml", "--seeds", "0", "1", "2", "--output", "fixture.json",
                "--cifar100-mobile-eps0316"]), 0)
        self.assertTrue(build.call_args.args[0].cifar100_mobile_eps0316)

    def test_gate_requires_every_original_cifar10_authority(self):
        from alpha_sweep_training.coverage import TaskObservation, collect_coverage
        def observation(task, **kwargs):
            science = task.config.scientific
            complete = task.phase == 1 and task.global_index != 30
            return TaskObservation(task.run_id, task.fingerprint, task.global_index, task.phase,
                task.phase_index, science["dataset"], science["architecture"], science["alpha"],
                str(science["alpha"]), science["training_seed"], condition="ok",
                training_complete=complete, stored_status="training_completed" if complete else "training")
        with patch("alpha_sweep_training.coverage.inspect_task", side_effect=observation):
            coverage = collect_coverage(self.updated, base_dir=".")
        self.assertEqual(coverage["phases"][0]["training_completed"], 119)
        self.assertFalse(coverage["phases"][1]["training_eligible"])
        # Missing alpha-zero MobileNet seed 0 still names the ORIGINAL CIFAR-10 run.
        self.assertEqual(self.updated.tasks[30].run_id, self.original.tasks[30].run_id)
        self.assertNotIn("eps0316", self.updated.tasks[30].run_id)

    def test_manual_cifar100_execution_rejects_before_construction_or_writes(self):
        from alpha_sweep_training.coverage import TaskObservation
        from alpha_sweep_training.task_execution import execute_task, ExitCode
        task = self.updated.tasks[150]
        missing = TaskObservation(task.run_id, task.fingerprint, task.global_index, task.phase,
            task.phase_index, "cifar100", "mobilenetv3_large", 0., "0", 0, condition="missing")
        with patch("alpha_sweep_training.execution_provenance.collect_provenance", return_value={}), \
             patch("alpha_sweep_training.task_execution.inspect_task", return_value=missing), \
             patch("alpha_sweep_training.task_execution.collect_coverage", return_value={
                 "phases": [{"training_successfully_completed": False}]}), \
             patch("alpha_sweep_training.task_execution.open_run") as writer:
            outcome = execute_task(self.updated, task, base_dir=".", source_root=".")
        self.assertEqual(outcome.exit_code, ExitCode.REJECTED)
        self.assertIn("All 120 CIFAR-10 training completions", outcome.diagnostic)
        writer.assert_not_called()


def native_blockers():
    blockers = []
    for name, expected in (("torch", "2.5.1"), ("torchvision", "0.20.1"), ("numpy", "1.26.4")):
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = "missing"
        if actual.split("+")[0] != expected:
            blockers.append(f"{name}={actual} (requires {expected})")
    return "; ".join(blockers)


NATIVE_BLOCKERS = native_blockers()


@unittest.skipIf(NATIVE_BLOCKERS, "Native checks pending: " + NATIVE_BLOCKERS)
class NativeCifar100Recipe(unittest.TestCase):
    def test_real_models_heads_forwards_and_native_optimizer_options(self):
        import torch
        from alpha_sweep_training.models import build_model
        from alpha_sweep_training.optimizers import build_optimizer, describe_optimizer, validate_optimizer
        from alpha_sweep_training.component_contracts import require_runtime_versions
        import torchvision
        require_runtime_versions(torch.__version__, torchvision.__version__)
        for architecture in ARCHITECTURES:
            for version in ((SPECIFICATION_VERSION, CIFAR100_MOBILE_EPS0316_VERSION)
                            if architecture == "mobilenetv3_large" else (SPECIFICATION_VERSION,)):
                with self.subTest(architecture=architecture, version=version):
                    config = RunConfig(ready_document("cifar100", architecture))
                    if version == CIFAR100_MOBILE_EPS0316_VERSION:
                        config = revised(config)
                    model = build_model(config).to("cpu")
                    optimizer = build_optimizer(model, config)
                    metadata = describe_optimizer(optimizer).to_dict()
                    self.assertEqual(metadata["declared_recipe"], config.scientific["optimizer"])
                    group = optimizer.param_groups[0]
                    self.assertEqual(len(optimizer.param_groups), 1)
                    self.assertEqual(group["lr"], config.scientific["optimizer"]["learning_rate"])
                    self.assertEqual(group["weight_decay"], config.scientific["optimizer"]["weight_decay"])
                    if architecture in ("mobilenetv3_large", "efficientnet_b0"):
                        self.assertIs(type(optimizer), torch.optim.RMSprop)
                        self.assertEqual(group["eps"], .0316 if version == CIFAR100_MOBILE_EPS0316_VERSION else .001)
                        self.assertEqual((group["momentum"], group["alpha"]), (.9, .9))
                    elif architecture == "resnet50":
                        self.assertIs(type(optimizer), torch.optim.SGD)
                        self.assertEqual(group["momentum"], .9)
                    else:
                        self.assertIs(type(optimizer), torch.optim.AdamW)
                        self.assertEqual(group["betas"], (.9, .999))
                    model.eval()
                    with torch.inference_mode():
                        self.assertEqual(tuple(model(torch.zeros(2, 3, 32, 32)).shape), (2, 100))
                    if version == CIFAR100_MOBILE_EPS0316_VERSION:
                        with self.assertRaises(ValueError):
                            validate_optimizer(optimizer, RunConfig(ready_document("cifar100", architecture)))


HISTORICAL_FINGERPRINTS = {
    ('cifar10', 'resnet50'): 'e5f6eb8eaa99e558eb865a3867b7c0c3071d7f7d6ad5a2a325968c1ef2559665',
    ('cifar10', 'mobilenetv3_large'): '9a62babed9cabf667b37474a35727542d69fb5cb2fd85cde7bc773d2d2ee2f8b',
    ('cifar10', 'efficientnet_b0'): '67e009af43c060027bfd4036c11dc3342424f0ce41cfff0c404bceb95f960919',
    ('cifar10', 'convnext_tiny'): 'f80dd2b21bc415049f4a0bc2908854636d26a863435bda4abc31803db3b0a5be',
    ('cifar100', 'resnet50'): 'd853142473adae507c0b7bb9baad5a11cd60ba001b4329c73cec4daa59b36f28',
    ('cifar100', 'mobilenetv3_large'): '54b1fee7049d24db526715380806483edfddc55140a7772e90fcc1a66bea33d3',
    ('cifar100', 'efficientnet_b0'): '9bcf82213a39c941b23616f783f47dca6086670592b5ae50275546fb95512c78',
    ('cifar100', 'convnext_tiny'): '6dcf49b1feb76e4d8c23acb10b78ca52b88f0ab3435e6f6d78ee561cacecaa97',
}


if __name__ == "__main__":
    unittest.main()
