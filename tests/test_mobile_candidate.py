"""Focused opt-in CPU compatibility checks; no dataset/model/GPU execution.

fixtures/main_v1_spec.py is the exact specification from audited commit 71922f3,
retained as a historical wire/fingerprint oracle, never a production import.
"""

from copy import deepcopy
from contextlib import ExitStack, nullcontext, redirect_stderr, redirect_stdout
from dataclasses import replace
import hashlib
import importlib.util
import io
import json
import os
import sys
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

fixture = importlib.util.spec_from_file_location("frozen_main_v1_spec", Path(__file__).parent / "fixtures" / "main_v1_spec.py")
historical = importlib.util.module_from_spec(fixture)
fixture.loader.exec_module(historical)
from alpha_sweep_training.config import ConfigError


def resolved(document):
    document = deepcopy(document)
    s = document["scientific"]
    s.update(training_seed=2, training_seeds=[0, 1, 2])
    s["data"]["split"]["identity"] = "sha256:" + "a" * 64
    s["data"]["normalization"].update(mean=[.5] * 3, std=[.25] * 3,
        source="normalization-v1:sha256:" + "b" * 64)
    s["training"].update(drop_last=False, warmup_start_factor=.01, cosine_min_lr=0., scheduler_interval="epoch")
    s["precision"].update(mode="amp_bfloat16", float32_matmul_precision="highest", allow_tf32=False,
        deterministic_algorithms=True, cudnn_deterministic=True, cudnn_benchmark=False)
    document["execution"]["storage"]["split_file"] = "/unit/prepared/split.json"
    return document


def candidate():
    from alpha_sweep_training.config import RunConfig
    from alpha_sweep_training.spec import default_document, MOBILE_EPSILON_PROFILE
    return RunConfig(resolved(default_document("cifar10", "mobilenetv3_large", MOBILE_EPSILON_PROFILE)))


def study():
    from alpha_sweep_training.config import RunConfig
    from alpha_sweep_training.spec import ALPHAS
    from alpha_sweep_training.data_contract import content_identity, PREPROCESSING
    from alpha_sweep_training.prepared import preparation_identity
    from alpha_sweep_training.manifests import Task, Manifest, MOBILE_GRID_VERSION, _grid_id, manifest_from_dict
    doc = candidate().to_dict()
    prepared = {"split_identity": doc["scientific"]["data"]["split"]["identity"],
        "normalization_identity": "sha256:" + "b" * 64, "preprocessing_identity": content_identity(PREPROCESSING),
        "official_training_identity": "sha256:" + "c" * 64, "official_test_identity": "sha256:" + "d" * 64}
    prepared["preparation_identity"] = preparation_identity("cifar10", prepared["official_training_identity"],
        prepared["official_test_identity"], prepared["split_identity"], prepared["normalization_identity"])
    tasks = []
    for alpha in ALPHAS:
        for seed in (0, 1, 2):
            document = deepcopy(doc)
            document["scientific"].update(alpha=alpha, training_seed=seed)
            index = len(tasks)
            tasks.append(Task("", 1, index, index, RunConfig(document), prepared))
    grid = _grid_id(tasks, MOBILE_GRID_VERSION)
    tasks = tuple(Task(grid, 1, t.global_index, t.phase_index, t.config, t.prepared_data) for t in tasks)
    return manifest_from_dict(Manifest(tasks, (0, 1, 2), grid, MOBILE_GRID_VERSION).to_dict())


@unittest.skipUnless(os.environ.get("ALPHA_SWEEP_DIAGNOSTIC_TESTS") == "1", "Explicit pinned CPU checks required")
class CandidateContracts(unittest.TestCase):
    def test_historical_documents_names_fingerprints_and_native_recipes(self):
        from alpha_sweep_training.spec import default_document, optimizer_recipe
        from alpha_sweep_training.config import RunConfig, from_declaration
        from alpha_sweep_training.identity import scientific_fingerprint, run_identity
        from alpha_sweep_training.optimizers import build_optimizer
        import torch
        for dataset in historical.DATASETS:
            for architecture in historical.ARCHITECTURES:
                with self.subTest(dataset=dataset, architecture=architecture):
                    old = resolved(historical.default_document(dataset, architecture))
                    current = resolved(default_document(dataset, architecture))
                    self.assertEqual(current, old)
                    config = from_declaration(current)
                    self.assertEqual(config.to_dict(), RunConfig(old).to_dict())
                    payload = {"schema_version": 4, "fingerprint_version": 4, "scientific": old["scientific"]}
                    expected_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                        ensure_ascii=False, allow_nan=False).encode()).hexdigest()
                    self.assertEqual(scientific_fingerprint(config), expected_hash)
                    self.assertEqual(run_identity(config), f"{dataset}__{architecture}__alpha-0__seed-2")
                    self.assertEqual(optimizer_recipe(architecture), historical.optimizer_recipe(architecture))
                    info = SimpleNamespace(architecture=architecture, dataset=dataset,
                                           num_classes=historical.DATASETS[dataset], training_seed=2)
                    model = torch.nn.Linear(2, 2)
                    with patch("alpha_sweep_training.optimizers.model_construction", return_value=info):
                        optimizer = build_optimizer(model, config)
                    recipe = historical.optimizer_recipe(architecture)
                    group = optimizer.param_groups[0]
                    self.assertEqual(group["lr"], recipe["learning_rate"])
                    if recipe["name"] == "rmsprop":
                        self.assertEqual(group["eps"], .001)

    def test_native_candidate_epsilon_single_group_and_cross_profile_rejection(self):
        import torch
        from alpha_sweep_training.config import RunConfig
        from alpha_sweep_training.optimizers import build_optimizer, validate_optimizer, OptimizerConstructionError
        from alpha_sweep_training.scheduler import WarmupCosineScheduler
        config = candidate()
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))
        info = SimpleNamespace(architecture="mobilenetv3_large", dataset="cifar10", num_classes=10, training_seed=2)
        with patch("alpha_sweep_training.optimizers.model_construction", return_value=info):
            optimizer = build_optimizer(model, config)
            group = optimizer.param_groups[0]
            self.assertEqual((group["eps"], group["lr"], group["momentum"], group["alpha"], group["weight_decay"]),
                             (.0316, .01, .9, .9, .00001))
            self.assertEqual(len(optimizer.param_groups), 1)
            self.assertEqual({id(p) for p in group["params"]}, {id(p) for p in model.parameters()})
            self.assertFalse(group["centered"])
            self.assertIsNone(group["foreach"])
            original = RunConfig(resolved(historical.default_document("cifar10", "mobilenetv3_large")))
            with self.assertRaises(OptimizerConstructionError):
                validate_optimizer(optimizer, original)
            schedule = WarmupCosineScheduler(optimizer, config, 352)
            # Epoch-cadence state restoration keeps all 200 units; no training loop.
            for _ in range(7):
                schedule.step()
            native, saved = deepcopy(optimizer.state_dict()), deepcopy(schedule.state_dict())
            other = build_optimizer(model, config)
            restored = WarmupCosineScheduler(other, config, 352, initialize=False)
            other.load_state_dict(native)
            restored.load_state_dict(saved)
            self.assertEqual((restored.completed_units, restored.total_units, restored.warmup_units), (7, 200, 5))
            self.assertEqual(restored.state_dict(), schedule.state_dict())

    def test_reject_unsupported_science_and_differing_candidate_options(self):
        from alpha_sweep_training.config import RunConfig, ConfigError, from_declaration
        for path, value in (("specification_version", "unknown"), ("dataset", "cifar100"),
                            ("architecture", "efficientnet_b0"), ("training_seeds", [1, 2, 3])):
            document = candidate().to_dict()
            document["scientific"][path] = value
            with self.subTest(path=path), self.assertRaises(ConfigError):
                RunConfig(document)
        for section, field, value in (("optimizer", "epsilon", .001), ("training", "epochs", 20),
                                       ("precision", "mode", "float32"), ("training", "scheduler_interval", "optimizer_step")):
            document = candidate().to_dict()
            document["scientific"][section][field] = value
            with self.subTest(field=field), self.assertRaises(ConfigError):
                RunConfig(document)
        original = resolved(historical.default_document("cifar10", "mobilenetv3_large"))
        with self.assertRaises(ConfigError):
            from_declaration(original, overrides={"scientific.specification_version": "cifar10-mobile-eps0316-v1"})

    def test_full_candidate_cartesian_grid_pilot_subset_and_strict_legacy_size(self):
        from alpha_sweep_training.manifests import (manifest_from_dict, select_task, ManifestError,
                                                  GRID_VERSION, _sealed)
        from alpha_sweep_training.mobile_candidate import pilot_tasks
        manifest = study()
        self.assertEqual(len(manifest.tasks), 30)
        tasks = pilot_tasks(manifest)
        self.assertEqual([(t.config.training_alpha, t.config.scientific["training_seed"]) for t in tasks],
                         [(0., 0), (0., 1), (0., 2)])
        for bad in (30, -1, True):
            with self.assertRaises(ManifestError):
                select_task(manifest, global_index=bad)
        with self.assertRaises(ManifestError):
            select_task(manifest, phase=2, phase_index=0)
        document = manifest.to_dict()
        document.pop("document_sha256")
        document["grid_version"] = GRID_VERSION
        with self.assertRaises(ManifestError):
            manifest_from_dict(_sealed(document))
        document = manifest.to_dict()
        document.pop("document_sha256")
        document["tasks"] = document["tasks"][:3]
        with self.assertRaises(ManifestError):
            manifest_from_dict(_sealed(document))

    def test_namespace_and_native_writer_reject_foreign_resume_and_overwrite(self):
        from alpha_sweep_training.artifacts import resolve_run_paths, _check_existing, RunDirectoryError
        from alpha_sweep_training.serialization import atomic_write_json
        from alpha_sweep_training.identity import identity_metadata
        from alpha_sweep_training.config import RunConfig
        from alpha_sweep_training.mobile_candidate import _claim
        config = candidate()
        original = RunConfig(resolved(historical.default_document("cifar10", "mobilenetv3_large")))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            new_paths = resolve_run_paths(config, root)
            old_paths = resolve_run_paths(original, root)
            self.assertTrue(set(new_paths.directories()).isdisjoint(old_paths.directories()))
            new_paths.metadata.mkdir(parents=True)
            atomic_write_json(new_paths.metadata / "run_metadata.json", identity_metadata(original))
            atomic_write_json(new_paths.metadata / "resolved_config.yaml", original.to_dict())
            # Both operations must preserve a copied main-v1 owner as evidence.
            for operation in ("continue", "overwrite"):
                with self.assertRaises(RunDirectoryError):
                    _check_existing(new_paths, config, operation, {})
            with self.assertRaises(ConfigError):
                _claim(root / "original" / "candidate", [root / "original"], kind="unit")

    def test_receipts_keep_best_final_latest_and_profile_authority_distinct(self):
        from alpha_sweep_training.training_contracts import CommitReceipt, CheckpointReference, PersistenceContractError
        from alpha_sweep_training.metrics import BestSelection, Counters
        from alpha_sweep_training.identity import identity_metadata
        from alpha_sweep_training.config import RunConfig
        config = candidate()
        best = BestSelection(18, 4145)
        latest = CheckpointReference("latest", "generations/latest.pt", "a" * 64, 200)
        final = CheckpointReference("final", "generations/final.pt", "b" * 64, 200)
        selected = CheckpointReference("best", "generations/best.pt", "c" * 64, 19)
        receipt = CommitReceipt(identity_metadata(config), "unit", 200, latest, selected, final)
        receipt.validate(config, Counters(completed_epochs=200), best)
        with self.assertRaises(PersistenceContractError):
            CommitReceipt(identity_metadata(config), "unit", 200, latest, selected, None).validate(config, Counters(completed_epochs=200), best)
        original = RunConfig(resolved(historical.default_document("cifar10", "mobilenetv3_large")))
        with self.assertRaises(PersistenceContractError):
            receipt.validate(original, Counters(completed_epochs=200), best)

    def test_actual_audited_baseline_is_required_before_candidate_generation(self):
        from alpha_sweep_training.config import RunConfig, ConfigError, from_declaration, _read_yaml
        from alpha_sweep_training.identity import scientific_fingerprint
        from alpha_sweep_training.mobile_candidate import candidate_config
        original = RunConfig(resolved(historical.default_document("cifar10", "mobilenetv3_large")))
        with self.assertRaises(ConfigError):
            candidate_config(original, "/new/candidate")
        with patch("alpha_sweep_training.mobile_candidate.AUDITED_BASELINE_FINGERPRINT", scientific_fingerprint(original)):
            actual = candidate_config(original, "/new/candidate", seed=1)
        declaration = _read_yaml(Path(__file__).resolve().parents[1] / "configs" / "candidates" / "mobile-eps0316.yaml")
        declared = from_declaration(original.to_dict(), declaration, overrides={"scientific.training_seed": 1})
        self.assertEqual(declared.scientific, actual.scientific)
        from alpha_sweep_training.randomness import runtime_seeds
        original_seed1 = from_declaration(original.to_dict(), overrides={"scientific.training_seed": 1})
        self.assertEqual(runtime_seeds(original_seed1), runtime_seeds(actual))
        before, after = original.scientific, actual.scientific
        for key in ("specification_version", "optimizer", "training_seed"):
            before.pop(key)
            after.pop(key)
        self.assertEqual(before, after)

    def test_journal_selection_retains_earlier_epoch_on_exact_tie(self):
        from alpha_sweep_training.checkpoint_records import validate_epoch
        from alpha_sweep_training.metrics import EpochRecord, Counters, BestSelection, ClassificationMetrics
        from alpha_sweep_training.scheduler import _Schedule
        from alpha_sweep_training.training_contracts import PersistenceContractError
        config = candidate()
        descriptor = _Schedule("mobilenetv3_large", "epoch", 200, 5, None, (.01,), .01, 0.)
        previous = Counters(completed_epochs=1, attempted_batches=352, processed_examples=45000, successful_updates=352)
        totals = Counters(completed_epochs=2, attempted_batches=704, processed_examples=90000, successful_updates=704)
        earlier = BestSelection(0, 4000)
        record = EpochRecord(epoch_index=1, completed_epochs=2, training_kind="clean_training",
            training=ClassificationMetrics(.5, 35000, 45000, 352), validation=ClassificationMetrics(.7, 4000, 5000, 40),
            attempted_batches=352, successful_updates=352, skipped_updates=0, totals=totals,
            lr_first_attempt=descriptor.rates_at(1)[0], lr_last_attempt=descriptor.rates_at(1)[0],
            lr_at_boundary=descriptor.rates_at(2)[0], scheduler_position_before=1, scheduler_position_after=2,
            scheduler_interval="epoch", duration_seconds=1., selected_best=False, best=earlier, amp_skips=(), status="training")
        components = {"data": {"counts": {"training_batches": 352, "training_processed_per_epoch": 45000,
                        "validation_batches": 40}}, "optimizer": {"parameter_names": []},
                      "scheduler": descriptor.to_dict(), "model": {"eligible_weights": []}}
        validate_epoch(record, (), previous=previous, previous_best=earlier, config=config, components=components)
        with self.assertRaisesRegex(PersistenceContractError, "strict best decision"):
            validate_epoch(replace(record, selected_best=True, best=BestSelection(1, 4000)), (),
                           previous=previous, previous_best=earlier, config=config, components=components)

    def test_report_retains_all_seed_trajectories_and_separates_health_completion(self):
        from alpha_sweep_training.mobile_candidate import validation_report
        from alpha_sweep_training.metrics import LayerStatistic
        from alpha_sweep_training.training_contracts import CommitReceipt, CheckpointReference
        from alpha_sweep_training.identity import identity_metadata
        manifest = study()
        ref = {"root": "metadata", "relative_path": "state/transactions/unit.json", "sha256": "a" * 64}
        receipt = CommitReceipt(identity_metadata(manifest.tasks[0].config), "unit", 2,
            CheckpointReference("latest", "generations/latest.pt", "b" * 64, 2),
            CheckpointReference("best", "generations/best.pt", "c" * 64, 1), None)
        records = tuple((None, SimpleNamespace(epoch_index=i, to_dict=lambda i=i: {"unit_epoch": i,
                        "training": {"loss": .5}, "validation": {"loss": .7}}),
                        (LayerStatistic("head.weight", "Linear", (2, 2), 4, .1),)) for i in (0, 1))
        current = SimpleNamespace(reference=SimpleNamespace(to_dict=lambda: ref), receipt=receipt,
            records=records, document={"records": [ref, ref], "artifacts": {"failure": None}, "best": {"epoch_index": 0}})
        observation = SimpleNamespace(condition="ok", transaction=ref, training_complete=False,
            to_dict=lambda: {"evidence_failures": [{"preserved": "unit failure evidence"}]})
        with patch("alpha_sweep_training.coverage.inspect_task", return_value=observation), \
             patch("alpha_sweep_training.coverage._stamp", return_value=("unchanged",)), \
             patch("alpha_sweep_training.coverage._lock_paths", return_value=()), \
             patch("alpha_sweep_training.mobile_candidate.read_json", return_value={"provenance": {}}), \
             patch("alpha_sweep_training.checkpoint_journal.read_committed", return_value=current) as reader:
            report = validation_report({"base_dir": "/unit"}, manifest)
        self.assertEqual([r["training_seed"] for r in report["seeds"]], [0, 1, 2])
        self.assertFalse(report["pilot_training_complete"])
        self.assertIsNone(report["study_training_complete"])
        for row in report["seeds"]:
            self.assertEqual(len(row["trajectory"]), 2)
            self.assertEqual(row["selected_epoch"], 1)
            self.assertIsNone(row["final"])
            self.assertEqual(row["health"]["assessment"], "unreviewed")
            self.assertTrue(row["failures"])
        self.assertTrue(all("metadata_only" not in call.kwargs for call in reader.call_args_list))


@unittest.skipUnless(os.environ.get("ALPHA_SWEEP_DIAGNOSTIC_TESTS") == "1", "Explicit pinned CPU checks required")
class PilotWiring(unittest.TestCase):
    """Real declaration/manifest/file wiring; prepared data, provenance and compute are doubles."""

    def setUp(self):
        from alpha_sweep_training.config import RunConfig
        from alpha_sweep_training.identity import scientific_fingerprint
        from alpha_sweep_training.serialization import save_resolved_config
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        directory = self.stack.enter_context(tempfile.TemporaryDirectory(prefix="mobile-pilot-wiring-"))
        self.root = Path(directory).resolve()
        self.original = self.root / "original"
        self.original.mkdir()
        self.source = self.root / "source"
        self.source.mkdir()
        self.candidate_root = self.root / "candidate"
        self.saved = self.original / "resolved_config.yaml"
        original = RunConfig(resolved(historical.default_document("cifar10", "mobilenetv3_large")))
        save_resolved_config(self.saved, original)
        self.original_bytes = self.saved.read_bytes()
        # The fixture has synthetic identities. Only this scientific evidence
        # gate is replaced; production still requires the real audited hash.
        self.stack.enter_context(patch("alpha_sweep_training.mobile_candidate.AUDITED_BASELINE_FINGERPRINT",
                                      scientific_fingerprint(original)))
        prepared = study().tasks[0].prepared_data
        bundle = SimpleNamespace(manifest={**prepared, "identity": prepared["preparation_identity"]},
            sources=SimpleNamespace(training_record={"identity": prepared["official_training_identity"]},
                                    test_record={"identity": prepared["official_test_identity"]}))
        self.loader = self.stack.enter_context(patch("alpha_sweep_training.prepared.load_prepared_dataset", return_value=bundle))
        self.pin = {"code": {"source_sha256": "e" * 64}, "software": {"python": "3.11.15",
            "torch": "2.5.1+cu124", "torchvision": "0.20.1+cu124", "numpy": "1.26.4",
            "PyYAML": "6.0.3", "cuda": "12.4", "cudnn": "90100"}}
        self.provenance = self.stack.enter_context(patch("alpha_sweep_training.execution_provenance.collect_provenance",
                                                       return_value=self.pin))
        self.inputs = {"activation_script": str(self.original / "bin" / "activate"), "unit_launch_pin": True}
        self.stack.enter_context(patch("alpha_sweep_training.mobile_candidate._launch_inputs", return_value=self.inputs))
        self.build_argv = ["build", "--saved-config", str(self.saved), "--base-dir", str(self.original),
            "--source-root", str(self.source), "--candidate-root", str(self.candidate_root),
            "--activation-script", self.inputs["activation_script"], "--protected-root", str(self.original / "workflow")]

    def build(self):
        from alpha_sweep_training.mobile_candidate import main, load_pilot
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(self.build_argv), 0)
        self.plan_path = self.candidate_root / "pilot.json"
        plan, manifest = load_pilot(self.plan_path)
        self.assertEqual(json.loads(output.getvalue()), plan)
        return plan, manifest

    def test_build_load_roundtrip_keeps_thirty_tasks_three_configs_and_original_bytes(self):
        from alpha_sweep_training.mobile_candidate import main
        from alpha_sweep_training.serialization import read_json
        plan, manifest = self.build()
        self.assertEqual(len(manifest.tasks), 30)
        self.assertEqual(plan["pilot_run_ids"], [t.run_id for t in manifest.tasks[:3]])
        for seed in (0, 1, 2):
            self.assertEqual(read_json(self.candidate_root / f"seed-{seed}.config.json"), manifest.tasks[seed].config.to_dict())
        self.assertTrue((self.candidate_root / "allocation-logs").is_dir())
        self.assertTrue(self.loader.call_args.kwargs["metadata_only"])
        before = {p.relative_to(self.candidate_root): p.read_bytes()
                  for p in self.candidate_root.rglob("*") if p.is_file()}
        with redirect_stderr(io.StringIO()):
            self.assertEqual(main(self.build_argv), 22)  # No replacement of an owned plan.
        after = {p.relative_to(self.candidate_root): p.read_bytes()
                 for p in self.candidate_root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(self.saved.read_bytes(), self.original_bytes)

    def test_load_rejects_bad_seals_wrong_pilot_membership_and_redirected_storage(self):
        from alpha_sweep_training.mobile_candidate import load_pilot
        from alpha_sweep_training.manifests import _sealed
        from alpha_sweep_training.serialization import atomic_write_json
        plan, manifest = self.build()
        corrupted = deepcopy(plan)
        corrupted["pilot_indices"] = [0, 1, 3]
        atomic_write_json(self.plan_path, corrupted)
        with self.assertRaises(ConfigError):
            load_pilot(self.plan_path)
        changed = {k: v for k, v in plan.items() if k != "document_sha256"}
        changed["pilot_run_ids"] = [t.run_id for t in manifest.tasks[3:6]]
        atomic_write_json(self.plan_path, _sealed(changed))
        with self.assertRaises(ConfigError):
            load_pilot(self.plan_path)
        document = manifest.to_dict()
        document.pop("document_sha256")
        document["tasks"][29]["config"]["execution"]["storage"]["output_root"] = str(self.original)
        document = _sealed(document)
        atomic_write_json(self.candidate_root / "study.json", document)
        changed = {k: v for k, v in plan.items() if k != "document_sha256"}
        changed["manifest_document_sha256"] = document["document_sha256"]
        atomic_write_json(self.plan_path, _sealed(changed))
        with self.assertRaisesRegex(ConfigError, "owned run root"):
            load_pilot(self.plan_path)
        self.assertEqual(self.saved.read_bytes(), self.original_bytes)

    def test_gpu_cli_dispatches_each_pilot_seed_with_stop_using_only_mocked_cuda(self):
        from alpha_sweep_training.mobile_candidate import main, PILOT_OPERATIONS
        from alpha_sweep_training.interruption import StopRequest
        _, manifest = self.build()
        backend = ModuleType("torch")
        backend.set_num_threads = Mock()
        backend.set_num_interop_threads = Mock()
        backend.cuda = SimpleNamespace(device_count=Mock(return_value=1), get_device_properties=Mock(return_value=
            SimpleNamespace(name="NVIDIA H100 80GB HBM3", major=9, minor=0, total_memory=85009367040)))
        env = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "SLURM_NTASKS": "1", "SLURM_CPUS_PER_TASK": "8",
               "SLURM_JOB_PARTITION": "dgxh", "SLURM_JOB_NUM_NODES": "1", "SLURM_JOB_ID": "123",
               "CUDA_VISIBLE_DEVICES": "unit-visible-gpu"}
        result = SimpleNamespace(to_dict=lambda: {"outcome": "unit_interrupted", "exit_code": 10}, exit_code=10)
        with patch("alpha_sweep_training.task_execution.execute_task", return_value=result) as execute, \
             patch("alpha_sweep_training.allocation_signals.allocation_signal_handlers", side_effect=lambda stop: nullcontext()) as signals, \
             patch.dict(sys.modules, {"torch": backend}), patch.dict(os.environ, env, clear=True):
            for operation in PILOT_OPERATIONS:
                for seed in (0, 1, 2):
                    os.environ["SLURM_ARRAY_TASK_ID"] = str(seed)
                    output = io.StringIO()
                    with redirect_stdout(output):
                        code = main(["run", "--plan", str(self.plan_path), "--seed", str(seed), "--operation", operation])
                    self.assertEqual(code, 10)
                    self.assertEqual(json.loads(output.getvalue()), result.to_dict())
                    call = execute.call_args
                    self.assertEqual(call.args[1].run_id, manifest.tasks[seed].run_id)
                    self.assertEqual(call.kwargs["operation"], operation)
                    self.assertEqual(call.kwargs["source_root"], str(self.source))
                    self.assertEqual(call.kwargs["base_dir"], str(self.original))
                    self.assertIsInstance(call.kwargs["stop"], StopRequest)
                    self.assertIs(call.kwargs["stop"], signals.call_args.args[0])
            self.assertEqual(execute.call_count, 9)
            execute.reset_mock()
            os.environ["SLURM_ARRAY_TASK_ID"] = "2"
            with redirect_stderr(io.StringIO()):
                self.assertEqual(main(["run", "--plan", str(self.plan_path), "--seed", "0"]), 22)
            execute.assert_not_called()
        backend.set_num_threads.assert_called_with(8)
        backend.set_num_interop_threads.assert_called_with(112)

    def test_recover_is_rejected_before_gpu_runner_side_effects(self):
        from alpha_sweep_training.mobile_candidate import main, _run
        with patch("alpha_sweep_training.mobile_candidate._run") as run, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                main(["run", "--plan", "/not/opened/pilot.json", "--seed", "0", "--operation", "recover"])
            self.assertEqual(error.exception.code, 2)
            run.assert_not_called()
        with patch("alpha_sweep_training.allocation_signals.allocation_signal_handlers") as signals, \
             patch("alpha_sweep_training.mobile_candidate.load_pilot") as load:
            with self.assertRaisesRegex(ConfigError, "CPU controller"):
                _run(SimpleNamespace(operation="recover"))
            signals.assert_not_called()
            load.assert_not_called()

    def test_cpu_controller_full_budget_recovery_uses_publication_plan_without_stop_or_models(self):
        from alpha_sweep_training.controller import main
        from alpha_sweep_training.continuation import TrainingFinalizationPlan
        from alpha_sweep_training.coverage import TaskObservation
        from alpha_sweep_training.interruption import StopRequest
        _, manifest = self.build()
        task = manifest.tasks[0]
        observation = TaskObservation(run_id=task.run_id, scientific_fingerprint=task.fingerprint,
            global_index=0, phase=1, phase_index=0, dataset="cifar10", architecture="mobilenetv3_large",
            alpha=0., canonical_alpha="0", training_seed=0, condition="ok", full_budget_committed=True,
            completed_epochs=200, stored_status="training_pending_completion")
        service = Mock()
        with patch("alpha_sweep_training.task_execution.inspect_task", return_value=observation), \
             patch("alpha_sweep_training.task_execution.open_run", return_value=nullcontext(object())) as open_run, \
             patch("alpha_sweep_training.persistence.DurableTrainingPersistence", return_value=service), \
             patch("alpha_sweep_training.data.build_loaders", side_effect=AssertionError("No loaders in publication recovery")) as loaders, \
             patch("alpha_sweep_training.models.build_model", side_effect=AssertionError("No model in publication recovery")) as model, \
             patch("alpha_sweep_training.runtime.build_runtime", side_effect=AssertionError("No CUDA in publication recovery")) as runtime, \
             patch("alpha_sweep_training.training.train_run", side_effect=AssertionError("No training in publication recovery")) as train, \
             patch.object(StopRequest, "__init__", side_effect=AssertionError("No StopRequest in CPU recovery")), \
             redirect_stdout(io.StringIO()) as output:
            code = main(["run", "--manifest", str(self.candidate_root / "study.json"),
                "--run-id", manifest.tasks[0].run_id, "--base-dir", str(self.original),
                "--source-root", str(self.source), "--operation", "recover"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue())["outcome"], "training_publication_repaired")
        self.assertIsInstance(open_run.call_args.kwargs["continuation_plan"], TrainingFinalizationPlan)
        self.assertEqual(open_run.call_args.kwargs["operation"], "recover")
        service.repair_publication.assert_called_once_with()
        service.publish_training_completion.assert_not_called()
        loaders.assert_not_called()
        model.assert_not_called()
        runtime.assert_not_called()
        train.assert_not_called()
