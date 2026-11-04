"""Behavioral checks of the real external functions with isolated native adapters.

Only stdlib and the external bundle are required. The package namespace is closed
to real native imports; evaluator, checkpoint, runtime and scheduler boundaries
are mocks. Real StopRequest/signal-handler source is loaded with mocked OS signal
registration. All operational files are confined to TemporaryDirectory.
"""

from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass
import csv
from datetime import datetime, timezone
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import launcher as op
import report
import figures

ALPHAS = (0., .01, .02, .05, .1, .2, .3, .5, .75, 1.)
BUNDLE = Path(op.__file__).resolve().parent
REPOSITORY = Path(os.environ.get("ALPHA_SWEEP_CHECK_NATIVE_SOURCE_ROOT", BUNDLE.parents[1]))
LARGE_SEED = 2**64 - 59


@dataclass(frozen=True)
class Slot:
    evaluation_alpha: float
    evaluation_replicate: int

    @property
    def slot_id(self):
        return f"evaluation-alpha-{self.evaluation_alpha:g}__replicate-{self.evaluation_replicate}"

    def to_dict(self):
        return {"evaluation_alpha": self.evaluation_alpha, "evaluation_replicate": self.evaluation_replicate,
                "canonical_evaluation_alpha": f"{self.evaluation_alpha:g}", "slot_id": self.slot_id}


SLOTS = tuple(Slot(a, r) for a in ALPHAS for r in ((0,) if a == 0 else (1, 2, 3)))


class OperatorCase(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(patch.object(op.subprocess, "run", side_effect=AssertionError("External command forbidden")))
        # Windows cannot exercise POSIX directory durability. File writes,
        # hard-link publication and exclusivity are still real temporary I/O.
        if os.name != "posix":
            self.stack.enter_context(patch.object(op, "fsync", return_value=None))
        for key, value in {"BASE": self.root, "NATIVE": self.root / "native", "STUDY": self.root / "study",
                           "OPS": self.root / "operations", "EVALUATIONS": self.root / "evaluations",
                           "OLD_OPERATOR": self.root / "old-operator", "OLD_PLAN": self.root / "old-plan",
                           "EVIDENCE": self.root / "evidence", "HOME": self.root / "bundle",
                           "PYTHON": Path(sys.executable).absolute()}.items():
            self.stack.enter_context(patch.object(op, key, value))
        op.HOME.mkdir()
        for name in op.BUNDLE_FILES:
            shutil.copyfile(BUNDLE / name, op.HOME / name)
        op.STUDY.mkdir()
        (op.STUDY / "study.json").write_text('{}', encoding="utf-8")
        (op.STUDY / "pilot.json").write_text('{}', encoding="utf-8")
        self.modules = {}
        package = ModuleType("alpha_sweep_training")
        package.__path__ = []  # Never fall through to a real native package.
        self.modules[package.__name__] = package
        self.forbidden = []
        self.module("hpc_policy", WorkflowError=ValueError)
        for module, names in {"training": ("train_run",), "task_execution": ("execute_task",),
                "data": ("build_training_loader", "build_train_loader", "build_loaders", "build_test_loader"),
                "optimizers": ("build_optimizer",), "scheduler": ("build_scheduler",),
                "checkpoints": ("read_checkpoint", "copy_clean_model"), "models": ("build_model",)}.items():
            guards = {name: Mock(side_effect=AssertionError(f"Forbidden native invocation: {module}.{name}")) for name in names}
            self.forbidden.extend(guards.values())
            self.module(module, **guards)
        self.runtime_metadata = {"master_dtype": "float32", "precision": {"mode": "amp_bfloat16"}, "grad_scaler": None}
        self.source = {"commit": op.COMMIT, "source_sha256": op.SOURCE_SHA, "dirty": False,
                       "inventory": {"native.py": "a" * 64}, "git_root": str(op.NATIVE)}
        self.launch = {"activation_script": str(self.root / "activate"), "python_sha256": "pinned",
                       "python_configured": str(op.PYTHON), "modules": []}
        self.pilot = {"base_dir": str(self.root / "original"), "source_root": str(op.NATIVE),
                      "candidate_root": str(op.STUDY), "source_pin": deepcopy(self.source),
                      "software_pin": deepcopy(op.SOFTWARE), "launch_inputs": deepcopy(self.launch),
                      "protected_roots": [str(self.root / "original")],
                      "thread_policy": {"torch_intraop": 8, "torch_interop": 112}}
        self.tasks, self.inputs, self.observations, self.coverages = [], [], [], []
        self.read_indices = []
        self.slot_errors = {}
        self.history_errors = set()
        self.validation_errors = set()
        self.stamps = {}
        self.evidence_calls = {}
        self.evidence_unstable = set()
        self.training_unstable = set()
        self.coverage_unstable = set()
        self.coverage_calls = {}
        self.binding_errors = {}
        self.observation_errors = {}
        for i in range(30):
            science = {"alpha": ALPHAS[i // 3], "training_seed": i % 3, "dataset": "cifar10",
                       "architecture": "mobilenetv3_large", "specification_version": op.PROFILE,
                       "optimizer": {"epsilon": .0316}}
            config = NS(scientific=science, execution={"storage": {"output_root": str(op.STUDY / "runs"),
                         "checkpoint_root": None, "result_root": None, "log_root": None,
                         "data_root": str(self.root / "original/data"), "split_file": None}})
            task = NS(global_index=i, run_id=f"fixture-run-{i}", fingerprint=f"fingerprint-{i}", config=config)
            transaction = {"boundary": i}
            binding = {"best_checkpoint_identity": {"generation": str(i), "sha256": f"digest-{i}"},
                       "best_validation": {"epoch_index": 123, "correct": 4500, "examples": 5000},
                       "training_provenance": {"code": deepcopy(self.source), "software": deepcopy(op.SOFTWARE),
                                                "allocation": {"runtime": deepcopy(self.runtime_metadata)}},
                       "noise_seed_evidence": LARGE_SEED}
            protocol = {"protocol_version": "weight_robustness_eval_v1",
                        "randomness_version": "sha256-best-checkpoint-realization-v1"}
            paths = NS(index=i, directories=lambda i=i: [op.STUDY / "runs" / str(i)])
            inputs = NS(index=i, binding=binding, config=config, paths=paths, training_transaction=transaction,
                        protocol=NS(to_dict=lambda: deepcopy(protocol)))
            observation = NS(condition="ok", training_complete=True, completed_epochs=200, transaction=transaction,
                             to_dict=lambda i=i: {"condition": "ok", "training_complete": True, "index": i})
            slots = [{**slot.to_dict(), "condition": "ok", "stored_status": "evaluation_completed",
                      "reference": {"relative_path": slot.slot_id + "/current.json", "sha256": "c" * 64},
                      "orphan_failures": [], "failure": None, "diagnostic": None} for slot in SLOTS]
            coverage = {"namespace": str(self.root / f"namespace-{i}"), "slots": slots, "robustness_completed": True,
                        "counts": {k: 0 for k in ("locked", "partial", "incompatible", "unstable", "inspection_error", "failed", "unacknowledged_failures")},
                        "completion_inspection_error": None, "completion_inspection_deferred": False}
            self.tasks.append(task)
            self.inputs.append(inputs)
            self.observations.append(observation)
            self.coverages.append(coverage)
        self.manifest = NS(tasks=self.tasks, seeds=(0, 1, 2), grid_id=op.GRID, document_sha256=op.MANIFEST_SHA)
        self.module("mobile_candidate", load_pilot=Mock(side_effect=lambda _: (deepcopy(self.pilot), self.manifest)),
                    _launch_inputs=Mock(side_effect=lambda _: deepcopy(self.launch)))
        self.provenance = Mock(side_effect=lambda *a, **k: {"code": deepcopy(self.source), "software": deepcopy(op.SOFTWARE)})
        self.module("execution_provenance", collect_provenance=self.provenance)
        self.module("coverage", inspect_task=self.inspect_task, _stamp=self.training_stamp, _lock_paths=lambda _: [])
        self.module("artifacts", resolve_run_paths=lambda config, base: self.inputs[self.tasks.index(next(t for t in self.tasks if t.config is config))].paths)
        self.binder = Mock(side_effect=self.bind)
        self.module("evaluation_inputs", bind_evaluation_checkpoint=self.binder)
        self.module("evaluation_protocol", prescribed_slots=lambda: SLOTS,
                    perturbation_seed=lambda identity, slot: None if slot.evaluation_alpha == 0 else LARGE_SEED)
        self.module("evaluation_coverage", collect_evaluation_coverage=self.coverage)
        self.module("evaluation_persistence", read_slot=self.read_slot, _read_reference=self.read_reference,
                    validate_result=self.validate_result)
        self.module("planning_inspection", evidence_stamp=self.evidence_stamp)
        self.module("manifests", load_manifest=lambda _: self.manifest)
        self.runtime_builder = Mock(return_value=NS(metadata=lambda: deepcopy(self.runtime_metadata)))
        self.module("runtime", build_runtime=self.runtime_builder)
        self.evaluator = Mock(return_value=NS(to_dict=lambda: {"outcome": "completed"}, exit_code=0))
        self.module("evaluation_execution", execute_evaluation=self.evaluator)
        torch = ModuleType("torch")
        torch.set_num_threads, torch.set_num_interop_threads = Mock(), Mock()
        torch.cuda = NS(device_count=Mock(return_value=1), get_device_properties=Mock(return_value=NS(
            name="NVIDIA H100 80GB HBM3", major=9, minor=0, total_memory=85009367040)))
        torch.optim = Mock(side_effect=AssertionError("Optimizer forbidden"))
        torch.amp = NS(GradScaler=Mock(side_effect=AssertionError("Scaler forbidden")))
        self.forbidden.append(torch.amp.GradScaler)
        self.modules["torch"] = torch
        self.stack.enter_context(patch.dict(sys.modules, self.modules))
        for name in ("interruption", "allocation_signals"):
            path = REPOSITORY / "src/alpha_sweep_training" / (name + ".py")
            spec = importlib.util.spec_from_file_location("alpha_sweep_training." + name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
        self.handlers = {}
        self.stack.enter_context(patch.object(signal, "SIGUSR1", 10, create=True))
        self.stack.enter_context(patch.object(signal, "getsignal", return_value=signal.SIG_DFL))
        self.stack.enter_context(patch.object(signal, "signal", side_effect=lambda n, h: self.handlers.update({n: h})))
        self.plan = {"native": {"base_dir": self.pilot["base_dir"], "bindings": [deepcopy(x.binding) for x in self.inputs]},
                     "evaluation_root": str(op.EVALUATIONS), "document_sha256": "fixture-plan", "operator_pin": {}}
        self.addCleanup(self.assert_no_training)

    def module(self, name, **attrs):
        module = ModuleType("alpha_sweep_training." + name)
        module.__dict__.update(attrs)
        self.modules[module.__name__] = module
        return module

    def assert_no_training(self):
        for guard in self.forbidden:
            guard.assert_not_called()

    def inspect_task(self, task, **kwargs):
        if task.global_index in self.observation_errors:
            raise self.observation_errors[task.global_index]
        return self.observations[task.global_index]

    def bind(self, task, **kwargs):
        if task.global_index in self.binding_errors:
            raise self.binding_errors[task.global_index]
        return self.inputs[task.global_index]

    def training_stamp(self, paths):
        count = self.stamps.get(paths.index, 0)
        self.stamps[paths.index] = count + 1
        return count if paths.index in self.training_unstable else 0

    def evidence_stamp(self, paths):
        index = int(Path(paths[0]).name.split("-")[-1])
        count = self.evidence_calls.get(index, 0)
        self.evidence_calls[index] = count + 1
        return count if index in self.evidence_unstable else 0

    def coverage(self, inputs, **kwargs):
        count = self.coverage_calls.get(inputs.index, 0)
        self.coverage_calls[inputs.index] = count + 1
        value = deepcopy(self.coverages[inputs.index])
        if count and inputs.index in self.coverage_unstable:
            value["robustness_completed"] = False
        return value

    def document(self, index, slot):
        return {"status": "evaluation_completed", "test_metrics": {"accuracy": .8, "loss": .6, "examples": 10000, "correct": 8000},
                "previous_record": None, "slot_id": slot.slot_id, "index": index, "seed": LARGE_SEED}

    def read_slot(self, namespace, inputs, slot):
        self.read_indices.append((inputs.index, SLOTS.index(slot)))
        error = self.slot_errors.get((inputs.index, SLOTS.index(slot)))
        if error is not None:
            raise error
        observed = self.coverages[inputs.index]["slots"][SLOTS.index(slot)]
        return {"current": observed["reference"]}, self.document(inputs.index, slot)

    def read_reference(self, namespace, reference):
        index = int(Path(namespace).name.split("-")[-1])
        slot = next(s for s in SLOTS if reference["relative_path"].startswith(s.slot_id + "/"))
        if (index, SLOTS.index(slot)) in self.history_errors:
            raise PermissionError("Unreadable ancestry")
        return self.document(index, slot)

    def validate_result(self, document, inputs, slot):
        if (inputs.index, SLOTS.index(slot)) in self.validation_errors:
            raise ValueError("Invalid raw result")
        return document

    def collect(self):
        return report.collect(self.plan, self.manifest)

    def seed_group(self, result, alpha=0., beta=0., seed=0):
        return next(g for g in result["summaries"]["per_training_seed"] if
                    (g["training_alpha"], g["evaluation_alpha"], g["training_seed"]) == (alpha, beta, seed))

    def raw_files(self, index=0, slot_index=0):
        slot = SLOTS[slot_index]
        base = Path(self.coverages[index]["namespace"]) / "slots" / slot.slot_id
        for name in ("records", "attempts"):
            (base / name).mkdir(parents=True)
        (base / "records/good.json").write_text(json.dumps(self.document(index, slot)), encoding="utf-8")
        (base / "records/bad.json").write_text('{broken', encoding="utf-8")
        (base / "attempts/attempt.json").write_text(json.dumps({"seed": LARGE_SEED}), encoding="utf-8")
        self.coverages[index]["slots"][slot_index]["orphan_failures"] = [{"status": "failed_execution", "sha256": "orphan"}]

    def prepare_plan(self):
        op.own_bundle()
        native, _, _ = op.native_inputs()
        export = self.root / "bindings.json"
        export.write_text(json.dumps([{**row, "evaluation_binding": binding,
                "selected_epoch": binding["best_validation"]["epoch_index"] + 1,
                "best_checkpoint_identity": binding["best_checkpoint_identity"]}
            for row, binding in zip(native["selection"], native["bindings"])]), encoding="utf-8")
        evidence = {}
        for key in ("accounts", "nodes", "active_work", "storage", "walltime"):
            path = self.root / (key + ".txt")
            path.write_text("CPU fixture only", encoding="utf-8")
            evidence[key] = str(path)
        inspection = {"inspected_at_utc": datetime.now(timezone.utc).isoformat(), "operator": "CPU-test", "reason": "fixture",
                      "account": "fixture", "qos": "fixture", "partition": "dgxh", "node": "fixture-node", "gres": "gpu:1",
                      "concurrency": 4, "other_project_reserved_tasks": 0, "all_prior_submissions_resolved": True,
                      "exclusive_project_window": True, "wall_time": "00:30:00", "evidence_files": evidence}
        self.inspection_path = self.root / "inspection.json"
        self.inspection_path.write_text(json.dumps(inspection), encoding="utf-8")
        self.plan = op.prepare(NS(export_bindings=str(export), inspection=str(self.inspection_path)))
        return self.plan

    def wave(self, name="first", selected=None):
        selected = list(range(30)) if selected is None else selected
        wave = op.wave_path(name)
        wave.mkdir()
        for part in ("logs", "claims", "outcomes"):
            (wave / part).mkdir()
        observed = op.inspection(self.inspection_path, fresh=True)
        intent = op.seal({"version": op.VERSION, "plan_sha256": self.plan["document_sha256"], "wave": name,
                          "inspection": observed, "indices": selected, "retry_failed": False, "reason": "CPU fixture",
                          "argv": op.argv_for(self.plan, wave, observed, selected)})
        op.publish(wave / "intent.json", intent)
        op.acknowledge(self.plan, wave, "12345", "CPU fixture", None)
        return wave

    def worker_environment(self, index):
        return {"SLURM_ARRAY_TASK_ID": str(index), "SLURM_JOB_NUM_NODES": "1", "SLURM_NTASKS": "1",
                "SLURM_CPUS_PER_TASK": "8", "SLURM_GPUS_ON_NODE": "1", "SLURM_JOB_PARTITION": "dgxh",
                "SLURM_JOB_ACCOUNT": "fixture", "SLURM_JOB_QOS": "fixture", "SLURMD_NODENAME": "fixture-node",
                "SLURM_MEM_PER_NODE": "32768", "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "SLURM_ARRAY_JOB_ID": "12345",
                "SLURM_JOB_ID": str(20000 + index), "CUDA_VISIBLE_DEVICES": "0"}

    def submit_args(self, **changes):
        return NS(**{**dict(plan=str(op.OPS / "plan.json"), indices="0-29", inspection=str(self.inspection_path),
                           wave="first", retry_failed=False, reason="CPU fixture", execute=True,
                           confirmed_project_window=True), **changes})


class ReportingChecks(OperatorCase):
    def test_complete_grid_and_official_aggregates(self):
        result = self.collect()
        self.assertEqual((len(result["training"]), len(result["evaluation"])), (30, 840))
        self.assertTrue(result["robustness_study_complete"])
        self.assertTrue(all(report.verified_measurement(r) for r in result["evaluation"]))
        self.assertEqual(self.seed_group(result)["mean_accuracy"], .8)
        self.assertEqual(result["summaries"]["validation_selected_training_alphas"], list(ALPHAS))

    def test_late_slot_error_retains_earlier_evidence_and_continues(self):
        self.raw_files()
        self.slot_errors[(0, 26)] = PermissionError("late unreadable HEAD")
        result = self.collect()
        first, failed, last = result["evaluation"][0], result["evaluation"][26], result["evaluation"][27]
        self.assertIsNotNone(first["test_metrics"])
        self.assertTrue(first["raw_history"])
        self.assertTrue(first["raw_records"])
        self.assertTrue(first["attempts"])
        self.assertTrue(first["orphan_failures"])
        self.assertTrue(first["unparsed_evidence"])
        self.assertEqual(first["attempts"][0]["document"]["seed"], LARGE_SEED)
        self.assertFalse(first["measurement_verified"])  # malformed raw evidence fails closed locally
        self.assertEqual(failed["inspection_state"], "unreadable")
        self.assertIsNotNone(failed["reference"])
        self.assertTrue(report.verified_measurement(last))
        self.assertIn((0, 27), self.read_indices)
        self.assertFalse(result["robustness_study_complete"])
        self.assertEqual(len(result["evaluation"]), 840)

    def test_slot_error_does_not_discard_or_invalidate_independent_successes(self):
        self.slot_errors[(0, 13)] = FileNotFoundError("HEAD removed")
        result = self.collect()
        self.assertEqual(result["evaluation"][13]["inspection_state"], "missing")
        self.assertTrue(report.verified_measurement(result["evaluation"][0]))
        self.assertTrue(report.verified_measurement(result["evaluation"][27]))
        self.assertFalse(report.verified_measurement(result["evaluation"][13]))
        self.assertTrue(result["training"][0]["evaluation_snapshot_verified"])

    def test_late_history_error_retains_metrics_but_excludes_individual_point(self):
        self.history_errors.add((0, 27))
        result = self.collect()
        item = result["evaluation"][27]
        self.assertIsNotNone(item["test_metrics"])
        self.assertIsNotNone(item["observed_head"])
        self.assertFalse(item["measurement_verified"])
        group = self.seed_group(result, beta=1.)
        self.assertFalse(group["complete"])
        self.assertEqual(len(figures.individual_points(group, "accuracy")), 2)

    def test_late_ancestry_validation_failure_preserves_unvalidated_document(self):
        self.validation_errors.add((0, 27))
        result = self.collect()
        item = result["evaluation"][27]
        self.assertEqual(item["raw_history"][0]["document"]["seed"], LARGE_SEED)
        self.assertFalse(item["raw_history"][0]["validated"])
        self.assertFalse(report.verified_measurement(item))

    def test_checkpoint_wide_stability_failures_preserve_all_raw_reads(self):
        for cause in ("evidence_unstable", "training_unstable", "coverage_unstable"):
            with self.subTest(cause=cause):
                for state in (self.stamps, self.evidence_calls, self.coverage_calls, self.evidence_unstable,
                              self.training_unstable, self.coverage_unstable):
                    state.clear()
                getattr(self, cause).add(0)
                result = self.collect()
                affected = result["evaluation"][:28]
                self.assertTrue(all(r["test_metrics"] is not None and r["raw_history"] for r in affected))
                self.assertTrue(all(r["inspection_state"] == "unstable" for r in affected))
                self.assertTrue(all(not report.verified_measurement(r) for r in affected))
                self.assertFalse(result["training"][0]["evaluation_snapshot_verified"])
                self.assertFalse(result["robustness_study_complete"])
                self.assertTrue(report.verified_measurement(result["evaluation"][28]))
                self.assertEqual((len(result["training"]), len(result["evaluation"])), (30, 840))
                self.assertIsNone(self.seed_group(result)["mean_accuracy"])
                if cause == "coverage_unstable":
                    self.assertTrue(result["training"][0]["evaluation_coverage"]["robustness_completed"])
                    self.assertFalse(result["training"][0]["final_evaluation_coverage"]["robustness_completed"])
                if cause == "training_unstable":
                    self.assertFalse(result["training"][0]["training_completion_verified"])
                    self.assertIsNotNone(result["training"][0]["best_validation"])
                    self.assertFalse(result["summaries"]["all_ten_validation_candidates_complete"])

    def test_final_stability_error_retains_attempts_orphans_and_unparsed_evidence(self):
        self.raw_files(slot_index=27)
        self.evidence_unstable.add(0)
        item = self.collect()["evaluation"][27]
        for key in ("test_metrics", "raw_history", "raw_records", "attempts", "orphan_failures", "unparsed_evidence", "reference"):
            self.assertTrue(item[key], key)
        self.assertFalse(item["measurement_verified"])

    def test_final_coverage_read_exception_retains_all_observed_slots(self):
        native = sys.modules["alpha_sweep_training.evaluation_coverage"]
        def fail_final_read(inputs, **kwargs):
            if inputs.index == 0 and self.coverage_calls.get(0, 0):
                raise PermissionError("Final coverage read unavailable")
            return self.coverage(inputs, **kwargs)
        with patch.object(native, "collect_evaluation_coverage", side_effect=fail_final_read):
            result = self.collect()
        affected = result["evaluation"][:28]
        self.assertTrue(all(r["test_metrics"] and r["raw_history"] for r in affected))
        self.assertTrue(all(r["inspection_state"] == "unreadable" and not report.verified_measurement(r) for r in affected))
        self.assertFalse(result["robustness_study_complete"])
        self.assertIsNone(self.seed_group(result)["mean_accuracy"])

    def test_binding_mismatch_preserves_actual_identity_without_success(self):
        self.inputs[0].binding["best_checkpoint_identity"]["sha256"] = "changed-best"
        result = self.collect()
        self.assertEqual(result["training"][0]["actual_binding"]["best_checkpoint_identity"]["sha256"], "changed-best")
        self.assertEqual(result["evaluation"][0]["best_checkpoint_identity"]["sha256"], "changed-best")
        self.assertTrue(all(r["inspection_state"] == "incompatible" for r in result["evaluation"][:28]))
        self.assertFalse(result["training"][0]["binding_verified"])
        self.assertFalse(result["robustness_study_complete"])

    def test_missing_unreadable_pending_and_unstable_are_distinct(self):
        self.binding_errors[0] = FileNotFoundError("checkpoint absent")
        self.binding_errors[1] = PermissionError("checkpoint unreadable")
        self.evidence_unstable.add(2)
        self.coverages[3]["robustness_completed"] = False
        self.coverages[3]["slots"][0].update(condition="pending", stored_status=None, reference=None)
        result = self.collect()
        self.assertEqual([result["evaluation"][i * 28]["inspection_state"] for i in range(4)],
                         ["missing", "unreadable", "unstable", "missing"])
        self.assertIsNone(result["evaluation"][0]["best_checkpoint_identity"])
        pending = result["evaluation"][84]
        self.assertTrue(pending["observation_verified"])
        self.assertFalse(pending["measurement_verified"])

    def test_summary_and_plot_inputs_fail_closed_and_ignore_cached_summaries(self):
        result = self.collect()
        target = result["evaluation"][0]
        for flag in ("measurement_verified", "observation_verified", "binding_verified"):
            with self.subTest(flag=flag):
                changed = deepcopy(result)
                changed["evaluation"][0][flag] = False
                inputs = figures.plot_inputs(changed)
                group = inputs["per_training_seed"][0]
                self.assertIsNone(group["mean_accuracy"])
                self.assertEqual(figures.individual_points(group, "accuracy"), [])
                self.assertIsNone(inputs["across_training_seeds"][0]["mean_accuracy"])
                self.assertIsNone(inputs["paired_clean_relative"][0]["accuracy_retention_from_own_clean"])
        self.assertIsNotNone(target["test_metrics"])
        old = deepcopy(result)
        old.pop("verification_version")
        with self.assertRaisesRegex(ValueError, "verification schema"):
            figures.plot_inputs(old)
        old["evaluation"][0].pop("measurement_verified")
        self.assertFalse(report.verified_measurement(old["evaluation"][0]))

    def test_aggregation_averages_realizations_before_three_seed_summary(self):
        result = self.collect()
        chosen = [r for r in result["evaluation"] if r["training_alpha"] == 0 and r["evaluation_alpha"] == .01]
        for row in chosen:
            row["test_metrics"]["accuracy"] = [.1, .4, .9][row["training_seed"]]
        summary = report.summaries(result["training"], result["evaluation"])
        group = summary["across_training_seeds"][1]
        self.assertAlmostEqual(group["mean_accuracy"], (.1 + .4 + .9) / 3)
        self.assertAlmostEqual(group["between_seed_sample_sd_accuracy"], report.statistics.stdev([.1, .4, .9]))

    def test_report_does_not_write_or_repair_research_evidence(self):
        self.raw_files()
        namespace = Path(self.coverages[0]["namespace"])
        before = {p.relative_to(namespace).as_posix(): p.read_bytes() for p in namespace.rglob("*") if p.is_file()}
        with patch.object(op, "publish", side_effect=AssertionError("Report collector wrote storage")), \
             patch.object(op, "claim", side_effect=AssertionError("Report collector claimed storage")):
            result = self.collect()
        after = {p.relative_to(namespace).as_posix(): p.read_bytes() for p in namespace.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertFalse(result["robustness_study_complete"])

    def test_actual_figure_calls_exclude_retained_unverified_metrics(self):
        result = self.collect()
        result["evaluation"][0]["measurement_verified"] = False
        result["evaluation"][0]["test_metrics"].update(accuracy=999., loss=999.)
        # Keep the old cached summaries: plotting must recompute, not trust them.
        result = op.seal({k: v for k, v in result.items() if k != "document_sha256"})
        source = self.root / "derived-report"
        source.mkdir()
        op.publish(source / "report.json", result)
        axes = []
        def subplots(*args, **kwargs):
            figure = Mock()
            figure.savefig.side_effect = lambda path, **kw: Path(path).write_bytes(b"mock-render-only")
            if args == (2, 5):
                values = [Mock() for _ in range(10)]
                axes.extend(values)
                return figure, NS(flat=values)
            axis = Mock()
            axes.append(axis)
            return figure, axis
        matplotlib, pyplot = ModuleType("matplotlib"), ModuleType("matplotlib.pyplot")
        matplotlib.use = Mock()
        pyplot.get_cmap = lambda _: lambda i: str(i)
        pyplot.subplots = subplots
        pyplot.close = Mock()
        output = self.root / "mock-figures"
        with patch.dict(sys.modules, {"matplotlib": matplotlib, "matplotlib.pyplot": pyplot}), \
             patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(figures.main(["--report", str(source / "report.json"), "--output-dir", str(output)]), 0)
        scatter_values = [call.args[1] for axis in axes for call in axis.scatter.call_args_list]
        self.assertTrue(scatter_values)
        self.assertNotIn(999., scatter_values)
        self.assertNotIn(99900., scatter_values)
        self.assertEqual(op.check(op.read(output / "FIGURES.json"))["unverified_measurements"], 1)

    def test_changed_slot_head_retains_returned_metrics_but_is_unstable(self):
        native = sys.modules["alpha_sweep_training.evaluation_persistence"]
        def changed_head(namespace, inputs, slot):
            head, record = self.read_slot(namespace, inputs, slot)
            if inputs.index == 0 and slot == SLOTS[-1]:
                head = {"current": {"relative_path": "other-record.json", "sha256": "other"}}
            return head, record
        with patch.object(native, "read_slot", side_effect=changed_head):
            result = self.collect()
        item = result["evaluation"][27]
        self.assertIsNotNone(item["test_metrics"])
        self.assertEqual(item["observed_head"]["current"]["sha256"], "other")
        self.assertEqual(item["inspection_state"], "unstable")
        self.assertFalse(report.verified_measurement(item))
        self.assertTrue(report.verified_measurement(result["evaluation"][0]))

    def test_raw_csv_retains_evidence_with_explicit_false_verification(self):
        self.prepare_plan()
        self.evidence_unstable.add(0)
        result = self.collect()
        output = self.root / "temporary-report"
        with patch.object(op, "load_plan", return_value=(self.plan, self.manifest)), \
             patch.object(report, "collect", return_value=result), patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(report.main(["--plan", str(op.OPS / "plan.json"), "--output-dir", str(output)]), 0)
        with (output / "slots.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 840)
        self.assertEqual(rows[0]["measurement_verified"], "False")
        self.assertEqual(rows[0]["inspection_state"], "unstable")
        self.assertEqual(float(rows[0]["accuracy"]), .8)
        self.assertEqual(int(rows[1]["perturbation_seed"]), LARGE_SEED)
        self.assertFalse(op.check(op.read(output / "report.json"))["robustness_study_complete"])


class WiringChecks(OperatorCase):
    def test_all_indices_and_native_slot_delegation_including_pilots(self):
        self.prepare_plan()
        self.wave()
        self.assertEqual(op.indices("0-29"), list(range(30)))
        self.assertEqual(len(SLOTS), 28)
        self.assertEqual([(s.evaluation_alpha, s.evaluation_replicate) for s in SLOTS],
                         [(0., 0)] + [(a, r) for a in ALPHAS[1:] for r in (1, 2, 3)])
        for i in range(30):
            with self.subTest(index=i), patch.dict(os.environ, self.worker_environment(i), clear=True):
                self.assertEqual(op.mapping(i), (ALPHAS[i // 3], i % 3))
                result, code = op.run(NS(global_index=i, plan=str(op.OPS / "plan.json"), wave="first"))
                self.assertEqual(code, 0)
                args, kw = self.evaluator.call_args
                self.assertIs(args[0], self.manifest)
                self.assertIs(args[1], self.tasks[i])
                self.assertEqual(kw["expected_binding"], self.plan["native"]["bindings"][i])
                self.assertEqual(kw["expected_binding"]["noise_seed_evidence"], LARGE_SEED)
                self.assertEqual((kw["base_dir"], kw["source_root"], kw["evaluation_root"]),
                                 (self.pilot["base_dir"], op.NATIVE, str(op.EVALUATIONS)))
                self.assertIsNone(kw["evaluation_alphas"])
                self.assertIsNone(kw["evaluation_replicates"])
                self.assertFalse(kw["retry_failed"])
                self.assertFalse(kw["reuse_legacy_clean"])
                self.assertTrue(self.runtime_builder.call_args.kwargs["evaluation_only"])
        self.assertEqual(self.evaluator.call_count, 30)
        for value in (-1, 30, True, .5, "0"):
            with self.assertRaises(ValueError):
                op.mapping(value)
        for value in ("0,0", "-1", "30", "1-0", "0-999999999"):
            with self.assertRaises(ValueError):
                op.indices(value)

    def test_native_source_and_bad_native_slot_inventory_reject(self):
        for key, value in (("commit", "wrong"), ("source_sha256", "wrong"), ("dirty", True)):
            with self.subTest(key=key), patch.dict(self.source, {key: value}):
                with self.assertRaisesRegex(ValueError, "native source"):
                    op.native_inputs()
        module = sys.modules["alpha_sweep_training.evaluation_protocol"]
        with patch.object(module, "prescribed_slots", return_value=SLOTS[:-1]):
            with self.assertRaisesRegex(ValueError, "28 native slots"):
                op.native_inputs()
        self.evaluator.assert_not_called()

    def test_external_code_and_actual_checkpoint_binding_mismatch_reject(self):
        self.prepare_plan()
        path = op.HOME / "report.py"
        saved = path.read_bytes()
        path.write_bytes(saved + b'\n# changed external bytes\n')
        with self.assertRaisesRegex(ValueError, "external file"):
            op.load_plan(op.OPS / "plan.json")
        path.write_bytes(saved)
        self.inputs[0].binding["best_checkpoint_identity"]["sha256"] = "other-best"
        with self.assertRaisesRegex(ValueError, "checkpoint bindings"):
            op.load_plan(op.OPS / "plan.json")
        self.evaluator.assert_not_called()

    def test_worker_rechecks_binding_after_plan_preflight(self):
        self.prepare_plan()
        self.wave(selected=[0])
        self.binder.reset_mock()
        def bind_after_plan(task, **kwargs):
            inputs = self.bind(task, **kwargs)
            if self.binder.call_count > 30:
                inputs = deepcopy(inputs)
                inputs.binding["best_checkpoint_identity"]["sha256"] = "changed-after-preflight"
            return inputs
        self.binder.side_effect = bind_after_plan
        with patch.dict(os.environ, self.worker_environment(0), clear=True):
            with self.assertRaisesRegex(ValueError, "allocation full binding"):
                op.run(NS(global_index=0, plan=str(op.OPS / "plan.json"), wave="first"))
        self.runtime_builder.assert_not_called()
        self.evaluator.assert_not_called()

    def test_effective_runtime_mismatch_rejects_before_evaluation(self):
        self.prepare_plan()
        self.wave(selected=[0])
        self.runtime_metadata["precision"]["mode"] = "float32"
        with patch.dict(os.environ, self.worker_environment(0), clear=True):
            with self.assertRaisesRegex(ValueError, "effective runtime"):
                op.run(NS(global_index=0, plan=str(op.OPS / "plan.json"), wave="first"))
        self.evaluator.assert_not_called()

    def test_stop_request_signal_forwarding_and_native_nonzero_outcomes(self):
        self.prepare_plan()
        stop_type = sys.modules["alpha_sweep_training.interruption"].StopRequest
        for i, code in enumerate((10, 20, 21, 22)):
            self.wave(f"case-{code}", [i])
            def evaluate(*args, **kwargs):
                stop = kwargs["stop"]
                self.assertIsInstance(stop, stop_type)
                self.handlers[signal.SIGUSR1](signal.SIGUSR1, None)
                self.assertEqual(stop.policy, "discard_epoch")
                return NS(to_dict=lambda: {"exit_code": code}, exit_code=code)
            self.evaluator.side_effect = evaluate
            with patch.dict(os.environ, self.worker_environment(i), clear=True):
                result, actual_code = op.run(NS(global_index=i, plan=str(op.OPS / "plan.json"), wave=f"case-{code}"))
            self.assertEqual((result["exit_code"], actual_code), (code, code))

    def test_lossless_seed_json_and_exclusive_owner(self):
        root, owner = op.claim(self.root / "owned", [], "CPU fixture")
        op.publish(root / "seed.json", op.seal({"seed": LARGE_SEED}))
        value = op.check(op.read(root / "seed.json"))
        self.assertEqual(value["seed"], LARGE_SEED)
        self.assertIs(type(value["seed"]), int)
        with self.assertRaises(FileExistsError):
            op.publish(root / "seed.json", {"seed": 0})
        with self.assertRaises(FileExistsError):
            op.claim(root, [], "CPU fixture")
        with op.controller_lock(root):
            with self.assertRaises(FileExistsError):
                with op.controller_lock(root):
                    self.fail("Contended owner entered")
        self.assertEqual(op.read(root / "OWNER.json"), owner)

    def test_exact_intent_precedes_sbatch_and_acknowledgment_is_persisted(self):
        self.prepare_plan()
        def submit(argv, **kwargs):
            self.assertEqual(argv[0], "sbatch")
            intent = op.check(op.read(op.OPS / "waves/first/intent.json"))
            self.assertEqual(intent["argv"], argv)
            self.assertEqual(intent["indices"], list(range(30)))
            self.assertIn("--signal=B:USR1@300", argv)
            self.assertIn("--no-requeue", argv)
            return NS(returncode=0, stdout="12345\n", stderr="")
        with patch.object(op, "queue", return_value=[]), patch.object(op.subprocess, "run", side_effect=submit) as sbatch:
            result = op.preview_or_submit(self.submit_args())
        self.assertEqual(sbatch.call_count, 1)
        self.assertEqual(result["array_job_id"], "12345")
        self.assertEqual(op.check(op.read(op.OPS / "waves/first/ack.json"))["array_job_id"], "12345")

    def test_lost_acknowledgment_preserves_reservation_and_never_resubmits(self):
        self.prepare_plan()
        def lost(argv, **kwargs):
            self.assertTrue((op.OPS / "waves/first/intent.json").is_file())
            raise subprocess.TimeoutExpired(argv, 45)
        with patch.object(op, "queue", return_value=[]), patch.object(op.subprocess, "run", side_effect=lost) as sbatch:
            with self.assertRaisesRegex(ValueError, "Acknowledgment unknown"):
                op.preview_or_submit(self.submit_args())
            for name in ("first", "second"):
                with self.assertRaisesRegex(ValueError, "unresolved"):
                    op.preview_or_submit(self.submit_args(wave=name))
            self.assertEqual(sbatch.call_count, 1)
        self.assertFalse((op.OPS / "waves/first/ack.json").exists())
        self.assertTrue((op.OPS / "waves/first/intent.json").exists())

    def test_preview_never_writes_or_calls_sbatch(self):
        self.prepare_plan()
        result = op.preview_or_submit(self.submit_args(execute=False))
        self.assertFalse(result["submission_performed"])
        self.assertEqual(list((op.OPS / "waves").iterdir()), [])

    def test_failed_slots_require_explicit_audited_retry(self):
        self.prepare_plan()
        self.coverages[0]["counts"]["failed"] = 1
        with self.assertRaisesRegex(ValueError, "explicit audited retry"):
            op.preview_or_submit(self.submit_args(execute=False))
        result = op.preview_or_submit(self.submit_args(execute=False, retry_failed=True, reason="Reviewed fixture failure"))
        self.assertTrue(result["retry_failed"])

    def test_unreserved_account_work_blocks_before_intent(self):
        self.prepare_plan()
        with patch.object(op, "queue", return_value=[{"job": "other"}]):
            with self.assertRaisesRegex(ValueError, "exceeds reserved"):
                op.preview_or_submit(self.submit_args())
        self.assertEqual(list((op.OPS / "waves").iterdir()), [])
