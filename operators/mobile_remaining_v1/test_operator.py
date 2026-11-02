"""CPU-only wiring doubles. No real CUDA, model, datasets, sbatch, or native writers."""

from contextlib import ExitStack, nullcontext
from copy import deepcopy
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import launcher as op
import report


@unittest.skipUnless(os.environ.get("ALPHA_SWEEP_DIAGNOSTIC_TESTS") == "1", "Use the compatible CPU entry point")
class OperatorChecks(unittest.TestCase):
    def setUp(self):
        from test_mobile_candidate import study
        from alpha_sweep_training.config import from_declaration
        from alpha_sweep_training.manifests import Manifest, Task, manifest_from_dict
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory(prefix="remaining27-"))).resolve()
        self.native, self.study, self.original = (self.root / n for n in ("native", "study", "original"))
        for p in (self.native, self.study, self.original):
            p.mkdir()
        self.env = self.original / ".venv"
        original = study()  # Frozen native unit fixture, not research preparation.
        tasks = []
        for t in original.tasks:
            cfg = from_declaration(t.config.to_dict(), overrides={"execution.num_workers": 4,
                "execution.device": "cuda:0", "execution.storage.output_root": str(self.study / "runs"),
                "execution.storage.data_root": str(self.original / "data"),
                "execution.storage.split_file": str(self.original / "prepared" / "split.json")})
            tasks.append(Task(t.grid_id, t.phase, t.global_index, t.phase_index, cfg, t.prepared_data))
        self.manifest = manifest_from_dict(Manifest(tuple(tasks), original.seeds, original.grid_id, original.grid_version).to_dict())
        for key, value in {"NATIVE": str(self.native), "STUDY": str(self.study), "ENVIRONMENT": str(self.env),
                           "GRID": self.manifest.grid_id, "MANIFEST_SHA": self.manifest.document_sha256}.items():
            self.stack.enter_context(patch.object(op, key, value))
        self.stack.enter_context(patch.object(sys, "executable", str(self.env / "bin" / "python")))
        self.launch = {"activation_script": str(self.env / "bin" / "activate"), "mocked_existing_launch_bytes": True}
        self.pilot = {"source_root": str(self.native), "candidate_root": str(self.study), "base_dir": str(self.original),
            "source_pin": {"source_sha256": op.SOURCE_SHA}, "software_pin": deepcopy(op.SOFTWARE),
            "launch_inputs": self.launch, "protected_roots": [str(self.original)]}
        op.publish(self.study / "pilot.json", self.pilot)
        op.publish(self.study / "study.json", self.manifest.to_dict())
        self.pilot_bytes = (self.study / "pilot.json").read_bytes()
        self.provenance = {"code": {"commit": op.COMMIT, "source_sha256": op.SOURCE_SHA, "dirty": False}, "software": deepcopy(op.SOFTWARE)}
        self.stack.enter_context(patch("alpha_sweep_training.mobile_candidate.load_pilot", return_value=(self.pilot, self.manifest)))
        self.stack.enter_context(patch("alpha_sweep_training.mobile_candidate._launch_inputs", return_value=self.launch))
        self.stack.enter_context(patch("alpha_sweep_training.execution_provenance.collect_provenance", return_value=self.provenance))
        self.stack.enter_context(patch("alpha_sweep_training.coverage.inspect_task", return_value=
            SimpleNamespace(condition="ok", training_complete=True, completed_epochs=200)))
        evidence = {}
        for name in ("accounts", "nodes", "active_work"):
            path = self.root / f"{name}.txt"
            path.write_text("unit evidence; not actual HPC availability", encoding="utf-8")
            evidence[name] = str(path)
        self.inspection_path = self.root / "inspection.json"
        self.record = {"inspected_at_utc": datetime.now(timezone.utc).isoformat(), "operator": "unit operator",
            "reason": "unit-only reviewed evidence", "account": "unit-account", "qos": None,
            "node": "unit-node", "gres": "gpu:unit-full:1", "concurrency": 2,
            "other_active_experiment_tasks": 2, "all_prior_submissions_resolved": True, "evidence_files": evidence}
        op.publish(self.inspection_path, self.record)
        self.plan = op.prepare(SimpleNamespace(inspection=str(self.inspection_path), output_dir=str(self.root / "ops"),
                                              protected_root=[str(self.root / "evaluation")]))
        self.plan_path = Path(self.plan["root"]) / "plan.json"

    def test_all_twenty_seven_native_mappings_and_rejected_indices(self):
        rows = op.task_rows(self.manifest)
        self.assertEqual([(r["training_alpha"], r["training_seed"]) for r in rows],
                         [(a, s) for a in op.ALPHAS[1:] for s in (0, 1, 2)])
        self.assertEqual([r["global_index"] for r in rows], list(range(3, 30)))
        for index in (-1, 0, 1, 2, 30, 100, True, "3"):
            with self.subTest(index=index), self.assertRaises(ValueError):
                op.mapping(index)
        self.assertEqual((self.study / "pilot.json").read_bytes(), self.pilot_bytes)

    def test_native_source_software_manifest_runtime_and_plan_pins(self):
        op.load_plan(self.plan_path)
        for section, key, value in (("code", "commit", "wrong"), ("code", "source_sha256", "f" * 64),
                                    ("software", "torch", "wrong"), ("code", "dirty", True)):
            before = self.provenance[section][key]
            self.provenance[section][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                op.load_plan(self.plan_path)
            self.provenance[section][key] = before
        with patch.object(op, "MANIFEST_SHA", "f" * 64), self.assertRaises(ValueError):
            op.load_plan(self.plan_path)
        with patch.object(op, "operator_pin", return_value={"foreign": "launcher"}), self.assertRaises(ValueError):
            op.load_plan(self.plan_path)
        self.pilot["launch_inputs"] = {"changed": "activation"}
        with self.assertRaises(ValueError):
            op.load_plan(self.plan_path)

    def test_runtime_declaration_storage_and_combined_capacity_reject(self):
        from alpha_sweep_training.config import from_declaration
        from alpha_sweep_training.manifests import Task
        for override in ({"execution.num_workers": 8}, {"execution.storage.output_root": str(self.original)}):
            tasks = list(self.manifest.tasks)
            old = tasks[29]
            cfg = from_declaration(old.config.to_dict(), overrides=override)
            tasks[29] = Task(old.grid_id, old.phase, old.global_index, old.phase_index, cfg, old.prepared_data)
            with self.assertRaises(ValueError):
                op.task_rows(SimpleNamespace(tasks=tasks))
        bad = {**self.record, "concurrency": 3}  # 3 + 2 > 4, not a site entitlement.
        path = self.root / "bad-inspection.json"
        op.publish(path, bad)
        with self.assertRaises(ValueError):
            op.inspection(path)

    def test_exact_intent_precedes_submission_and_lost_ack_cannot_resubmit(self):
        args = SimpleNamespace(plan=str(self.plan_path), inspection=str(self.inspection_path), execute=True,
                               confirmed_current_inspection=True)
        def sbatch(argv, **kwargs):
            self.assertTrue((Path(self.plan["root"]) / "submission-intent.json").is_file())
            self.assertIn("--array=3-29%2", argv)
            self.assertIn("--signal=B:USR1@300", argv)
            self.assertIn("--no-requeue", argv)
            raise subprocess.TimeoutExpired(argv, 60)
        with patch.object(op.subprocess, "run", side_effect=sbatch) as call:
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                op.submit(args)
            with self.assertRaises(FileExistsError):
                op.submit(args)
            self.assertEqual(call.call_count, 1)
        result = op.acknowledge(self.plan, "123", "unit accounting reconciliation; not an actual HPC job")
        self.assertEqual(result["array_job_id"], "123")

    def test_native_dispatch_signal_ownership_auto_continuation_and_no_pilot_writes(self):
        from alpha_sweep_training.interruption import StopRequest
        backend = ModuleType("torch")
        backend.set_num_threads, backend.set_num_interop_threads = Mock(), Mock()
        backend.cuda = SimpleNamespace(device_count=Mock(return_value=1), get_device_properties=Mock(return_value=
            SimpleNamespace(name="NVIDIA H100 80GB HBM3", major=9, minor=0, total_memory=85009367040)))
        result = SimpleNamespace(to_dict=lambda: {"outcome": "interrupted", "exit_code": 10}, exit_code=10)
        intent = op.sealed({"plan_sha256": self.plan["document_sha256"], "argv": op.sbatch_argv(self.plan)})
        op.publish(Path(self.plan["root"]) / "submission-intent.json", intent)
        env = {"SLURM_ARRAY_JOB_ID": "123", "SLURM_JOB_ID": "124", "SLURM_JOB_NUM_NODES": "1",
            "SLURM_NTASKS": "1", "SLURM_CPUS_PER_TASK": "8", "SLURM_GPUS_ON_NODE": "1",
            "SLURM_JOB_PARTITION": "dgxh", "SLURM_JOB_ACCOUNT": "unit-account", "SLURMD_NODENAME": "unit-node",
            "CUDA_VISIBLE_DEVICES": "unit-visible-gpu", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
        with patch("alpha_sweep_training.task_execution.execute_task", return_value=result) as execute, \
             patch("alpha_sweep_training.allocation_signals.allocation_signal_handlers", side_effect=lambda stop: nullcontext()) as signals, \
             patch.dict(sys.modules, {"torch": backend}), patch.dict(os.environ, env, clear=True):
            for index in op.INDICES:
                os.environ["SLURM_ARRAY_TASK_ID"] = str(index)
                _, code = op.run(SimpleNamespace(plan=str(self.plan_path), global_index=index))
                self.assertEqual(code, 10)
                self.assertEqual(execute.call_args.args[1].run_id, self.manifest.tasks[index].run_id)
                self.assertEqual(execute.call_args.kwargs["operation"], "auto")
                self.assertIsInstance(execute.call_args.kwargs["stop"], StopRequest)
                self.assertIs(execute.call_args.kwargs["stop"], signals.call_args.args[0])
            self.assertEqual(execute.call_count, 27)
            # Claim prevents another allocation for the same index in this plan.
            with self.assertRaises(FileExistsError):
                op.run(SimpleNamespace(plan=str(self.plan_path), global_index=29))
            self.assertEqual(execute.call_count, 27)
        self.assertEqual((self.study / "pilot.json").read_bytes(), self.pilot_bytes)

    def test_recovery_is_not_a_gpu_operator_command_and_native_auto_preserves_failures(self):
        from alpha_sweep_training.task_execution import dispatch
        from alpha_sweep_training.coverage import TaskObservation
        from contextlib import redirect_stderr
        import io
        with patch.object(op, "run") as run, redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            op.main(["run", "--plan", str(self.plan_path), "--global-index", "3", "--operation", "recover"])
        run.assert_not_called()
        task = self.manifest.tasks[3]
        common = dict(run_id=task.run_id, scientific_fingerprint=task.fingerprint, global_index=3, phase=1,
            phase_index=3, dataset="cifar10", architecture="mobilenetv3_large", alpha=.01,
            canonical_alpha="0.01", training_seed=0, condition="ok")
        self.assertEqual(dispatch(TaskObservation(**common, training_complete=True), operation="auto"), "already_training_completed")
        self.assertEqual(dispatch(TaskObservation(**common, terminal_numerical=True), operation="auto"), "blocked_numerical")
        self.assertEqual(dispatch(TaskObservation(**common, execution_failed=True), operation="auto"), "blocked_execution")
        self.assertEqual(dispatch(TaskObservation(**common, stored_status="interrupted"), operation="auto"), "continue")
        self.assertEqual(dispatch(TaskObservation(**common, full_budget_committed=True), operation="auto"), "finalize_training")

    def test_all30_report_retains_missing_rows_unbound_slots_and_no_partial_selection(self):
        from alpha_sweep_training.coverage import TaskObservation
        def missing(task, **kwargs):
            s = task.config.scientific
            return TaskObservation(run_id=task.run_id, scientific_fingerprint=task.fingerprint,
                global_index=task.global_index, phase=1, phase_index=task.global_index, dataset="cifar10",
                architecture="mobilenetv3_large", alpha=s["alpha"], canonical_alpha=str(s["alpha"]),
                training_seed=s["training_seed"], condition="missing")
        with patch("alpha_sweep_training.coverage.inspect_task", side_effect=missing), \
             patch("alpha_sweep_training.evaluation_inputs.bind_evaluation_checkpoint") as bind:
            result = report.collect(self.plan, self.manifest, str(self.root / "evaluation"))
        self.assertEqual((len(result["training"]), len(result["evaluation"])), (30, 840))
        self.assertTrue(all(r["best_checkpoint_identity"] is None and r["perturbation_seed"] is None for r in result["evaluation"]))
        self.assertFalse(result["training_study_complete"])
        self.assertFalse(result["robustness_study_complete"])
        self.assertEqual(result["summaries"]["best_training_alphas"], [])
        bind.assert_not_called()

    def test_validation_selection_requires_all_ten_complete_seed_sets_and_retains_exact_ties(self):
        rows = [{"training_alpha": a, "training_seed": s, "training_completion_verified": True,
                 "best": {"record": {"validation": {"accuracy": .9}}}} for a in op.ALPHAS for s in (0, 1, 2)]
        summary = report.summaries(rows)
        self.assertEqual(summary["best_training_alphas"], list(op.ALPHAS))
        self.assertEqual(summary["groups"][0]["best_validation_sample_sd"], 0.)
        rows[-1]["training_completion_verified"] = False
        summary = report.summaries(rows)
        self.assertEqual(summary["best_training_alphas"], [])
        self.assertIsNone(summary["groups"][-1]["best_validation_mean"])

    def test_all30_report_preserves_every_epoch_layer_and_historical_failure(self):
        from alpha_sweep_training.coverage import TaskObservation
        from alpha_sweep_training.checkpoint_io import ArtifactReference
        from alpha_sweep_training.metrics import LayerStatistic
        from alpha_sweep_training.training_contracts import CommitReceipt, CheckpointReference
        from alpha_sweep_training.identity import identity_metadata
        reference = {"root": "metadata", "relative_path": "state/transactions/unit.json", "sha256": "a" * 64}
        failure = ArtifactReference("results", "failures/unit.json", "d" * 64)
        def observation(task, **kwargs):
            s = task.config.scientific
            return TaskObservation(run_id=task.run_id, scientific_fingerprint=task.fingerprint,
                global_index=task.global_index, phase=1, phase_index=task.global_index, dataset="cifar10",
                architecture="mobilenetv3_large", alpha=s["alpha"], canonical_alpha=str(s["alpha"]),
                training_seed=s["training_seed"], condition="ok", completed_epochs=2,
                stored_status="training", transaction=reference)
        def committed(paths, cfg, saved):
            receipt = CommitReceipt(identity_metadata(cfg), "unit", 2,
                CheckpointReference("latest", "generations/latest.pt", "b" * 64, 2),
                CheckpointReference("best", "generations/best.pt", "c" * 64, 1), None)
            records = tuple((None, SimpleNamespace(epoch_index=i, to_dict=lambda i=i: {"unit_epoch": i,
                "training": {"accuracy": .8, "loss": .5}, "validation": {"accuracy": .75, "loss": .7}}),
                (LayerStatistic("head.weight", "Linear", (2, 2), 4, .1),)) for i in (0, 1))
            return SimpleNamespace(reference=SimpleNamespace(to_dict=lambda: reference), receipt=receipt,
                records=records, document={"records": [reference, reference], "best": {"epoch_index": 0}, "artifacts": {"failure": None}})
        with patch("alpha_sweep_training.coverage.inspect_task", side_effect=observation), \
             patch("alpha_sweep_training.coverage._stamp", return_value=("stable",)), \
             patch("alpha_sweep_training.coverage._lock_paths", return_value=()), \
             patch("alpha_sweep_training.planning_inspection.evidence_stamp", return_value=("stable-failures",)), \
             patch("alpha_sweep_training.serialization.read_json", return_value={"provenance": {}}), \
             patch("alpha_sweep_training.checkpoint_journal.read_committed", side_effect=committed), \
             patch("alpha_sweep_training.checkpoint_journal._failure_evidence", side_effect=
                   lambda *a, status: (failure,) if status == "interrupted" else ()), \
             patch("alpha_sweep_training.checkpoint_io.read_document", return_value={"payload": {"status": "interrupted"}}):
            result = report.collect(self.plan, self.manifest, str(self.root / "evaluation"))
        self.assertEqual(len(result["training"]), 30)
        for row in result["training"]:
            self.assertNotIn("training_inspection_error", row)
            self.assertEqual(len(row["epochs"]), 2)
            self.assertEqual(len(row["epochs"][1]["weight_statistics"]), 1)
            self.assertEqual(row["selected_epoch"], 1)
            self.assertIsNone(row["final"])
            self.assertEqual(row["historical_failure_records"][0]["document"]["payload"]["status"], "interrupted")
