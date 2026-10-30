"""CPU mocked wiring checks, NOT native MobileNet/CUDA/checkpoint verification.

Run only via run_diagnostic_cpu_checks.py in the pinned environment. The actual
runner, RMSprop updates, schedule, accumulator, observer and state baseline run
on tiny CPU tensors. Model/data/runtime/RNG/checkpoint/full-evaluator boundaries
are explicit doubles. These checks cannot reproduce or explain the HPC collapse.
"""

from contextlib import ExitStack, nullcontext, redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


@unittest.skipUnless(os.environ.get("ALPHA_SWEEP_DIAGNOSTIC_TESTS") == "1", "Explicit CPU diagnostic checks required")
class DiagnosticRunnerWiring(unittest.TestCase):
    def setUp(self):
        import torch
        from alpha_sweep_training.config import RunConfig
        from alpha_sweep_training.spec import default_document
        from alpha_sweep_training.optimizers import build_optimizer, describe_optimizer
        from alpha_sweep_training.checkpoints import _DATA_FIELDS, data_identity
        from alpha_sweep_training.metrics import ClassificationMetrics
        self.t = torch
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(torch.random.fork_rng(devices=[]))
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        self.addCleanup(lambda: torch.set_default_dtype(previous_dtype))
        self.temporary = self.stack.enter_context(tempfile.TemporaryDirectory(prefix="alpha-diagnostic-wiring-"))
        self.root = Path(self.temporary)
        self.original = self.root / "original"
        self.original.mkdir()
        (self.original / "checkpoint-evidence.bin").write_bytes(b"original immutable evidence")
        doc = default_document("cifar10", "mobilenetv3_large")
        science = doc["scientific"]
        science.update(training_seed=101, training_seeds=[101, 102, 103])
        science["data"]["split"]["identity"] = "sha256:" + "a" * 64
        science["data"]["normalization"].update(mean=[.5] * 3, std=[.25] * 3, source="mocked-unit-data")
        science["training"].update(drop_last=False, warmup_start_factor=.1, cosine_min_lr=0., scheduler_interval="optimizer_step")
        science["precision"].update(mode="amp_bfloat16", float32_matmul_precision="highest", allow_tf32=False,
            deterministic_algorithms=True, cudnn_deterministic=True, cudnn_benchmark=False)
        self.config = RunConfig(doc)
        self.config_before = self.config.to_dict()
        info = SimpleNamespace(dataset="cifar10", architecture="mobilenetv3_large", num_classes=10, training_seed=101)
        self.stack.enter_context(patch("alpha_sweep_training.optimizers.model_construction", return_value=info))
        self.stack.enter_context(patch("alpha_sweep_training.perturbation.model_construction", return_value=info))
        self.models, self.model_configs, self.runtimes, self.validation_calls, self.steps, self.epoch_calls = [], [], [], [], [], []
        self.descriptor = {"kind": "mocked-tiny-builder; not MobileNet"}
        sample = self.make_model(self.config)
        original_opt = build_optimizer(sample, self.config)
        options = describe_optimizer(original_opt).to_dict()
        self.models.clear()
        self.model_configs.clear()
        metadata = {name: f"mocked-{name}" for name in _DATA_FIELDS}
        metadata.update(dataset="cifar10", num_classes=10, training_seed=101, counts={}, drop_last=False, data_rng_software={})
        self.metadata = metadata
        self.images = torch.tensor([[.1, .2, .3], [.4, .5, .6]])
        self.labels = torch.tensor([0, 1])
        with torch.no_grad():
            expected_correct = 2 * int((sample(self.images).argmax(1) == self.labels).sum())
        self.validation_loader = [(self.images, self.labels)] * 2
        self.data = SimpleNamespace(train=[(self.images, self.labels)] * 2, validation=self.validation_loader,
            test=object(), metadata=SimpleNamespace(to_dict=lambda: deepcopy(metadata)), set_epoch=self.epoch_calls.append)
        self.audit = {"source_comparison": {"core_file_differences": [], "software_equal": True,
                "recorded_file_differences": [], "actual_provenance": {"kind": "mocked_wiring"}},
            "latest_metadata": {"data_identity": data_identity(metadata), "component_metadata": {
                "runtime": {"device_properties": {"name": "CPU double", "capability": [0, 0]}, "effective_backend": {}},
                "model": self.descriptor, "optimizer": options}},
            "identity": {"kind": "mocked_original_identity"}, "transaction_reference": {"kind": "mocked_reference"},
            "diagnostic_process_threads": {"kind": "mocked"}, "saved_run_metadata": {"provenance": {"kind": "saved_mock"}},
            "transaction": {"status": "training_completed", "best": {"epoch_index": 0, "correct": expected_correct, "examples": 4}},
            "receipt": {"completed_epochs": 200},
            "checkpoints": {"best": dict(role="best", relative_path="generations/unit.best.pt", sha256="b" * 64, completed_epochs=1)},
            "epochs": [{"document": {"record": {"validation": ClassificationMetrics(.7, expected_correct, 4, 2).to_dict()}}}]}
        self.payload = {"metadata": {"data_identity": data_identity(metadata)}, "clean_model": deepcopy(sample.state_dict())}
        self.stack.enter_context(patch("alpha_sweep_training.models.build_model", side_effect=self.make_model))
        self.stack.enter_context(patch("alpha_sweep_training.models.describe_model", return_value=SimpleNamespace(to_dict=lambda: deepcopy(self.descriptor))))
        self.stack.enter_context(patch("alpha_sweep_training.data.build_loaders", return_value=self.data))
        self.runtime_builder = self.stack.enter_context(patch("alpha_sweep_training.runtime.build_runtime", side_effect=self.make_runtime))
        self.evaluator = self.stack.enter_context(patch("alpha_sweep_training.evaluation.evaluate_clean", side_effect=self.evaluate))
        class Streams:
            def __init__(inner, config, device, *, initialize):
                self.assertTrue(initialize)  # This fixture uses seeded reconstruction, not a saved RNG.
                self.assertEqual(device.type, "cpu")
                inner.noise_generator = torch.Generator(device="cpu").manual_seed(123)
            def claim(inner, config, device, *, continuation):
                self.assertFalse(continuation)
            def metadata(inner):
                return {"kind": "mocked_CPU_streams; not CUDA RNG verification"}
        self.stack.enter_context(patch("alpha_sweep_training.randomness.RuntimeStreams", Streams))
        native_step = torch.optim.RMSprop.step
        def step(opt, *args, **kwargs):
            self.steps.append({k: deepcopy(v) for k, v in opt.param_groups[0].items() if k != "params"})
            return native_step(opt, *args, **kwargs)
        self.stack.enter_context(patch.object(torch.optim.RMSprop, "step", step))
        # No GPU availability/fallback; accidental CUDA initialization fails.
        self.stack.enter_context(patch.object(torch.cuda, "is_available", return_value=False))
        self.stack.enter_context(patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden in CPU wiring checks")))

    def make_model(self, config):
        with self.t.random.fork_rng(devices=[]), self.t.device("cpu"):
            self.t.set_rng_state(self.t.Generator(device="cpu").manual_seed(config.scientific["training_seed"]).get_state())
            model = self.t.nn.Linear(3, 10)
        self.models.append(model)
        self.model_configs.append(config.to_dict())
        return model

    def make_runtime(self, config, *, evaluation_only=False):
        runtime = SimpleNamespace(device=self.t.device("cpu"), device_metadata={"name": "CPU double", "capability": [0, 0]},
            effective_backend={}, scaler=None, evaluation_only=evaluation_only, batch_calls=[], active_calls=[])
        def batch(pair, classes):
            self.assertEqual(classes, 10)
            runtime.batch_calls.append(pair)
            return pair
        runtime.batch, runtime.autocast = batch, nullcontext
        runtime.require_active = runtime.active_calls.append
        runtime.metadata = lambda: {"kind": "mocked_CPU_runtime", "requested_mode": config.scientific["precision"]["mode"]}
        self.runtimes.append(runtime)
        return runtime

    def evaluate(self, model, loader, *, split, config, runtime):
        from alpha_sweep_training.evaluation import classification_loss
        from alpha_sweep_training.metrics import MetricAccumulator
        self.assertIs(loader, self.validation_loader)
        self.assertEqual(split, "validation")
        self.validation_calls.append((model, config, runtime))
        runtime.require_active(config)
        modes = [(module, module.training) for module in model.modules()]
        try:
            model.eval()
            metrics = MetricAccumulator(runtime.device)
            with self.t.no_grad():
                for pair in loader:
                    images, labels = runtime.batch(pair, 10)
                    with runtime.autocast():
                        logits = model(images)
                    metrics.add(classification_loss(logits, labels, 10), logits, labels)
            return metrics.finish()
        finally:
            for module, mode in modes:
                module.training = mode

    def protocol(self, **overrides):
        from alpha_sweep_training.identity import scientific_fingerprint
        value = dict(protocol_version=1, kind="optimization_diagnostic_v1", base_scientific_fingerprint=scientific_fingerprint(self.config),
            arm="baseline", epsilon=.001, learning_rate=.01, precision_mode="amp_bfloat16", epochs=2,
            max_batches_per_epoch=None, trace_every=0, activation_modules=[], initial_checkpoint=None,
            source_difference_review={})
        return {**value, **overrides}

    def run_arm(self, **overrides):
        from alpha_sweep_training.diagnostic_protocol import new_output
        from alpha_sweep_training.diagnostic_training import run_diagnostic
        output = new_output(self.root / f"diagnostic-{len(list(self.root.iterdir()))}", [self.original])
        result = run_diagnostic(self.config, self.protocol(**overrides), self.audit, base_dir=self.original, output=output)
        self.assertEqual(self.config.to_dict(), self.config_before)
        self.assertEqual((self.original / "checkpoint-evidence.bin").read_bytes(), b"original immutable evidence")
        self.assertEqual(json.loads((output / "diagnostic.json").read_text(encoding="utf-8")), json.loads(json.dumps(result, allow_nan=False)))
        return result, output

    def test_baseline_native_options_and_actual_step_warmup_cosine_lr_sequence(self):
        result, _ = self.run_arm(epochs=6)
        self.assertEqual(result["status"], "diagnostic_budget_finished")
        self.assertFalse(result["main_v1_completion"])
        self.assertEqual(result["effective_optimizer_options"], {
            **self.audit["latest_metadata"]["component_metadata"]["optimizer"]["effective_options"], "lr": .01})
        expected = [.01 * (.1 + .9 * p / 10) for p in range(10)] + [.01, .01 * (1 + math.cos(math.pi / 390)) / 2]
        for observed, wanted in zip(self.steps, expected):
            self.assertAlmostEqual(observed["lr"], wanted, places=15)
            self.assertEqual({k: observed[k] for k in ("eps", "momentum", "alpha", "weight_decay")},
                             dict(eps=.001, momentum=.9, alpha=.9, weight_decay=1e-5))
        self.assertEqual(len(self.steps), 12)
        self.assertEqual(result["schedule"]["original_epochs"], 200)
        self.assertEqual(result["scheduler_position"], 12)
        self.assertEqual(self.epoch_calls, list(range(6)))
        self.assertTrue(all(runtime is self.runtimes[0] for _, _, runtime in self.validation_calls))

    def test_all_comparison_arms_share_initial_bytes_and_change_only_declared_options(self):
        results = [self.run_arm(epochs=1, **arm)[0] for arm in ({}, {"arm": "epsilon", "epsilon": .0316},
            {"arm": "lower_lr", "learning_rate": .001}, {"arm": "float32", "precision_mode": "float32"})]
        self.assertTrue(all(row["status"] == "diagnostic_budget_finished" for row in results))
        self.assertEqual(len({row["initial_state_sha256"] for row in results}), 1)
        self.assertEqual(len({row["diagnostic_id"] for row in results}), 4)
        self.assertEqual(self.model_configs, [self.config_before] * 4)
        baseline = results[0]["effective_optimizer_options"]
        for index, change in ((1, {"eps": .0316}), (2, {"lr": .001}), (3, {})):
            self.assertEqual(results[index]["effective_optimizer_options"], {**baseline, **change})
        self.assertEqual(self.runtimes[-1].metadata()["requested_mode"], "float32")

    def test_epoch_cadence_and_traced_validation_delegate_original_runtime(self):
        from alpha_sweep_training.config import RunConfig
        document = self.config.to_dict()
        document["scientific"]["training"]["scheduler_interval"] = "epoch"
        self.config = RunConfig(document)
        self.config_before = self.config.to_dict()
        result, output = self.run_arm(trace_every=1)
        self.assertEqual(result["scheduler_position"], 2)
        for step, expected in zip(self.steps, [.001, .001, .0028, .0028]):
            self.assertAlmostEqual(step["lr"], expected, places=15)
        self.assertTrue(all(runtime is not self.runtimes[0] for _, _, runtime in self.validation_calls))
        rows = [json.loads(line) for line in (output / "batches.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(sum(row["split"] == "clean_validation" for row in rows), 4)
        self.assertEqual(len(self.runtimes[0].batch_calls), 8)  # 4 train + 4 delegated validation.

    def test_bounded_prefix_traces_training_only_and_still_delegates_full_validation(self):
        result, output = self.run_arm(epochs=1, max_batches_per_epoch=1, trace_every=1)
        rows = [json.loads(line) for line in (output / "batches.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["split"] for row in rows], ["training"])
        self.assertEqual(result["completed_full_original_epochs"], 0)
        self.assertEqual(len(self.validation_calls), 1)
        self.assertIs(self.validation_calls[0][2], self.runtimes[0])
        self.assertEqual(len(self.runtimes[0].batch_calls), 3)

    def test_numerical_forward_failure_retains_location_and_named_tensors(self):
        from alpha_sweep_training.numerical import NumericalFailure
        from alpha_sweep_training.perturbation import training_forward
        calls = []
        def forward(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise NumericalFailure("unit_logits", "injected numerical failure", ("head.weight",))
            return training_forward(*args, **kwargs)
        with patch("alpha_sweep_training.perturbation.training_forward", side_effect=forward):
            result, _ = self.run_arm()
        self.assertEqual(result["status"], "diagnostic_failed")
        failure = result["failure"]
        self.assertEqual((failure["origin"], failure["stage"], failure["epoch_index"], failure["batch_index"]),
                         ("model_computation", "training_forward", 0, 1))
        self.assertEqual(failure["numerical_failure"], dict(stage="unit_logits", reason="injected numerical failure", tensors=["head.weight"]))
        self.assertEqual(result["successful_updates"], 1)

    def test_activation_hook_and_epoch_output_failures_are_not_computation_failures(self):
        with patch("alpha_sweep_training.diagnostic_observer.tensor_summary", side_effect=RuntimeError("injected observation")):
            observed, _ = self.run_arm(trace_every=1, activation_modules=["*"])
        self.assertEqual(observed["failure"]["origin"], "observation")
        self.assertTrue(observed["failure"]["stage"].startswith("activation:"))
        self.assertEqual((observed["failure"]["epoch_index"], observed["failure"]["batch_index"]), (0, 0))
        from alpha_sweep_training.diagnostic_training import atomic_write_json
        def publish(path, document):
            if path.name.startswith("diagnostic_epoch_"):
                raise OSError("injected disk failure")
            return atomic_write_json(path, document)
        with patch("alpha_sweep_training.diagnostic_training.atomic_write_json", side_effect=publish):
            output, _ = self.run_arm()
        self.assertEqual((output["failure"]["origin"], output["failure"]["stage"]), ("output", "epoch_publication"))
        self.assertIsNone(output["failure"]["numerical_failure"])

    def test_failure_trace_output_error_keeps_primary_numerical_origin(self):
        from alpha_sweep_training.numerical import NumericalFailure
        native_dumps = json.dumps
        def dumps(document, *args, **kwargs):
            if isinstance(document, dict) and "batch_index" in document and document.get("failure") is not None:
                raise OSError("injected failure trace write")
            return native_dumps(document, *args, **kwargs)
        with patch("alpha_sweep_training.perturbation.training_forward", side_effect=NumericalFailure("unit_forward", "injected", ("weight",))), \
             patch("alpha_sweep_training.diagnostic_training.json.dumps", side_effect=dumps):
            result, _ = self.run_arm(trace_every=1)
        self.assertEqual(result["failure"]["origin"], "model_computation")
        self.assertEqual(result["failure"]["numerical_failure"]["tensors"], ["weight"])
        self.assertEqual(result["secondary_failures"][0]["origin"], "output")
        self.assertEqual(result["secondary_failures"][0]["stage"], "batch_trace_publication")
        self.assertIsNone(result["secondary_failures"][0]["numerical_failure"])

    def test_final_publication_failure_propagates_with_prior_computation_failure(self):
        from alpha_sweep_training.diagnostic_failures import DiagnosticStageFailure
        from alpha_sweep_training.numerical import NumericalFailure
        from alpha_sweep_training.diagnostic_training import atomic_write_json
        def publish(path, document):
            if path.name == "diagnostic.json":
                raise OSError("injected final publication failure")
            return atomic_write_json(path, document)
        with patch("alpha_sweep_training.perturbation.training_forward", side_effect=NumericalFailure("unit_forward", "injected", ("weight",))), \
             patch("alpha_sweep_training.diagnostic_training.atomic_write_json", side_effect=publish):
            with self.assertRaises(DiagnosticStageFailure) as captured:
                self.run_arm()
        record = captured.exception.record
        self.assertEqual((record["origin"], record["stage"]), ("output", "diagnostic_publication"))
        self.assertEqual(record["prior_failure"]["origin"], "model_computation")
        self.assertEqual(record["prior_failure"]["numerical_failure"]["stage"], "unit_forward")

    def recheck(self):
        from alpha_sweep_training.diagnostic_training import recheck_validation
        with patch("alpha_sweep_training.checkpoints.read_checkpoint", return_value=self.payload) as read, \
             patch("alpha_sweep_training.checkpoints.copy_clean_model", side_effect=lambda model, payload: model.load_state_dict(payload["clean_model"])):
            result = recheck_validation(self.config, self.audit, base_dir=self.original, source_review={})
        self.assertEqual(read.call_args.args[1].role, "best")
        self.assertIs(read.call_args.args[3], self.audit["saved_run_metadata"]["provenance"])
        return result

    def test_validation_recheck_loads_selected_best_and_delegates_original_config(self):
        original_state = deepcopy(self.payload["clean_model"])
        result = self.recheck()
        self.assertTrue(result["exact_correct_and_examples_match"])
        self.runtime_builder.assert_called_once_with(self.config, evaluation_only=True)
        self.assertIs(self.validation_calls[0][1], self.config)
        self.assertIs(self.validation_calls[0][2], self.runtimes[0])
        self.assertFalse(result["original_artifact_writes"])
        self.assertFalse(result["test_used"])
        for name, value in original_state.items():
            self.assertTrue(self.t.equal(value, self.payload["clean_model"][name]))

    def test_recheck_reports_count_mismatch_without_publishing_completion(self):
        expected = self.audit["epochs"][0]["document"]["record"]["validation"]
        expected["correct"] = (expected["correct"] + 1) % 5
        result = self.recheck()
        self.assertFalse(result["exact_correct_and_examples_match"])
        self.assertFalse(result["research_authority"])
        self.assertFalse(result["original_artifact_writes"])

    def test_recheck_failure_propagates_numerical_location_and_cli_nonzero(self):
        from alpha_sweep_training.diagnostic_failures import DiagnosticStageFailure
        from alpha_sweep_training.evaluation import EvaluationFailure
        from alpha_sweep_training.numerical import NumericalFailure
        def fail(*args, **kwargs):
            try:
                raise NumericalFailure("forward_loss", "injected validation failure", ("logits",))
            except NumericalFailure as exc:
                raise EvaluationFailure("validation", 1, exc) from exc
        self.evaluator.side_effect = fail
        with self.assertRaises(DiagnosticStageFailure) as captured:
            self.recheck()
        error = captured.exception
        self.assertEqual((error.record["origin"], error.record["stage"], error.record["epoch_index"], error.record["batch_index"]),
                         ("model_computation", "best_clean_validation", 0, 1))
        self.assertEqual(error.record["numerical_failure"]["tensors"], ["logits"])
        self.assertTrue(self.models[-1].training)  # Owned baseline restored the prior mode on failure.
        from alpha_sweep_training.diagnostics import main
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch("alpha_sweep_training.diagnostics.load_config", return_value=self.config), \
             patch("alpha_sweep_training.diagnostics.file_digest", return_value="c" * 64), \
             patch("alpha_sweep_training.diagnostic_audit.audit_run", return_value=self.audit), \
             patch("alpha_sweep_training.diagnostic_training.recheck_validation", side_effect=error), \
             patch("alpha_sweep_training.diagnostics.protected_config_roots", return_value=[self.original]), \
             redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(["recheck-validation", "--execute", "--saved-config", str(self.original / "config.yaml"),
                "--base-dir", str(self.original), "--source-root", str(self.original),
                "--output-dir", str(self.root / "cli-error"), "--protected-root", str(self.original)])
        self.assertEqual(code, 2)
        self.assertEqual(stdout.getvalue(), "")
        document = json.loads((self.root / "cli-error" / "DIAGNOSTIC_FAILURE.json").read_text(encoding="utf-8"))
        self.assertEqual(document["failure"]["batch_index"], 1)
        self.assertEqual(document["failure"]["origin"], "model_computation")
        self.assertIn("diagnostic_execution_error", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
