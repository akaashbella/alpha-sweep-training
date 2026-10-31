"""Opt-in reference checks; implementation does not execute these tests.

ALPHA_SWEEP_DIAGNOSTIC_TESTS=1 enables CPU checks; CUDA checks additionally require
ALPHA_SWEEP_CUDA_TESTS=1 and
the pinned GPU environment (absence is an error once explicitly enabled).
Tiny modules isolate arithmetic/ownership. Builder and full-loader boundaries are
mocked explicitly; these checks do not certify complete CIFAR models or datasets.
"""

from copy import deepcopy
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ENABLED = os.environ.get("ALPHA_SWEEP_DIAGNOSTIC_TESTS") == "1"


def tiny_model(torch, device="cpu"):
    nn = torch.nn
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, 3, padding=1)
            self.bn = nn.BatchNorm2d(4)
            self.dropout = nn.Dropout(.25)
            self.head = nn.Linear(4, 10)
            self.fail = False

        def forward(self, value):
            value = self.bn(self.conv(value))
            if self.fail:
                raise RuntimeError("intentional forward failure")
            return self.head(self.dropout(value.mean((2, 3))))
    with torch.device("cpu"):
        model = Tiny()
    return model.to(device=device, dtype=torch.float32)


class ReferenceChecks(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        original_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        self.addCleanup(lambda: torch.set_default_dtype(original_dtype))
        self.rng = torch.random.fork_rng(devices=[])
        self.rng.__enter__()
        self.addCleanup(lambda: self.rng.__exit__(None, None, None))
        torch.set_rng_state(torch.Generator().manual_seed(593).get_state())

    def equal_tree(self, left, right):
        if isinstance(left, self.torch.Tensor):
            self.torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.equal_tree(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.equal_tree(a, b)
        else:
            self.assertEqual(left, right)


@unittest.skipUnless(ENABLED, "Computation execution requires explicit ALPHA_SWEEP_DIAGNOSTIC_TESTS=1 authorization")
class CpuContracts(ReferenceChecks):
    def test_sample_weighted_metrics_with_unequal_batches(self):
        from alpha_sweep_training.metrics import MetricAccumulator
        t = self.torch
        logits = t.tensor([[4., 0.], [0., 2.], [2., 0.], [0., 3.]])
        labels = t.tensor([0, 1, 1, 1])
        accumulator = MetricAccumulator(t.device("cpu"))
        losses = []
        for a, b in ((0, 3), (3, 4)):
            loss = t.nn.functional.cross_entropy(logits[a:b], labels[a:b])
            accumulator.add(loss, logits[a:b], labels[a:b])
            losses.append(float(loss))
        result = accumulator.finish()
        self.assertEqual((result.correct, result.examples, result.batches), (3, 4, 2))
        self.assertAlmostEqual(result.loss, (3 * losses[0] + losses[1]) / 4, places=12)
        self.assertAlmostEqual(result.loss, float(t.nn.functional.cross_entropy(logits, labels)), places=6)
        self.assertNotAlmostEqual(result.loss, sum(losses) / 2, places=4)

    def test_observer_does_not_change_rng_gradients_buffers_optimizer_or_step(self):
        import random
        import numpy as np
        from alpha_sweep_training.diagnostic_observer import BatchObserver
        t = self.torch
        first = tiny_model(t)
        second = deepcopy(first)
        opts = [t.optim.RMSprop(m.parameters(), lr=.01, alpha=.9, eps=.001, momentum=.9, weight_decay=1e-5)
                for m in (first, second)]
        images = t.arange(4 * 3 * 4 * 4, dtype=t.float32).reshape(4, 3, 4, 4) / 50
        labels = t.tensor([0, 1, 2, 3])
        start = t.get_rng_state().clone()
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        reference = first(images)
        loss = t.nn.functional.cross_entropy(reference, labels)
        loss.backward()
        gradients = {n: p.grad.clone() for n, p in first.named_parameters()}
        opts[0].step()
        after_rng = t.get_rng_state().clone()
        t.set_rng_state(start)
        rows = []
        with BatchObserver(second, opts[1], every=1, activation_modules=["*"], emit=rows.append) as observer:
            observer.start(0, 0)
            self.assertEqual(len(opts[1].state), 0)  # Observation must not initialize optimizer state.
            actual = second(images)
            loss = t.nn.functional.cross_entropy(actual, labels)
            observer.forward(actual, labels, loss)
            loss.backward()
            observer.gradients()
            for name, p in second.named_parameters():
                t.testing.assert_close(p.grad, gradients[name], rtol=0, atol=0)
            opts[1].step()
            observer.finish()
        t.testing.assert_close(actual, reference, rtol=0, atol=0)
        self.equal_tree(first.state_dict(), second.state_dict())
        self.equal_tree(opts[0].state_dict(), opts[1].state_dict())
        self.assertTrue(t.equal(after_rng, t.get_rng_state()))
        self.assertEqual(python_rng, random.getstate())
        current_numpy = np.random.get_state()
        self.assertEqual(numpy_rng[0], current_numpy[0])
        np.testing.assert_array_equal(numpy_rng[1], current_numpy[1])
        self.assertEqual(numpy_rng[2:], current_numpy[2:])
        self.assertTrue(rows[0]["updates"])
        self.assertTrue(all(not m._forward_hooks for m in second.modules()))

    def test_native_optimizer_scheduler_continuation_keeps_original_horizon(self):
        from alpha_sweep_training.config import RunConfig
        from alpha_sweep_training.spec import default_document
        from alpha_sweep_training.optimizers import build_optimizer
        from alpha_sweep_training.scheduler import WarmupCosineScheduler, SchedulerError
        from alpha_sweep_training.training import _backward_and_update
        from alpha_sweep_training.metrics import Counters
        t = self.torch
        doc = default_document("cifar10", "resnet50")
        doc["scientific"]["training_seed"] = 593
        doc["scientific"]["training"].update(drop_last=False, scheduler_interval="optimizer_step", warmup_start_factor=.01, cosine_min_lr=0.)
        config = RunConfig(doc)
        info = SimpleNamespace(architecture="resnet50", dataset="cifar10", num_classes=10, training_seed=593)
        runtime = SimpleNamespace(scaler=None, device=t.device("cpu"))
        with patch("alpha_sweep_training.optimizers.model_construction", return_value=info):
            first, second = tiny_model(t), tiny_model(t)
            optimizer = build_optimizer(first, config)
            schedule = WarmupCosineScheduler(optimizer, config, 352)
            for _ in range(7):
                optimizer.zero_grad(set_to_none=True)
                updated, *_ = _backward_and_update(sum(p.square().sum() for p in first.parameters()), first, optimizer, runtime)
                self.assertTrue(updated)  # Native SGD returns None, but actually updates.
                schedule.step()
            saved_opt, saved_schedule = deepcopy(optimizer.state_dict()), deepcopy(schedule.state_dict())
            counters = Counters(attempted_batches=7, successful_updates=7)
            counters.validate(complete_boundary=True)
            second.load_state_dict(first.state_dict())
            other = build_optimizer(second, config)
            restored = WarmupCosineScheduler(other, config, 352, initialize=False)
            with self.assertRaises(SchedulerError):
                restored.load_state_dict(saved_schedule)  # Wrong order must reject, not restart warmup.
            other.load_state_dict(saved_opt)
            restored.load_state_dict(saved_schedule)
            self.assertEqual(restored.completed_units, counters.successful_updates)
            self.assertEqual(counters.completed_epochs, 0)
            self.assertEqual(restored.total_units, 200 * 352)
            self.assertEqual(restored.warmup_units, 5 * 352)
            for _ in range(3):
                for model, opt, sched in ((first, optimizer, schedule), (second, other, restored)):
                    opt.zero_grad(set_to_none=True)
                    _backward_and_update(sum(p.square().sum() for p in model.parameters()), model, opt, runtime)
                    sched.step()
            self.equal_tree(first.state_dict(), second.state_dict())
            self.equal_tree(optimizer.state_dict(), other.state_dict())
            self.assertEqual(schedule.state_dict(), restored.state_dict())

    def test_data_epoch_restore_replays_order_without_consuming_model_rng(self):
        from alpha_sweep_training.data import DataLoaders, EpochSampler
        from alpha_sweep_training.data_contract import DatasetArtifactError
        t = self.torch
        metadata = SimpleNamespace(dataset="cifar10", training_seed=593, split_identity="unit-split",
            normalization_identity="unit-normalization", preprocessing_identity="unit-preprocessing",
            official_training_identity="unit-training", official_test_identity="unit-test",
            randomness_version="unit", counts=SimpleNamespace(batch_size=128), drop_last=False,
            data_rng_software={"torch": str(t.__version__)})
        sampler = EpochSampler(list(range(17)), "cifar10", "unit-split", 593)
        data = DataLoaders(None, None, None, metadata, sampler, t.Generator(device="cpu"))
        before = t.get_rng_state().clone()
        data.set_epoch(6)
        expected = list(sampler)
        boundary = data.continuation_metadata(6)
        data.set_epoch(2)
        self.assertEqual(data.restore_next_epoch(boundary), 6)
        self.assertEqual(list(sampler), expected)
        self.assertTrue(t.equal(before, t.get_rng_state()))
        data.restore_next_epoch(data.continuation_metadata(200))
        with self.assertRaises(DatasetArtifactError):
            list(sampler)  # No epoch 201 or implicit restart at zero.


@unittest.skipUnless(ENABLED and os.environ.get("ALPHA_SWEEP_CUDA_TESTS") == "1",
                     "Separate ALPHA_SWEEP_CUDA_TESTS=1 authorization required; excluded from CPU stage")
class CudaContracts(ReferenceChecks):
    @classmethod
    def setUpClass(cls):
        import torch
        import torchvision
        from alpha_sweep_training.component_contracts import require_runtime_versions
        require_runtime_versions(torch.__version__, torchvision.__version__)
        if not torch.cuda.is_available():
            raise RuntimeError("Explicit CUDA regressions require the approved assigned GPU; no silent CPU/skip fallback")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
            raise RuntimeError("Set CUBLAS_WORKSPACE_CONFIG before this explicitly authorized CUDA test process")

    def setUp(self):
        super().setUp()
        t = self.torch
        self.device = t.device("cuda", t.cuda.current_device())
        self.cuda_rng = t.cuda.get_rng_state(self.device).clone()
        self.addCleanup(lambda: t.cuda.set_rng_state(self.cuda_rng, self.device))
        previous = (t.are_deterministic_algorithms_enabled(), t.is_deterministic_algorithms_warn_only_enabled(),
                    t.backends.cudnn.benchmark, t.backends.cudnn.deterministic)
        def restore():
            t.use_deterministic_algorithms(previous[0], warn_only=previous[1])
            t.backends.cudnn.benchmark, t.backends.cudnn.deterministic = previous[2:]
        self.addCleanup(restore)
        t.use_deterministic_algorithms(True)
        t.backends.cudnn.benchmark = False
        t.backends.cudnn.deterministic = True
        t.cuda.set_rng_state(t.Generator(device=self.device).manual_seed(593).get_state(), self.device)

    def forward(self, model, images, alpha, generator):
        from alpha_sweep_training.perturbation import training_forward
        with patch("alpha_sweep_training.perturbation.model_construction", return_value=None):
            return training_forward(model, images, alpha=alpha, noise_generator=generator)

    def inputs(self):
        t = self.torch
        return (t.arange(4 * 3 * 4 * 4, device=self.device, dtype=t.float32).reshape(4, 3, 4, 4) / 50,
                t.tensor([0, 1, 2, 3], device=self.device))

    def test_alpha_zero_logits_gradients_step_bn_and_noise_rng_equivalence(self):
        t = self.torch
        actual = tiny_model(t, self.device)
        reference = deepcopy(actual)
        inputs, labels = self.inputs()
        generator = t.Generator(device=self.device).manual_seed(81)
        noise_before = generator.get_state().clone()
        model_rng = t.cuda.get_rng_state(self.device).clone()
        expected = reference(inputs)
        t.nn.functional.cross_entropy(expected, labels).backward()
        rng_after = t.cuda.get_rng_state(self.device).clone()
        t.cuda.set_rng_state(model_rng, self.device)
        observed = self.forward(actual, inputs, 0., generator)
        t.nn.functional.cross_entropy(observed, labels).backward()
        t.testing.assert_close(observed, expected, rtol=0, atol=0)
        self.assertTrue(t.equal(generator.get_state(), noise_before))
        self.assertTrue(t.equal(t.cuda.get_rng_state(self.device), rng_after))
        for a, b in zip(actual.parameters(), reference.parameters()):
            t.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
        for model in (actual, reference):
            t.optim.RMSprop(model.parameters(), lr=.01, alpha=.9, eps=.001, momentum=.9, weight_decay=1e-5).step()
        self.equal_tree(actual.state_dict(), reference.state_dict())

    def test_runtime_rng_restore_replays_all_streams_without_reseeding(self):
        import random
        import numpy as np
        from alpha_sweep_training.config import RunConfig
        from alpha_sweep_training.spec import default_document
        from alpha_sweep_training.randomness import RuntimeStreams
        from alpha_sweep_training.component_contracts import ComponentError
        t = self.torch
        python_before, numpy_before = random.getstate(), np.random.get_state()
        self.addCleanup(lambda: random.setstate(python_before))
        self.addCleanup(lambda: np.random.set_state(numpy_before))
        doc = default_document("cifar10", "mobilenetv3_large")
        doc["scientific"]["training_seed"] = 593
        config = RunConfig(doc)
        source = RuntimeStreams(config, self.device, initialize=True)
        source.claim(config, self.device, continuation=False)
        saved = source.state_dict()
        def draw(streams):
            return (random.random(), float(np.random.random()), t.rand(3, device="cpu"),
                    t.rand(3, device=self.device), t.randn(3, device=self.device, generator=streams.noise_generator))
        expected = draw(source)
        restored = RuntimeStreams(config, self.device, initialize=False)
        restored.load_state_dict(saved)
        with self.assertRaises(ComponentError):
            restored.claim(config, self.device, continuation=False)
        restored.claim(config, self.device, continuation=True)
        self.equal_tree(expected, draw(restored))

    def test_cuda_observer_noninterference_float32_and_bfloat16(self):
        from alpha_sweep_training.diagnostic_observer import BatchObserver
        from alpha_sweep_training.evaluation import classification_loss
        t = self.torch
        if not t.cuda.is_bf16_supported():
            raise RuntimeError("The original bfloat16 observer regression requires native bfloat16 support")
        images, labels = self.inputs()
        for use_amp in (False, True):
            with self.subTest(bfloat16=use_amp):
                actual = tiny_model(t, self.device)
                reference = deepcopy(actual)
                reference_opt = t.optim.RMSprop(reference.parameters(), lr=.01, alpha=.9, eps=.001, momentum=.9, weight_decay=1e-5)
                actual_opt = t.optim.RMSprop(actual.parameters(), lr=.01, alpha=.9, eps=.001, momentum=.9, weight_decay=1e-5)
                before_rng = t.cuda.get_rng_state(self.device).clone()
                with t.autocast(device_type="cuda", enabled=use_amp, dtype=t.bfloat16):
                    expected = reference(images)
                classification_loss(expected, labels, 10).backward()
                gradients = {n: p.grad.clone() for n, p in reference.named_parameters()}
                reference_opt.step()
                after_rng = t.cuda.get_rng_state(self.device).clone()
                t.cuda.set_rng_state(before_rng, self.device)
                rows = []
                with BatchObserver(actual, actual_opt, every=1, activation_modules=["*"], emit=rows.append) as observer:
                    observer.start(0, 0)
                    with t.autocast(device_type="cuda", enabled=use_amp, dtype=t.bfloat16):
                        observed = actual(images)
                    observer.forward(observed, labels, None)
                    loss = classification_loss(observed, labels, 10)
                    observer.mean_loss(loss)
                    loss.backward()
                    observer.gradients()
                    for name, parameter in actual.named_parameters():
                        t.testing.assert_close(parameter.grad, gradients[name], rtol=0, atol=0)
                    actual_opt.step()
                    observer.finish()
                t.testing.assert_close(expected, observed, rtol=0, atol=0)
                self.equal_tree(reference.state_dict(), actual.state_dict())
                self.equal_tree(reference_opt.state_dict(), actual_opt.state_dict())
                self.assertTrue(t.equal(after_rng, t.cuda.get_rng_state(self.device)))
                self.assertEqual(len(rows), 1)
                reference.eval()
                actual.eval()
                validation_rows = []
                runtime = SimpleNamespace(batch=lambda pair, classes: pair)
                with t.inference_mode(), t.autocast(device_type="cuda", enabled=use_amp, dtype=t.bfloat16):
                    expected_validation = reference(images)
                    with BatchObserver(actual, actual_opt, every=1, activation_modules=["*"], emit=validation_rows.append) as observer:
                        with observer.validation_runtime(runtime, 0) as observed_runtime:
                            same_images, same_labels = observed_runtime.batch((images, labels), 10)
                            self.assertIs(same_images, images)
                            self.assertIs(same_labels, labels)
                            observed_validation = actual(same_images)
                t.testing.assert_close(expected_validation, observed_validation, rtol=0, atol=0)
                self.equal_tree(reference.state_dict(), actual.state_dict())
                self.equal_tree(reference_opt.state_dict(), actual_opt.state_dict())
                self.assertTrue(t.equal(after_rng, t.cuda.get_rng_state(self.device)))
                self.assertEqual(validation_rows[0]["split"], "clean_validation")
                self.assertEqual(sum(validation_rows[0]["prediction_histogram"]), len(labels))

    def test_positive_alpha_same_detached_offsets_manual_gradient_reference(self):
        from alpha_sweep_training.weights import discover_eligible_weights
        from torch.func import functional_call
        t = self.torch
        actual = tiny_model(t, self.device)
        reference = deepcopy(actual)
        original = {name: p.detach().clone() for name, p in actual.named_parameters()}
        owned = dict(actual.named_parameters())
        buffers = dict(actual.named_buffers())
        entries = discover_eligible_weights(actual)
        generator = t.Generator(device=self.device).manual_seed(83)
        scales = [t.std(e.parameter.detach(), correction=0) for e in entries]
        offsets = [(t.randn(e.shape, device=self.device, generator=generator) * (.1 * s)).detach() for e, s in zip(entries, scales)]
        calls = []
        def held_offset(shape, *, strength, clean_population_std, **kwargs):
            index = len(calls)
            self.assertEqual(strength, .1)
            self.assertFalse(clean_population_std.requires_grad)
            t.testing.assert_close(clean_population_std, scales[index], rtol=0, atol=0)
            calls.append(shape)
            return offsets[index]
        inputs, labels = self.inputs()
        start = t.cuda.get_rng_state(self.device).clone()
        with patch("alpha_sweep_training.perturbation.gaussian_weight_offset", side_effect=held_offset):
            output = self.forward(actual, inputs, .1, generator)
            t.nn.functional.cross_entropy(output, labels).backward()
        t.cuda.set_rng_state(start, self.device)
        parameters = dict(reference.named_parameters())
        for entry, offset in zip(entries, offsets):
            parameters[entry.name] = parameters[entry.name] + offset  # Do not recompute a detached scale in this reference.
        expected = functional_call(reference, (parameters, dict(reference.named_buffers())), (inputs,), strict=True)
        t.nn.functional.cross_entropy(expected, labels).backward()
        t.testing.assert_close(output, expected, rtol=0, atol=0)
        self.assertEqual(len(calls), len(entries))
        for name, p in actual.named_parameters():
            self.assertIs(p, owned[name])
            t.testing.assert_close(p.detach(), original[name], rtol=0, atol=0)
            t.testing.assert_close(p.grad, dict(reference.named_parameters())[name].grad, rtol=0, atol=0)
        for name, buffer in actual.named_buffers():
            self.assertIs(buffer, buffers[name])
            t.testing.assert_close(buffer, dict(reference.named_buffers())[name], rtol=0, atol=0)

    def test_training_exception_restores_clean_bindings_without_undoing_native_bn(self):
        t = self.torch
        model = tiny_model(t, self.device)
        model.fail = True
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        parameters, buffers = dict(model.named_parameters()), dict(model.named_buffers())
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            self.forward(model, self.inputs()[0], .1, t.Generator(device=self.device).manual_seed(84))
        self.assertFalse(model._alpha_sweep_perturbation_active)
        for name, p in model.named_parameters():
            self.assertIs(p, parameters[name])
            t.testing.assert_close(p, before[name], rtol=0, atol=0)
        for name, value in model.named_buffers():
            self.assertIs(value, buffers[name])
        self.assertEqual(model.bn.num_batches_tracked.item(), 1)  # Native partial-forward buffer update is intentional.

    def test_evaluation_one_network_per_pass_and_exception_state_preservation(self):
        from alpha_sweep_training.evaluation import evaluate_realization
        from alpha_sweep_training.evaluation_protocol import EvaluationSlot
        from alpha_sweep_training.evaluation_state import CleanStateBaseline
        from alpha_sweep_training.perturbation import gaussian_weight_offset
        from alpha_sweep_training.weights import discover_eligible_weights
        t = self.torch
        model = tiny_model(t, self.device)
        model.train()
        model.bn.eval()  # Preserve mixed prior modes, not just the root flag.
        modes = [m.training for m in model.modules()]
        before = deepcopy(model.state_dict())
        baseline = CleanStateBaseline(model)
        runtime = SimpleNamespace(device=self.device, evaluation_only=True, require_active=lambda config: None)
        inputs, _ = self.inputs()
        @t.inference_mode()
        def whole_pass(_model, loader, *, forward, **kwargs):
            model.eval()
            first = forward(inputs).detach().clone()
            second = forward(inputs).detach().clone()
            t.testing.assert_close(first, second, rtol=0, atol=0)
            return first
        rng = t.cuda.get_rng_state(self.device).clone()
        with patch("alpha_sweep_training.evaluation._evaluate_split", side_effect=whole_pass), \
             patch("alpha_sweep_training.perturbation.gaussian_weight_offset", wraps=gaussian_weight_offset) as draw:
            evaluate_realization(model, None, config=None, runtime=runtime, slot=EvaluationSlot(.1, 1), seed=85, baseline=baseline)
            self.assertEqual(draw.call_count, len(discover_eligible_weights(model)))
        self.assertTrue(t.equal(rng, t.cuda.get_rng_state(self.device)))
        self.equal_tree(model.state_dict(), before)
        self.assertEqual([m.training for m in model.modules()], modes)
        model.fail = True
        with patch("alpha_sweep_training.evaluation._evaluate_split", side_effect=whole_pass):
            with self.assertRaisesRegex(RuntimeError, "intentional"):
                evaluate_realization(model, None, config=None, runtime=runtime, slot=EvaluationSlot(.1, 1), seed=85, baseline=baseline)
        baseline.verify()
        self.assertEqual([m.training for m in model.modules()], modes)

    def test_owned_evaluation_baseline_repairs_mutation_and_raises_on_exception(self):
        from alpha_sweep_training.evaluation_state import CleanStateBaseline, CleanStatePreservationError
        t = self.torch
        model = tiny_model(t, self.device)
        model.bn.eval()
        original = model.conv.weight
        before = deepcopy(model.state_dict())
        modes = [m.training for m in model.modules()]
        baseline = CleanStateBaseline(model)
        with self.assertRaises(CleanStatePreservationError):
            with baseline.realization():
                model.conv.weight = t.nn.Parameter(model.conv.weight.detach().clone() + 1)
                with t.no_grad():
                    model.bn.running_mean.add_(1)
                model.eval()
                raise RuntimeError("intentional mutation then exception")
        baseline.verify()
        self.assertIs(model.conv.weight, original)
        self.equal_tree(model.state_dict(), before)
        self.assertEqual([m.training for m in model.modules()], modes)

    def test_native_amp_skip_observes_no_step_and_no_success_counter_increment(self):
        from alpha_sweep_training.training import _backward_and_update
        from alpha_sweep_training.metrics import Counters
        t = self.torch
        model = tiny_model(t, self.device)
        before = deepcopy(model.state_dict())
        optimizer = t.optim.SGD(model.parameters(), lr=.1, momentum=.9)
        scaler = t.amp.GradScaler("cuda")
        # Finite original loss; scaling produces overflow in BN-affine gradients.
        loss = sum(p.square().sum() for p in model.parameters()) * 1e34
        self.assertTrue(bool(t.isfinite(loss)))
        updated, bad, scale_before, scale_after = _backward_and_update(loss, model, optimizer,
            SimpleNamespace(scaler=scaler, device=self.device))
        self.assertFalse(updated)
        self.assertTrue(bad)
        self.assertLess(scale_after, scale_before)
        self.assertEqual(len(optimizer.state), 0)
        self.equal_tree(model.state_dict(), before)
        counters = Counters(attempted_batches=1, successful_updates=int(updated), skipped_updates=int(not updated))
        counters.validate(complete_boundary=True)
        self.assertEqual(counters.successful_updates, 0)


if __name__ == "__main__":
    unittest.main()
