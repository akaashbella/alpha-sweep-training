"""Analytical and small-neural-network tests for full second derivatives."""

import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.autograd.function import once_differentiable
from torch.utils.data import DataLoader, TensorDataset

from alpha_sweep_training.evaluation import classification_loss
from alpha_sweep_training.weights import discover_eligible_weights

from ..estimator import DerivativeError, StateGuard, evaluation, hvp, measure_probe, vectors
from ..protocol import decomposition, validate_layout
from ..storage import Stage3Error
from ..summary import stats

BEST = {"generation": "a"*32, "sha256": "b"*64}


class HessianTests(unittest.TestCase):
    def test_indefinite_nondiagonal_hvp_and_block_restricted(self):
        matrix = torch.tensor([[4., -1., 2.], [-1., -5., 1.], [2., 1., 3.]], dtype=torch.float64)
        a = torch.tensor([1., -2.], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([1.], dtype=torch.float64, requires_grad=True)
        first = torch.tensor([.25, -.25], dtype=torch.float64)
        second = torch.tensor([2.], dtype=torch.float64)

        def objective(a, b):
            x = torch.cat([a, b])
            return .5*x @ matrix @ x

        explicit = torch.autograd.functional.hessian(lambda x: .5*x @ matrix @ x, torch.cat([a, b]))
        self.assertTrue(torch.equal(explicit, matrix))
        product = hvp(objective(a, b), (a, b), (first, second))
        self.assertTrue(torch.allclose(torch.cat(product), explicit @ torch.cat([first, second])))
        isolated = hvp(objective(a, b), (a, b), (first, torch.zeros_like(second)))
        self.assertTrue(torch.allclose(isolated[0], matrix[:2,:2] @ first))
        self.assertNotEqual(float(first @ isolated[0]), float(first @ product[0]))
        negative = torch.tensor([0., 1.], dtype=torch.float64)
        curvature, _ = hvp(objective(a, b), (a, b), (negative, torch.zeros_like(second)))
        self.assertEqual(float(negative @ curvature), -5.)

    def test_weighted_trace_rademacher_convergence_and_fixed_seed(self):
        a = torch.tensor([1., 2.], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([4.], dtype=torch.float64, requires_grad=True)
        entries = (SimpleNamespace(parameter=a, name="a"), SimpleNamespace(parameter=b, name="b"))
        # Deliberately different scales. Covariance is diagonal in the probe.
        scales = {"a": .5, "b": 1.5}
        matrix = torch.tensor([[3., 1., -1.], [1., 1., 2.], [-1., 2., -2.]], dtype=torch.float64)
        v = vectors(entries, scales, BEST, 1)
        state = torch.get_rng_state().clone()
        same = vectors(entries[::-1], scales, BEST, 1)
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(v, reversed(same))))
        self.assertFalse(all(torch.equal(x, y) for x, y in zip(v, vectors(entries, scales, BEST, 2))))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertTrue(torch.equal(v[0].abs(), torch.full_like(v[0], .5)))
        self.assertEqual(float(v[1].abs()), 1.5)
        self.assertEqual(float(vectors(entries, {**scales, "b": 0}, BEST, 1)[1]), 0.)
        draws = torch.stack([torch.cat(vectors(entries, scales, BEST, i)) for i in range(1, 4097)])
        result = ((draws @ matrix)*draws).sum(1).mean().item()
        weighted_trace = .25*(3+1) + 2.25*(-2)
        self.assertAlmostEqual(result, weighted_trace, delta=.17)

    def test_nonfinite_and_unsupported_derivatives_fail(self):
        x = torch.zeros(1, requires_grad=True)
        with self.assertRaisesRegex(DerivativeError, "stage3_gradient"):
            hvp(x.sqrt().sum(), (x,), (torch.ones_like(x),))
        with self.assertRaisesRegex(DerivativeError, "stage3_hessian_vector"):
            hvp(x.pow(1.5).sum(), (x,), (torch.ones_like(x),))

        class FirstOrderOnly(torch.autograd.Function):
            @staticmethod
            def forward(ctx, y):
                return y.square()

            @staticmethod
            @once_differentiable
            def backward(ctx, upstream):
                return upstream

        with self.assertRaisesRegex(DerivativeError, "second derivative|double backward"):
            hvp(FirstOrderOnly.apply(x).sum(), (x,), (torch.ones_like(x),))
        unused = torch.ones_like(x, requires_grad=True)
        with self.assertRaises(DerivativeError):
            hvp(x.square().sum(), (x, unused), (torch.ones_like(x), torch.ones_like(unused)))

    def test_partition_and_group_uncertainty_use_paired_sums(self):
        layout = make_layout((SimpleNamespace(to_dict=lambda: entry("first")), SimpleNamespace(to_dict=lambda: entry("second"))))
        validate_layout(layout)
        row = decomposition(layout, {"first.weight": 4., "second.weight": -7.})
        self.assertEqual(row["whole"], -3.)
        self.assertEqual(row["partition_absolute_error"], 0.)
        self.assertEqual(row["diagnostics"]["both"], -3.)
        self.assertEqual(stats([a+b for a, b in [(1., -1.), (-1., 1.)]], 2)["mc_se"], 0.)
        layout["stages"]["second"].append("first.weight")
        with self.assertRaises(Stage3Error):
            validate_layout(layout)

    def test_cnn_batch_weighting_gradients_and_state(self, device="cpu"):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(100)
            model = nn.Sequential(nn.Conv2d(1, 2, 1), nn.BatchNorm2d(2), nn.Tanh(), nn.Flatten(), nn.Linear(8, 3)).to(device)
            images = torch.randn(7, 1, 2, 2)
            labels = torch.tensor([0, 1, 2, 0, 2, 0, 1])
        model[1].eval()
        model[0].weight.grad = torch.ones_like(model[0].weight)
        previous_grad = model[0].weight.grad
        original_modes = [m.training for m in model.modules()]
        entries = discover_eligible_weights(model)
        layout = make_layout(entries)
        scales = {e.name: e.parameter.detach().std(correction=0) for e in entries}
        guard = StateGuard(model)
        loader = DataLoader(TensorDataset(images, labels), batch_size=3)
        to_device = lambda batch: tuple(t.to(device) for t in batch)
        with patch("analysis.hessian_geometry_stage3.estimator.vectors", wraps=vectors) as generate:
            result = measure_probe(model, loader, to_device, entries, scales, layout, BEST, 1,
                                   classes=3, expected=7, batches=3, guard=guard)
            self.assertEqual(generate.call_count, 1)
        with evaluation(guard, torch.device(device)):
            parameters = tuple(e.parameter for e in entries)
            v = vectors(entries, scales, BEST, 1)
            x, y = to_device((images, labels))
            full = classification_loss(model(x), y, 3)
            explicit = hvp(full, parameters, v)
            q = math.fsum(float((d*h).sum(dtype=torch.float64)) for d, h in zip(v, explicit))
            self.assertAlmostEqual(result["clean_loss"], float(full.detach()), places=6)
            self.assertAlmostEqual(result["contributions"]["whole"], q, places=6)
        guard.verify()
        self.assertIs(model[0].weight.grad, previous_grad)
        self.assertEqual(original_modes, [m.training for m in model.modules()])
        with self.assertRaisesRegex(Stage3Error, "Incomplete"):
            measure_probe(model, loader, to_device, entries, scales, layout, BEST, 1, classes=3, expected=8, batches=3, guard=guard)
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            with evaluation(guard, torch.device(device)):
                raise RuntimeError("synthetic")
        self.assertEqual(original_modes, [m.training for m in model.modules()])
        guard.verify()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_synthetic_cnn(self):
        self.test_cnn_batch_weighting_gradients_and_state("cuda")


def entry(name):
    return {"name": name+".weight", "module_name": name, "module_type": "torch.nn.modules.linear.Linear",
            "shape": [1, 1], "numel": 1, "requires_grad": True}


def make_layout(entries):
    tensors = [e.to_dict() for e in entries]
    names = [r["name"] for r in tensors]
    return {"tensors": tensors, "stages": {"first": [names[0]], "second": names[1:]},
            "stage_order": ["first", "second"], "diagnostics": {"both": names}}
