import math
import unittest

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from alpha_sweep_training.evaluation_state import CleanStateBaseline
from alpha_sweep_training.perturbation import gaussian_weight_offset
from alpha_sweep_training.weights import discover_eligible_weights

from ..groups import model_layout, partition
from ..measurement import changes, inventory, measure, normalization, parameters_for
from ..storage import Stage2Error

CHECKPOINT = {"generation": "synthetic", "sha256": "a"*64}


def fixture():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4)
        model = nn.Sequential(nn.Conv2d(1, 2, 3), nn.BatchNorm2d(2), nn.Flatten(), nn.Linear(8, 3))
        data = torch.randn(7, 1, 4, 4)
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 1])
    return model, data, labels


class MeasurementTests(unittest.TestCase):
    def test_native_architecture_layouts(self):
        from torchvision import models

        for architecture, constructor, count in (
            ("resnet50", models.resnet50, 6), ("mobilenetv3_large", models.mobilenet_v3_large, 8),
            ("efficientnet_b0", models.efficientnet_b0, 10), ("convnext_tiny", models.convnext_tiny, 9),
        ):
            with self.subTest(architecture=architecture), torch.device("meta"):
                # Native stage tree only. Authoritative discovery rejects meta
                # tensors, so materialize empty CPU parameters before discovery.
                model = constructor(weights=None).to_empty(device="cpu")
                result = model_layout(model, architecture)
                self.assertEqual(len(result["stages"]), count)
                self.assertEqual(sum(len(group) for group in result["stages"].values()), len(result["tensors"]))
                if architecture in ("mobilenetv3_large", "efficientnet_b0"):
                    self.assertTrue({"depthwise", "pointwise", "squeeze_excitation", "expansion", "projection"} <= result["diagnostics"].keys())
        with self.assertRaises(Stage2Error):
            partition(["a", "b"], {"one": ["a"], "two": ["a", "b"]})
        with self.assertRaises(Stage2Error):
            partition(["a", "b"], {"one": ["a"]})

    def setup_pass(self, device="cpu"):
        model, data, labels = fixture()
        model.to(device)
        model[1].eval()
        layout = {"tensors": [e.to_dict() for e in discover_eligible_weights(model)]}
        inv, scales = inventory(model, layout)
        options = dict(baseline=CleanStateBaseline(model), device=device, classes=3, expected=len(labels),
                       scales=scales, checkpoint=CHECKPOINT, prepare_batch=lambda batch: tuple(t.to(device) for t in batch))
        return model, data, labels, inv, options

    def test_selected_full_noise_and_original_scaling(self):
        model, _, _, inv, settings = self.setup_pass()
        scales = settings["scales"]
        alpha = .2
        original = dict(model.named_parameters())
        self.assertEqual(set(scales), {"0.weight", "3.weight"})
        state = torch.get_rng_state().clone()
        clean, seeds = parameters_for(model, scales, [], CHECKPOINT, 0., 0)
        self.assertEqual(seeds, {})
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertTrue(all(clean[k] is p for k, p in original.items()))
        one, seeds = parameters_for(model, scales, ["0.weight"], CHECKPOINT, alpha, 1)
        full, all_seeds = parameters_for(model, scales, ["3.weight", "0.weight"], CHECKPOINT, alpha, 1)
        later, later_seeds = parameters_for(model, scales, ["0.weight"], CHECKPOINT, alpha, 2)
        self.assertEqual(seeds["0.weight"], all_seeds["0.weight"])
        self.assertNotEqual(seeds["0.weight"], all_seeds["3.weight"])
        self.assertNotEqual(seeds, later_seeds)
        self.assertTrue(torch.equal(one["0.weight"], full["0.weight"]))
        self.assertFalse(torch.equal(one["0.weight"], later["0.weight"]))
        for name, value in original.items():
            if name != "0.weight":
                self.assertIs(one[name], value)
            if name not in all_seeds:
                self.assertIs(full[name], value)
            else:
                generator = torch.Generator().manual_seed(all_seeds[name])
                expected = value + gaussian_weight_offset(value.shape, strength=alpha,
                    clean_population_std=value.detach().std(correction=0), generator=generator, device="cpu")
                self.assertTrue(torch.equal(full[name], expected))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        norm = normalization(inv, ["0.weight"], alpha, -.2)
        self.assertLess(norm["delta_loss_per_million_parameters"], 0)
        self.assertAlmostEqual(norm["expected_squared_perturbation_norm"], alpha**2*original["0.weight"].numel()*float(scales["0.weight"])**2)
        self.assertIsNone(normalization(inv, [], 0., 0.)["delta_loss_per_expected_squared_norm"])

    def test_fixed_batches_metrics_and_preservation(self, device="cpu"):
        model, data, labels, _, options = self.setup_pass(device)
        loader = DataLoader(TensorDataset(data, labels), batch_size=3)
        original_modes = [m.training for m in model.modules()]
        captured = []
        hook = model[0].register_forward_pre_hook(lambda layer, _: captured.append(layer.weight.detach().clone()))
        measured, ordered, preds, _ = measure(model, loader, selected=["0.weight"], alpha=.1, realization=1, **options)
        hook.remove()
        self.assertEqual(len(captured), len(loader))
        self.assertTrue(all(torch.equal(captured[0], item) for item in captured[1:]))
        self.assertEqual(ordered, bytes(labels.tolist()))
        self.assertEqual(measured["correct"], sum(a == b for a, b in zip(preds, ordered)))
        options["baseline"].verify()
        self.assertEqual([m.training for m in model.modules()], original_modes)
        clean, _, clean_predictions, _ = measure(model, loader, selected=[], alpha=0., realization=0, **options)
        with options["baseline"].realization(), torch.inference_mode():
            logits = model.eval()(data.to(device))
            expected = F.cross_entropy(logits.float(), labels.to(device)).item()
        self.assertAlmostEqual(clean["loss"], expected, places=6)
        self.assertEqual(len(clean_predictions), len(labels))
        with self.assertRaises(Stage2Error):
            measure(model, loader, selected=[], alpha=0., realization=0, **dict(options, expected=len(labels)+1))
        hook = model[0].register_forward_pre_hook(lambda *_: (_ for _ in ()).throw(RuntimeError("synthetic forward failure")))
        try:
            with self.assertRaisesRegex(RuntimeError, "synthetic forward failure"):
                measure(model, loader, selected=["0.weight"], alpha=.1, realization=1, **options)
        finally:
            hook.remove()
        self.assertEqual([m.training for m in model.modules()], original_modes)
        options["baseline"].verify()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_small_cuda_pass(self):
        self.test_fixed_batches_metrics_and_preservation(device="cuda")

    def test_metrics_flips_include_transitions_between_wrong_classes(self):
        labels = bytes([0, 0, 0, 0])
        a, b = bytes([0, 1, 1, 0]), bytes([2, 2, 0, 0])
        clean, perturbed = dict(examples=4, correct=2, loss=2.), dict(examples=4, correct=2, loss=1.)
        result = changes(clean, perturbed, labels, a, b)
        self.assertEqual(result["prediction_flip_rate"], .75)
        self.assertEqual((result["correct_to_incorrect"], result["incorrect_to_correct"]), (1, 1))
        self.assertEqual((result["delta_loss"], result["accuracy_change_pp"]), (-1., 0.))
        with self.assertRaises(Stage2Error):
            changes(clean, perturbed, labels, a, b[:-1])
