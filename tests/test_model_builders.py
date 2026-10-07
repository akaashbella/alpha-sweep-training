import unittest

import torch
from torch import nn

from src.models.builders import build_model


class ModelBuilderTests(unittest.TestCase):
    def test_resnet50_has_exact_cifar_stem(self) -> None:
        model = build_model("resnet50", num_classes=10)
        self.assertEqual(model.conv1.kernel_size, (3, 3))
        self.assertEqual(model.conv1.stride, (1, 1))
        self.assertEqual(model.conv1.padding, (1, 1))
        self.assertIsNone(model.conv1.bias)
        self.assertIsInstance(model.maxpool, nn.Identity)

    def test_resnet50_has_requested_output_class_count(self) -> None:
        model = build_model("resnet50", num_classes=100)
        model.eval()
        self.assertEqual(model.fc.out_features, 100)
        with torch.inference_mode():
            output = model(torch.zeros(1, 3, 32, 32))
        self.assertEqual(output.shape, (1, 100))
