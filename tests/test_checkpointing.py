from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from src.config import load_config
from src.training.checkpointing import load_checkpoint, save_checkpoint
from src.training.optimizers import build_scheduler


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_round_trip(self) -> None:
        config = load_config(
            REPOSITORY_ROOT / "configs" / "smoke" / "cifar10_resnet50.yaml"
        )
        model = nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        scheduler = build_scheduler(optimizer, config.training)
        expected_weight = model.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                epoch=1,
                validation_accuracy=0.5,
                validation_loss=1.0,
            )
            with torch.no_grad():
                model.weight.zero_()

            checkpoint = load_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            self.assertTrue(path.exists())
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertEqual(checkpoint["dataset"], "cifar10")
            self.assertEqual(checkpoint["perturbation_alpha"], 0.0)
            self.assertTrue(torch.equal(model.weight, expected_weight))
