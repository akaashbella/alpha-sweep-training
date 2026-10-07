from pathlib import Path
import unittest

import yaml

from src.config import load_config


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = REPOSITORY_ROOT / "configs" / "smoke" / "cifar10_resnet50.yaml"


class ConfigTests(unittest.TestCase):
    def test_smoke_configuration_loads(self) -> None:
        config = load_config(SMOKE_CONFIG)
        self.assertEqual(config.dataset.name, "cifar10")
        self.assertEqual(config.architecture, "resnet50")
        self.assertEqual(config.num_classes, 10)
        self.assertEqual(config.training.epochs, 2)
        self.assertEqual(config.perturbation_alpha, 0.0)
        self.assertEqual(
            config.dataset.normalization.status, "provisional_smoke_test_only"
        )

    def test_nonzero_perturbation_is_rejected(self) -> None:
        import tempfile

        raw = yaml.safe_load(SMOKE_CONFIG.read_text(encoding="utf-8"))
        raw["run"]["perturbation_alpha"] = 0.1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not implemented"):
                load_config(path)

    def test_research_template_keeps_unresolved_values_explicit(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing required configuration value"):
            load_config(REPOSITORY_ROOT / "configs" / "base.yaml")
