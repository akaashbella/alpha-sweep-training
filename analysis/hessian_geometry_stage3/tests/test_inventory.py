"""Stage 3 contract checks against actual inventory producers."""

from copy import deepcopy
from itertools import product
from tempfile import TemporaryDirectory
import unittest

import torch
from torch import nn
from torchvision import models

from alpha_sweep_training.models import _adapt
from alpha_sweep_training.weights import discover_eligible_weights
from analysis.layer_sensitivity_stage2.groups import model_layout
from analysis.layer_sensitivity_stage2.measurement import inventory
from analysis.layer_sensitivity_stage2.planning import make_plan
from analysis.layer_sensitivity_stage2.records import linkage, read_inventory
from analysis.layer_sensitivity_stage2.storage import install, publish_document, seal, source_digest

from ..protocol import DATASETS, PILOT_ALPHAS, corrected, validate_layout
from ..provenance import stage2_plan
from ..records import binding, validate_inventory
from ..storage import Stage3Error, plain
from .test_records import row


class InventoryContractTests(unittest.TestCase):
    def test_published_stage2_plan_uses_native_cifar_inventories_for_all_architectures(self):
        constructors = {
            "resnet50": models.resnet50,
            "mobilenetv3_large": models.mobilenet_v3_large,
            "efficientnet_b0": models.efficientnet_b0,
            "convnext_tiny": models.convnext_tiny,
        }
        layouts = {}
        for dataset, (architecture, constructor) in product(DATASETS, constructors.items()):
            with self.subTest(dataset=dataset, architecture=architecture):
                classes = 10 if dataset == "cifar10" else 100
                # Use the repository's actual CIFAR adaptations. Materialize
                # storage for discovery without running a forward pass.
                with torch.device("meta"):
                    model = constructor(weights=None, num_classes=classes)
                    _adapt(model, architecture, classes)
                model.to_empty(device="cpu")
                discovered = [tensor.to_dict() for tensor in discover_eligible_weights(model)]
                layout = model_layout(model, architecture)
                self.assertEqual(layout["tensors"], discovered)
                self.assertEqual({tensor["module_type"] for tensor in discovered},
                                 {"torch.nn.modules.conv.Conv2d", "torch.nn.modules.linear.Linear"})
                validate_layout(layout)
                layouts[dataset, architecture] = layout
                del model

        candidates = []
        for dataset, arch, alpha in product(DATASETS, constructors, PILOT_ALPHAS):
            native = row(dataset, arch, alpha)["stage2_row"]
            native["layout"] = layouts[dataset, arch]
            candidates.append(native)
        audit = seal("audit", {"checkpoints": candidates, "selected_seeds": [3], "protected_roots": [],
                             "source_sha256": source_digest()})
        plan = make_plan(audit, split="validation", alphas=(.01, .02, .05, .10, .20),
                         realizations=8, scopes=("stages",), max_passes=8904)
        with TemporaryDirectory(prefix="stage3-inventory-contract-") as temp:
            directory = plain(temp) / "plan"
            publish_document(directory, [], "plan", plan)
            loaded = stage2_plan(directory)
        self.assertEqual(loaded, plan)
        for native in loaded["checkpoints"]:
            with self.subTest(checkpoint=native["key"]):
                corrected(native)

    def test_stage2_persisted_inventory_includes_population_scales(self):
        with torch.random.fork_rng(devices=[]):
            model = nn.Sequential(nn.Conv2d(1, 2, 1), nn.Flatten(), nn.Linear(2, 3))
        native = row()["stage2_row"]
        tensors = [tensor.to_dict() for tensor in discover_eligible_weights(model)]
        native["layout"] = {"tensors": tensors, "stages": {"stem": ["0.weight"], "head": ["2.weight"]},
                            "stage_order": ["stem", "head"], "diagnostics": {}}
        layout, scales = inventory(model, native["layout"])
        stage2 = {"document_sha256": "a"*64, "source_sha256": "b"*64}
        with TemporaryDirectory(prefix="stage3-inventory-contract-") as temp:
            path = plain(temp) / "inventory.json"
            install(path, seal("inventory", {"binding": linkage(stage2, native), "inventory": layout}))
            published = read_inventory(stage2, native, path)
        checkpoint = row()
        checkpoint["stage2_row"] = native
        checkpoint["stage2"]["inventory"] = published["inventory"]
        plan = {"work_sha256": "c"*64, "core": {"stage2_plan_sha256": stage2["document_sha256"]}}
        accepted = {"binding": binding(plan, checkpoint), "layout": native["layout"],
                    "scales": {name: scale.item() for name, scale in scales.items()}}
        validate_inventory(accepted, plan, checkpoint)

        # Even a self-consistent reduced partition cannot replace the audited inventory.
        reduced = deepcopy(accepted)
        reduced["layout"]["tensors"].pop()
        del reduced["layout"]["stages"]["head"], reduced["scales"]["2.weight"]
        reduced["layout"]["stage_order"].remove("head")
        validate_layout(reduced["layout"])
        with self.assertRaisesRegex(Stage3Error, "inventory binding mismatch"):
            validate_inventory(reduced, plan, checkpoint)

    def test_invalid_inventory_and_partitions_are_rejected(self):
        layout = row()["stage2_row"]["layout"]
        mutations = {
            "unsupported qualified type": lambda value: value["tensors"][0].update(module_type="torch.nn.modules.batchnorm.BatchNorm2d"),
            "unsupported short type": lambda value: value["tensors"][0].update(module_type="Linear"),
            "missing tensor descriptor": lambda value: value["tensors"].pop(),
            "duplicate tensor descriptor": lambda value: value["tensors"].append(value["tensors"][0]),
            "missing stage member": lambda value: value["stages"].update(head=[]),
            "overlapping stages": lambda value: value["stages"]["head"].append("first.weight"),
            "foreign tensor name": lambda value: value["stages"].update(head=["foreign.weight"]),
            "non-list stage": lambda value: value["stages"].update(stem="first.weight"),
            "repeated stage in order": lambda value: value.update(stage_order=["stem", "stem"]),
            "boolean shape dimension": lambda value: value["tensors"][0].update(shape=[True, 1]),
            "wrong tensor rank": lambda value: value["tensors"][0].update(shape=[1]),
            "bad diagnostic member": lambda value: value["diagnostics"].update(combined=["foreign.weight"]),
        }
        for name, change in mutations.items():
            with self.subTest(name=name):
                invalid = deepcopy(layout)
                change(invalid)
                with self.assertRaises(Stage3Error):
                    validate_layout(invalid)
