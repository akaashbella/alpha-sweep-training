"""Disjoint architectural stages and overlapping operation diagnostics."""

from collections import Counter, defaultdict

from .storage import require


def partition(names, groups):
    counts = Counter(name for members in groups.values() for name in members)
    require(all(groups.values()) and counts == Counter({name: 1 for name in names}),
            "Stage partition has missing, duplicate, empty or foreign tensor ownership")
    return {key: sorted(members) for key, members in groups.items()}


def model_layout(model, architecture):
    from torch import nn
    from torchvision.models.convnext import CNBlock
    from torchvision.models.efficientnet import MBConv
    from torchvision.models.mobilenetv3 import InvertedResidual
    from torchvision.models.resnet import Bottleneck
    from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation
    from alpha_sweep_training.weights import discover_eligible_weights

    eligible = discover_eligible_weights(model)
    modules = dict(model.named_modules())
    roots, operation = {}, {}
    if architecture == "resnet50":
        require(isinstance(model.conv1, nn.Conv2d) and isinstance(model.fc, nn.Linear), "Unexpected ResNet stem/head")
        roots["stem"] = ["conv1"]
        for i, count in enumerate((3, 4, 6, 3), 1):
            stage = getattr(model, f"layer{i}")
            require(len(stage) == count and all(isinstance(block, Bottleneck) for block in stage), "Unexpected ResNet50 bottleneck tree")
            roots[f"residual_{i}"] = [f"layer{i}"]
        roots["classifier"] = ["fc"]
    else:
        features = model.features
        require(isinstance(features[0], Conv2dNormActivation), "Unexpected feature stem")
        roots["stem"] = ["features.0"]
        if architecture == "mobilenetv3_large":
            require(len(features) == 17 and isinstance(features[-1], Conv2dNormActivation), "Unexpected MobileNetV3-Large feature count/head")
            stage = 1
            for i in range(1, len(features)-1):
                block = features[i]
                require(isinstance(block, InvertedResidual), "Unexpected inverted residual feature")
                depthwise = [m for m in block.modules() if isinstance(m, nn.Conv2d) and m.groups == m.in_channels and m.groups > 1]
                require(len(depthwise) == 1, "Expected exactly one depthwise convolution per inverted residual")
                if depthwise[0].stride == (2, 2):
                    stage += 1
                roots.setdefault(f"inverted_resolution_{stage}", []).append(f"features.{i}")
            require(stage == 5, "Unexpected MobileNetV3-Large resolution boundaries")
            roots["feature_head"] = ["features.16"]
        elif architecture == "efficientnet_b0":
            require(len(features) == 9 and isinstance(features[-1], Conv2dNormActivation), "Unexpected EfficientNet-B0 feature count/head")
            for i, count in enumerate((1, 2, 2, 3, 3, 4, 1), 1):
                require(len(features[i]) == count and all(isinstance(block, MBConv) for block in features[i]), "Unexpected MBConv stage")
                roots[f"mbconv_{i}"] = [f"features.{i}"]
            roots["feature_head"] = ["features.8"]
        elif architecture == "convnext_tiny":
            require(len(features) == 8, "Unexpected ConvNeXt-Tiny feature count")
            for stage, count in enumerate((3, 3, 9, 3), 1):
                i = 2*stage - 1
                require(len(features[i]) == count and all(isinstance(block, CNBlock) for block in features[i]), "Unexpected ConvNeXt stage")
                roots[f"feature_{stage}"] = [f"features.{i}"]
                if i < 7:
                    down = features[i+1]
                    require(len(down) == 2 and isinstance(down[1], nn.Conv2d) and down[1].stride == (2, 2), "Unexpected ConvNeXt downsample")
                    roots[f"downsample_{stage}"] = [f"features.{i+1}"]
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        roots["classifier"] = ["classifier"]

    for root, module in modules.items():
        if not isinstance(module, (InvertedResidual, MBConv)):
            continue
        convs = [(name, layer) for name, layer in module.block.named_children() if isinstance(layer, Conv2dNormActivation)]
        require(len(convs) in (2, 3) and all(isinstance(layer[0], nn.Conv2d) for _, layer in convs), "Unexpected MBConv/inverted residual operation tree")
        labels = ("depthwise", "projection") if len(convs) == 2 else ("expansion", "depthwise", "projection")
        for (name, _), label in zip(convs, labels):
            qualified = f"{root}.block.{name}.0"
            layer = modules[qualified]
            if label == "depthwise":
                require(layer.groups == layer.in_channels and layer.groups > 1, "Unexpected depthwise operation")
            else:
                require(layer.kernel_size == (1, 1) and layer.groups == 1, "Unexpected pointwise expansion or projection")
            operation[qualified] = label

    names = [entry.name for entry in eligible]

    def belongs(name, root):
        return name == root or name.startswith(root + ".")

    stages = partition(names, {key: [entry.name for entry in eligible if any(belongs(entry.module_name, root) for root in items)]
                               for key, items in roots.items()})
    diagnostics = defaultdict(list)
    for entry in eligible:
        layer = modules[entry.module_name]
        parents = [modules[name] for name in modules if belongs(entry.module_name, name)]
        if any(isinstance(parent, SqueezeExcitation) for parent in parents):
            diagnostics["squeeze_excitation"].append(entry.name)
        if entry.module_name in operation:
            diagnostics[operation[entry.module_name]].append(entry.name)
        if isinstance(layer, nn.Conv2d):
            if layer.groups == layer.in_channels and layer.groups > 1 and operation.get(entry.module_name) != "depthwise":
                diagnostics["depthwise"].append(entry.name)
            if layer.kernel_size == (1, 1):
                diagnostics["pointwise"].append(entry.name)
        elif any(isinstance(parent, CNBlock) for parent in parents):
            diagnostics["pointwise"].append(entry.name)
    return {"tensors": [entry.to_dict() for entry in eligible], "stages": stages, "stage_order": list(stages),
            "diagnostics": dict(sorted(diagnostics.items()))}


def targets(layout, scopes):
    by_scope = {
        "stages": [("stage", name, layout["stages"][name]) for name in layout["stage_order"]],
        "tensors": [("tensor", row["name"], [row["name"]]) for row in layout["tensors"]],
        "diagnostics": [("diagnostic", name, names) for name, names in sorted(layout["diagnostics"].items())],
    }
    selected = [("network", "all", [row["name"] for row in layout["tensors"]])]
    for scope in scopes:
        selected.extend(by_scope[scope])
    return [{"id": f"{scope}:{name}", "scope": scope, "name": name, "tensors": names} for scope, name, names in selected]
