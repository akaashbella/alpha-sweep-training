"""Read-only native checkpoint loading and the isolated FP32 HVP runtime."""

from contextlib import contextmanager
import json
import math
import os
import platform
import sys

from .protocol import numerical_policy
from .provenance import bind, pin_in_process
from .storage import identity, require


def scientific_environment():
    import torch
    import torchvision
    from alpha_sweep_training.component_contracts import require_runtime_versions

    require((3, 10) <= sys.version_info[:2] <= (3, 12), "Scientific workers require Python 3.10–3.12")
    require_runtime_versions(str(torch.__version__), str(torchvision.__version__))
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__), "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version()}


def effective_backend():
    import torch

    return {"allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cudnn_enabled": torch.backends.cudnn.enabled, "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "autocast": torch.is_autocast_enabled()}


def validate_runtime(metadata, numerics):
    require(numerics == numerical_policy(numerics["backend"]) and metadata["numerics"] == numerics, "Runtime precision differs from the plan")
    effective = metadata["effective"]
    checks = {"allow_tf32_matmul": False, "allow_tf32_cudnn": False, "autocast": False,
              "cudnn_enabled": numerics["backend"] == "cudnn", "cudnn_benchmark": False, "cudnn_deterministic": True,
              "deterministic_algorithms": True, "deterministic_warn_only": False}
    require(all(effective[key] is value for key, value in checks.items()) and
            effective["float32_matmul_precision"] == "highest" and
            effective["cublas_workspace_config"] in (":4096:8", ":16:8"), "Incorrect effective FP32/TF32/backend policy")
    software = metadata["software"]
    require(software["torch"].split("+")[0] == "2.5.1" and software["torchvision"].split("+")[0] == "0.20.1"
            and software["python"].split(".")[:2] in (["3", "10"], ["3", "11"], ["3", "12"])
            and software["cuda"], "Scientific runtime does not use the pinned CUDA environment")
    require(isinstance(metadata["gpu"]["name"], str) and len(metadata["gpu"]["capability"]) == 2 and
            metadata["gpu"]["total_memory_bytes"] > 0, "Missing device provenance")


def runtime(numerics, requested):
    import torch

    software = scientific_environment()
    device = torch.device(requested)
    require(device.type == "cuda" and torch.cuda.is_available(), "Use an assigned CUDA GPU")
    require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") in (":4096:8", ":16:8"),
            "Set CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8 before launching Python")
    torch.cuda.set_device(device)
    device = torch.device("cuda", torch.cuda.current_device())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.enabled = numerics["backend"] == "cudnn"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)
    gpu = torch.cuda.get_device_properties(device)
    properties = {"name": gpu.name, "capability": [gpu.major, gpu.minor], "total_memory_bytes": gpu.total_memory}
    metadata = {"numerics": numerics, "effective": effective_backend(), "software": software, "gpu": properties}
    validate_runtime(metadata, numerics)
    return device, metadata


def build_data(bound, audit_row, split, batch_size, expected):
    import torch
    from torch.utils.data import DataLoader
    from alpha_sweep_training.cifar_source import dataset_class, _source_record, _validate_structure
    from alpha_sweep_training.data import EvaluationView, _initialize_worker
    from alpha_sweep_training.manifests import DATA_KEYS
    from alpha_sweep_training.prepared import load_prepared_dataset

    row = audit_row["stage2_row"]
    prepared = load_prepared_dataset(bound.config, base_dir=row["source"]["base_dir"], metadata_only=True)
    data_ids = {**prepared.manifest, "preparation_identity": prepared.manifest["identity"],
                "official_training_identity": prepared.sources.training_record["identity"],
                "official_test_identity": prepared.sources.test_record["identity"]}
    data_ids = {k: data_ids[k] for k in DATA_KEYS}
    require(data_ids == row["evidence"]["prepared_data"], "Prepared data differs from verified checkpoint")
    dataset = row["key"]["dataset"]
    validation = split == "validation"
    cls = dataset_class(dataset)
    source = cls(root=str(prepared.data_root), train=validation, transform=None, target_transform=None, download=False)
    _validate_structure(source, dataset, 50000 if validation else 10000)
    record = prepared.sources.training_record if validation else prepared.sources.test_record
    require(_source_record(source, dataset, "official_train" if validation else "official_test", prepared.data_root, cls) == record,
            "Decoded CIFAR content differs from finalized preparation")
    indices = prepared.split["validation_indices"] if validation else range(expected)
    require(len(indices) == expected, "Incomplete ordered CIFAR split")
    data = {"split": split, "prepared": data_ids, "examples": expected, "ordered_indices_sha256": identity(list(indices))}
    if validation:
        require(data == audit_row["stage2"]["data"], "Ordered validation split differs from Stage 2")
    view = EvaluationView(source, indices, prepared.normalization["mean"], prepared.normalization["std"])
    workers = bound.config.execution["num_workers"]
    loader = DataLoader(view, batch_size=batch_size, num_workers=workers, shuffle=False, drop_last=False,
                        generator=torch.Generator().manual_seed(0), worker_init_fn=_initialize_worker,
                        persistent_workers=workers > 0, prefetch_factor=2 if workers else None)
    return loader, data


@contextmanager
def components(core, audit_row, device_arg):
    import torch
    from alpha_sweep_training.checkpoints import copy_clean_model
    from alpha_sweep_training.models import build_model
    from alpha_sweep_training.numerical import require_finite
    from alpha_sweep_training.weights import discover_eligible_weights, named_parameters_once
    from analysis.layer_sensitivity_stage2.groups import model_layout
    from .estimator import StateGuard

    row = audit_row["stage2_row"]
    bound, _ = bind(row, audit_row["reader"], metadata_only=False)
    loader, data = build_data(bound, audit_row, core["split"], core["batch_size"], core["examples"])
    device, metadata = runtime(core["numerics"], device_arg)
    classes = row["evidence"]["scientific"]["model"]["num_classes"]
    model = build_model(bound.config).to(device=device, dtype=torch.float32)
    copy_clean_model(model, bound.payload)
    del bound
    entries = discover_eligible_weights(model)
    layout = model_layout(model, row["key"]["architecture"])
    require(layout == row["layout"], "Runtime model grouping differs from the audited Stage 2 inventory")
    require(all(p.dtype == torch.float32 for _, p in named_parameters_once(model)) and
            all(not b.is_floating_point() or b.dtype == torch.float32 for b in model.buffers()), "Model state is not FP32")
    with torch.no_grad():
        scales = {e.name: e.parameter.detach().std(correction=0) for e in entries}
    require_finite(scales.items(), "stage3_population_std")
    floats = {name: float(value.cpu()) for name, value in scales.items()}
    stage2_scales = {t["name"]: t["population_std"] for t in audit_row["stage2"]["inventory"]["tensors"]}
    require(set(floats) == set(stage2_scales) and
            all(math.isclose(value, stage2_scales[name], rel_tol=2e-6, abs_tol=1e-12) for name, value in floats.items()),
            "Clean tensor population standard deviations differ from Stage 2 beyond tolerance")
    guard = StateGuard(model)

    def prepare(batch):
        x, y = batch
        require(x.dtype == torch.float32 and y.dtype == torch.int64 and
                x.shape == (len(y), 3, 32, 32) and len(y.shape) == 1 and len(y) <= core["batch_size"],
                "Invalid prepared batch")
        require_finite((("inputs", x),), "stage3_input")
        require(bool(((0 <= y) & (y < classes)).all()), "Prepared labels outside class range")
        return x.to(device), y.to(device)

    yield {"model": model, "loader": loader, "data": data, "runtime": metadata, "entries": entries,
           "layout": layout, "scales": scales, "floats": floats, "guard": guard, "device": device, "prepare": prepare, "classes": classes}


def corrected_test_evidence(row, pin, path):
    """Read the existing corrected native test sweep for the Stage 1 method."""
    from collections import defaultdict
    from alpha_sweep_training.evaluation_coverage import collect_evaluation_coverage, inspect_robustness_completion
    from alpha_sweep_training.evaluation_persistence import evaluation_directory, read_slot
    from alpha_sweep_training.evaluation_protocol import prescribed_slots

    require(pin["historical"], "Historical test evidence requires the audited historical reader")
    bound, _ = bind(row, pin, metadata_only=True)
    root = evaluation_directory(bound, evaluation_root=path)
    coverage = collect_evaluation_coverage(bound, evaluation_root=path)
    require(coverage["robustness_completed"], "Corrected native test sweep does not have verified complete authority")
    before = inspect_robustness_completion(root, bound)
    measurements = defaultdict(list)
    references, runtime, signature = [], None, None
    for slot in prescribed_slots():
        head, record = read_slot(root, bound, slot)
        require(record["status"] == "evaluation_completed" and record["test_metrics"]["examples"] == 10000,
                "Corrected test sweep contains an incomplete realization")
        actual = record["runtime"]
        software = record["evaluation_provenance"]["software"]
        fingerprint = identity({"precision": actual["precision"], "backend": actual["effective_backend"],
                                "torch": software["torch"], "torchvision": software["torchvision"]})
        require(signature in (None, fingerprint), "Incompatible historical test evaluation settings")
        signature = fingerprint
        runtime = {**actual, "torch": software["torch"], "torchvision": software["torchvision"]}
        measurements[slot.evaluation_alpha].append(record["test_metrics"]["loss"])
        references.append(head["current"])
    require(before == inspect_robustness_completion(root, bound) and coverage == collect_evaluation_coverage(bound, evaluation_root=path),
            "Corrected native test sweep changed during read")
    require(len(measurements[0.]) == 1 and all(len(values) == 3 for alpha, values in measurements.items() if alpha),
            "Incomplete corrected stable test realization coverage")
    points = {str(alpha): math.fsum(value/3 for value in values)-measurements[0.][0]
              for alpha, values in measurements.items() if alpha}
    return {"points": points, "runtime": runtime, "references": references}


def serve():
    """Private subprocess protocol. The parent checks JSON success and reader pin."""
    request = json.load(sys.stdin)
    try:
        pin = pin_in_process(request["pin"])
        software = scientific_environment()
        if request["operation"] == "audit":
            from alpha_sweep_training.models import build_model, describe_model
            from analysis.layer_sensitivity_stage2.groups import model_layout

            row = request["row"]
            bound, roots = bind(row, pin, metadata_only=True)
            model = build_model(bound.config)
            require(describe_model(model).to_dict() == bound.payload["metadata"]["component_metadata"]["model"] and
                    model_layout(model, row["key"]["architecture"]) == row["layout"],
                    "Native model or eligible stage partition differs from the completed Stage 2 audit")
            result = {"reader": pin, "protected_roots": roots, "software": software}
        elif request["operation"] == "run":
            from .records import execute

            result = execute(request, pin)
        elif request["operation"] == "test_evidence":
            result = corrected_test_evidence(request["row"], pin, request["evaluation_root"])
        else:
            raise ValueError("Unknown isolated reader operation")
    except Exception as exc:
        print(json.dumps({"ok": False, "reason": f"{type(exc).__name__}: {exc}"}), flush=True)
    else:
        print(json.dumps({"ok": True, "result": result}, allow_nan=False), flush=True)
