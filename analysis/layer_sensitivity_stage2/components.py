"""Bind scientific checkpoints separately from evaluation computation."""

from contextlib import contextmanager

from .groups import model_layout
from .planning import rebind
from .storage import identity, require


@contextmanager
def prepared_worker(row, split, expected):
    import torch
    import torchvision
    from torch.utils.data import DataLoader
    from alpha_sweep_training.checkpoints import copy_clean_model
    from alpha_sweep_training.cifar_source import dataset_class, _source_record, _validate_structure
    from alpha_sweep_training.data import EvaluationView, _initialize_worker
    from alpha_sweep_training.evaluation_state import CleanStateBaseline
    from alpha_sweep_training.models import build_model
    from alpha_sweep_training.prepared import load_prepared_dataset
    from alpha_sweep_training.runtime import build_runtime
    from alpha_sweep_training.spec import BATCH_SIZE, TEST_SIZE, TRAIN_SIZE, VALIDATION_SIZE

    from .measurement import inventory

    bound = rebind(row)
    config = bound.config
    prepared = load_prepared_dataset(config, base_dir=row["source"]["base_dir"], metadata_only=True)
    data_identity = {
        "split_identity": prepared.split["identity"], "normalization_identity": prepared.normalization["identity"],
        "preprocessing_identity": prepared.manifest["preprocessing_identity"], "preparation_identity": prepared.manifest["identity"],
        "official_training_identity": prepared.sources.training_record["identity"], "official_test_identity": prepared.sources.test_record["identity"],
    }
    require(data_identity == row["evidence"]["prepared_data"], "Runtime prepared data differ from audited checkpoint")
    validation = split == "validation"
    dataset = row["key"]["dataset"]
    constructor = dataset_class(dataset)
    source = constructor(root=str(prepared.data_root), train=validation, transform=None, target_transform=None, download=False)
    size = TRAIN_SIZE + VALIDATION_SIZE if validation else TEST_SIZE
    partition = "official_train" if validation else "official_test"
    _validate_structure(source, dataset, size)
    descriptor = prepared.sources.training_record if validation else prepared.sources.test_record
    require(_source_record(source, dataset, partition, prepared.data_root, constructor) == descriptor,
            "Decoded source bytes or labels disagree with finalized preparation")
    indices = prepared.split["validation_indices"] if validation else range(TEST_SIZE)
    require(len(indices) == expected, "Incorrect split length")
    view = EvaluationView(source, indices, prepared.normalization["mean"], prepared.normalization["std"])
    workers = config.execution["num_workers"]
    loader = DataLoader(view, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=workers,
                        generator=torch.Generator().manual_seed(0), worker_init_fn=_initialize_worker,
                        persistent_workers=workers > 0, prefetch_factor=2 if workers else None)
    runtime = build_runtime(config, evaluation_only=True)
    model = build_model(config).to(device=runtime.device, dtype=torch.float32)
    copy_clean_model(model, bound.payload)
    del bound
    layout = model_layout(model, row["key"]["architecture"])
    require(layout == row["layout"], "Constructed architecture stages differ from audited plan")
    inv, scales = inventory(model, layout)
    runtime_info = runtime.metadata()
    # Device index can differ between allocations. Compare actual effective policy
    # and device properties without treating ordinal labels as scientific identity.
    runtime_info.pop("device")
    runtime_info.update(torchvision=str(torchvision.__version__), data_loader_workers=workers)
    data = {"prepared": data_identity, "split": split, "ordered_indices_sha256": identity(list(indices)), "examples": expected}
    baseline = CleanStateBaseline(model)
    yield model, loader, runtime, baseline, inv, scales, data, runtime_info
