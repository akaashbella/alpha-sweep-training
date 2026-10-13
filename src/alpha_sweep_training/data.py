"""Read-only full CIFAR loaders with epoch/sample keyed local randomness."""

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, get_worker_info

from .config import RunConfig
from .data_contract import (
    CONTINUATION_VERSION, DatasetArtifactError, content_identity, require_keys, require_seed,
)
from .prepared import load_prepared_dataset
from .spec import BATCH_SIZE, DATA_RANDOMNESS_VERSION, DATASETS, EPOCHS, TEST_SIZE, TRAIN_SIZE, VALIDATION_SIZE


def _data_seed(stream: str, dataset: str, split_identity: str, training_seed: int,
               epoch: int, example_index: int | None = None) -> int:
    payload = {
        "version": DATA_RANDOMNESS_VERSION, "stream": stream, "dataset": dataset,
        "split_identity": split_identity, "training_seed": training_seed,
        "epoch": epoch, "example_index": example_index,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little", signed=False)


def _generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def _epoch_index(epoch: int, *, include_budget_end: bool = False) -> int:
    limit = EPOCHS if include_budget_end else EPOCHS - 1
    if type(epoch) is not int or not 0 <= epoch <= limit:
        raise DatasetArtifactError(f"Epoch index must be an integer in [0, {limit}]")
    return epoch


@dataclass(frozen=True)
class TrainSampleKey:
    """The sampler transmits epoch with the official index into worker copies."""
    epoch: int
    official_index: int


class EpochSampler(Sampler[TrainSampleKey]):
    def __init__(self, indices: list[int], dataset: str, split_identity: str, training_seed: int):
        self.indices = tuple(indices)
        self.dataset = dataset
        self.split_identity = split_identity
        self.training_seed = training_seed
        self.epoch: int | None = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = _epoch_index(epoch)

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self) -> Iterator[TrainSampleKey]:
        if self.epoch is None:
            raise DatasetArtifactError("Call loaders.set_epoch(zero_based_epoch) before iterating the training loader")
        epoch = self.epoch
        seed = _data_seed("sample_order", self.dataset, self.split_identity, self.training_seed, epoch)
        positions = torch.randperm(len(self.indices), generator=_generator(seed), device="cpu", dtype=torch.int64).tolist()
        for position in positions:
            yield TrainSampleKey(epoch, self.indices[position])


class _NormalizedView(Dataset):
    def __init__(self, source, mean: list[float], std: list[float]):
        self.source = source
        self.mean = torch.tensor(mean, dtype=torch.float32, device="cpu").reshape(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32, device="cpu").reshape(3, 1, 1)
        if (not bool(torch.isfinite(self.mean).all()) or not bool(torch.isfinite(self.std).all())
                or not bool((self.std > 0).all())):
            raise DatasetArtifactError("Normalization constants cannot be represented as finite positive-std float32 tensors")

    def _tensor(self, image: np.ndarray) -> torch.Tensor:
        # Ensure flipped negative-stride arrays are contiguous. Conversion to
        # float32 creates separate storage, so normalization never changes source.
        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).to(device="cpu", dtype=torch.float32)
        return tensor.div_(255.0).sub_(self.mean).div_(self.std)


class TrainingView(_NormalizedView):
    def __init__(self, source, indices: list[int], mean: list[float], std: list[float],
                 dataset: str, split_identity: str, training_seed: int):
        super().__init__(source, mean, std)
        self.indices = frozenset(indices)
        self.dataset = dataset
        self.split_identity = split_identity
        self.training_seed = training_seed

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, key: TrainSampleKey) -> tuple[torch.Tensor, int]:
        if not isinstance(key, TrainSampleKey):
            raise DatasetArtifactError("Training view requires an epoch-carrying TrainSampleKey from EpochSampler")
        _epoch_index(key.epoch)
        if type(key.official_index) is not int or key.official_index not in self.indices:
            raise DatasetArtifactError("Training sample index is not in the persisted training partition")
        crop_seed = _data_seed("augmentation_crop", self.dataset, self.split_identity,
                               self.training_seed, key.epoch, key.official_index)
        flip_seed = _data_seed("augmentation_flip", self.dataset, self.split_identity,
                               self.training_seed, key.epoch, key.official_index)
        top, left = torch.randint(0, 9, (2,), generator=_generator(crop_seed), device="cpu", dtype=torch.int64).tolist()
        image = np.pad(self.source.data[key.official_index], ((4, 4), (4, 4), (0, 0)),
                       mode="constant", constant_values=0)[top:top + 32, left:left + 32]
        if torch.rand((), generator=_generator(flip_seed), device="cpu", dtype=torch.float32).item() < 0.5:
            image = image[:, ::-1, :]
        return self._tensor(image), self.source.targets[key.official_index]


class EvaluationView(_NormalizedView):
    def __init__(self, source, indices: list[int] | range, mean: list[float], std: list[float]):
        super().__init__(source, mean, std)
        self.indices = tuple(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        official_index = self.indices[index]
        return self._tensor(self.source.data[official_index]), self.source.targets[official_index]


def _initialize_worker(worker_id: int) -> None:
    """Initialize worker-only incidental RNGs once; sample choices never use them."""
    from .allocation_signals import configure_data_worker_signals
    configure_data_worker_signals()
    info = get_worker_info()
    if info is None:
        return
    random.seed(info.seed)
    np.random.seed(info.seed % 2**32)
    # Torch DataLoader initializes that worker's torch RNG itself. No main-process
    # model/dropout/noise RNG is reseeded, including when num_workers == 0.


@dataclass(frozen=True)
class LoaderCounts:
    official_training: int
    training_partition: int
    validation_partition: int
    test_partition: int
    training_processed_per_epoch: int
    training_batches: int
    validation_batches: int
    test_batches: int
    batch_size: int


@dataclass(frozen=True)
class LoaderMetadata:
    dataset: str
    num_classes: int
    training_seed: int
    split_identity: str
    normalization_identity: str
    preprocessing_identity: str
    preparation_identity: str
    official_training_identity: str
    official_test_identity: str
    randomness_version: str
    counts: LoaderCounts
    drop_last: bool
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int | None
    data_rng_software: dict

    def to_dict(self) -> dict:
        return deepcopy(asdict(self))


@dataclass
class DataLoaders:
    train: DataLoader
    validation: DataLoader
    test: DataLoader
    metadata: LoaderMetadata
    _sampler: EpochSampler
    _worker_generator: torch.Generator

    def set_epoch(self, epoch: int) -> None:
        """Set only before starting a new epoch, after any previous iterator ends."""
        self._sampler.set_epoch(epoch)
        self._worker_generator.manual_seed(_data_seed(
            "training_workers", self.metadata.dataset, self.metadata.split_identity,
            self.metadata.training_seed, epoch,
        ))

    def continuation_metadata(self, next_epoch_index: int) -> dict:
        """Minimal data state at a committed epoch boundary; no mutable RNG cursor."""
        next_epoch_index = _epoch_index(next_epoch_index, include_budget_end=True)
        metadata = self.metadata
        return {
            "format_version": CONTINUATION_VERSION, "boundary": "completed_epoch",
            "next_epoch_index": next_epoch_index, "original_epochs": EPOCHS,
            "dataset": metadata.dataset, "training_seed": metadata.training_seed,
            "split_identity": metadata.split_identity, "normalization_identity": metadata.normalization_identity,
            "preprocessing_identity": metadata.preprocessing_identity,
            "official_training_identity": metadata.official_training_identity,
            "official_test_identity": metadata.official_test_identity,
            "randomness_version": metadata.randomness_version,
            "batch_size": metadata.counts.batch_size, "drop_last": metadata.drop_last,
            "data_rng_software": deepcopy(metadata.data_rng_software),
        }

    def restore_next_epoch(self, record: dict) -> int:
        """Validate the data boundary before reconstructing its next epoch.

        Epoch 200 means no training epoch remains; future trainers must proceed
        to pending clean evaluation rather than start a 201st training epoch.
        """
        if type(record) is not dict or "next_epoch_index" not in record:
            raise DatasetArtifactError("Missing data continuation metadata")
        expected = self.continuation_metadata(record["next_epoch_index"])
        require_keys(record, set(expected), "data continuation")
        if content_identity(record) != content_identity(expected):
            raise DatasetArtifactError("Data continuation metadata is incompatible with these loaders")
        epoch = record["next_epoch_index"]
        if epoch < EPOCHS:
            self.set_epoch(epoch)
        else:
            self._sampler.epoch = None
        return epoch


def build_loaders(config: RunConfig, *, base_dir: str | Path,
                  pin_memory: bool = False, persistent_workers: bool = False) -> DataLoaders:
    """Build read-only full loaders without requiring unrelated training decisions.

    Operational worker count comes from RunConfig. CPU samples only; optional
    pinning and persistent workers apply to all loaders, defaulting explicitly
    to false. Prefetch is two batches per worker when workers are present.
    """
    if not isinstance(config, RunConfig):
        raise DatasetArtifactError("build_loaders requires a validated RunConfig")
    if type(pin_memory) is not bool or type(persistent_workers) is not bool:
        raise DatasetArtifactError("pin_memory and persistent_workers must be bool")
    science = config.scientific
    seed = require_seed(science["training_seed"], "scientific.training_seed for data loading")
    drop_last = science["training"]["drop_last"]
    if type(drop_last) is not bool:
        raise DatasetArtifactError("Resolve scientific.training.drop_last explicitly before building loaders")
    workers = config.execution["num_workers"]
    if persistent_workers and workers == 0:
        raise DatasetArtifactError("persistent_workers=True requires execution.num_workers > 0")
    prepared = load_prepared_dataset(config, base_dir=base_dir)
    split = prepared.split
    normalization = prepared.normalization
    mean, std = normalization["mean"], normalization["std"]
    train_view = TrainingView(prepared.sources.training, split["train_indices"], mean, std,
                              science["dataset"], split["identity"], seed)
    val_view = EvaluationView(prepared.sources.training, split["validation_indices"], mean, std)
    test_view = EvaluationView(prepared.sources.test, range(TEST_SIZE), mean, std)
    sampler = EpochSampler(split["train_indices"], science["dataset"], split["identity"], seed)
    worker_generator = _generator(_data_seed("training_workers", science["dataset"], split["identity"], seed, 0))
    settings = {
        "batch_size": science["training"]["batch_size"], "num_workers": workers,
        "pin_memory": pin_memory, "persistent_workers": persistent_workers,
        "worker_init_fn": _initialize_worker, "prefetch_factor": 2 if workers else None,
    }
    train = DataLoader(train_view, sampler=sampler, shuffle=False, drop_last=drop_last,
                       generator=worker_generator, **settings)
    validation = DataLoader(val_view, shuffle=False, drop_last=False, generator=_generator(
        _data_seed("validation_workers", science["dataset"], split["identity"], seed, 0)), **settings)
    test = DataLoader(test_view, shuffle=False, drop_last=False, generator=_generator(
        _data_seed("test_workers", science["dataset"], split["identity"], seed, 0)), **settings)
    counts = LoaderCounts(
        official_training=TRAIN_SIZE + VALIDATION_SIZE, training_partition=len(train_view),
        validation_partition=len(val_view), test_partition=len(test_view),
        training_processed_per_epoch=(len(train) * BATCH_SIZE if drop_last else len(train_view)),
        training_batches=len(train), validation_batches=len(validation), test_batches=len(test),
        batch_size=science["training"]["batch_size"],
    )
    metadata = LoaderMetadata(
        dataset=science["dataset"], num_classes=DATASETS[science["dataset"]], training_seed=seed,
        split_identity=split["identity"], normalization_identity=normalization["identity"],
        preprocessing_identity=prepared.manifest["preprocessing_identity"],
        preparation_identity=prepared.manifest["identity"],
        official_training_identity=prepared.sources.training_record["identity"],
        official_test_identity=prepared.sources.test_record["identity"], randomness_version=DATA_RANDOMNESS_VERSION,
        counts=counts, drop_last=drop_last, num_workers=workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers, prefetch_factor=settings["prefetch_factor"],
        data_rng_software={"torch": str(torch.__version__)},
    )
    return DataLoaders(train, validation, test, metadata, sampler, worker_generator)


@dataclass(frozen=True)
class TestData:
    loader: DataLoader
    metadata: dict


def build_test_loader(config: RunConfig, *, base_dir: str | Path) -> TestData:
    """Read-only full test, without training views/sampler/iterators or RNG restore."""
    prepared = load_prepared_dataset(config, base_dir=base_dir)
    science, workers = config.scientific, config.execution["num_workers"]
    normalization = prepared.normalization
    view = EvaluationView(prepared.sources.test, range(TEST_SIZE), normalization["mean"], normalization["std"])
    # A private loader generator only seeds incidental workers; samples are deterministic.
    generator = _generator(_data_seed("test_workers", science["dataset"], prepared.split["identity"],
                                      require_seed(science["training_seed"], "training checkpoint seed"), 0))
    loader = DataLoader(view, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=workers,
                        generator=generator, worker_init_fn=_initialize_worker, pin_memory=False,
                        persistent_workers=False, prefetch_factor=2 if workers else None)
    return TestData(loader, {"dataset": science["dataset"], "num_classes": DATASETS[science["dataset"]],
        "split_identity": prepared.split["identity"], "normalization_identity": normalization["identity"],
        "preprocessing_identity": prepared.manifest["preprocessing_identity"],
        "preparation_identity": prepared.manifest["identity"],
        "official_training_identity": prepared.sources.training_record["identity"],
        "official_test_identity": prepared.sources.test_record["identity"],
        "test_examples": TEST_SIZE, "test_batches": len(loader), "batch_size": BATCH_SIZE,
        "num_workers": workers, "torch": str(torch.__version__)})
