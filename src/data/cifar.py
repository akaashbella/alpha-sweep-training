"""CIFAR-10/CIFAR-100 data loaders with one fixed train/validation split."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.config import DatasetConfig

CIFAR_TRAIN_SIZE = 50_000
CIFAR_VALIDATION_SIZE = 5_000
CIFAR_TEST_SIZE = 10_000


@dataclass(frozen=True)
class CifarDataLoaders:
    train: DataLoader
    validation: DataLoader
    test: DataLoader
    full_train_size: int
    full_validation_size: int
    full_test_size: int


def make_cifar_split_indices(
    *,
    total_size: int = CIFAR_TRAIN_SIZE,
    validation_size: int = CIFAR_VALIDATION_SIZE,
    split_seed: int,
) -> tuple[list[int], list[int]]:
    """Create a fixed random split controlled only by ``split_seed``."""
    if total_size <= validation_size or validation_size <= 0:
        raise ValueError("Require 0 < validation_size < total_size")
    generator = torch.Generator().manual_seed(split_seed)
    permutation = torch.randperm(total_size, generator=generator).tolist()
    training_size = total_size - validation_size
    return permutation[:training_size], permutation[training_size:]


def _limit(indices: Sequence[int], size: int | None) -> list[int]:
    if size is None:
        return list(indices)
    if size > len(indices):
        raise ValueError(f"Requested subset size {size} exceeds available size {len(indices)}")
    return list(indices[:size])


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_cifar_dataloaders(
    config: DatasetConfig,
    *,
    training_seed: int,
    batch_size: int,
) -> CifarDataLoaders:
    """Build native-resolution CIFAR loaders without coupling split and run seeds."""
    normalize = transforms.Normalize(config.normalization.mean, config.normalization.std)
    training_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    evaluation_transform = transforms.Compose([transforms.ToTensor(), normalize])

    dataset_type = {"cifar10": datasets.CIFAR10, "cifar100": datasets.CIFAR100}.get(
        config.name
    )
    if dataset_type is None:
        raise ValueError(f"Unsupported CIFAR dataset: {config.name}")

    training_source = dataset_type(
        root=config.data_root,
        train=True,
        transform=training_transform,
        download=config.download,
    )
    validation_source = dataset_type(
        root=config.data_root,
        train=True,
        transform=evaluation_transform,
        download=False,
    )
    test_source = dataset_type(
        root=config.data_root,
        train=False,
        transform=evaluation_transform,
        download=config.download,
    )
    if len(training_source) != CIFAR_TRAIN_SIZE or len(test_source) != CIFAR_TEST_SIZE:
        raise RuntimeError("Unexpected official CIFAR dataset size")

    training_indices, validation_indices = make_cifar_split_indices(
        split_seed=config.split_seed
    )
    training_subset = Subset(
        training_source, _limit(training_indices, config.train_subset_size)
    )
    validation_subset = Subset(
        validation_source, _limit(validation_indices, config.validation_subset_size)
    )
    test_indices = _limit(range(CIFAR_TEST_SIZE), config.test_subset_size)
    test_subset = Subset(test_source, test_indices)

    loader_generator = torch.Generator().manual_seed(training_seed)
    common = {
        "batch_size": batch_size,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
        "worker_init_fn": _seed_worker,
        "persistent_workers": config.num_workers > 0,
    }
    return CifarDataLoaders(
        train=DataLoader(
            training_subset,
            shuffle=True,
            drop_last=False,
            generator=loader_generator,
            **common,
        ),
        validation=DataLoader(
            validation_subset,
            shuffle=False,
            drop_last=False,
            **common,
        ),
        test=DataLoader(test_subset, shuffle=False, drop_last=False, **common),
        full_train_size=CIFAR_TRAIN_SIZE - CIFAR_VALIDATION_SIZE,
        full_validation_size=CIFAR_VALIDATION_SIZE,
        full_test_size=CIFAR_TEST_SIZE,
    )

