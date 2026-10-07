"""Configuration loading and validation for a single experiment run."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class NormalizationConfig:
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    status: str


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    data_root: Path
    split_seed: int
    normalization: NormalizationConfig
    download: bool
    num_workers: int
    pin_memory: bool
    train_subset_size: int | None = None
    validation_subset_size: int | None = None
    test_subset_size: int | None = None


@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    learning_rate: float
    momentum: float
    weight_decay: float


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    warmup_epochs: int
    optimizer: OptimizerConfig


@dataclass(frozen=True)
class PathsConfig:
    checkpoints: Path
    results: Path
    logs: Path


@dataclass(frozen=True)
class RunConfig:
    label: str
    dataset: DatasetConfig
    architecture: str
    num_classes: int
    seed: int
    perturbation_alpha: float
    training: TrainingConfig
    paths: PathsConfig

    @property
    def run_id(self) -> str:
        return (
            f"{self.label}__{self.dataset.name}__{self.architecture}"
            f"__seed-{self.seed}"
        )

    def as_dict(self) -> dict[str, Any]:
        return _paths_to_strings(asdict(self))


def _paths_to_strings(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _paths_to_strings(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_paths_to_strings(item) for item in value]
    return value


def _required(mapping: Mapping[str, Any], key: str, section: str) -> Any:
    if key not in mapping or mapping[key] is None:
        raise ValueError(f"Missing required configuration value: {section}.{key}")
    return mapping[key]


def _triplet(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three numbers")
    return tuple(float(item) for item in value)  # type: ignore[return-value]


def _optional_positive_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive when provided")
    return parsed


def load_config(config_path: str | Path) -> RunConfig:
    """Load one YAML run configuration and reject unsupported science settings."""
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping")

    run_raw = _required(raw, "run", "root")
    dataset_raw = _required(raw, "dataset", "root")
    training_raw = _required(raw, "training", "root")
    optimizer_raw = _required(training_raw, "optimizer", "training")
    paths_raw = _required(raw, "paths", "root")
    normalization_raw = _required(dataset_raw, "normalization", "dataset")

    dataset_name = str(_required(run_raw, "dataset", "run")).lower()
    expected_classes = {"cifar10": 10, "cifar100": 100}
    if dataset_name not in expected_classes:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    architecture = str(_required(run_raw, "architecture", "run")).lower()
    if architecture != "resnet50":
        raise ValueError(
            "Only the clean ResNet50 path is implemented at this stage; "
            f"received {architecture!r}"
        )

    perturbation_alpha = float(run_raw.get("perturbation_alpha", 0.0))
    if perturbation_alpha != 0.0:
        raise ValueError(
            "Gaussian perturbation is intentionally not implemented yet; "
            "perturbation_alpha must be 0.0"
        )

    num_classes = int(run_raw.get("num_classes", expected_classes[dataset_name]))
    if num_classes != expected_classes[dataset_name]:
        raise ValueError(
            f"{dataset_name} requires num_classes={expected_classes[dataset_name]}"
        )

    epochs = int(_required(training_raw, "epochs", "training"))
    warmup_epochs = int(_required(training_raw, "warmup_epochs", "training"))
    if epochs <= 0 or warmup_epochs < 0 or warmup_epochs > epochs:
        raise ValueError("Require epochs > 0 and 0 <= warmup_epochs <= epochs")

    optimizer_name = str(_required(optimizer_raw, "name", "training.optimizer")).lower()
    if optimizer_name != "sgd":
        raise ValueError("ResNet50 must use SGD according to AGENTS.md")

    return RunConfig(
        label=str(_required(run_raw, "label", "run")),
        dataset=DatasetConfig(
            name=dataset_name,
            data_root=Path(_required(dataset_raw, "data_root", "dataset")).expanduser(),
            split_seed=int(_required(dataset_raw, "split_seed", "dataset")),
            normalization=NormalizationConfig(
                mean=_triplet(
                    _required(normalization_raw, "mean", "dataset.normalization"),
                    "dataset.normalization.mean",
                ),
                std=_triplet(
                    _required(normalization_raw, "std", "dataset.normalization"),
                    "dataset.normalization.std",
                ),
                status=str(_required(normalization_raw, "status", "dataset.normalization")),
            ),
            download=bool(dataset_raw.get("download", False)),
            num_workers=int(dataset_raw.get("num_workers", 4)),
            pin_memory=bool(dataset_raw.get("pin_memory", True)),
            train_subset_size=_optional_positive_int(
                dataset_raw.get("train_subset_size"), "dataset.train_subset_size"
            ),
            validation_subset_size=_optional_positive_int(
                dataset_raw.get("validation_subset_size"),
                "dataset.validation_subset_size",
            ),
            test_subset_size=_optional_positive_int(
                dataset_raw.get("test_subset_size"), "dataset.test_subset_size"
            ),
        ),
        architecture=architecture,
        num_classes=num_classes,
        seed=int(_required(run_raw, "seed", "run")),
        perturbation_alpha=perturbation_alpha,
        training=TrainingConfig(
            epochs=epochs,
            batch_size=int(_required(training_raw, "batch_size", "training")),
            warmup_epochs=warmup_epochs,
            optimizer=OptimizerConfig(
                name=optimizer_name,
                learning_rate=float(
                    _required(optimizer_raw, "learning_rate", "training.optimizer")
                ),
                momentum=float(_required(optimizer_raw, "momentum", "training.optimizer")),
                weight_decay=float(
                    _required(optimizer_raw, "weight_decay", "training.optimizer")
                ),
            ),
        ),
        paths=PathsConfig(
            checkpoints=Path(_required(paths_raw, "checkpoints", "paths")).expanduser(),
            results=Path(_required(paths_raw, "results", "paths")).expanduser(),
            logs=Path(_required(paths_raw, "logs", "paths")).expanduser(),
        ),
    )


def replace_storage_paths(
    config: RunConfig,
    *,
    data_root: str | Path | None = None,
    checkpoints: str | Path | None = None,
    results: str | Path | None = None,
    logs: str | Path | None = None,
) -> RunConfig:
    """Return a config with optional CLI/HPC storage overrides."""
    from dataclasses import replace

    dataset = replace(
        config.dataset,
        data_root=Path(data_root).expanduser() if data_root is not None else config.dataset.data_root,
    )
    paths = replace(
        config.paths,
        checkpoints=Path(checkpoints).expanduser() if checkpoints is not None else config.paths.checkpoints,
        results=Path(results).expanduser() if results is not None else config.paths.results,
        logs=Path(logs).expanduser() if logs is not None else config.paths.logs,
    )
    return replace(config, dataset=dataset, paths=paths)

