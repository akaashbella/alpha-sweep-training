"""Explicit shared CIFAR preparation CLI, implemented for later HPC execution."""

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import socket
from typing import Iterator
import uuid

import numpy as np
import torch
import torchvision

from .cifar_source import load_official_sources
from .config import ConfigError, RunConfig, load_config
from .data_contract import (
    ARTIFACT_FOLDER, PREPARATION_VERSION, PREPROCESSING, DatasetArtifactError,
    NormalizationRequest, content_identity, file_digest, require_seed,
)
from .dataset_artifacts import (
    generate_normalization, generate_split, normalization_config, require_normalization_request,
)
from .prepared import (
    ARTIFACT_FILES, preparation_directory, preparation_identity, preparation_lock_path, read_prepared_bundle,
)
from .serialization import atomic_write_json, read_json, save_resolved_config, strict_json_text
from .spec import DATASETS


@dataclass(frozen=True)
class PreparationResult:
    config: RunConfig
    config_path: Path
    manifest_path: Path
    reused: bool


def _provenance() -> dict:
    directory = Path(__file__).parent
    return {
        "format_version": 1,
        "software": {"python": platform.python_version(), "torch": str(torch.__version__),
                     "torchvision": str(torchvision.__version__), "numpy": np.__version__},
        "implementation_sha256": {name: file_digest(directory / name) for name in
            ("prepare.py", "prepared.py", "cifar_source.py", "dataset_artifacts.py", "data_contract.py",
             "spec.py", "config.py", "serialization.py")},
    }


@contextmanager
def _preparation_lock(root: Path, dataset: str) -> Iterator[None]:
    parent = root / ARTIFACT_FOLDER
    if parent.is_symlink() or getattr(parent, "is_junction", lambda: False)():
        raise DatasetArtifactError(f"Refusing a linked preparation artifact parent: {parent}")
    parent.mkdir(parents=True, exist_ok=True)
    lock = preparation_lock_path(root, dataset)
    token = uuid.uuid4().hex
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise DatasetArtifactError(f"Preparation lock exists: {lock}; inspect its owner before manual recovery") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(strict_json_text({"dataset": dataset, "token": token, "hostname": socket.gethostname(),
                                           "pid": os.getpid(), "created_at_utc": datetime.now(timezone.utc).isoformat()}))
            stream.flush()
            os.fsync(stream.fileno())
        yield
    finally:
        # A damaged/changed lock is retained, never stolen or automatically expired.
        try:
            owner = read_json(lock) if not lock.is_symlink() and lock.is_file() else None
        except (OSError, ValueError):
            owner = None
        if isinstance(owner, dict) and owner.get("token") == token:
            lock.unlink()


def _resolved_data_config(config: RunConfig, root: Path, split: dict, normalization: dict) -> RunConfig:
    document = config.to_dict()
    document["scientific"]["data"]["split"]["identity"] = split["identity"]
    document["scientific"]["data"]["normalization"] = normalization_config(normalization)
    document["execution"]["storage"]["data_root"] = str(root)
    document["execution"]["storage"]["split_file"] = str(preparation_directory(root, config.scientific["dataset"]) / "split.json")
    return RunConfig(document)


def _declared_data_compatible(config: RunConfig, split: dict, normalization: dict) -> None:
    data = config.scientific["data"]
    if data["split"]["identity"] not in (None, split["identity"]):
        raise DatasetArtifactError("Declared split identity conflicts with the prepared split")
    for key in ("mean", "std"):
        if data["normalization"][key] is not None and data["normalization"][key] != normalization[key]:
            raise DatasetArtifactError(f"Declared normalization {key} conflicts with prepared constants")
    source = data["normalization"]["source"]
    allowed = (None, normalization_config(normalization)["source"], normalization["source"])
    if source not in allowed:
        if not (source == "derive_training" and normalization["policy"] == "derive_training"
                and data["normalization"]["mean"] is None and data["normalization"]["std"] is None):
            raise DatasetArtifactError("Declared normalization source conflicts with the selected preparation policy")


def prepare_dataset(
    config: RunConfig,
    *,
    base_dir: str | Path,
    normalization: NormalizationRequest,
    split_seed: int | None = None,
    download: bool = False,
) -> PreparationResult:
    """Prepare/reuse one full official dataset without unrelated research readiness.

    Requires an explicit normalization policy and, for a new split, an explicit
    preparation seed. Never uses training_seed to select a split. No overwrite
    or repair mode exists. Staged/incompatible artifacts require manual recovery.
    """
    if not isinstance(config, RunConfig) or not isinstance(normalization, NormalizationRequest):
        raise DatasetArtifactError("Provide a validated RunConfig and explicit NormalizationRequest")
    normalization.validate()
    if type(download) is not bool:
        raise DatasetArtifactError("download must be explicitly true or false")
    if split_seed is not None:
        require_seed(split_seed, "split generation seed (--split-seed)")
    dataset = config.scientific["dataset"]
    base = Path(base_dir).expanduser().resolve()
    root = (base / Path(config.execution["storage"]["data_root"]).expanduser()).resolve()
    directory = preparation_directory(root, dataset)
    declared_split_file = config.execution["storage"]["split_file"]
    if declared_split_file is not None:
        selected = (base / Path(declared_split_file).expanduser()).resolve()
        if selected != (directory / "split.json").resolve():
            raise DatasetArtifactError("Preparation must use the single dataset-level split location for this data root")
    if not directory.exists() and split_seed is None:
        raise DatasetArtifactError("A new split requires --split-seed <approved-preparation-integer>; training seeds are unrelated")
    if not root.is_dir() and not download:
        raise DatasetArtifactError(f"Official data root must exist: {root}; acquisition requires explicit --download")
    with _preparation_lock(root, dataset):
        stages = list(directory.parent.glob(f".{dataset}.stage-*"))
        if stages:
            raise DatasetArtifactError(f"Incomplete preparation staging directories require manual recovery: {stages}")
        if directory.exists() or directory.is_symlink():
            # Finalized preparations never acquire or alter raw/shared files.
            sources = load_official_sources(dataset, root, download=False)
            prepared = read_prepared_bundle(dataset, root, sources)
            if split_seed is not None and prepared.split["generation"]["seed"] != split_seed:
                raise DatasetArtifactError("Requested split seed conflicts with the existing persisted split; refusing replacement")
            require_normalization_request(prepared.normalization, normalization)
            _declared_data_compatible(config, prepared.split, prepared.normalization)
            # Preserve the original shared config; return effective settings for
            # this caller without rewriting another architecture's declaration.
            resolved = _resolved_data_config(config, root, prepared.split, prepared.normalization)
            return PreparationResult(resolved, directory / "dataset_config.yaml", directory / "manifest.json", True)

        provenance = _provenance()
        sources = load_official_sources(dataset, root, download=download)
        split = generate_split(dataset, sources.training_record["identity"], split_seed, provenance)
        norm = generate_normalization(dataset, sources.training_record["identity"], split,
                                      sources.training.data, normalization, provenance)
        _declared_data_compatible(config, split, norm)
        resolved = _resolved_data_config(config, root, split, norm)
        # Publish a complete artifact directory through a same-parent rename.
        # Any failure retains staging files and stops a later automatic retry.
        stage = directory.parent / f".{dataset}.stage-{uuid.uuid4().hex}"
        stage.mkdir(exist_ok=False)
        atomic_write_json(stage / "split.json", split)
        atomic_write_json(stage / "normalization.json", norm)
        save_resolved_config(stage / "dataset_config.yaml", resolved)
        manifest = {
            "format_version": PREPARATION_VERSION, "status": "prepared", "dataset": dataset,
            "training_source": sources.training_record, "test_source": sources.test_record,
            "split_identity": split["identity"], "normalization_identity": norm["identity"],
            "preprocessing_identity": content_identity(PREPROCESSING), "preprocessing": PREPROCESSING,
            "files": {name: file_digest(stage / name) for name in ARTIFACT_FILES},
            "provenance": provenance, "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
            "identity": preparation_identity(dataset, sources.training_record["identity"], sources.test_record["identity"],
                                             split["identity"], norm["identity"]),
        }
        atomic_write_json(stage / "manifest.json", manifest)
        if directory.exists() or directory.is_symlink():
            raise DatasetArtifactError(f"Preparation target appeared during staging; refusing replacement: {directory}")
        stage.rename(directory)
        if os.name == "posix":
            descriptor = os.open(directory.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        return PreparationResult(resolved, directory / "dataset_config.yaml", directory / "manifest.json", False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare one full official CIFAR dataset for the main alpha experiment")
    parser.add_argument("--config", required=True, type=Path, help="Main-experiment base YAML; pending training decisions are allowed")
    parser.add_argument("--dataset", required=True, choices=tuple(DATASETS))
    parser.add_argument("--data-root", required=True, type=Path, help="Persistent shared raw/prepared data storage")
    parser.add_argument("--split-seed", type=int, help="Explicit approved preparation seed; required only for a new split")
    parser.add_argument("--normalization", required=True, choices=("explicit", "derive_training"))
    parser.add_argument("--mean", type=float, nargs=3, metavar=("R", "G", "B"))
    parser.add_argument("--std", type=float, nargs=3, metavar=("R", "G", "B"))
    parser.add_argument("--normalization-source", help="Documented source for explicitly supplied RGB constants")
    parser.add_argument("--download", action="store_true", help="Explicitly permit official acquisition/extraction; never repairs existing partial data")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, overrides={
            "scientific.dataset": args.dataset, "scientific.model.num_classes": DATASETS[args.dataset],
            "execution.storage.data_root": str(args.data_root),
            "execution.storage.split_file": str(args.data_root / ARTIFACT_FOLDER / args.dataset / "split.json"),
        })
        request = NormalizationRequest(args.normalization, tuple(args.mean) if args.mean is not None else None,
                                       tuple(args.std) if args.std is not None else None, args.normalization_source)
        result = prepare_dataset(config, base_dir=Path.cwd(), normalization=request,
                                 split_seed=args.split_seed, download=args.download)
    except (ConfigError, DatasetArtifactError, OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"Preparation failed: {exc}\n")
    print(f"{'Reused' if result.reused else 'Prepared'} {args.dataset}: {result.manifest_path}")
    print(f"Shared loader-compatible configuration (original declaration on reuse): {result.config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
