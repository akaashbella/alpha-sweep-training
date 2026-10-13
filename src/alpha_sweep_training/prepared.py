"""Read-only validation of finalized shared preparation artifacts for any run."""

from dataclasses import dataclass
from pathlib import Path

from .cifar_source import OfficialSources, load_official_sources
from .config import ConfigError, RunConfig, load_config
from .data_contract import (
    ARTIFACT_FOLDER, PREPARATION_VERSION, PREPROCESSING, DatasetArtifactError,
    content_identity, file_digest, read_artifact_json, require_keys, require_provenance, require_version,
)
from .dataset_artifacts import normalization_config, validate_normalization, validate_split

ARTIFACT_FILES = ("split.json", "normalization.json", "dataset_config.yaml")


@dataclass(frozen=True)
class PreparedDataset:
    sources: OfficialSources
    split: dict
    normalization: dict
    manifest: dict
    data_root: Path
    artifact_directory: Path


def preparation_directory(data_root: Path, dataset: str) -> Path:
    return data_root / ARTIFACT_FOLDER / dataset


def preparation_lock_path(data_root: Path, dataset: str) -> Path:
    return data_root / ARTIFACT_FOLDER / f".{dataset}.lock"


def _plain_directory(path: Path) -> None:
    if path.is_symlink() or getattr(path, "is_junction", lambda: False)() or not path.is_dir():
        raise DatasetArtifactError(f"Missing or linked preparation directory: {path}")


def _plain_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise DatasetArtifactError(f"Missing, incomplete, or linked preparation artifact: {path}")


def preparation_identity(dataset: str, training_identity: str, test_identity: str,
                         split_identity: str, normalization_identity: str) -> str:
    return content_identity({
        "format_version": PREPARATION_VERSION, "dataset": dataset,
        "official_training_identity": training_identity, "official_test_identity": test_identity,
        "split_identity": split_identity, "normalization_identity": normalization_identity,
        "preprocessing_identity": content_identity(PREPROCESSING),
    })


def read_prepared_bundle(dataset: str, data_root: Path, sources: OfficialSources) -> PreparedDataset:
    """Require a final manifest and validate contents; never generate/repair data."""
    directory = preparation_directory(data_root, dataset)
    _plain_directory(directory.parent)
    _plain_directory(directory)
    manifest_path = directory / "manifest.json"
    _plain_file(manifest_path)
    manifest = read_artifact_json(manifest_path)
    require_keys(manifest, {
        "format_version", "status", "dataset", "training_source", "test_source",
        "split_identity", "normalization_identity", "preprocessing_identity", "preprocessing",
        "files", "provenance", "prepared_at_utc", "identity",
    }, "preparation manifest")
    require_version(manifest["format_version"], PREPARATION_VERSION, "preparation manifest")
    if manifest["status"] != "prepared" or manifest["dataset"] != dataset:
        raise DatasetArtifactError("Preparation is not complete for the selected dataset")
    for key, actual in (("training_source", sources.training_record), ("test_source", sources.test_record)):
        if content_identity(manifest[key]) != content_identity(actual):
            raise DatasetArtifactError(f"Preparation {key} disagrees with current official dataset content")
    if (content_identity(manifest["preprocessing"]) != content_identity(PREPROCESSING)
            or manifest["preprocessing_identity"] != content_identity(PREPROCESSING)):
        raise DatasetArtifactError("Preparation uses an incompatible preprocessing/randomness contract")
    require_provenance(manifest["provenance"], "preparation manifest")
    if type(manifest["prepared_at_utc"]) is not str or not manifest["prepared_at_utc"]:
        raise DatasetArtifactError("Preparation manifest is missing its creation timestamp")
    files = require_keys(manifest["files"], set(ARTIFACT_FILES), "manifest files")
    for name in ARTIFACT_FILES:
        path = directory / name
        _plain_file(path)
        if type(files[name]) is not str or file_digest(path) != files[name]:
            raise DatasetArtifactError(f"Preparation artifact checksum mismatch: {path}")
    split = validate_split(read_artifact_json(directory / "split.json"), dataset, sources.training_record["identity"])
    normalization = validate_normalization(read_artifact_json(directory / "normalization.json"), dataset,
                                            sources.training_record["identity"], split["identity"])
    if (manifest["split_identity"] != split["identity"]
            or manifest["normalization_identity"] != normalization["identity"]
            or manifest["identity"] != preparation_identity(dataset, sources.training_record["identity"],
                    sources.test_record["identity"], split["identity"], normalization["identity"])):
        raise DatasetArtifactError("Preparation manifest identities disagree with its artifacts")
    try:
        saved_config = load_config(directory / "dataset_config.yaml")
    except (ConfigError, OSError) as exc:
        raise DatasetArtifactError(f"Invalid preparation configuration {directory / 'dataset_config.yaml'}: {exc}") from exc
    require_config_association(saved_config, dataset, split, normalization)
    return PreparedDataset(sources, split, normalization, manifest, data_root, directory)


def require_config_association(config: RunConfig, dataset: str, split: dict, normalization: dict) -> None:
    science = config.scientific
    if (science["dataset"] != dataset or science["data"]["split"]["identity"] != split["identity"]
            or science["data"]["normalization"] != normalization_config(normalization)):
        raise DatasetArtifactError("Configuration does not match the prepared dataset/split/normalization identities")


def load_prepared_dataset(config: RunConfig, *, base_dir: str | Path) -> PreparedDataset:
    """Read-only data gate, independent of training/controller research readiness."""
    storage = config.execution["storage"]
    base = Path(base_dir).expanduser().resolve()
    root = (base / Path(storage["data_root"]).expanduser()).resolve()
    dataset = config.scientific["dataset"]
    if storage["split_file"] is None:
        raise DatasetArtifactError("execution.storage.split_file is unresolved; use the prepared dataset configuration")
    selected_split = (base / Path(storage["split_file"]).expanduser()).resolve()
    expected_split = preparation_directory(root, dataset) / "split.json"
    if selected_split != expected_split.resolve():
        raise DatasetArtifactError(f"Use the one shared split for this dataset/data root: {expected_split}")
    lock = preparation_lock_path(root, dataset)
    if lock.exists() or lock.is_symlink():
        raise DatasetArtifactError(f"Preparation lock exists: {lock}; do not load while shared data are being prepared")
    # Check final completion before opening the comparatively expensive sources.
    _plain_directory(preparation_directory(root, dataset).parent)
    _plain_directory(preparation_directory(root, dataset))
    _plain_file(preparation_directory(root, dataset) / "manifest.json")
    sources = load_official_sources(dataset, root, download=False)
    prepared = read_prepared_bundle(dataset, root, sources)
    require_config_association(config, dataset, prepared.split, prepared.normalization)
    if lock.exists() or lock.is_symlink():
        raise DatasetArtifactError(f"Preparation started during artifact inspection: {lock}; retry after it finishes")
    return prepared
