"""Official TorchVision sources, integrity checks, and content-based identities."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

from .data_contract import (DatasetArtifactError, SOURCE_VERSION, content_identity, file_digest,
                            read_artifact_json, require_keys, require_version, ARTIFACT_FOLDER)
from .spec import DATASETS, TEST_SIZE, TRAIN_SIZE, VALIDATION_SIZE


@dataclass(frozen=True)
class OfficialSources:
    training: CIFAR10 | CIFAR100
    test: CIFAR10 | CIFAR100
    training_record: dict
    test_record: dict


@dataclass(frozen=True)
class SourceMetadata:
    """Verified prepared source descriptors, without unpickling image arrays."""
    training_record: dict
    test_record: dict


def read_source_metadata(dataset: str, root: Path) -> SourceMetadata:
    """Validate descriptors against the same official files/checksums as loaders.

    The preparation descriptor binds decoded-image/label identities to raw file
    digests. Full loaders additionally decode and recompute those identities.
    This path never constructs a dataset, model, loader or CUDA object.
    """
    cls = dataset_class(dataset)
    _check_files(root, cls)
    manifest = read_artifact_json(root / ARTIFACT_FOLDER / dataset / "manifest.json")
    records = []
    for key, partition, length, files in (
        ("training_source", "official_train", TRAIN_SIZE + VALIDATION_SIZE, cls.train_list),
        ("test_source", "official_test", TEST_SIZE, cls.test_list),
    ):
        record = manifest[key]
        require_keys(record, {"format_version", "dataset", "partition", "length", "num_classes",
            "image_shape", "image_encoding", "images_sha256", "label_encoding", "labels_sha256",
            "classes", "files", "identity"}, key)
        require_version(record["format_version"], SOURCE_VERSION, key)
        fixed = {"dataset": dataset, "partition": partition, "length": length,
            "num_classes": DATASETS[dataset], "image_shape": [32, 32, 3],
            "image_encoding": "uint8_HWC_C_order_official_example_order",
            "label_encoding": "uint32_little_endian"}
        if any(content_identity(record[k]) != content_identity(v) for k, v in fixed.items()):
            raise DatasetArtifactError(f"Invalid {key} descriptor")
        for name in ("images_sha256", "labels_sha256"):
            digest = record[name]
            if type(digest) is not str or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise DatasetArtifactError(f"Invalid {key}.{name}")
        classes = record["classes"]
        if (type(classes) is not list or len(classes) != DATASETS[dataset]
                or any(type(name) is not str or not name for name in classes)
                or len(set(classes)) != len(classes)):
            raise DatasetArtifactError(f"Invalid {key} class vocabulary")
        expected = []
        for name, md5 in [*(tuple(item) for item in files), (cls.meta["filename"], cls.meta["md5"])]:
            path = root / cls.base_folder / name
            if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
                raise DatasetArtifactError(f"Linked official source: {path}")
            expected.append({"name": f"{cls.base_folder}/{name}", "official_md5": md5,
                             "sha256": file_digest(path)})
        if content_identity(record["files"]) != content_identity(expected):
            raise DatasetArtifactError(f"Current official files disagree with {key}")
        if record["identity"] != content_identity({k: v for k, v in record.items() if k != "identity"}):
            raise DatasetArtifactError(f"Invalid {key} content identity")
        records.append(record)
    if records[0]["classes"] != records[1]["classes"]:
        raise DatasetArtifactError("Official source class vocabularies disagree")
    return SourceMetadata(*records)


def dataset_class(dataset: str):
    if dataset == "cifar10":
        return CIFAR10
    if dataset == "cifar100":
        return CIFAR100
    raise DatasetArtifactError(f"Unsupported main-experiment dataset: {dataset!r}")


def _required_files(cls) -> list[tuple[str, str]]:
    return [*(tuple(item) for item in cls.train_list),
            *(tuple(item) for item in cls.test_list), (cls.meta["filename"], cls.meta["md5"])]


def _check_files(root: Path, cls) -> None:
    for name, checksum in _required_files(cls):
        path = root / cls.base_folder / name
        if not check_integrity(str(path), md5=checksum):
            raise DatasetArtifactError(
                f"Missing or invalid official CIFAR file: {path}. Prepare shared data first; "
                "existing partial/corrupt files are never repaired automatically."
            )


def _acquire_if_requested(root: Path, cls, download: bool) -> None:
    try:
        _check_files(root, cls)
        return
    except DatasetArtifactError:
        if not download:
            raise
    base = root / cls.base_folder
    archive = root / cls.filename
    if base.exists() or base.is_symlink():
        raise DatasetArtifactError(f"Refusing to extract over an existing incomplete/corrupt official directory: {base}")
    if archive.exists() or archive.is_symlink():
        if not check_integrity(str(archive), md5=cls.tgz_md5):
            raise DatasetArtifactError(f"Refusing to replace an incompatible official archive: {archive}")
    # Only the explicit preparation path reaches download=True. The caller holds
    # the dataset-level lock through acquisition, extraction, and final commit.
    download_and_extract_archive(cls.url, str(root), filename=cls.filename, md5=cls.tgz_md5)
    _check_files(root, cls)


def _validate_structure(source, dataset: str, expected_size: int) -> None:
    if source.data.dtype != np.uint8 or source.data.shape != (expected_size, 32, 32, 3):
        raise DatasetArtifactError(f"{dataset}: expected {expected_size} native 32x32 RGB uint8 images")
    if len(source) != expected_size or len(source.targets) != expected_size:
        raise DatasetArtifactError(f"{dataset}: official image/label counts disagree")
    classes = DATASETS[dataset]
    if (len(source.classes) != classes or len(set(source.classes)) != classes
            or any(type(name) is not str or not name for name in source.classes)):
        raise DatasetArtifactError(f"{dataset}: official class vocabulary disagrees with classifier count")
    if source.class_to_idx != {name: index for index, name in enumerate(source.classes)}:
        raise DatasetArtifactError(f"{dataset}: invalid class-name/label association")
    if any(type(label) is not int or not 0 <= label < classes for label in source.targets):
        raise DatasetArtifactError(f"{dataset}: invalid official class labels")
    if set(source.targets) != set(range(classes)):
        raise DatasetArtifactError(f"{dataset}: official partition does not contain all classes")
    if source.transform is not None or source.target_transform is not None:
        raise DatasetArtifactError("Official source objects must be untransformed views")


def _source_record(source, dataset: str, partition: str, root: Path, cls) -> dict:
    import hashlib

    image_digest = hashlib.sha256()
    # TorchVision's HWC array may be non-contiguous. Hash small C-order blocks,
    # avoiding another complete 50,000-image contiguous copy.
    for start in range(0, len(source), 256):
        image_digest.update(np.ascontiguousarray(source.data[start:start + 256]).tobytes(order="C"))
    labels = np.asarray(source.targets, dtype="<u4")
    label_digest = hashlib.sha256(labels.tobytes(order="C")).hexdigest()
    files = cls.train_list if partition == "official_train" else cls.test_list
    files = [*(tuple(item) for item in files), (cls.meta["filename"], cls.meta["md5"])]
    record = {
        "format_version": SOURCE_VERSION, "dataset": dataset, "partition": partition,
        "length": len(source), "num_classes": DATASETS[dataset], "image_shape": [32, 32, 3],
        "image_encoding": "uint8_HWC_C_order_official_example_order",
        "images_sha256": image_digest.hexdigest(), "label_encoding": "uint32_little_endian",
        "labels_sha256": label_digest, "classes": list(source.classes),
        "files": [
            {"name": f"{cls.base_folder}/{name}", "official_md5": checksum,
             "sha256": file_digest(root / cls.base_folder / name)}
            for name, checksum in files
        ],
    }
    return {**record, "identity": content_identity(record)}


def load_official_sources(dataset: str, root: Path, *, download: bool = False) -> OfficialSources:
    """Load one training and one test source after checking all official checksums.

    Workers/loaders always pass download=False. Preparation is the only caller
    allowed to opt into acquisition, while holding the preparation lock.
    """
    cls = dataset_class(dataset)
    _acquire_if_requested(root, cls, download)
    # Never delegate downloading to either source constructor here.
    training = cls(root=str(root), train=True, transform=None, target_transform=None, download=False)
    test = cls(root=str(root), train=False, transform=None, target_transform=None, download=False)
    _validate_structure(training, dataset, TRAIN_SIZE + VALIDATION_SIZE)
    _validate_structure(test, dataset, TEST_SIZE)
    if training.classes != test.classes:
        raise DatasetArtifactError(f"{dataset}: training and test class vocabularies disagree")
    return OfficialSources(
        training=training, test=test,
        training_record=_source_record(training, dataset, "official_train", root, cls),
        test_record=_source_record(test, dataset, "official_test", root, cls),
    )
