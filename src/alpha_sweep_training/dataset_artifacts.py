"""Shared split/normalization validation; generation is preparation-only."""

import numpy as np

from .data_contract import (
    DatasetArtifactError, NormalizationRequest, NORMALIZATION_VERSION,
    SPLIT_ALGORITHM, SPLIT_VERSION, content_identity, require_keys,
    require_provenance, require_seed, require_version, rgb_values,
)
from .spec import TRAIN_SIZE, VALIDATION_SIZE

_SPLIT_PAYLOAD_KEYS = {"format_version", "dataset", "official_training_identity",
                       "train_indices", "validation_indices"}
_NORMALIZATION_PAYLOAD_KEYS = {
    "format_version", "dataset", "official_training_identity", "split_identity",
    "policy", "mean", "std", "source", "calculation",
}


def _payload(document: dict, keys: set[str]) -> dict:
    return {key: document[key] for key in keys}


def validate_split(document: dict, dataset: str, training_identity: str) -> dict:
    require_keys(document, _SPLIT_PAYLOAD_KEYS | {"identity", "generation", "provenance"}, "split")
    require_version(document["format_version"], SPLIT_VERSION, "split")
    if document["dataset"] != dataset or document["official_training_identity"] != training_identity:
        raise DatasetArtifactError("Split dataset/official-training-source association is incompatible")
    partitions = []
    for key, size in (("train_indices", TRAIN_SIZE), ("validation_indices", VALIDATION_SIZE)):
        values = document[key]
        if type(values) is not list or len(values) != size:
            raise DatasetArtifactError(f"split.{key}: expected exactly {size} indices")
        if any(type(value) is not int or not 0 <= value < TRAIN_SIZE + VALIDATION_SIZE for value in values):
            raise DatasetArtifactError(f"split.{key}: require integer indices in [0, 50000)")
        unique = set(values)
        if len(unique) != size:
            raise DatasetArtifactError(f"split.{key}: duplicate indices")
        partitions.append(unique)
    if partitions[0] & partitions[1] or partitions[0] | partitions[1] != set(range(TRAIN_SIZE + VALIDATION_SIZE)):
        raise DatasetArtifactError("Split must be disjoint and cover all 50,000 official training examples")
    generation = require_keys(document["generation"], {"seed", "algorithm"}, "split generation")
    require_seed(generation["seed"], "split generation seed")
    if generation["algorithm"] != SPLIT_ALGORITHM:
        raise DatasetArtifactError("Unsupported split generation algorithm")
    if document["identity"] != content_identity(_payload(document, _SPLIT_PAYLOAD_KEYS)):
        raise DatasetArtifactError("Split content fingerprint is invalid")
    require_provenance(document["provenance"], "split")
    return document


def generate_split(dataset: str, training_identity: str, seed: int, provenance: dict) -> dict:
    seed = require_seed(seed, "split generation seed (--split-seed)")
    # No stratification: one seeded permutation; persist the resulting order.
    indices = np.random.Generator(np.random.PCG64(seed)).permutation(TRAIN_SIZE + VALIDATION_SIZE).tolist()
    payload = {"format_version": SPLIT_VERSION, "dataset": dataset,
               "official_training_identity": training_identity,
               "train_indices": indices[:TRAIN_SIZE], "validation_indices": indices[TRAIN_SIZE:]}
    document = {**payload, "identity": content_identity(payload),
                "generation": {"seed": seed, "algorithm": SPLIT_ALGORITHM}, "provenance": provenance}
    return validate_split(document, dataset, training_identity)


def calculation_contract(policy: str) -> dict:
    if policy == "explicit":
        return {"algorithm": "supplied_rgb_constants", "pixel_scale": "uint8_div_255"}
    return {
        "algorithm": "chan_image_blocks-v1", "accumulation_dtype": "float64",
        "order": "persisted_training_indices_then_HWC_C_order",
        "pixel_scale": "uint8_div_255", "correction": 0,
        "images": TRAIN_SIZE, "pixels_per_channel": TRAIN_SIZE * 32 * 32,
    }


def _derived_statistics(images: np.ndarray, indices: list[int]) -> tuple[list[float], list[float]]:
    """Chan's merge of per-image moments in persisted index order, float64 CPU.

    Each image contributes 1024 values per channel. No augmentation, validation,
    test images, GPU, or additional complete image-array copy is used.
    """
    count = 0
    mean = np.zeros(3, dtype=np.float64)
    m2 = np.zeros(3, dtype=np.float64)
    for index in indices:
        pixels = images[index].astype(np.float64).reshape(-1, 3) / np.float64(255.0)
        block_count = pixels.shape[0]
        block_mean = pixels.mean(axis=0, dtype=np.float64)
        block_m2 = np.sum((pixels - block_mean) ** 2, axis=0, dtype=np.float64)
        total = count + block_count
        delta = block_mean - mean
        m2 += block_m2 + delta * delta * (count * block_count / total)
        mean += delta * (block_count / total)
        count = total
    return mean.tolist(), np.sqrt(m2 / count).tolist()


def generate_normalization(dataset: str, training_identity: str, split: dict,
                           images: np.ndarray, request: NormalizationRequest, provenance: dict) -> dict:
    request.validate()
    validate_split(split, dataset, training_identity)
    if request.policy == "explicit":
        mean = rgb_values(request.mean, "normalization mean")
        std = rgb_values(request.std, "normalization std", positive=True)
        source = request.source.strip()
    else:
        mean, std = _derived_statistics(images, split["train_indices"])
        source = "Population RGB moments of the persisted 45000-image training partition; no augmentation"
    payload = {
        "format_version": NORMALIZATION_VERSION, "dataset": dataset,
        "official_training_identity": training_identity, "split_identity": split["identity"],
        "policy": request.policy, "mean": rgb_values(mean, "normalization mean"),
        "std": rgb_values(std, "normalization std", positive=True), "source": source,
        "calculation": calculation_contract(request.policy),
    }
    document = {**payload, "identity": content_identity(payload), "provenance": provenance}
    return validate_normalization(document, dataset, training_identity, split["identity"])


def validate_normalization(document: dict, dataset: str, training_identity: str, split_identity: str) -> dict:
    require_keys(document, _NORMALIZATION_PAYLOAD_KEYS | {"identity", "provenance"}, "normalization")
    require_version(document["format_version"], NORMALIZATION_VERSION, "normalization")
    if (document["dataset"] != dataset or document["official_training_identity"] != training_identity
            or document["split_identity"] != split_identity):
        raise DatasetArtifactError("Normalization dataset/source/split association is incompatible")
    if document["policy"] not in ("explicit", "derive_training"):
        raise DatasetArtifactError("Unsupported normalization policy")
    if type(document["mean"]) is not list or type(document["std"]) is not list:
        raise DatasetArtifactError("Persisted normalization constants must be RGB lists")
    rgb_values(document["mean"], "normalization mean")
    rgb_values(document["std"], "normalization std", positive=True)
    if type(document["source"]) is not str or not document["source"].strip():
        raise DatasetArtifactError("Missing normalization source")
    # Compare canonical encodings as well as values, rejecting bool/int aliases.
    if content_identity(document["calculation"]) != content_identity(calculation_contract(document["policy"])):
        raise DatasetArtifactError("Unsupported normalization calculation contract")
    if document["identity"] != content_identity(_payload(document, _NORMALIZATION_PAYLOAD_KEYS)):
        raise DatasetArtifactError("Normalization content fingerprint is invalid")
    require_provenance(document["provenance"], "normalization")
    return document


def require_normalization_request(document: dict, request: NormalizationRequest) -> None:
    request.validate()
    if document["policy"] != request.policy:
        raise DatasetArtifactError("Existing normalization policy conflicts with the explicit preparation request")
    if request.policy == "explicit":
        if (document["mean"] != rgb_values(request.mean, "normalization mean")
                or document["std"] != rgb_values(request.std, "normalization std", positive=True)
                or document["source"] != request.source.strip()):
            raise DatasetArtifactError("Existing normalization constants/source conflict; refusing replacement")


def normalization_config(document: dict) -> dict:
    """Keep the human source in the artifact; bind config source to its content."""
    return {"mean": list(document["mean"]), "std": list(document["std"]),
            "source": "normalization-v1:" + document["identity"]}
