"""Strict research declarations, independent of builder/runtime availability.

Only YAML parsing requires PyYAML. No datasets, torch objects, or project execution
are imported here. A valid development declaration may contain unresolved fields;
the execution boundary must additionally call require_research_ready().
"""

from copy import deepcopy
import math
from pathlib import Path
import re
from typing import Any, Mapping

from .spec import (
    ALPHAS, ARCHITECTURES, DATASETS, SCHEMA_VERSION, SEED_COUNT,
    PREPROCESSING_VERSION, DATA_RANDOMNESS_VERSION, default_document,
)


class ConfigError(ValueError):
    """A declaration violates the versioned research schema."""


class UnresolvedConfigError(ConfigError):
    """A development declaration is not ready to launch a research run."""


# These fields may be null in development; optimizer fields may additionally be
# inactive. Types are checked in every input layer, not only after overriding.
_NULLABLE_TYPES = {
    "scientific.training_seed": int,
    "scientific.training_seeds": list,
    "scientific.data.split.identity": str,
    "scientific.data.normalization.mean": list,
    "scientific.data.normalization.std": list,
    "scientific.data.normalization.source": str,
    "scientific.optimizer.momentum": float,
    "scientific.optimizer.rmsprop_alpha": float,
    "scientific.optimizer.epsilon": float,
    "scientific.optimizer.betas": list,
    "scientific.training.warmup_start_factor": float,
    "scientific.training.cosine_min_lr": float,
    "scientific.training.scheduler_interval": str,
    "scientific.training.drop_last": bool,
    "scientific.precision.mode": str,
    "scientific.precision.float32_matmul_precision": str,
    "scientific.precision.allow_tf32": bool,
    "scientific.precision.deterministic_algorithms": bool,
    "scientific.precision.cudnn_deterministic": bool,
    "scientific.precision.cudnn_benchmark": bool,
    "execution.storage.checkpoint_root": str,
    "execution.storage.result_root": str,
    "execution.storage.log_root": str,
    "execution.storage.split_file": str,
    "execution.slurm.account": str,
}
_NUMERIC_LISTS = {
    "scientific.data.normalization.mean",
    "scientific.data.normalization.std",
    "scientific.optimizer.betas",
}


def _declared_type(value: Any, expected: Any, path: str) -> None:
    field = path.removeprefix("config.")
    if field in _NULLABLE_TYPES and value is None:
        return
    kind = _NULLABLE_TYPES.get(field, type(expected))
    if kind is float:
        _number(value, path)
    elif kind is int:
        _integer(value, path)
    elif kind is str:
        _string(value, path)
    elif kind is list:
        if type(value) is not list:
            raise ConfigError(f"{path}: expected a list")
        for index, item in enumerate(value):
            if field in _NUMERIC_LISTS:
                _number(item, f"{path}[{index}]")
            elif field == "scientific.training_seeds":
                _integer(item, f"{path}[{index}]")
            else:
                _string(item, f"{path}[{index}]")
    elif type(value) is not kind:
        raise ConfigError(f"{path}: expected {kind.__name__}")


def _mapping(value: Any, path: str) -> dict:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path}: expected a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ConfigError(f"{path}: keys must be strings")
    return dict(value)


def _merge(target: dict, layer: Mapping, shape: dict, path: str = "config") -> None:
    """Overlay known fields only; mappings cannot be replaced by scalar values."""
    for key, value in _mapping(layer, path).items():
        here = f"{path}.{key}"
        if key not in shape:
            raise ConfigError(f"{here}: unknown key")
        if isinstance(shape[key], dict):
            _merge(target[key], value, shape[key], here)
        else:
            _declared_type(value, shape[key], here)
            target[key] = deepcopy(value)


def _number(value: Any, path: str) -> float:
    if type(value) not in (int, float):
        raise ConfigError(f"{path}: expected a finite number, not a string or bool")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ConfigError(f"{path}: expected a finite number") from exc
    if not math.isfinite(result):
        raise ConfigError(f"{path}: NaN and infinity are forbidden")
    return result


def _integer(value: Any, path: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ConfigError(f"{path}: expected an integer >= {minimum}")
    return value


def _string(value: Any, path: str) -> str:
    if type(value) is not str or not value.strip():
        raise ConfigError(f"{path}: expected a nonempty string")
    return value


def _choice(value: Any, choices: tuple | dict, path: str) -> str:
    _string(value, path)
    if value not in choices:
        raise ConfigError(f"{path}: expected one of {tuple(choices)}")
    return value


def _fixed(actual: Any, expected: Any, path: str) -> None:
    """Enforce fixed settings without Python's bool/integer equality loophole."""
    if isinstance(expected, dict):
        for key, value in expected.items():
            _fixed(actual[key], value, f"{path}.{key}")
    elif isinstance(expected, list):
        if type(actual) is not list or len(actual) != len(expected):
            raise ConfigError(f"{path}: required research value is {expected!r}")
        for index, (item, value) in enumerate(zip(actual, expected)):
            _fixed(item, value, f"{path}[{index}]")
    elif type(expected) is float:
        if _number(actual, path) != expected:
            raise ConfigError(f"{path}: required research value is {expected!r}")
    elif type(actual) is not type(expected) or actual != expected:
        raise ConfigError(f"{path}: required research value is {expected!r}")


def _seed(value: Any, path: str) -> None:
    if value is not None:
        _integer(value, path)
        if value > 2**32 - 1:
            raise ConfigError(f"{path}: must fit an unsigned 32-bit seed")


def _validate(document: dict) -> None:
    _fixed(document["schema_version"], SCHEMA_VERSION, "schema_version")
    science = document["scientific"]
    dataset = _choice(science["dataset"], DATASETS, "scientific.dataset")
    architecture = _choice(science["architecture"], ARCHITECTURES, "scientific.architecture")
    alpha = _number(science["alpha"], "scientific.alpha")
    if alpha < 0:
        raise ConfigError("scientific.alpha: must be nonnegative")
    if alpha not in ALPHAS:
        raise ConfigError(f"scientific.alpha: main research values are {ALPHAS}")
    science["alpha"] = 0.0 if alpha == 0 else alpha

    _seed(science["training_seed"], "scientific.training_seed")
    seeds = science["training_seeds"]
    if seeds is not None:
        if type(seeds) is not list or len(seeds) != SEED_COUNT:
            raise ConfigError(f"scientific.training_seeds: expected exactly {SEED_COUNT} seed integers or null")
        for index, seed in enumerate(seeds):
            if seed is None:
                raise ConfigError("scientific.training_seeds: use null for the entire pending decision")
            _seed(seed, f"scientific.training_seeds[{index}]")
        if len(set(seeds)) != SEED_COUNT:
            raise ConfigError("scientific.training_seeds: seeds must be distinct")
        if science["training_seed"] is not None and science["training_seed"] not in seeds:
            raise ConfigError("scientific.training_seed: must belong to scientific.training_seeds")

    expected = default_document(dataset, architecture)["scientific"]
    # Everything outside this explicit list is fixed by main-v1, including budget,
    # full split sizes, classifier shape, perturbation, and optimizer-specific fields.
    for key in ("specification_version", "model", "optimizer", "perturbation", "evaluation"):
        _fixed(science[key], expected[key], f"scientific.{key}")
        science[key] = deepcopy(expected[key])
    for key in ("preprocessing_version", "randomness_version",
                "image_size", "train_size", "validation_size", "test_size",
                "training_transforms", "evaluation_transforms"):
        _fixed(science["data"][key], expected["data"][key], f"scientific.data.{key}")
        science["data"][key] = deepcopy(expected["data"][key])
    split = science["data"]["split"]
    _fixed(split["policy"], expected["data"]["split"]["policy"], "scientific.data.split.policy")
    if split["identity"] is not None:
        if type(split["identity"]) is not str or not re.fullmatch(r"sha256:[0-9a-f]{64}", split["identity"]):
            raise ConfigError("scientific.data.split.identity: expected sha256:<64 lowercase hex digits> or null")
    normalization = science["data"]["normalization"]
    for key in ("mean", "std"):
        values = normalization[key]
        if values is not None:
            if type(values) is not list or len(values) != 3:
                raise ConfigError(f"scientific.data.normalization.{key}: expected three RGB values or null")
            values = [_number(value, f"scientific.data.normalization.{key}[{index}]")
                      for index, value in enumerate(values)]
            if key == "std" and any(value <= 0 for value in values):
                raise ConfigError("scientific.data.normalization.std: values must be positive")
            normalization[key] = values
    if normalization["source"] is not None:
        _string(normalization["source"], "scientific.data.normalization.source")

    training = science["training"]
    pending_training = {"drop_last", "warmup_start_factor", "cosine_min_lr", "scheduler_interval"}
    for key in training.keys() - pending_training:
        _fixed(training[key], expected["training"][key], f"scientific.training.{key}")
        training[key] = deepcopy(expected["training"][key])
    if training["drop_last"] is not None and type(training["drop_last"]) is not bool:
        raise ConfigError("scientific.training.drop_last: expected bool or null")
    if training["warmup_start_factor"] is not None:
        factor = _number(training["warmup_start_factor"], "scientific.training.warmup_start_factor")
        if not 0 <= factor <= 1:
            raise ConfigError("scientific.training.warmup_start_factor: must lie in [0, 1]")
        training["warmup_start_factor"] = factor
    if training["cosine_min_lr"] is not None:
        minimum = _number(training["cosine_min_lr"], "scientific.training.cosine_min_lr")
        if not 0 <= minimum <= science["optimizer"]["learning_rate"]:
            raise ConfigError("scientific.training.cosine_min_lr: must lie between zero and the recipe learning rate")
        training["cosine_min_lr"] = minimum
    if training["scheduler_interval"] is not None:
        _choice(training["scheduler_interval"], ("epoch", "optimizer_step"), "scientific.training.scheduler_interval")

    precision = science["precision"]
    for key, choices in {
        "mode": ("float32", "amp_float16", "amp_bfloat16"),
        "float32_matmul_precision": ("highest", "high", "medium"),
    }.items():
        if precision[key] is not None:
            _choice(precision[key], choices, f"scientific.precision.{key}")
    for key in ("allow_tf32", "deterministic_algorithms", "cudnn_deterministic", "cudnn_benchmark"):
        if precision[key] is not None and type(precision[key]) is not bool:
            raise ConfigError(f"scientific.precision.{key}: expected bool or null")

    execution = document["execution"]
    storage = execution["storage"]
    for key in ("output_root", "data_root"):
        _string(storage[key], f"execution.storage.{key}")
    for key in ("checkpoint_root", "result_root", "log_root", "split_file"):
        if storage[key] is not None:
            _string(storage[key], f"execution.storage.{key}")
    _integer(execution["num_workers"], "execution.num_workers")
    _string(execution["device"], "execution.device")
    slurm = execution["slurm"]
    _fixed(slurm["partition"], "dgxh", "execution.slurm.partition")
    _fixed(slurm["gpus"], 1, "execution.slurm.gpus")
    if slurm["account"] is not None:
        _string(slurm["account"], "execution.slurm.account")
    for key in ("cpus", "memory_gb", "max_concurrent"):
        _integer(slurm[key], f"execution.slurm.{key}", minimum=1)
    wall_time = _string(slurm["wall_time"], "execution.slurm.wall_time")
    if not re.fullmatch(r"(?:[0-9]+-)?[0-9]+:[0-5][0-9]:[0-5][0-9]", wall_time):
        raise ConfigError("execution.slurm.wall_time: expected [days-]hours:mm:ss")


class RunConfig:
    """Validated effective settings, exposed through defensive copies.

    Direct construction accepts a full document. Use load_config/from_declaration
    for partial declarations. No component-availability check belongs here.
    """

    __slots__ = ("_document",)

    def __init__(self, document: Mapping):
        document = _migrate_v1(_mapping(document, "config"))
        try:
            science = _mapping(document["scientific"], "scientific")
            dataset = _choice(science["dataset"], DATASETS, "scientific.dataset")
            architecture = _choice(science["architecture"], ARCHITECTURES, "scientific.architecture")
        except KeyError as exc:
            raise ConfigError(f"missing required key: {exc.args[0]}") from exc
        shape = default_document(dataset, architecture)
        # Enforce full-document keys at every depth before accessing fields.
        _require_full_shape(document, shape)
        _validate(document)
        self._document = document

    @property
    def scientific(self) -> dict:
        return deepcopy(self._document["scientific"])

    @property
    def execution(self) -> dict:
        return deepcopy(self._document["execution"])

    def to_dict(self) -> dict:
        return deepcopy(self._document)

    def unresolved_settings(self) -> tuple[str, ...]:
        science = self._document["scientific"]
        fields = {
            "scientific.training_seeds": science["training_seeds"],
            "scientific.training_seed": science["training_seed"],
            "scientific.data.split.identity": science["data"]["split"]["identity"],
        }
        for key, value in science["data"]["normalization"].items():
            fields[f"scientific.data.normalization.{key}"] = value
        for key in ("drop_last", "warmup_start_factor", "cosine_min_lr", "scheduler_interval"):
            fields[f"scientific.training.{key}"] = science["training"][key]
        for key, value in science["precision"].items():
            fields[f"scientific.precision.{key}"] = value
        return tuple(path for path, value in fields.items() if value is None)

    def require_research_ready(self) -> None:
        """Mandatory gate before artifact initialization, training, or submission."""
        pending = self.unresolved_settings()
        if pending:
            raise UnresolvedConfigError("Research settings remain unresolved: " + ", ".join(pending))


def _require_full_shape(document: Mapping, shape: dict, path: str = "config") -> None:
    document = _mapping(document, path)
    unknown = document.keys() - shape.keys()
    missing = shape.keys() - document.keys()
    if unknown or missing:
        raise ConfigError(f"{path}: unknown keys={sorted(unknown)}, missing keys={sorted(missing)}")
    for key, value in shape.items():
        if isinstance(value, dict):
            _require_full_shape(document[key], value, f"{path}.{key}")


def _read_yaml(path: str | Path) -> dict:
    import yaml

    class UniqueKeyLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if type(key) is not str:
                raise ConfigError(f"{path}: YAML mapping keys must be strings")
            if key in result:
                raise ConfigError(f"{path}: duplicate YAML key {key!r}")
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping)
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            value = yaml.load(stream, Loader=UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    return _mapping(value, str(path))


def _override_layer(overrides: Mapping[str, Any]) -> dict:
    layer = {}
    for path, value in _mapping(overrides, "overrides").items():
        parts = path.split(".")
        if len(parts) < 2 or parts[0] not in ("scientific", "execution") or any(not part for part in parts):
            raise ConfigError(f"override {path!r}: use a scientific.* or execution.* field path")
        target = layer
        for part in parts[:-1]:
            if part in target and not isinstance(target[part], dict):
                raise ConfigError(f"override {path!r}: conflicting parent override")
            target = target.setdefault(part, {})
        if parts[-1] in target:
            raise ConfigError(f"override {path!r}: duplicate/conflicting override")
        target[parts[-1]] = deepcopy(value)
    return layer


def _migrate_v1(document: Mapping) -> dict:
    """Explicit additive migration; never resolve a pending scientific decision.

    Version 2 records preprocessing/randomness contracts and spells out tensor
    conversion in the transform lists. All other fields stay unchanged and strict.
    """
    result = deepcopy(dict(document))
    if type(result.get("schema_version")) is int and result["schema_version"] == 1:
        result["schema_version"] = SCHEMA_VERSION
        if "scientific" in result:
            science = _mapping(result["scientific"], "scientific")
            if "data" in science:
                data = _mapping(science["data"], "scientific.data")
                data.setdefault("preprocessing_version", PREPROCESSING_VERSION)
                data.setdefault("randomness_version", DATA_RANDOMNESS_VERSION)
                if data.get("training_transforms") == ["random_crop_padding_4", "random_horizontal_flip", "normalize"]:
                    data["training_transforms"] = ["random_crop_padding_4", "random_horizontal_flip",
                                                   "to_float32_chw_uint8_div_255", "normalize"]
                if data.get("evaluation_transforms") == ["normalize"]:
                    data["evaluation_transforms"] = ["to_float32_chw_uint8_div_255", "normalize"]
                science["data"] = data
            result["scientific"] = science
    return result


def from_declaration(*layers: Mapping, overrides: Mapping[str, Any] | None = None) -> RunConfig:
    """Apply base, dataset, model, then typed CLI overrides, in that order.

    Selector-derived defaults are filled only for omitted fields. Explicit values
    never disappear when a selector changes; an incompatible recipe/classifier is
    rejected and must be overridden explicitly.
    """
    declarations = [_migrate_v1(_mapping(layer, "declaration")) for layer in layers]
    if not declarations or "schema_version" not in declarations[0]:
        raise ConfigError("base declaration must contain schema_version")
    for layer in declarations:
        if "schema_version" in layer:
            _fixed(layer["schema_version"], SCHEMA_VERSION, "schema_version")
    declarations.append(_override_layer({} if overrides is None else overrides))
    selectors = {}
    for layer in declarations:
        if "scientific" in layer:
            scientific = _mapping(layer["scientific"], "scientific")
            for key in ("dataset", "architecture"):
                if key in scientific:
                    selectors[key] = scientific[key]
    if set(selectors) != {"dataset", "architecture"}:
        raise ConfigError("declarations must select scientific.dataset and scientific.architecture")
    dataset = _choice(selectors["dataset"], DATASETS, "scientific.dataset")
    architecture = _choice(selectors["architecture"], ARCHITECTURES, "scientific.architecture")
    document = default_document(dataset, architecture)
    shape = deepcopy(document)
    for layer in declarations:
        _merge(document, layer, shape)
    return RunConfig(document)


def load_config(
    base_path: str | Path,
    *,
    dataset_path: str | Path | None = None,
    model_path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> RunConfig:
    """Load declarations. A saved resolved_config.yaml is a valid base_path."""
    layers = [_read_yaml(base_path)]
    for path in (dataset_path, model_path):
        if path is not None:
            layers.append(_read_yaml(path))
    return from_declaration(*layers, overrides=overrides)
