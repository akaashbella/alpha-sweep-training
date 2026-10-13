"""Strict main-grid documents and selectors, with no compute imports on import."""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

from .config import ConfigError, RunConfig, from_declaration
from .data_contract import content_identity, require_seed
from .identity import canonical_alpha, run_identity, scientific_fingerprint
from .serialization import atomic_write_json, durable_mkdir, fsync_directory, read_json, strict_json_text
from .spec import ALPHAS, ARCHITECTURES, DATASETS, SEED_COUNT, TOTAL_RUNS

MANIFEST_VERSION = 1
GRID_VERSION = "ordered-scientific-tasks-v1"
PHASE_SIZE = TOTAL_RUNS // len(DATASETS)
DATA_KEYS = {"split_identity", "normalization_identity", "preprocessing_identity",
             "preparation_identity", "official_training_identity", "official_test_identity"}
TASK_KEYS = {"format_version", "grid_id", "phase", "global_index", "phase_index", "dataset",
             "architecture", "alpha", "canonical_alpha", "training_seed", "run_id",
             "scientific_fingerprint", "config", "prepared_data"}
FULL_KEYS = {"format_version", "kind", "grid_version", "grid_id", "seeds", "tasks", "document_sha256"}
PHASE_KEYS = {"format_version", "kind", "grid_id", "parent_document_sha256", "phase",
              "dataset", "tasks", "document_sha256"}


class ManifestError(ConfigError):
    pass


def _exact(value, keys, label):
    if type(value) is not dict or set(value) != keys:
        raise ManifestError(f"{label}: require exactly {sorted(keys)}")
    return value


def _same(actual, expected, label):
    if strict_json_text(actual) != strict_json_text(expected):
        raise ManifestError(f"Inconsistent {label}")


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _sealed(document):
    return {**document, "document_sha256": _digest(document)}


def _check_seal(document):
    _same(document["document_sha256"], _digest({k: v for k, v in document.items()
                                             if k != "document_sha256"}), "document integrity digest")


def seed_plan(seeds):
    if type(seeds) is not list or len(seeds) != SEED_COUNT:
        raise ManifestError("Supply the explicit scientific.training_seeds plan: three distinct uint32 integers")
    for seed in seeds:
        require_seed(seed, "scientific.training_seeds")
    if len(set(seeds)) != SEED_COUNT:
        raise ManifestError("Training seed plan must contain three distinct seeds in approved order")
    return list(seeds)


def combinations(seeds):
    """The single ordering definition; never sorts the supplied seed plan."""
    for phase, dataset in enumerate(DATASETS, start=1):
        for architecture in ARCHITECTURES:
            for alpha in ALPHAS:
                for seed in seeds:
                    yield phase, dataset, architecture, alpha, seed


@dataclass(frozen=True)
class Task:
    grid_id: str
    phase: int
    global_index: int
    phase_index: int
    config: RunConfig
    # Keep nested identity data private and return copies.
    _prepared_data: dict

    @property
    def run_id(self):
        return run_identity(self.config)

    @property
    def fingerprint(self):
        return scientific_fingerprint(self.config)

    @property
    def prepared_data(self):
        return deepcopy(self._prepared_data)

    def to_dict(self):
        science = self.config.scientific
        return {"format_version": MANIFEST_VERSION, "grid_id": self.grid_id,
            "phase": self.phase, "global_index": self.global_index, "phase_index": self.phase_index,
            "dataset": science["dataset"], "architecture": science["architecture"],
            "alpha": science["alpha"], "canonical_alpha": canonical_alpha(science["alpha"]),
            "training_seed": science["training_seed"], "run_id": self.run_id,
            "scientific_fingerprint": self.fingerprint, "config": self.config.to_dict(),
            "prepared_data": self.prepared_data}


def _grid_id(tasks):
    return _digest({"grid_version": GRID_VERSION, "manifest_version": MANIFEST_VERSION,
                   "tasks": [{"run_id": t.run_id, "scientific_fingerprint": t.fingerprint} for t in tasks]})


def validate_prepared_identity(config, identities):
    """Bind descriptors to existing preparation identity definitions."""
    from .prepared import preparation_identity
    from .data_contract import PREPROCESSING
    _exact(identities, DATA_KEYS, "prepared data identities")
    for key, value in identities.items():
        if (type(value) is not str or not value.startswith("sha256:") or len(value) != 71
                or any(c not in "0123456789abcdef" for c in value[7:])):
            raise ManifestError(f"Invalid prepared_data.{key}")
    science = config.scientific
    _same(identities["split_identity"], science["data"]["split"]["identity"], "prepared split")
    _same("normalization-v1:" + identities["normalization_identity"],
          science["data"]["normalization"]["source"], "prepared normalization")
    _same(identities["preprocessing_identity"], content_identity(PREPROCESSING), "preprocessing")
    _same(identities["preparation_identity"], preparation_identity(science["dataset"],
          identities["official_training_identity"], identities["official_test_identity"],
          identities["split_identity"], identities["normalization_identity"]), "preparation")


@dataclass(frozen=True)
class Manifest:
    tasks: tuple[Task, ...]
    seeds: tuple[int, ...]
    grid_id: str

    def to_dict(self):
        return _sealed({"format_version": MANIFEST_VERSION, "kind": "full", "grid_version": GRID_VERSION,
                        "grid_id": self.grid_id, "seeds": list(self.seeds),
                        "tasks": [task.to_dict() for task in self.tasks]})

    @property
    def document_sha256(self):
        return self.to_dict()["document_sha256"]

    def phase_view(self, phase: int):
        phase = require_phase(phase)
        return _sealed({"format_version": MANIFEST_VERSION, "kind": "phase", "grid_id": self.grid_id,
            "parent_document_sha256": self.document_sha256, "phase": phase,
            "dataset": tuple(DATASETS)[phase - 1], "tasks": [
                {"global_index": t.global_index, "phase_index": t.phase_index,
                 "run_id": t.run_id, "scientific_fingerprint": t.fingerprint}
                for t in self.tasks if t.phase == phase]})


def manifest_from_dict(document) -> Manifest:
    document = _exact(document, FULL_KEYS, "full manifest")
    _same(document["format_version"], MANIFEST_VERSION, "manifest format version (no implicit migration)")
    _same(document["kind"], "full", "manifest kind")
    _same(document["grid_version"], GRID_VERSION, "grid version")
    _check_seal(document)
    seeds = seed_plan(document["seeds"])
    rows = document["tasks"]
    if type(rows) is not list or len(rows) != TOTAL_RUNS:
        raise ManifestError(f"Full manifest requires exactly {TOTAL_RUNS} tasks")
    tasks, ids, recipes, dataset_data = [], set(), {}, {}
    for index, (row, combo) in enumerate(zip(rows, combinations(seeds))):
        _exact(row, TASK_KEYS, f"task {index}")
        phase, dataset, architecture, alpha, seed = combo
        config = RunConfig(row["config"])
        config.require_research_ready()
        if config.execution["storage"]["split_file"] is None:
            raise ManifestError(f"Task {index}: execution.storage.split_file requires finalized preparation")
        task = Task(document["grid_id"], phase, index, index % PHASE_SIZE, config, deepcopy(row["prepared_data"]))
        expected_science = {"dataset": dataset, "architecture": architecture, "alpha": alpha,
                            "training_seed": seed, "training_seeds": seeds}
        for key, value in expected_science.items():
            _same(config.scientific[key], value, f"task {index} scientific.{key}/Cartesian order")
        validate_prepared_identity(config, task.prepared_data)
        _same(row, task.to_dict(), f"task {index} linkage/indices/identity/config")
        if task.run_id in ids:
            raise ManifestError("Duplicate task run ID")
        ids.add(task.run_id)
        science = config.scientific
        del science["alpha"], science["training_seed"]
        recipe_key = (dataset, architecture)
        if recipe_key in recipes:
            _same(science, recipes[recipe_key], f"fixed recipe for {recipe_key}")
        recipes[recipe_key] = science
        if dataset in dataset_data:
            _same({"data": config.scientific["data"], "prepared": task.prepared_data},
                  dataset_data[dataset], f"shared prepared data for {dataset}")
        dataset_data[dataset] = {"data": config.scientific["data"], "prepared": task.prepared_data}
        tasks.append(task)
    _same(document["grid_id"], _grid_id(tasks), "scientific grid identity")
    return Manifest(tuple(tasks), tuple(seeds), document["grid_id"])


def load_manifest(path) -> Manifest:
    return manifest_from_dict(read_json(path))


def load_phase(path, parent: Manifest) -> tuple[Task, ...]:
    parent = manifest_from_dict(parent.to_dict())
    return validate_phase_document(read_json(path), parent)


def validate_phase_document(document, parent: Manifest) -> tuple[Task, ...]:
    parent = manifest_from_dict(parent.to_dict())
    document = _exact(document, PHASE_KEYS, "phase view")
    _check_seal(document)
    phase = require_phase(document["phase"])
    _same(document, parent.phase_view(phase), "phase view/parent linkage")
    return tuple(t for t in parent.tasks if t.phase == phase)


def build_manifest(dataset_configs: Mapping[str, RunConfig], model_declarations: Mapping[str, Mapping],
                   *, seeds: list[int], base_dir) -> Manifest:
    """Prepared base -> architecture declaration -> alpha/seed, never raw dataset overlay."""
    from .prepared import load_prepared_dataset
    seeds = seed_plan(seeds)
    if set(dataset_configs) != set(DATASETS) or set(model_declarations) != set(ARCHITECTURES):
        raise ManifestError("Supply both prepared dataset bases and exactly the four architecture declarations")
    configs, missing = [], set()
    for _, dataset, architecture, alpha, seed in combinations(seeds):
        base = dataset_configs[dataset]
        if not isinstance(base, RunConfig) or base.scientific["dataset"] != dataset:
            raise ManifestError(f"Wrong prepared dataset base for {dataset}")
        config = from_declaration(base.to_dict(), model_declarations[architecture], overrides={
            "scientific.alpha": alpha, "scientific.training_seed": seed, "scientific.training_seeds": seeds})
        if config.scientific["dataset"] != dataset or config.scientific["architecture"] != architecture:
            raise ManifestError("Model declaration changed dataset or selected the wrong architecture")
        missing.update(f"{dataset}/{architecture}: {key}" for key in config.unresolved_settings())
        if config.execution["storage"]["split_file"] is None:
            missing.add(f"{dataset}: execution.storage.split_file (finalized preparation required)")
        configs.append(config)
    if missing:
        raise ManifestError("Finalize these decisions before manifest generation:\n" + "\n".join(sorted(missing)))
    prepared = {}
    for dataset in DATASETS:
        bundle = load_prepared_dataset(dataset_configs[dataset], base_dir=base_dir, metadata_only=True)
        prepared[dataset] = {key: bundle.manifest[key] for key in DATA_KEYS
                            if key not in ("official_training_identity", "official_test_identity", "preparation_identity")}
        prepared[dataset].update(preparation_identity=bundle.manifest["identity"],
            official_training_identity=bundle.sources.training_record["identity"],
            official_test_identity=bundle.sources.test_record["identity"])
    tasks = [Task("", phase, index, index % PHASE_SIZE, config, prepared[dataset])
             for index, ((phase, dataset, _, _, _), config) in enumerate(zip(combinations(seeds), configs))]
    grid = _grid_id(tasks)
    tasks = tuple(Task(grid, t.phase, t.global_index, t.phase_index, t.config, t.prepared_data) for t in tasks)
    return manifest_from_dict(Manifest(tasks, tuple(seeds), grid).to_dict())


def require_phase(phase):
    if type(phase) is not int or not 1 <= phase <= len(DATASETS):
        raise ManifestError("phase must be the integer 1 (CIFAR-10) or 2 (CIFAR-100)")
    return phase


def select_task(manifest: Manifest, *, global_index=None, phase=None, phase_index=None, run_id=None) -> Task:
    manifest = manifest_from_dict(manifest.to_dict())
    selectors = sum((global_index is not None, phase_index is not None, run_id is not None))
    if selectors != 1 or (phase is not None) != (phase_index is not None):
        raise ManifestError("Select exactly one global index, phase+phase index, or exact run ID")
    if run_id is not None:
        if type(run_id) is not str or not run_id:
            raise ManifestError("run_id must be a nonempty exact string")
        matches = [t for t in manifest.tasks if t.run_id == run_id]
        if len(matches) != 1:
            raise ManifestError("Unknown or ambiguous run ID")
        return matches[0]
    index, size = (global_index, TOTAL_RUNS) if global_index is not None else (phase_index, PHASE_SIZE)
    if type(index) is not int or not 0 <= index < size:
        raise ManifestError(f"Task index must be an integer in [0, {size})")
    if phase_index is not None:
        index += (require_phase(phase) - 1) * PHASE_SIZE
    return manifest.tasks[index]


EXECUTION_OVERRIDES = frozenset({
    "execution.device", "execution.num_workers",
    *(f"execution.storage.{key}" for key in ("output_root", "checkpoint_root", "result_root", "log_root", "data_root", "split_file")),
    *(f"execution.slurm.{key}" for key in ("partition", "account", "gpus", "cpus", "memory_gb", "wall_time", "max_concurrent")),
})


def task_config(task: Task, overrides: Mapping | None = None) -> RunConfig:
    validate_prepared_identity(task.config, task.prepared_data)
    if overrides is not None and not isinstance(overrides, Mapping):
        raise ManifestError("Execution overrides must be a typed dotted-path mapping")
    overrides = dict(overrides or {})
    if set(overrides) - EXECUTION_OVERRIDES:
        raise ManifestError("Only documented execution overrides are permitted; scientific changes require a new manifest")
    config = from_declaration(task.config.to_dict(), overrides=overrides)
    config.require_research_ready()
    _same(scientific_fingerprint(config), task.fingerprint, "execution override fingerprint")
    return config


def save_manifest_document(path, document: dict, *, parent: Manifest, replace: bool = False,
                           previous_parent: Manifest | None = None) -> str:
    """Cooperative exclusive publication; invalid existing evidence is never replaced.

    Each file is atomic, the three-file bundle is not. Reinvoke to reuse completed
    files after an interrupted bundle publication. No stale publication-lock steal.
    """
    if type(replace) is not bool:
        raise ManifestError("replace must be bool")
    parent = manifest_from_dict(parent.to_dict())
    if previous_parent is not None:
        previous_parent = manifest_from_dict(previous_parent.to_dict())
    if document.get("kind") == "full":
        _same(document, parent.to_dict(), "full output")
    else:
        _same(document, parent.phase_view(document.get("phase")), "phase output")
    path = Path(path).expanduser().absolute()
    if any(p.is_symlink() or getattr(p, "is_junction", lambda: False)() for p in (path, *path.parents)):
        raise ManifestError("Refusing linked manifest output")
    durable_mkdir(path.parent)
    lock = path.with_name(f".{path.name}.publish.lock")
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(strict_json_text({"pid": os.getpid(), "output": str(path)}))
            stream.flush()
            os.fsync(stream.fileno())
        existed = path.exists()
        if existed:
            old = read_json(path)
            _exact(old, FULL_KEYS if document["kind"] == "full" else PHASE_KEYS, "existing manifest")
            _same(old["kind"], document["kind"], "existing manifest kind")
            if old.get("kind") == "full":
                manifest_from_dict(old)
            else:
                _exact(old, PHASE_KEYS, "existing phase")
                _check_seal(old)
                # A phase cannot be independently validated without its parent.
                expected = [parent.phase_view(old["phase"])]
                if previous_parent is not None:
                    expected.append(previous_parent.phase_view(old["phase"]))
                if strict_json_text(old) not in [strict_json_text(value) for value in expected]:
                    raise ManifestError("Existing phase has no validated old or target parent")
            if strict_json_text(old) == strict_json_text(document):
                return "reused"
            if not replace:
                raise ManifestError("Valid conflicting manifest exists; use explicit --replace-manifests")
        atomic_write_json(path, document)
        return "replaced" if existed else "created"
    finally:
        lock.unlink()
        fsync_directory(lock.parent)


def evaluation_plan(manifest):
    """Logical slots, no fabricated checkpoint digests or scheduler job grouping."""
    from .evaluation_protocol import evaluation_protocol, prescribed_slots
    from .spec import TOTAL_EVALUATION_SLOTS, EVALUATION_SLOTS_PER_CHECKPOINT, EVALUATION_PROTOCOL_VERSION
    manifest = manifest_from_dict(manifest.to_dict())
    rows = []
    for task in manifest.tasks:
        protocol = evaluation_protocol(task.config)
        rows.append({"training_global_index": task.global_index, "training_phase": task.phase,
            "training_run_id": task.run_id, "training_scientific_fingerprint": task.fingerprint,
            "training_alpha": task.config.scientific["alpha"], "training_seed": task.config.scientific["training_seed"],
            "evaluation_protocol": protocol.to_dict(), "evaluation_protocol_fingerprint": protocol.fingerprint,
            "checkpoint_binding": "bind_verified_best_when_full_training_budget_available",
            "slots": [{"logical_evaluation_index": task.global_index * EVALUATION_SLOTS_PER_CHECKPOINT + index,
                       **slot.to_dict()} for index, slot in enumerate(prescribed_slots())]})
    return {"evaluation_plan_version": 1, "parent_grid_id": manifest.grid_id,
        "parent_document_sha256": manifest.document_sha256, "protocol_version": EVALUATION_PROTOCOL_VERSION,
        "training_tasks": len(manifest.tasks), "logical_evaluation_slots": TOTAL_EVALUATION_SLOTS,
        "job_grouping": "Step 7; logical slots are not job counts", "tasks": rows}


