# Configuration, identity, and artifact ownership

Step 7 adds separate operational policy version 1 in `configs/workflow.yaml`,
validated by hpc_policy.py. It does not alter this scientific schema/fingerprint.
Resources/environment/shared paths/signals/continuation and reviewed account/QOS
evidence are archived in immutable work mappings and workflow state. Typed
execution-only overrides preserve existing run identity; see
[slurm-workflow.md](slurm-workflow.md) for required future inputs and commands.

## Schema and current scope

Step 8 adds a separate derived reporting-v1 contract, without changing this schema,
scientific hash or evaluation seeds. Sample SD, complete prescribed observations,
two-stage robustness means and validation-only co-winner rules are in
[results-and-figures.md](results-and-figures.md). Reporting storage/options/provenance
are outside research identity; unresolved science remains unresolved.

`schema_version: 4` describes the file structure; `scientific.specification_version:
main-v1` describes the experiment. The effective document has exactly three top
keys: `schema_version`, `scientific`, `execution`. No arbitrary extra settings,
unknown names, duplicate YAML keys, non-string mapping keys, YAML merge keys,
or bool-as-integer coercion are accepted. Numeric strings are not numbers.
Configuration validation does not import torch or check whether a builder exists.

Step 2 adds two fixed scientific fields: `data.preprocessing_version` and
`data.randomness_version`. They identify the exact approved padding/flip/tensor
semantics and local epoch/sample keyed RNG contract. Defaults and declarations
record them explicitly; alternate values are rejected. Schema 1 declarations
and full documents receive an explicit additive migration through schema 2, filling
only fixed contract fields and spelling out tensor conversion in the previously
implicit transform lists. No pending choice is resolved by migration.
Step 3 adds fixed `model.builder_version`, `model.initialization_version` and
`training.scheduler_version` fields. Schema 2 -> 3 migration fills these contract
fields; schema 1 -> 2 -> 3 preserves the earlier data migration. Missing sections
in partial declarations are filled by central defaults. Explicit incompatible
versions are rejected; pending values, existing normalization/split values and
operational paths are preserved. Step 4 adds fixed `training.engine_version =
functional-fp32-training-v1` and `training.randomness_version =
sha256-runtime-streams-v1`. Schema 3 -> 4 adds only those contracts; supported
versions 1/2 chain through earlier migrations. Saved effective YAML now uses
schema 4. Fingerprint format is version 4, so older run artifacts are incompatible with
blind continuation/output reuse. This does not invalidate shared prepared data
or authorize rewriting its original checksummed YAML.

`spec.py` centrally defines datasets/class counts, architecture names, ten
approved alphas, training/split sizes, and all optimizer recipes. `config.py`
implements strict schema validation, layering, and a separate research-readiness
gate. `RunConfig.scientific`, `.execution`, and `.to_dict()` return defensive
copies. Direct `RunConfig(document)` requires the full schema;
`from_declaration(...)` and `load_config(...)` accept partial declarations.
The approved independent evaluation protocol is separate from this frozen training
schema. `RunConfig.training_alpha`/`identity.training_alpha` adapt scientific.alpha;
it and legacy alpha names always mean training alpha. Existing evaluation fields
now identify clean validation/selection and the mandatory baseline, preserving
fingerprint version 4. They do not prohibit weight_robustness_eval_v1. Protocol
fingerprints include its full fixed grid/replicates/math/lifetime/RNG/checkpoint/
preprocessing/precision declaration without changing training identity. See
[weight-robustness-evaluation.md](weight-robustness-evaluation.md).

The schema rejects negative/NaN/infinite alpha and alpha outside the approved
main grid. It enforces classifier/dataset agreement, scratch initialization,
all fixed optimizer fields (inactive fields must be null), 200 epochs, batch
size 128, five warmup epochs, full dataset sizes, transforms, and the approved
perturbation/evaluation contract. There is no subset or reduced-budget option.
Finalized normalization requires three finite RGB means and three positive
finite RGB standard deviations plus a nonempty source/derivation identity.

## Precedence and serialization

`load_config(base_path, dataset_path=None, model_path=None, overrides=None)`
applies these layers in increasing precedence:

1. Central defaults derived from the **final selected** dataset and architecture.
2. Base YAML (`configs/research.yaml`); it must declare schema_version.
3. Optional dataset YAML (`configs/datasets/cifar10.yaml` or `cifar100.yaml`).
4. Optional model YAML (one of the four declarations in `configs/models/`).
5. Typed CLI override mapping, keyed by full dotted field paths such as
   `scientific.alpha` and `execution.storage.output_root`.

This is a small loader API, not a CLI or general-purpose configuration framework.
A future CLI must parse arguments into actual numeric/bool values before passing
overrides; strings must not bypass type checking. Layers cannot replace mappings
with scalars. Unknown keys and invalid types are rejected in every layer. Explicit values retain
precedence even when selectors change: switching a fully resolved CIFAR-10
config to CIFAR-100 requires changing its explicit classifier. Supply the correct
normalization/split identity for that dataset; Step 2 verifies their artifact
association. Switching an explicit model recipe requires an
appropriate model layer or explicit compatible fields. Incompatible fields are
rejected rather than silently reset. Dataset/model files are applied only when
requested; their filenames are not interpreted as settings.

`save_resolved_config(path, config)` writes the complete effective document in
the **same** schema that `load_config` accepts. It introduces no `paths`,
`resolved`, identity, or provenance wrapper into that YAML. Effective execution
overrides are retained. `open_run` converts storage paths to absolute paths
against its explicitly supplied `base_dir` before saving, so the file can be
loaded later from a different working directory. A saved complete YAML can be
used as the next base without a dataset/model layer.

## Central pending-decision register

No prior source or instructions existed in this working tree to establish these
choices. `configs/research.yaml` centrally lists pending fields as `null`; the
dataset declarations keep each dataset's normalization/split decision separate.
Null means unresolved, never "use library/default/smoke value."

| Decision | Schema fields | Required finalization |
| --- | --- | --- |
| Exact three training seeds | scientific.training_seeds | Three distinct unsigned 32-bit integers; shared across alpha values and architectures |
| Selected task seed | scientific.training_seed | One of the finalized three seeds; task selection belongs to Step 6 |
| Dataset normalization | scientific.data.normalization.mean / std / source | Final constants and source, or an approved derivation policy that Step 2 resolves to constants/source before execution |
| Final incomplete training batch | scientific.training.drop_last | Explicit true or false; validation/test always retain all examples |
| Precision | scientific.precision.mode | float32, amp_float16, or amp_bfloat16 |
| Backend/determinism | scientific.precision.float32_matmul_precision / allow_tf32 / deterministic_algorithms / cudnn_deterministic / cudnn_benchmark | Explicit supported string/bools; allow_tf32 controls both CUDA matmul and cuDNN TF32 |
| Persisted split identity | scientific.data.split.identity | Step 2-generated sha256 identity of the actual ordered training/validation indices and dataset |
| Split preparation seed | Explicit --split-seed preparation input, recorded in split generation metadata | Required to create a new split; separate from every training seed; no default |
| Warmup endpoints | scientific.training.warmup_start_factor | Explicit factor in [0,1] relative to the architecture's base LR |
| Cosine endpoint | scientific.training.cosine_min_lr | Explicit finite LR between zero and the architecture's base LR |
| Scheduler cadence | scientific.training.scheduler_interval | epoch or optimizer_step, with the exact indexing equation documented in Step 3 |

The split identity and scheduler rows are necessary to avoid inventing unspecified split/scheduler
details. There is no provisional split seed, guessed normalization, training
seed list, precision, batch policy, or warmup/cosine endpoint. A split's identity
is supplied by dataset preparation, not chosen by the training seed. A derivation
policy written in normalization.source alone does not resolve missing constants.

Partial decisions are valid during development. `unresolved_settings()` lists
the exact pending field paths. `require_research_ready()` raises
`UnresolvedConfigError` with all missing fields. Every future launcher/controller
must call it before initializing a run, launching training, or submitting a
research task. Artifact initialization and scientific hashing already call it.
Resolved settings must stay fixed across alphas/seeds for a given architecture
and dataset; Step 6 must also check that grid-wide consistency.

Operational `slurm.account` and `storage.split_file` have no invented values.
Step 7 validates the chosen account and resources before submission. Step 2
validates dataset files and the actual split artifact before loading. These
operational checks are separate from scientific readiness.

`prepare_dataset` deliberately does not call research readiness: it requires only
valid configuration, an explicit normalization request, and an explicit split
seed for a new split. `build_loaders` requires resolved selected training seed,
drop_last, split/normalization, and prepared artifacts; the three-seed plan,
precision, and scheduler may remain pending during development. Actual launchers
must still call `require_research_ready()` later.

`build_model` requires only validated model settings and a selected training seed.
It does not require the finalized three-seed list or scheduler/data/HPC decisions.
`build_optimizer` requires a compatible clean model and the fixed recipe.
`build_scheduler` requires explicit warmup_start_factor, cosine_min_lr and
scheduler_interval, plus the actual loader count for optimizer-step cadence.
Component-specific errors do not change declaration validity. The implemented
contracts and continuation call order are in
[models-and-optimization.md](models-and-optimization.md); no component was executed.

`build_runtime` requires explicit precision/backend fields and a compatible
assigned CUDA device; no CPU fallback exists. Pinned CUDA matmul precision and
TF32 must agree: highest/false or high-or-medium/true. Deterministic algorithms
require CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8 set before Python starts.
Float16 uses pinned native GradScaler defaults; native bfloat16 uses no scaler.
Train and clean evaluation share the configured forward precision, with float32
master parameters and float32 cross-entropy/statistics/noise. Effective settings
are recorded separately from declarations; no pending field is resolved silently.
Its independent `evaluation_only=True` mode uses the same policy without a
GradScaler. `build_test_loader` creates only deterministic full-test loading.

`RuntimeStreams` requires a selected seed and explicit fresh/restored mode.
`train_run` additionally requires full research readiness, original-budget
components, an active `open_run` handle and Step 5's `TrainingPersistence` service.
The compute and recording interfaces are described in
[training-and-perturbation.md](training-and-perturbation.md). Step 6's
[controller](manifests-and-controller.md) supplies the one-task production CLI,
implemented without execution.

## Scientific identity and provenance

Human-readable run IDs have the form
`cifar10__resnet50__alpha-0.1__seed-<finalized-integer>`.
`canonical_alpha` expands the shortest numeric representation without decimal
rounding: 0.10 and 0.1 map to `0.1`, 1 and 1.00 to `1`, and signed zero to `0`.
All ten approved values remain distinct. Numeric strings are rejected.

`scientific_fingerprint` returns a full SHA-256 hex digest of sorted canonical
finite JSON containing fingerprint version 4, schema version 4, and the entire
effective scientific section. It includes selected seed, the three-seed plan,
dataset/model, alpha, full training/optimizer/perturbation/evaluation settings,
split hash, normalization identity/constants, and resolved precision/backend
policy, including preprocessing/data randomness, builder/initialization/scheduler
and training engine/runtime randomness versions. Fixed numeric values
and alpha are normalized on validation. It excludes
all execution paths, worker/CPU/RAM requests, allocation wall time, account,
Slurm IDs, timestamps, and code/software provenance. A version or scientific
change therefore prevents compatible reuse despite an unchanged human name.

Code/software provenance is a separate JSON record. `software_provenance`
requires actual commit (or null), dirty flag (or null with an explicit
Git-unavailable reason), source-snapshot SHA-256, Python,
torch, torchvision, CUDA and cuDNN versions; allocation metadata/timestamps are
operational. Step 6 implements future execution collection of the source actually
imported, actual dependencies and allocation environment; runtime policy supplies
hardware details. No environment probes or provenance collection were executed
during implementation. The source digest must include
execution source/config-contract files and exclude datasets/output artifacts.

Before continuation, compare marker versions, full run ID, full scientific
fingerprint, the fingerprint recomputed from saved YAML, and markers in every
populated selected artifact directory. Checkpoint/result metadata must repeat
these identities. `open_run` also currently requires the same source digest and
software mapping. Step 5 preserves this conservative policy and provides no
automatic code/software migration.
Commit/allocation/storage changes alone do not change scientific identity.

## Safe directories and writes

`resolve_run_paths(config, base_dir)` returns `RunPaths(metadata, checkpoints,
results, logs)`. Default layout is:

```text
<output_root>/<run_id>/
  resolved_config.yaml
  run_metadata.json
  attempts/<unique-attempt-id>/resolved_config.yaml
  attempts/<unique-attempt-id>/metadata.json
  checkpoints/.run_owner.json
  results/.run_owner.json
  logs/.run_owner.json
```

Optional checkpoint/result/log roots use `<root>/<run_id>` instead. Distinct
directories are required; overlapping directories other than direct children
of metadata are rejected. Empty directories can be used for a new run, but any
existing entry—including a configuration, checkpoint, log, temporary file, or
subdirectory—protects the run regardless of `summary.json`.

`open_run(config, base_dir=..., provenance=..., operation=...)` is a context
manager. All run writers must hold its context for their complete access period.
The yielded `RunHandle` includes attempt provenance and `require_active(config=None)`;
its lease is invalidated when the context exits. Training/logging/service use
must retain this same guarded handle rather than invent a second lock or owner.
It preflights every selected directory, acquires exclusive cooperative locks at
each independent storage root, repeats checks under locks, and then initializes
configuration/metadata. Locks are outside the run directories and contain owner
token, host, PID, time, and identity. Locks are never expired or stolen based on
wall time. Inspect a stale lock's ownership and allocations before manual
recovery. Readers and noncooperating external programs cannot be made safe by
these cooperative locks.

| Operation | Behavior |
| --- | --- |
| new (default) | Refuse any existing run artifacts; create fresh owner/config/metadata records |
| continue | Require existing compatible metadata and resolved YAML, valid populated-directory owners, and matching provenance; preserve artifacts and add an attempt snapshot |
| recover | Require the same full checkpoint/component preflight and ownership; hold existing locks and record an attempt for explicit committed publication repair |
| overwrite | Require an existing selected directory and the exact overwrite_run_id; archive all managed run directories beside their original locations with a unique suffix; initialize a new run |

Continue refuses authoritative HEAD completion and terminal numerical failure.
New completion means training_completed; legacy completed retains its original
clean-test meaning and establishes training, never full robustness. Count-200
dispatch uses publication-only TrainingFinalizationPlan/open_run finalize_training
without live training components. Stored training provenance validates old evidence;
actual finalizer provenance is separate. Evaluation reads under its explicit pinned
component compatibility policy, without requiring new source == training source.
This exception does not authorize unfinished-training source/software migration.
Metadata/summary status is a derived view, never the continuation authority.
Unfinished continue/recover requires `ContinuationPlan`, fully checked before attempt/config
writes and again under the existing locks. Missing/incomplete ownership
or config records do not authorize reuse; stop for recovery or explicitly
selected archived overwrite. A context exit does not mark a run completed.
Initialization failure may leave an incomplete directory or a stale lock, which
remains protected. Multi-directory archiving is not one filesystem transaction;
failed moves attempt rollback and preserve old content. It never recursively
deletes artifacts. Linked run directories/ownership/config files are rejected.

Operational paths may change without changing a fingerprint. Continuing after
relocation requires compatible metadata/config and any populated artifact
directories to be copied/mounted at their selected new locations. Step 5 checks
actual checkpoint/record digests and complete component compatibility before any
restoration or attempt/config writes. Paths/visible device indices alone do not
invalidate compatible state. These helpers do not copy runs automatically.

`atomic_write_text`, `atomic_write_json`, and `save_resolved_config` serialize to
a temporary file in the target directory, flush/fsync it, then use `os.replace`;
directory fsync is attempted on POSIX. Low-level atomic writing does not grant
permission to replace run artifacts: callers first use the guarded run context.
`strict_json_text` refuses nonfinite values; `read_json` also rejects duplicate
keys, NaN/Infinity tokens, and overflowed numbers. Convert unavailable failed
metrics to null only while recording explicit failure status/reason.

Step 5 implements restricted checkpoint files, full restoration and durable
completion/failure publication with one metadata HEAD commit point. Its storage
versions leave scientific schema/fingerprint 4 unchanged. See
[checkpoints-and-recovery.md](checkpoints-and-recovery.md). Training launchers,
manifest expansion are now connected by Step 6; Slurm remains Step 7. The
[manifest/controller guide](manifests-and-controller.md) specifies finalized-base
layering, strict full/phase documents, selectors, exact execution override fields,
provenance inventory, typed outcomes and read-only coverage. Generating a manifest
does not resolve any pending decision in the checked-in declarations.

## Prepared dataset configurations

Preparation writes `<data_root>/.alpha_sweep_prepared/<dataset>/dataset_config.yaml`
with full loader-compatible schema 4 settings for new preparation. Existing
schema-2/3 files are checksummed in their original stored bytes, migrated in memory
on loading and remain unchanged on disk.
It supplies the actual split
identity, normalization mean/std, `normalization.source = normalization-v1:sha256:<digest>`,
and absolute operational data_root/split_file paths. The human constants source
or derivation description lives in normalization.json and is covered by its
identity. Unrelated pending fields remain pending. Shared artifacts are not
run-specific outputs and use a separate preparation lock.

The preparation manifest is the fixed sibling `manifest.json` of split.json,
not an unvalidated configuration dictionary or a new optional path. The loader
requires the one dataset-level location beneath the chosen data_root. Reuse
keeps the original shared YAML unchanged; the preparation API returns this
caller's effective RunConfig as well. Use the shared file as a base for later
model/task overrides, not as an unconditional high-priority overlay that might
replace an already-selected architecture/recipe.
Apply the corresponding model declaration after that base. Do not reapply a
dataset declaration containing unresolved split/normalization values over the
finalized preparation settings. Model/task changes alone do not invalidate the
source/split/normalization contracts or require preparation again.

After relocating raw/prepared files, override execution.storage.data_root and
execution.storage.split_file together. Manifest/config checksums validate the
preserved stored YAML, while the supplied run config's operational paths select
the new location. Content identities stay unchanged. No schema keys are needed
for normalization-policy or split-seed defaults: these are explicit preparation
inputs, and persisted identities determine effective research data settings.
