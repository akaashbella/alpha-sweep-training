# Full experiment manifests and one-task controller

Step 6 is implemented in source only. These commands are for authorized future
HPC use after all nine steps. No commands below have been executed. Runtime
correctness remains unverified. Slurm submission, signals, retry/requeue,
resubmission and concurrency orchestration belong to Step 7. The approved independent evaluation integration is documented in [weight-robustness-evaluation.md](weight-robustness-evaluation.md); training and robustness completion have distinct authorities.

## Finalized inputs

Supply two dataset bases loaded with the existing `load_config`, one for each
dataset. Start from its finalized preparation `dataset_config.yaml`, retain its
split/normalization and data paths, and finalize the other approved scientific
decisions in a separate full configuration if necessary. Preserve the original
checksummed shared YAML. The builder applies each architecture declaration from
`configs/models` after this base, then only the explicit alpha/selected seed and
three-seed plan. It never reapplies `configs/datasets` over prepared values.

The seed plan is exactly three distinct explicitly supplied uint32 integers in
approved order. Preparation seed is independent and comes from the persisted
split artifact. The checked-in declarations remain pending: exact training
seeds, split preparation seed/identities, dataset normalization policy/constants,
drop_last, precision/backend settings, warmup factor, cosine minimum and cadence.
`build_manifest` lists unresolved paths across dataset/architecture combinations
before checking preparation or publishing anything. There are no launchable
null-seed/placeholder tasks or provisional scientific defaults.

`load_prepared_dataset(..., metadata_only=True)` reuses bundle checksums,
split/normalization validators, configuration association and preparation identity.
It validates current raw source files against the pinned official checksum
declarations and stored SHA-256 descriptors without decoding the image arrays.
Full loaders still decode and recompute image/label identities before execution.
Neither path downloads or repairs. Manifest construction builds no models,
loaders or CUDA objects, computes no normalization and creates no split.

Finalize science before generation. Architecture-specific approved scheduler
choices can be declared in its model YAML using existing schema fields; the
optimizer recipes and all fixed scientific settings remain centrally validated.
There is no new scientific configuration schema.

## Format and ordering

Manifest format version 1 and grid contract `ordered-scientific-tasks-v1` are
independent of scientific schema/fingerprint version 4. Unsupported versions
fail explicitly; no implicit manifest migration exists.

The only grid order is dataset order from `spec.DATASETS`, architecture order
from `spec.ARCHITECTURES`, alpha order from `spec.ALPHAS`, then supplied seed-plan
order. The builder and loader share that definition. The exact full Cartesian
product has 240 tasks, global indices 0–239. Each phase has 120 tasks, phase-local
indices 0–119. Phase 1 is CIFAR-10; phase 2 is CIFAR-100. Duplicates, omissions,
extra combinations, reordering, bool indices/seeds, inconsistent fields and
unresolved configurations fail validation.

The full JSON has exactly `format_version`, `kind="full"`, `grid_version`,
`grid_id`, `seeds`, `tasks`, `document_sha256`. Each task contains:

- Format/grid linkage, phase, global and phase-local indices.
- Dataset, architecture, legacy numeric/canonical alpha (training alpha only) and selected training seed. The manifest format and training fingerprint stay unchanged; RunConfig.training_alpha is the explicit adapter.
- Existing canonical run ID and full scientific fingerprint.
- Complete validated effective RunConfig in the existing schema.
- Split, normalization, preprocessing, preparation and official train/test identities.

Loading reconstructs every `RunConfig`, calls readiness, recomputes its run ID
and fingerprint, validates data identity linkage, and compares the entire task
to its expected index/combination. Within each dataset/architecture, all science
except alpha and selected seed is identical. Each dataset's complete data
configuration and prepared identities are identical across architectures too.

`grid_id` is the lowercase SHA-256 of compact sorted finite UTF-8 JSON containing
grid version, manifest version and ordered `{run_id, scientific_fingerprint}`
records. Execution fields, paths, allocations and timestamps do not enter it.
`document_sha256` hashes the complete JSON document with that field omitted,
using the same encoding. It includes operational configurations. It is an
integrity check, not a signature or proof of authorship. Task entries link the
grid; their containing document supplies document-level linkage.

Phase views have exactly `format_version`, `kind="phase"`, `grid_id`,
`parent_document_sha256`, `phase`, `dataset`, `tasks`, `document_sha256`. Each
phase task is its global/phase-local index, run ID and scientific fingerprint.
They contain no independent configuration. `load_phase(path, parent)` compares
every field to the validated full parent, returning its original tasks. A phase
view cannot independently authorize execution or use a different recipe.

## Commands and output protection

`alpha-sweep` and `python -m alpha_sweep_training.controller` share these
subcommands. Arguments denoted with shell variables below must be supplied by
the future operator; they are not scientific choices or reduced examples.

```bash
alpha-sweep build --cifar10-config "$FINALIZED_CIFAR10_CONFIG" \
  --cifar100-config "$FINALIZED_CIFAR100_CONFIG" --models-dir configs/models \
  --seeds "$SEED_A" "$SEED_B" "$SEED_C" --base-dir "$SHARED_BASE" \
  --output "$FULL_MANIFEST"
alpha-sweep inspect --manifest "$FULL_MANIFEST" --phase-view "$PHASE1_VIEW"
alpha-sweep select --manifest "$FULL_MANIFEST" --phase 1 --phase-index "$TASK_INDEX"
alpha-sweep run --manifest "$FULL_MANIFEST" --run-id "$EXACT_RUN_ID" \
  --source-root "$EXECUTED_CHECKOUT" --base-dir "$SHARED_BASE" --operation auto
alpha-sweep status --manifest "$FULL_MANIFEST" --base-dir "$SHARED_BASE"
```

Build defaults phase output names to `<full-stem>.phase1.json` and
`<full-stem>.phase2.json`; explicit `--phase1-output`/`--phase2-output` are
available. It checks existing bundle contents before writing. Exact compatible
documents are reused without rewriting. Different valid documents require
`--replace-manifests`; invalid existing content is always preserved/rejected.
Old phase documents require their valid old full parent for replacement.
Output paths must be distinct. Each file uses a cooperative exclusive
`.<filename>.publish.lock` and atomic publication. Locks are never stolen;
interrupted publication may require operator review of its publication lock.
The three outputs are not one filesystem transaction. Phases publish first and
the full parent last, retaining the old parent until both views exist. Reinvocation
validates existing phases against the old or finalized target parent, so a partial
bundle can be completed by compatible reuse/explicit replacement. Phase inspection
still requires exact parent digest linkage and rejects a partly replaced bundle.

Configuration/model/override/manifest argument paths are resolved by the
invoking shell's current directory. `--base-dir` anchors storage fields inside
configurations; it does not alter scientific identities. Preparation generally
already supplies absolute data/split paths. Relative run roots require the same
explicit persistent base across allocations.

## Selection and execution overrides

Supply exactly one selector: `--global-index`, `--phase-index` together with
`--phase`, or an exact `--run-id`. `select_task` rejects conflicts, absent/extra
phase, unknown IDs, strings/bools used as indices and out-of-range values. Global
indices always address the full parent. Phase views are read-only linked views.

`--execution-overrides` reads a strict JSON object of typed dotted fields. Allowed
fields are all existing `execution.storage` fields, `execution.device`,
`execution.num_workers` and existing `execution.slurm` fields. Existing validation
still enforces `dgxh` and one GPU. Convenience flags cover device, workers,
output/checkpoint/result/log/data roots and split file. Duplicate fields supplied
through both sources are rejected. No alpha/seed/budget/batch/optimizer/data
science/precision/scheduler override is accepted. `task_config` asserts the
scientific fingerprint is unchanged. Relocation requires preserving all content
and updating data_root/split_file together; loaders revalidate identities.

## Authority and operations

`inspect_task` performs stored-state preflight before live compute construction.
Dispatch uses the following distinctions:

| Observation | `auto` action |
| --- | --- |
| All selected run directories absent | New run |
| Compatible initialized/training/interrupted authority | Full validated continuation |
| Valid count 200, training publication pending | CPU TrainingFinalizationPlan, zero additional epochs and no test |
| Valid immutable training completion or legacy completed transaction | `already_training_completed`, no model/runtime/training/test; robustness coverage is separate |
| Committed or unacknowledged numerical evidence before full training budget | Block, exit 20 |
| Execution failure or unacknowledged evidence at latest training boundary | Block, exit 21; explicit choice required |
| Lock, changing read, partial initialization, incompatible/corrupt state or inspection error | Preserve/report, exit 22 |

Explicit `new` never reuses existing directories, including empty partially
initialized selections. Explicit `continue` authorizes continuing an execution
failure after review; numerical evidence remains terminal. Failure records at
older training boundaries remain audit records but stop blocking after subsequent
durable training progress. If failure publication itself was impossible, CLI
failure remains explicit while stored status may be older; Step 7 must retain the
outcome and require an operator choice rather than equate old authority with a
successful allocation.

Explicit `overwrite` additionally requires `--overwrite-run-id` to exactly equal
the selected task. It delegates archival of all managed old directories to
`open_run`; it cannot bypass a lock or unstable observation. Corrupt/partial/
incompatible content is never silently replaced. Linked/unsafe paths remain
subject to the original artifact guards. There is no stale-lock release flag in
this CLI: use the separate Step 5 evidence/confirmed-inactive-owner APIs.

Explicit `recover` is publication repair, never scientific computation. A valid
full-budget boundary uses the CPU `TrainingFinalizationPlan` and stored training
provenance. Unfinished boundaries retain mandatory live-component
`ContinuationPlan` preflight and conservative source/software compatibility,
requiring the compatible runtime/data environment and assigned GPU. Both enter
`open_run(operation="recover")` and call `repair_publication`, without restoring
streams or invoking the trainer. Full-budget recovery rejects numerical-reference
selection for historical test failures. For unfinished orphan numerical evidence,
supply `--numerical-reference` as a strict JSON file holding the exact inspected
ArtifactReference, then delegate to `reconcile_numerical_failure` before repair.
Recovery does not turn a numerical failure into a resumable interruption.
Malformed derived summaries do not veto HEAD completion or explicit repair;
ownership and immutable authoritative artifacts still must validate.

## One-task construction and outcomes

`execute_task(manifest, task, base_dir=..., source_root=..., overrides=...,
operation=..., overwrite_run_id=..., numerical_reference=..., stop=...)` validates
parent membership and science. It collects actual provenance, then inspects
authority. Only an eligible compute operation constructs read-only full loaders,
the CUDA runtime, scratch isolated CPU float32 model, final device placement,
optimizer, original-budget scheduler and runtime/noise streams, in that order.

Fresh/overwrite scheduler and streams initialize once. The active `open_run`
lease encloses `DurableTrainingPersistence` and `train_run`; the trainer commits
its durable initialized boundary before its first batch. Continuation/recovery
uses `initialize=False` for both scheduler and streams. A concrete
`ContinuationPlan` is passed to `open_run`, checked before attempt/config writes
and again under existing root locks. Auto dispatch passes
`allow_execution_failure=False` to the plan, repeating the explicit-choice gate
under those locks if authority changed during construction. Explicit continue
permits that execution-failure choice. For continue, `plan.restore(run)` restores
all state with RNG last; the resulting `ContinuationState` goes to the original
trainer. Epoch zero restores rather than reseeds. Count 200 uses publication-only CPU finalization before live component construction, and never performs test. Training commands stop after durable training completion. No grid training loop exists.

The library accepts the existing `StopRequest`. It never registers a signal or
requests retries itself. `TrainingRunError` retains failure/publication details
and triggers a read of committed authority under the lease. Valid completion
dominates a lost acknowledgement; otherwise the failure remains explicit and any
inspection error is recorded. Pre-lease construction failures do not claim a
durable failure record. A cleanup error after acknowledged completion preserves
the completed outcome with its diagnostic.

| Exit | Run outcome |
| --- | --- |
| 0 | Training completed/already_training_completed, training completion after lost ack/diagnostic, or explicitly repaired publication; inspect outcome and lifecycle, never infer robustness completion |
| 10 | Interrupted; committed boundary preserved |
| 20 | Numerical failure or blocked terminal numerical evidence |
| 21 | Execution failure or blocked execution failure |
| 22 | Rejected configuration, selector, ownership, preflight/restoration or unsafe inspection |
| 2 | argparse usage error before dispatch |

All output is finite JSON. Build/inspect/select/status return 0 on successful
command completion, even if coverage contains missing/failed tasks; consult its
counts and individual conditions. Invalid inputs return 22. Exit code alone
never proves scientific completion.

## Provenance and read-only coverage

`execution_provenance.collect_provenance` runs only on the future execution path.
It requires an actually imported checkout, hashes sorted relative file/digest
records with inventory contract `checkout-python-and-declarations-v1`, includes
all package `.py` files (excluding caches), pyproject, research declaration, named
dataset/model declarations and authoritative scientific specification. It excludes
datasets, checkpoints/logs/results, generated manifests, environments and personal
files. Inventory is checked twice for change during collection. Execute from a
fixed checkout; there is no automatic source/software migration across attempts.

Actual Git commit/dirty state is read via Git without Slurm commands. If Git is
unavailable, commit/dirty are null with an explicit `git_unavailable` reason;
unborn HEAD may have a null commit and a verifiable dirty flag. Required source
digests and actual Python/torch/torchvision/NumPy/PyYAML/CUDA/cuDNN versions must
be available; CPU-build CUDA/cuDNN may be null and runtime subsequently rejects
unavailable CUDA. No hardware capability is fabricated. Runtime policy supplies
actual device/effective backend information in allocation/component metadata.
An allowlisted allocation environment, host/PID/interpreter, manifest selection
and operation are recorded; `open_run` adds the actual unique attempt ID/history.
Source/software/allocation remain outside scientific fingerprints.

`collect_coverage(manifest, base_dir=..., overrides=...)` joins independent
observations to every expected task, preserving all seeds. It uses saved
provenance to validate historical state, not an environment/provenance probe.
Unfinished training dispatch additionally checks compatibility with actual execution provenance. Full-budget reading/finalization and independent evaluation validate stored training provenance and record the actual new writer/evaluator separately; evaluation does not require historical training-source equality. Checkpoint validation restricted-loads CPU tensors and records; it
constructs no model/loader/CUDA runtime. Status never acquires writer locks,
repairs publications, releases locks, prepares data or writes run state.

`stored_status` is separate from `condition`. A visible lock suppresses status
claims. HEAD/ownership/config/lock tokens are observed before and after reads;
change reports `unstable`, never proven corruption or numerical failure. I/O or
environment inspection errors remain separate. Stable validation rejection is
reported as corrupt/incompatible/partial with its diagnostic. This is a bounded
cooperative observation, not an atomic grid snapshot or guarantee against
noncooperating writers. Reinspect after writers stop.

Training counts satisfy expected = training_completed + training_failed + training_unfinished. Failed divides into
terminal numerical and execution failures; interrupted is a subset of unfinished.
Inspection conditions contribute to unfinished, never fabricate scientific
failure. Counts, training_complete, full_budget_committed, training_finalization_pending, evaluation_eligible, legacy_clean_test, committed epoch counts, failure payloads,
evidence references and locks accompany every individual identity. Dataset,
architecture and dataset/architecture breakdowns remain coverage only; no result
means/std, alpha selection or plots are computed.

## Step 7 handoff

Use `load_manifest`, `load_phase`, `select_task`, `task_config`, `inspect_task`,
`dispatch`, `execute_task`, typed `TaskOutcome`/`ExitCode` and `collect_coverage`.
Phase coverage explicitly exposes training_successfully_completed, training_eligible, terminal scientific-failure count, settled
completed-or-numerical coverage, eligibility and automatic candidate run IDs.
Phase 1 is eligible first; phase 2's conservative eligibility requires every
phase-1 training task durably training-completed. Pending robustness does not block training phase eligibility. Terminal failure is visible and does not
imply phase success or silently unlock phase 2. Later orchestration must define
any explicitly approved alternate terminal-failure policy. The one-task executor
does not enforce cluster phase scheduling; Step 7 consumes eligibility, refreshes
observations before dispatch and enforces CIFAR-10-first/global concurrency.
Candidate lists exclude execution failures, numerical failures and unsafe reads.

Step 7 owns allocations/submission, resources/account checks, dependencies,
signal delivery to a per-attempt StopRequest, and continuation across allocations.
It must retain failure outcomes when publication fails and never infer completion
from an ended allocation. No part of those job/signal mechanisms is implemented
here. Step 8 owns seed-first results and aggregation conventions.

## Linked independent evaluation

`evaluation_plan(manifest)` keeps the parent at 240 training tasks and links 28
logical evaluation slots per task, 6,720 total. It records protocol contracts and
logical indices, not guessed checkpoint digests or scheduler jobs. Actual result
bindings are created only from a verified full-budget best checkpoint. Both
training alpha/seed and evaluation alpha/replicate remain explicit.

The controller adds `evaluation-plan`, `evaluation-inspect`, `evaluate` and
`evaluation-status`. Inspection/execution reuse the existing task selectors and
typed execution-only overrides. Evaluation selectors choose one slot or a group;
omitting them runs all missing slots. Completed compatible slots are validated
and skipped. Each interrupted pass restarts from its deterministic seed, discards
partial metrics and preserves raw failure evidence. Explicit `--retry-failed`
authorizes failed-slot retries; qualified `--reuse-legacy-clean` retains original
legacy references/provenance and never fabricates preservation evidence.

Independent namespace/slot locks and HEAD/SWEEP_HEAD authorities never change
training state. `collect_experiment_coverage` joins every training identity and
all logical evaluation slots, including unbound pending slots for untrained tasks.
Reports expose clean/positive success, pending/failed/interrupted/locked/
incompatible conditions and complete robustness authority separately from
training completion. See the evaluation guide for exact APIs, commands, versions,
raw schemas and Step 7/8 handoffs. No jobs, signals, means/std or plots are added.
