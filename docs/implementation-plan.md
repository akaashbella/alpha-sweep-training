# Shared contracts and remaining implementation steps

Steps 1–4 implement configuration, identity, finite atomic serialization,
provenance validation, guarded run initialization, full official CIFAR preparation,
reproducible loaders, all four scratch CIFAR models, owned eligible-weight
inventory, fixed optimizers, the original-budget scheduler, shared training,
functional perturbation, clean evaluation, RNG/precision policy, typed metrics
and numerical failures. These sources have not been executed or runtime-verified.
Checkpoint persistence/recovery, controllers, Slurm and reporting remain future
work. There is no runnable training command or default persistence service. See
[models-and-optimization.md](models-and-optimization.md) and
[training-and-perturbation.md](training-and-perturbation.md) for implemented contracts.

## Responsibilities and interfaces

| Component | Input → output / responsibility | Step |
| --- | --- | --- |
| Configuration | load_config(base_path, dataset_path, model_path, typed overrides) → RunConfig; strict declaration validity separate from require_research_ready | 1 |
| Dataset preparation | prepare_dataset(config, base_dir, normalization: NormalizationRequest, split_seed, download=False) → PreparationResult; independent explicit split seed; committed manifest and config | 2 implemented |
| Data loading | build_loaders(config, base_dir, pin_memory=False, persistent_workers=False) → DataLoaders with train/validation/test, metadata, set_epoch, continuation_metadata, restore_next_epoch | 2 implemented |
| Model construction | build_model(config) → scratch CIFAR-adapted model; describe_model(model), discover_eligible_weights(model) → actual metadata/inventory | 3 implemented |
| Optimizer/scheduler | build_optimizer(model, config), describe_optimizer(optimizer), build_scheduler(optimizer, config, steps_per_epoch, initialize=True) → fixed native recipe and original-budget schedule with in-memory state | 3 implemented |
| Perturbation | training_forward(model, inputs, *, alpha, noise_generator) → one ordinary/functional forward with detached float32 scale/noise and gradients to original clean parameters | 4 implemented |
| Runtime/randomness | build_runtime(config) → RuntimePolicy; RuntimeStreams(config, device, initialize=...) → separate model/noise streams and owned in-memory states | 4 implemented |
| Clean evaluation | evaluate_clean(model, loader, *, split, config, runtime) → full clean ClassificationMetrics and mode restoration | 4 implemented |
| Training | train_run(config, data, model, optimizer, scheduler, run, *, runtime, streams, persistence, continuation=None) → durably acknowledged TrainingResult or TrainingRunError | 4 implemented; requires Step 5 service |
| Records | Typed epochs/test/failures/statistics, finite JSON envelopes, epoch/layer CSV renderers and guarded diagnostic JSONL append | 4 implemented |
| Checkpoint/continuation | Implement TrainingPersistence: commit_epoch, load_best, publish_completion, publish_failure; validate/restore full live state before supplying ContinuationState | 5; protocol implemented in Step 4 |
| Manifest/task selection | build_manifest(finalized dataset configs, seed plan) → 240 uniquely identified scientific tasks; select_task(manifest, task index/run ID) → one ready RunConfig | 6 |
| Slurm submission | render_submission(manifest, selected tasks, execution policy) → reviewable dgxh submission sources; later submit only in authorized execution workflow | 7 |
| Aggregation | collect_results(manifest) → every individual seed row and explicit completion/failure coverage; validation-selected alpha plus supporting mean/std | 8 |
| Plotting | plot_results(aggregate) → shareable figures with seed observations visible, uncertainty convention and missing/failure annotations | 8 |

These are responsibility contracts, not a mandatory class hierarchy. Keep
interfaces small and share one configuration/identity implementation. Runtime
component absence must produce a component-specific error in future execution,
not make a scientifically valid declaration invalid.

## Dataset and reproducibility boundary

Step 2 must validate dataset identity/integrity, official sizes, full coverage,
disjoint train/validation indices, and fixed ordering. Persist split metadata
and indices on shared storage. Define `scientific.data.split.identity` as
`sha256:` plus the SHA-256 of canonical sorted finite JSON containing a split
format version, dataset name, official-training-set identity, and the ordered
training and validation index arrays. The implemented official-set identity is
defined precisely in [datasets.md](datasets.md).
Ordering belongs to the hash; a filesystem location does not. A generation
seed/algorithm, if selected, is recorded in split metadata; no provisional seed
is specified in Step 1. Verify the hash before building loaders.

Resolve normalization from approved constants or an approved reproducible
training-data derivation, persist actual constants and their source identity,
and do not infer constants from test images. Preserve native image dimensions.
The training seed controls initialization, shuffle, augmentation, and weight
noise, not split selection. Record the stateless data epoch contract and preserve
independent model/noise RNG state during Step 5 continuation.
The implemented versioned SHA-256 keyed order/crop/flip streams are independent
of alpha, architecture, operational workers, and global model RNGs. Sampler keys
carry epoch into worker copies. See [datasets.md](datasets.md) for exact semantics.
Call DataLoaders.set_epoch before each training epoch. The minimal completed-epoch
data continuation record reconstructs the next epoch without mutable data RNG
cursors; Step 5 also restores separate model/noise/optimizer/scheduler state.

## Training and scheduler boundary

Step 3 implements the scheduler equation and completed-unit indexing; warmup
factor, cosine minimum and cadence require explicit approved configuration and
remain pending. The first unit uses L*f, n=W reaches L, and n=T is the boundary
minimum after all T units; the last training unit uses lr(T-1). Epoch cadence
advances after completed epochs, step cadence after successful optimizer updates.
AMP-skipped updates do not advance step cadence, and epoch/update counts remain
separate. Actual loader batch count defines the step horizon, without fixed
guessed counts. Models use isolated CPU float32 initialization independent of
alpha; optimizers retain one group of all clean trainable parameters.
Step 4 consumes the same original 200-epoch schedule; there is no budget field
for "remaining epochs" or shortened continuation. LR, clean model statistics,
sample counts, train/validation loss/accuracy, duration, and finite/failure status
are written for every epoch. Any regularization inherent in the selected
architecture remains part of the builder, never silently disabled across runs.

Perturbation and evaluation must agree on clean/master parameter ownership.
Owned CPU snapshots and layer statistics never capture temporary noise. Best
checkpoint selection uses clean validation accuracy and strict `>` improvement.
No early stopping on poor metrics; a numerical failure is recorded explicitly
and is never a completed result. AMP scaler state and noise RNG streams belong
to continuation when the finalized precision policy uses them.

Step 4's `EpochBoundary` contains the complete epoch record, clean statistics,
owned clean/model/optimizer/scheduler/scaler/RNG/data snapshot and previous commit
ID. `commit_epoch` must durably publish coordinated state/records before returning
a validated `CommitReceipt`. Only acknowledgement advances durable progress;
observed completed epochs in a failure may be greater. CSVs derive from committed
records and `train.log` is diagnostic only. Unknown transaction outcomes must be
reconciled by Step 5 before replay. See the training guide for exact typed signatures,
relative checkpoint references, copy ownership and receipt semantics.

## Checkpoint roles and required metadata

| Role | Meaning |
| --- | --- |
| latest | Durable full continuation state at a documented safe boundary; not a validation-selected model |
| best | Clean model at the highest validation accuracy seen; an exact tie retains the earlier epoch |
| final | Clean model after completing the original 200-epoch training budget; not a replacement for best |

Use zero-based epoch indices internally and record `completed_epochs` as a
count from 0 through 200. `next_epoch_index == completed_epochs` for epoch-boundary
continuation. Human displays may use epochs 1–200 but must label the conversion.
Data continuation uses completed-epoch boundaries. Replay any unfinished epoch
from the last committed boundary, recording lost work since latest. Do not infer
batch position from wall time or add speculative mid-batch resume machinery.

Every checkpoint requires:

- Checkpoint format version, configuration schema/specification/fingerprint
  versions, run ID and full scientific fingerprint.
- Effective scientific configuration (or a content-addressed full reference),
  dataset/split/preprocessing identity, and original 200-epoch budget.
- Role, clean model state, architecture/classifier metadata, completed epoch
  count and next update/epoch position, and clean-weights assertion.
- Best validation accuracy and its epoch, current epoch/step, sample/optimizer
  update counts, and scheduler position.
- Code/software/source-snapshot provenance and creation/allocation metadata
  recorded separately from scientific identity.

Latest additionally requires optimizer, scheduler, any AMP scaler, Python/NumPy/
torch CPU/CUDA RNG state, dedicated weight-noise state, and
DataLoaders.continuation_metadata(next_epoch_index) for the completed-epoch data
boundary. Data order/augmentation use reconstructed local generators rather than
saved mutable RNG cursors. Worker-only incidental RNG states do not affect samples.
Best/final need model and selection metadata; they must not be used as implicit
latest continuation state. Checkpoint tensors must be clean; normalization
buffers (including trained BatchNorm statistics) are preserved.

Step 5 validates identity, source/software compatibility, checkpoint role,
original budget, model shape, dataset identities and state availability **before
restoring or changing any run-specific artifacts**. It must not reinitialize the
LR schedule, optimizer, RNG streams, or best validation record. Allocation changes
and path relocation alone remain compatible. Extend Step 1's conservative
code/software check only with an explicit reviewed migration policy. Stale-lock
and partially initialized-run recovery must preserve existing content.

Construct the clean model, move it to its final device and build its optimizer.
Construct Step 3's scheduler with initialize=False before restoring optimizer
state; preflight its descriptor with validate_state_dict. After full checkpoint
compatibility checks, restore clean model/buffers and optimizer, then scheduler
state. Scheduler restoration checks saved position, original horizon and actual
restored optimizer LR before changing its position/rates. Restore completed
epoch/successful-update counters, scaler, data and separate model/noise RNG state
before further iteration. See the model/optimization guide for exact order;
scheduler state handling alone is not full checkpoint compatibility or recovery.

## Result/status contract and completion

Run status uses an explicit vocabulary: `initialized`, `training`, `interrupted`,
`trained_pending_test`, `evaluating`, `completed`, `failed_numerical`, or
`failed_execution`. The Step 1 metadata status is only `initialized`; later
status/result writers own transitions. Allocation ending may mean interruption
or failure and is never itself completion.

Every epoch/result/status artifact includes result format version, schema and
fingerprint versions, specification version, run ID, full scientific fingerprint,
dataset, architecture, canonical alpha, numeric alpha, training seed, split and
preprocessing identity, and separate code/software provenance. Use finite JSON
numbers; unavailable failed metrics are null plus status/reason.

Required final `results/summary.json` fields include:

- Identity/version metadata repeated at the top level, effective scientific
  settings/reference, status, and completed epoch count.
- Best checkpoint path/content digest, best validation accuracy and epoch,
  final checkpoint reference, and latest continuation reference.
- Clean test evaluation marker, test checkpoint role `best`, test sample count
  exactly 10,000, clean test loss and accuracy, and evaluation timestamp.
- Paths/content references for complete epoch metrics and clean layer statistics.
- Completion/failure reason, failed epoch/step if any, and allocation/attempt
  history reference. Paths/timestamps are operational, not fingerprint inputs.

`completed` requires all 200 epochs, durable best and final clean checkpoints,
prescribed clean test evaluation of best on all 10,000 images, required metrics/
layer statistics, and an atomically written valid final summary. Epoch 200 with
test evaluation pending is `trained_pending_test`, resumable for evaluation
without additional training. Full resume implementation belongs to Step 5.

Step 8 must join against the complete manifest, prominently retain individual
seed rows, and report missing/interrupted/numerically failed runs rather than
filtering them out. Best alpha is based on the mean validation result from the
complete three-seed set. Specify cross-seed std convention and how exact
validation-mean ties are reported before generating final aggregates. Report
test results for every alpha/seed, not only the selected alpha.

## Steps 2–4 completed in source; remaining steps in order

2. **Dataset preparation and reproducibility (implemented, execution deferred).** Official dataset integrity,
   persistent fixed splits, finalized normalization, seeding/augmentation
   contracts, full-set loaders, and artifact identity verification.
3. **Four architectures and fixed optimizer recipes (implemented, execution deferred).** Scratch CIFAR builders,
   complete eligible-weight inventory, exact recipe construction, documented
   original-budget warmup/cosine scheduler, and explicit remaining decisions.
4. **Shared training, perturbation, and clean evaluation (implemented, execution deferred).** One engine across
   all architectures, fixed-global simultaneous perturbation, clean evaluation,
   epoch metrics, layer statistics, numerical failure recording and required
   persistence protocol with owned snapshots. No checkpoint files/recovery yet.
5. **Durable checkpoints and continuation.** Atomic latest/best/final states,
   identity/provenance checks, complete RNG/optimizer/scheduler/scaler restoration,
   interruption handling, evaluation-only continuation, and artifact recovery.
6. **Full experiment controller and manifests.** Finalized 240-run grid, unique
   per-run configs/identities, one-task selection, coverage/status tracking,
   fixed recipes/decisions across alphas/seeds, and two 120-run dataset manifests.
7. **OSU dgxh execution.** One process/GPU per run, configurable allocation
   policy, persistent shared paths, compatible continuation across allocations,
   global concurrency control, and reviewable operational submission sources.
8. **Results aggregation and figures.** Seed-first results for every task,
   validation-based alpha selection, transparent supporting mean/std summaries,
   failure/coverage reporting, and final figures.
9. **Final source integration and operating documentation.** Source-only
   consistency review, dependency/CLI/manifest integration, and complete HPC
   operating instructions. Runtime validation occurs after all nine steps.

Each implementation step ends with source and Git-diff inspection, not executable
verification. Do not create tests, smoke configs, demonstrations, preliminary
experiments, or placeholder modules. Step 4 stops before Step 5 persistence/recovery.
