# Shared contracts and remaining implementation steps

Step 1 implements configuration, identity, finite atomic serialization,
provenance record validation, path resolution, and guarded artifact initialization.
Everything described below for datasets/models/training/checkpoints/controllers/
Slurm/reporting is a future interface, not an implemented module. There are no
placeholder builders or executable experiment entry points.

## Responsibilities and interfaces

| Component | Input → output / responsibility | Step |
| --- | --- | --- |
| Configuration | load_config(base_path, dataset_path, model_path, typed overrides) → RunConfig; strict declaration validity separate from require_research_ready | 1 |
| Dataset preparation | prepare_dataset(dataset, data_root, approved split/normalization decisions) → persisted verified dataset/split/preprocessing metadata; no training-seed-dependent split | 2 |
| Data loading | build_loaders(config, split artifact, training RNG state) → train/validation/test loaders and dataset metadata; full validation/test sets, configured train drop_last | 2 |
| Model construction | build_model(config) → scratch CIFAR-adapted model with dataset classifier; report eligible parameter inventory | 3 |
| Optimizer/scheduler | build_optimizer(model, config) and build_scheduler(optimizer, config, steps_per_epoch) → exact fixed recipe and original-budget schedule | 3 |
| Perturbation | perturbed_training_weights(model, alpha, noise_generator) → exception-safe temporary perturbed view with detached scale/noise and gradients to clean/master parameters | 4 |
| Clean evaluation | evaluate_clean(model, loader) → full-set loss/accuracy/count with no temporary noise and correct model-mode restoration | 4 |
| Training | train_run(config, data, model, optimizer, scheduler, run_handle, continuation_state) → epoch records, clean statistics, checkpoint requests, explicit status | 4 |
| Checkpoint/continuation | save_checkpoint(role, state, run_handle); load_continuation(latest, config, provenance) → compatible full state at safe boundary; distinct clean best/final model loads | 5 |
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
training and validation index arrays. The dataset preparation implementation
must define that official-set identity precisely before creating artifacts.
Ordering belongs to the hash; a filesystem location does not. A generation
seed/algorithm, if selected, is recorded in split metadata; no provisional seed
is specified in Step 1. Verify the hash before building loaders.

Resolve normalization from approved constants or an approved reproducible
training-data derivation, persist actual constants and their source identity,
and do not infer constants from test images. Preserve native image dimensions.
The training seed controls initialization, shuffle, augmentation, and weight
noise, not split selection. Define independent RNG streams and save their states.
Sampler/augmentation identity must remain independent of operational worker
counts (e.g. deterministic epoch/sample keyed randomness); otherwise a worker
count cannot honestly remain outside scientific identity. Step 2 must settle
this contract and record exact transform semantics before execution.

## Training and scheduler boundary

Step 3 fixes the scheduler equation, first-update indexing, warmup endpoint,
cosine endpoint, and update cadence using explicit approved configuration.
Step 4 consumes the same original 200-epoch schedule; there is no budget field
for "remaining epochs" or shortened continuation. LR, clean model statistics,
sample counts, train/validation loss/accuracy, duration, and finite/failure status
are written for every epoch. Any regularization inherent in the selected
architecture remains part of the builder, never silently disabled across runs.

Perturbation and evaluation must agree on clean/master parameter ownership.
Snapshots and layer statistics must never capture temporary noise. Best
checkpoint selection uses clean validation accuracy and strict `>` improvement.
No early stopping on poor metrics; a numerical failure is recorded explicitly
and is never a completed result. AMP scaler state and noise RNG streams belong
to continuation when the finalized precision policy uses them.

## Checkpoint roles and required metadata

| Role | Meaning |
| --- | --- |
| latest | Durable full continuation state at a documented safe boundary; not a validation-selected model |
| best | Clean model at the highest validation accuracy seen; an exact tie retains the earlier epoch |
| final | Clean model after completing the original 200-epoch training budget; not a replacement for best |

Use zero-based epoch indices internally and record `completed_epochs` as a
count from 0 through 200. `next_epoch_index == completed_epochs` for epoch-boundary
continuation. Human displays may use epochs 1–200 but must label the conversion.
If later supporting a mid-epoch safe boundary, record sampler cursor and update
position explicitly; never approximate the next batch from a wall-clock cutoff.
Keeping continuation at complete epoch boundaries is acceptable, with lost work
since latest recorded explicitly.

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
torch CPU/CUDA RNG state, dedicated weight-noise and sampler/augmentation RNG
states, and any cursor/data-loader state necessary for its promised safe boundary.
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

## Remaining steps in order

2. **Dataset preparation and reproducibility.** Official dataset integrity,
   persistent fixed splits, finalized normalization, seeding/augmentation
   contracts, full-set loaders, and artifact identity verification.
3. **Four architectures and fixed optimizer recipes.** Scratch CIFAR builders,
   complete eligible-weight inventory, exact recipe construction, documented
   original-budget warmup/cosine scheduler, and explicit remaining decisions.
4. **Shared training, perturbation, and clean evaluation.** One engine across
   all architectures, fixed-global simultaneous perturbation, clean evaluation,
   epoch metrics, layer statistics, and numerical failure recording.
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
experiments, or placeholder modules. Step 1 stops before dataset implementation.
