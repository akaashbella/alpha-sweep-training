# Shared contracts and remaining implementation steps

Steps 1–8 implement configuration, identity, finite atomic serialization,
provenance validation, guarded run initialization, full official CIFAR preparation,
reproducible loaders, all four scratch CIFAR models, owned eligible-weight
inventory, fixed optimizers, the original-budget scheduler, shared training,
functional perturbation, clean evaluation, RNG/precision policy, typed metrics
and numerical failures, restricted durable checkpoints, full continuation,
coordinated publication/recovery, stop requests, full manifests, one-task execution
and complete read-only coverage. These sources have not been
executed or runtime-verified. Step 7 now adds source-only operational policy,
bounded Slurm waves, shared workflow authority, signal delivery and allocation
continuation. Step 8 implements authoritative result snapshots, complete summaries,
validation co-winners, paired comparisons, guarded reporting and headless figures.
Final integration remains Step 9. Commands are for later HPC use. See
[models-and-optimization.md](models-and-optimization.md) and
[training-and-perturbation.md](training-and-perturbation.md) and
[checkpoints-and-recovery.md](checkpoints-and-recovery.md) for implemented contracts.
Step 6 interfaces are in [manifests-and-controller.md](manifests-and-controller.md).
The approved integration after Step 6 separates training completion and independent
weight_robustness_eval_v1; its implemented contracts are in
[weight-robustness-evaluation.md](weight-robustness-evaluation.md). This supersedes
global clean-only testing while retaining clean validation and the mandatory baseline.

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
| Training | train_run(config, data, model, optimizer, scheduler, run, *, runtime, streams, persistence, continuation=None, stop=None) → durably acknowledged TrainingResult or TrainingRunError | 4–5 implemented |
| Records | Typed epochs/test/failures/statistics, finite JSON envelopes, epoch/layer CSV renderers and guarded diagnostic JSONL append | 4 implemented |
| Checkpoint/continuation | DurableTrainingPersistence(run): commit_initial, commit_epoch, publish_training_completion, publish_failure; ContinuationPlan preflight/restore and CPU TrainingFinalizationPlan → ContinuationState | 5 implemented |
| Recovery/interruption | inspect_training_completion (legacy inspect_completion adapter), recovery_diagnostics, explicit recover publication/numerical evidence and confirmed lock recovery; StopRequest finish/discard | 5 implemented; signal registration/orchestration later |
| Manifest/task selection | build_manifest(dataset_configs, model_declarations, seeds, base_dir) → strict 240-task Manifest; load_manifest/load_phase/select_task/task_config → validated Task/config with execution-only overrides | 6 implemented |
| One-task execution | execute_task(manifest, task, base_dir, source_root, operation, overrides, stop) → TaskOutcome/ExitCode through existing engine, lease and continuation plan | 6 implemented |
| Coverage/provenance | inspect_task/collect_coverage → complete identity join and phase eligibility; collect_provenance → actual future execution source/software/allocation record | 6 implemented |
| Independent evaluation | EvaluationProtocol/slots/hash seeds; build_test_loader, evaluation-only runtime, evaluate_realization and CleanStateBaseline; verified best binding, independent slot/sweep persistence, execute_evaluation and full logical coverage | approved integration after 6 |
| Slurm workflow | OperationalPolicy/resource_budget, build_work_plan/render_work, WorkflowLease, submit_bundle, SlurmClient, run_allocation/scoped handlers → sealed mappings, durable intents/outcomes and one-shot reconciliation/advancement | 7 implemented, execution deferred |
| Aggregation | collect_results(manifest, base_dir, source_root, execution-only overrides) → sealed 240/6720-row historical collection; summarize_results → complete ddof=1 summaries, validation-only co-winners and paired descriptions | 8 implemented, execution deferred |
| Plotting/report publication | plot_results(collection, output_dir, source_root); collection/summaries publishers and inspect_report → owned checksummed derived outputs, six headless figure families, individual-first observations and masked coverage | 8 implemented, execution deferred |

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

Step 5 commits the initialized epoch-zero latest before any training batch.
Immutable generations/records precede one metadata `state/HEAD.json` commit point.
Current/previous live checkpoint references are protected; only digest-validated
superseded committed generations are pruned. Small records and ambiguous staging
evidence remain. CSV/status/summary aliases derive from HEAD, with immutable full
completion artifacts. Cross-root replacement is never claimed atomic.

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

Current journal format 2 uses `initialized`, `training`, `interrupted`,
`training_pending_completion`, `training_completed`, `failed_numerical` and
`failed_execution`. Explicit legacy journal-1 readers retain `trained_pending_test`,
`evaluating` and `completed`; the latter proves historical training plus its clean
test, never the new robustness sweep. Allocation ending and exit code zero do not
prove either scientific lifecycle complete.

Epoch records retain version-1 legacy `alpha`/`canonical_alpha` fields meaning
training alpha only. New summaries, evaluation records and selectors explicitly
name training_alpha and evaluation_alpha, training seed and evaluation replicate.
Training schema/fingerprint 4 and training manifest 1 remain unchanged. The
independent protocol fixes the ten evaluation alphas, clean replicate 0 with null
seed and positive replicates 1-3, formula/inventory/fixed-pass lifetime, seed hash,
preprocessing, precision and checkpoint policy. Its fingerprint never changes a
training run's identity.

Training completion requires all 200 epochs, verified latest/best/final,
validation selection, complete epoch metrics/clean-layer statistics and immutable
training CSVs/summary referenced by the training HEAD. It requires no test.
Count-200 auto/continue performs CPU training finalization with zero additional
epochs, no runtime/optimizer/streams. Valid legacy full-budget boundaries are
independently evaluation-eligible; legacy completion/evidence is not rewritten.
Unfinished continuation keeps strict source/software and live-state checks.

Evaluation independently binds the saved immutable best checkpoint and its actual
digest using stored training provenance. Actual evaluator provenance is separate.
`evaluate_realization` constructs one simultaneous float32 fixed functional
network per complete 10,000-image pass, with exact whole-clean-state preservation
and a dedicated hash-derived device generator. Zero delegates to the clean
evaluator with no noise. Test-only loader/runtime require no optimizer, scheduler,
scaler continuation or training RNG restoration. Failed/partial metrics are null;
failed preservation prevents success. Training state never changes on eval failure.

Separate owned namespaces and per-slot leases publish immutable digest-validated
raw records through slot HEADs; SWEEP_HEAD establishes robustness completion only
after all 28 successful slots. Interrupted passes restart from the same seed;
completed compatible passes skip. Failures, incompatible/locked observations and
settled failed sweeps remain explicit. Qualified legacy clean imports preserve
original references and require immutable full-coverage, preprocessing, precision,
provenance and exact preservation/runtime evidence; standard old records lacking
that proof remain historical evidence and need a new clean pass, without retraining.

The training manifest remains 240 tasks/two 120-task views. Linked evaluation
planning describes 6,720 expected logical passes without fabricating checkpoint
digests or job counts. Training phase eligibility refers to training completion;
robustness pending cannot force training again. Exact APIs/commands, raw schemas,
version/compatibility policies and Step 7/8 consumers are documented in the
evaluation guide. No test results select checkpoints, alpha or hyperparameters.

Step 8 joins against the complete manifest, prominently retains individual
seed rows, and report missing/interrupted/numerically failed runs rather than
filtering them out. Best alpha is based on the mean validation result from the
complete three-seed set. Sample SD is ddof=1 and exact unrounded validation-mean
ties are all reported in grid order. Incomplete grids have no official winner. Report
test results for every alpha/seed, not only the selected alpha.

## Steps 2–8 completed in source; remaining step in order

2. **Dataset preparation and reproducibility (implemented, execution deferred).** Official dataset integrity,
   persistent fixed splits, finalized normalization, seeding/augmentation
   contracts, full-set loaders, and artifact identity verification.
3. **Four architectures and fixed optimizer recipes (implemented, execution deferred).** Scratch CIFAR builders,
   complete eligible-weight inventory, exact recipe construction, documented
   original-budget warmup/cosine scheduler, and explicit remaining decisions.
4. **Shared training, perturbation, and clean evaluation (implemented, execution deferred).** One engine across
   all architectures, fixed-global simultaneous perturbation, clean evaluation,
   epoch metrics, layer statistics, numerical failure recording and required
   persistence protocol with owned snapshots, consumed by Step 5.
5. **Durable checkpoints and continuation (implemented, execution deferred).** Restricted latest/best/final states,
   identity/provenance checks, complete RNG/optimizer/scheduler/scaler restoration,
   durable initial boundary, coordinated single-point transactions, bounded
   retention, stop requests, CPU training-completion finalization, read-only completion
   inspection and conservative artifact/lock recovery.
6. **Full experiment controller and manifests (implemented, execution deferred).** Finalized 240-run grid, unique
   per-run configs/identities, one-task selection, coverage/status tracking,
   fixed recipes/decisions across alphas/seeds, and two 120-run linked phase views;
   safe one-task execution, actual provenance collection and strict failure/ownership
   dispatch. No actual manifests generated; scientific inputs remain pending.
The approved **evaluation integration after Step 6** is implemented in source:
independent protocol, fixed full-network realizations, exact state preservation,
versioned seeds, training/evaluation completion, raw durable slots, resumable
selectors, logical coverage and documented Step 7/8 interfaces. Runtime verification
remains deferred.

7. **OSU dgxh execution (implemented in source, execution deferred).** One process/GPU per run, configurable allocation
   policy, persistent shared paths, compatible continuation across allocations,
   global concurrency across training and evaluation, independent evaluation grouping/dependencies/resubmission, and reviewable operational submission sources.
   Strict versioned operational policy, one unresolved shared wave, sealed work/index/
   manifest/policy/source/checkpoint linkage, durable pre-sbatch intent, exact scheduler
   reconciliation, scoped StopRequest delivery, safe bounded continuation, explicit
   failure/submission/lock recovery and immutable allocation outcomes. See
   [slurm-workflow.md](slurm-workflow.md). No manifests, bundles or jobs generated.
8. **Results aggregation and figures (implemented in source, execution deferred).** Seed-first results for every task,
   validation-based alpha selection, transparent supporting mean/std summaries,
   failure/coverage reporting, raw training-alpha × evaluation-alpha analysis retaining seeds and realizations separately, and final figures.
   Historical journal/slot validators, complete prescribed two-stage sample SD,
   all-ten validation-only co-winners, paired differences, read-only snapshots,
   derived namespace ownership and PNG/PDF/SVG families are documented in
   [results-and-figures.md](results-and-figures.md). No aggregates/figures generated.
9. **Final source integration and operating documentation.** Source-only
   consistency review, dependency/CLI/manifest integration, and complete HPC
   operating instructions. Runtime validation occurs after all nine steps.

Each implementation step ends with source and Git-diff inspection, not executable
verification. Do not create tests, smoke configs, demonstrations, preliminary
experiments, or placeholder modules. Step 8 stops before Step 9 final integration.
Actual HPC setup, accounting/driver/filesystem validation and runtime
verification remain authorized work after the complete implementation.
