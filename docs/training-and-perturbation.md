# Shared training, temporary perturbation, and clean evaluation

Step 4 is implemented in source only. No imports, models, forwards/backwards,
tests, compilation, linting, training, evaluation, installations, downloads,
artifacts, experiments or jobs were executed to verify it. Runtime correctness
remains unverified until the authorized OSU HPC work after all nine steps.
Checkpoint persistence/recovery is implemented in Step 5's
[checkpoint guide](checkpoints-and-recovery.md). The concrete service must be
supplied explicitly; Step 6's [controller](manifests-and-controller.md) now connects
one selected task without an implicit persistence bypass.
The approved [independent robustness integration](weight-robustness-evaluation.md)
now separates training completion from test. Clean validation and all training
mathematics below remain unchanged; the former combined clean-test handoff is retired.

## Interfaces and construction order

| Module/interface | Responsibility |
| --- | --- |
| `runtime.build_runtime(config) -> RuntimePolicy` | Explicit CUDA device/backend policy, autocast, scaler, batch validation/transfer |
| `randomness.RuntimeStreams(config, device, initialize=...)` | Process model RNGs, private noise RNG, owned in-memory states |
| `perturbation.training_forward(model, inputs, *, alpha, noise_generator)` | Exactly one ordinary or functional training forward |
| `evaluation.evaluate_clean(model, loader, *, split, config, runtime)` | Full validation/test `ClassificationMetrics` with safe mode restoration |
| `training.train_run(config, data, model, optimizer, scheduler, run, *, runtime, streams, persistence, continuation=None, stop=None)` | Durable initial boundary, remaining original epochs, validation selection, commits and training-only completion |
| `metrics` | Typed counters/results/statistics/failures, finite JSON envelopes and CSV rendering, diagnostic log append |
| `numerical` | Batched finite diagnostics and clean-state guards |
| `training_contracts` | Owned snapshots, commit/load/completion receipts and required `TrainingPersistence` protocol |

Finalize research settings and use existing read-only `build_loaders`. Apply
`build_runtime`, construct the scratch CPU float32 model with `build_model`, move
it to `runtime.device`, then build its optimizer and original-budget scheduler
using actual `len(data.train)`. The optimizer must own this exact model instance.
Construct fresh `RuntimeStreams(..., initialize=True)` once after model placement.
Keep `open_run` active throughout trainer/service use. Its handle carries attempt
provenance and an active lease; escaped handles cannot authorize later writes.
There is one owning process/thread and one assigned CUDA device per run.

The trainer requires the real Step 5 service before computation and checks whole-run
research readiness. Component constructors retain their narrower readiness gates;
pending declarations remain valid without importing compute modules. Configuration,
ownership and service preconditions can raise actionable errors before a training
attempt begins. Once the attempt context is established, computation/commit errors
produce a typed failure and request durable failure publication.

## Clean ownership and the temporary forward

`functional-fp32-training-v1` uses the sorted authoritative inventory from
`weights.discover_eligible_weights`. Every owned Conv2d/Linear weight participates,
including stems, projection, depthwise, pointwise, SE and classifier weights.
For each mini-batch and each entire current tensor:

`temporary = clean + alpha * std(clean.detach(), correction=0) * epsilon`

Standard deviation and independent standard-normal `epsilon` are float32,
computed outside autocast with the scale/noise branch detached. The final addition
remains tracked: its derivative with respect to the original clean parameter is
the identity. Backward evaluates the gradient at the temporary weight and delivers
it to the optimizer's original float32 Parameter. Native updates and weight decay
apply to clean values. There are no in-place noise additions/subtractions and no
replacement Parameters. Biases, normalization and other unperturbed trainable
parameters retain normal gradients, updates and declared weight decay.

All temporary weights enter a single strict `torch.func.functional_call` with the
original parameter/buffer names. Original buffer references are passed through,
so BatchNorm learns running statistics from the actual noisy activations. Native
dropout and stochastic depth remain active. There is no second clean training
forward. The pinned functional-call context restores parameter bindings on exit,
including exceptions; the wrapper checks parameter and buffer object ownership.
Mappings are discarded and never enter logs/snapshots. An interrupted forward can
have partially updated learned buffers; Step 5 replays from the prior committed
boundary rather than saving that partial epoch.

Alpha zero takes the ordinary model forward before inventory, perturbation std
or noise work. It consumes no dedicated noise draws. End-of-epoch **clean layer
statistics** are still computed for alpha zero, as required for all runs.
Clean evaluation uses ordinary model calls with learned parameters and learned
buffers, `eval()` and `inference_mode()`. It never accepts a noise generator,
recomputes BatchNorm, or updates BatchNorm statistics. Every prior submodule mode
is restored individually even on error; the trainer also restores its entry modes.

The implementation follows pinned PyTorch's
[functional call](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/_functorch/functional_call.py)
and [binding restoration](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/nn/utils/stateless.py)
source contracts. Reading these sources is not runtime verification.

## Randomness and continuation

`sha256-runtime-streams-v1` hashes sorted, compact UTF-8 JSON (`ensure_ascii=True`,
`allow_nan=False`) with exactly `version`, `stream`, `dataset`, `architecture`,
and selected `training_seed`. The first eight SHA-256 bytes interpreted
little-endian form the uint64 seed. Alpha, paths and allocation settings are absent.
Stream names are `python`, `numpy`, `model_runtime_cpu`, `model_runtime_cuda`, and
`weight_noise`; NumPy's legacy global seed is reduced modulo 2**32.

Python, NumPy, torch CPU and assigned-device CUDA globals belong to the one model
runtime owner. Dropout/stochastic depth use the model globals. Weight noise uses
its own CUDA `torch.Generator`; neither model operations nor artifact callbacks
consume it. Fresh streams are seeded once, never per epoch or batch. Initialization
retains Step 3's separate isolated CPU stream. Data order/crop/flip and worker
generators retain Step 2's versioned local keys, including epoch in sampler keys
for persistent workers; architecture and alpha do not enter those data keys.

Positive-alpha matched dataset/model/seed runs use comparable underlying noise
streams. Each attempted perturbation draws fresh noise regardless of a later AMP
skip; skipped updates do not rewind it. Alpha zero leaves its stream untouched.
Training metrics use the same actual forward, so logging adds no model draws.
`streams.preserve()` protects global and private RNG state around logging,
snapshot/persistence callbacks and separate test-model construction/loading.
Callbacks must run serially and must not leave asynchronous RNG consumers behind.

`state_dict()` owns copies of Python, NumPy, torch CPU, assigned CUDA and private
noise states plus version/seed/software descriptors. On continuation construct
with `initialize=False`, validate and `load_state_dict` **after** construction and
restoring other components. Do not repeat fresh-run initialization. `claim` checks
fresh/restored origin and prevents reusing streams for another trainer invocation.
Device indices may change across allocations; Step 5 must validate hardware/software
compatibility separately. No bitwise promise is made across arbitrary GPUs or
software versions. Data continuation uses the existing stateless epoch contract.

## Precision and update detection

All scientific precision fields must be explicit. Master parameters and floating
buffers remain float32. Train and clean evaluation use the same forward policy:

| Mode | Forward | Backward/update |
| --- | --- | --- |
| `float32` | Autocast disabled | Ordinary backward/native optimizer; nonfinite gradients terminate |
| `amp_float16` | CUDA float16 autocast | Native CUDA GradScaler, unscale, check gradients, scaler step/update |
| `amp_bfloat16` | Native CUDA bfloat16 autocast | No scaler; nonfinite gradients terminate |

Cross-entropy is unsmoothed mean loss calculated from float32 logits outside
autocast in every mode. Metric sums use float64 accumulation. Perturbation and
clean weight statistics always use float32 outside autocast. Float16 uses pinned
GradScaler defaults: initial scale 65536, growth factor 2, backoff 0.5 and growth
interval 2000; continuation restores current scale and growth tracker.

The runtime applies and records matmul precision, matmul/cuDNN TF32, deterministic
algorithms (without warn-only fallback), cuDNN deterministic/benchmark settings,
device properties and actual software. Pinned PyTorch couples CUDA matmul TF32
to matmul precision: `highest` requires `allow_tf32=False`; `high`/`medium` require
true. Set the matmul precision after the boolean setter to preserve `medium`.
Deterministic CUDA requires `CUBLAS_WORKSPACE_CONFIG=:4096:8` or `:16:8` in the
environment **before Python launches**; Step 4 never repairs it late. Native
bfloat16 support and advertised wheel/device architectures are checked at future
runtime. No CPU fallback or silent precision substitution exists. Later device
or backend-policy mutation is rejected.

These details follow pinned [backend setters](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/Context.cpp)
and [GradScaler](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/amp/grad_scaler.py).
A public [optimizer step post-hook](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/optim/optimizer.py)
counts actual native optimizer calls: normal steps may return None, so the return
value is never used as an update signal. The fixed unfused native optimizer
contract is enforced. A scaler-managed nonfinite-gradient skip must have zero
optimizer calls; finite gradients must have exactly one. Disagreement is terminal.
Every managed skip records batch location, gradient names, before/after scales and
counts. There is no skip threshold, clipping, catch-up update or early stopping.

## Epoch ordering and records

Before each zero-based epoch, `data.set_epoch(epoch_index)` and `model.train()`
apply. Each batch clears gradients, performs one forward/CE/backward, then updates
clean parameters or records a managed skip. Optimizer-step scheduling advances
only after a confirmed finite update. Epoch scheduling advances once after full
training coverage. Neither validation nor test advances scheduling. Both retain
the original 200-epoch horizon; skipped updates may leave step cadence short of
its mathematical endpoint at epoch 200. No extra work compensates for this.

The complete boundary is: training coverage, epoch scheduling when applicable,
clean model/optimizer checks, clean layer statistics, full clean validation,
strict best selection, counters, owned snapshot, then acknowledged persistence.
Clean layer std is computed after all training updates and before validation,
over every entire eligible clean tensor with correction zero. Records contain
qualified name, module type, shape, element count and std, including alpha zero.

Loss is sum(batch mean CE * actual batch size) / actual examples. Accuracy is
integer correct / actual examples, in fraction units. Positive-alpha metrics are
`perturbed_training`; alpha-zero metrics are `clean_training`. Batch/epoch counts
come from actual loader metadata and the explicit `drop_last` policy. Full clean
validation/test must cover exactly 5,000/10,000 examples. Best replacement uses
strictly greater validation correct count; equal counts keep the earlier epoch.
Test values never select checkpoints or alpha.

`Counters` separates attempted batches, processed examples, confirmed successful
updates, managed skips and completed epochs. Attempts increment before batch
validation; processed examples increment after a finite forward/loss is accumulated.
Success/skip counts increment after the update helper's checks return. A terminal
error inside an update may leave partial live changes and an incomplete observed
counter pair. These are failure diagnostics, never continuation state. A complete
boundary requires successes + skips == attempts and exact epoch coverage.

`EpochRecord` includes those per-epoch counts and cumulative totals, training and
clean validation metrics, best selection, AMP skip details, scheduler positions
before/after and cadence. `lr_first_attempt`/`lr_last_attempt` are the rates used
for those forwards/updates; `lr_at_boundary` is the installed next rate. Duration
covers training (including any AMP-skip diagnostics), statistics and validation,
excluding boundary snapshot/commit/log time.
`TestRecord`/CompletionReceipt remain legacy v1 read types. New independent raw
evaluation records name training_alpha and evaluation_alpha/replicate separately.
TrainingResult now contains counters, best, final commit and TrainingCompletionReceipt;
it contains no test result. Legacy training-record alpha fields mean training only.

All records use `RecordContext`: format/schema/specification/fingerprint versions,
run ID/full fingerprint, dataset/model/numeric and canonical alpha/seed, split/
normalization/preprocessing identity, attempt and separate provenance.
`context.envelope(kind, payload)` adds a UTC timestamp and rejects nonfinite JSON.

| Artifact | Format and authority |
| --- | --- |
| `logs/train.log` | Append-only finite UTF-8 JSONL envelopes: entry, managed skip, acknowledged epoch, failure. Historical best-test entries remain legacy evidence. Flushed/fsynced diagnostics; never a resume counter. |
| `results/epochs.csv` | `epoch_csv` renders one row per committed epoch with flattened training/validation/totals, fraction/loss units, LR/counters/selection/status; best and skips are JSON cells. |
| `results/layer_statistics.csv` | `layer_statistics_csv` renders rows by committed epoch and qualified weight name with clean marker, boundary, shape JSON, numel, std/correction zero and identity/provenance columns. |

CSV rendering is pure source functionality; Step 5 chooses header/publication
within its transaction. It must derive CSVs from committed records and prevent
duplicates on replay. Durable test/failure/summary documents use the same finite
envelopes and established final-summary fields in the implementation plan.

## Numerical failures and interruption

Finite checks cover input, perturbation scales/temporary weights, logits/loss,
unscaled gradients, scaler state and clean parameters/buffers. Clean state is
checked at training entry, after each batch update/skip, at epoch/snapshot
boundaries and evaluation entry/exit. Optimizer state is checked at entry and
epoch/snapshot boundaries. Tensor diagnostics reduce on-device and transfer stacked
flags per device, retaining tensor names without a host synchronization per layer.
Layer-statistic values transfer together and failures retain qualified names.

Only float16 scaler-managed overflow is a nonterminal recorded skip. Nonfinite
forward/loss, bad clean/optimizer state, invalid scaler state and nonfinite
unscaled gradients without a scaler yield `failed_numerical`. Other exceptions
yield `failed_execution`; caught KeyboardInterrupt/SystemExit yield `interrupted`.
No signal handler is registered. Step 5 adds StopRequest and explicit recovery;
abrupt termination can still prevent failure publication, so recovery follows
durable HEAD rather than diagnosing completion from logs or allocation end.

`FailureRecord` has phase/epoch/batch, numerical stage, reason/type/affected tensors,
observed counters, last acknowledged commit and durable completed count. Unavailable loss
and accuracy are null, never NaN/Infinity or fabricated scores. `EvaluationFailure`
preserves the validation/test batch location and original exception cause, including
its numerical classification. A failed epoch does not become a CSV success row.
Failure publication uses CPU records without
trying to checkpoint the failed live state or query a failed CUDA context.
`TrainingRunError` exposes the record, whether publication returned successfully,
and any publication/logging error; durability is not claimed if publication fails.

## Exact Step 5 handoff

The current training service implements the three methods below plus
`commit_initial(run, InitialBoundary) -> CommitReceipt`. Fresh training requests
this durable clean initialized boundary before its first batch: zero counters,
no best/final, initialized RNG state. The receipt remains a fresh-run receipt,
not a manufactured ContinuationState. A count-zero continuation restores it.

1. `commit_epoch(run, boundary: EpochBoundary) -> CommitReceipt`: commit latest
   full continuation state, epoch metrics, clean statistics and selection together;
   save best when `boundary.save_best`, final when `boundary.save_final`. A receipt
   means synchronous **durable** success, never merely queued work. It carries
   identity, new commit ID, completed count and validated latest/best/final relative
   paths with SHA-256 digests. Preserve earlier best on ties/non-improvement.
2. `publish_training_completion(run, context, final_commit) -> TrainingCompletionReceipt`:
   commit all full-budget checkpoint/selection/record/CSV authority and a training-only
   summary without test. Independent evaluation reads best and uses its own result
   namespace, preserving training's final optimizer/latest separately.
3. `publish_failure(run, context, failure: FailureRecord) -> None`: durably publish
   failure/interruption without saving failed live state, deleting checkpoints or
   destroying evidence of an unknown epoch/completion transaction.
   Reconcile unknown completion first; never downgrade a committed completion
   because a later diagnostic or acknowledgement path failed.

`TrainingSnapshot` owns detached CPU copies of clean model parameters/buffers and
optimizer tensors, copied scheduler/scaler state, all RNG states, data continuation,
counters, best selection, the guarded handle's effective config (resolved storage
paths), provenance and actual component metadata.
Copying finishes before further updates; `state_dict()` references alone are not
snapshots. No gradients or temporary tensors are included. Treat boundary/context
objects as read-only. The service can retain owned snapshots or pass them to a
writer, but must await durability before returning; all use must retain `open_run`.
This in-memory snapshot format does not define a checkpoint-file format.

For continuation, Step 5 validates full saved identity/provenance/data/recipe/
shape/precision/state compatibility before mutations, constructs/moves the model,
builds optimizer and scheduler with `initialize=False`, restores model and optimizer,
then scheduler/scaler/data/counters/best and RNG states. It passes `ContinuationState`
only after restoration, with the acknowledged receipt and data continuation record.
The trainer checks counter coverage, scheduler position and `next_epoch_index ==
completed_epochs`. There is no reduced remaining-budget schedule.

Observed progress may exceed acknowledged progress when a commit fails. The trainer
stops and reports `unknown_until_acknowledged`; Step 5 reconciles possible committed
transactions before any replay. A complete acknowledged epoch remains authoritative
even if its later diagnostic append fails. Replay an unfinished epoch from latest,
including its saved RNG/scaler/optimizer/scheduler/buffer state. Step 5 also
handles the durable initialized boundary and rejects partially initialized
artifacts without authority; existing files stay protected.

Epoch 200 commits `training_pending_completion`; training-only completion follows.
Count-200 command dispatch uses CPU TrainingFinalizationPlan/publication without
epochs, loaders, CUDA, optimizer/scheduler/scaler or stream restoration. Legacy
trained_pending_test/evaluating/completed are handled explicitly as v1 evidence;
they never imply complete robustness. Step 5 supplies checkpoint files, restoration,
publication repair, conservative lock recovery and finish/discard stop requests.
Step 6 connects runnable production task commands in source; Slurm/signal
orchestration remains Step 7. Independent evaluation provides its own test loader,
evaluation-only runtime, fixed whole-pass perturbation and exact whole-state guard.

## Decisions and versions

Schema/fingerprint 4 adds only fixed `training.engine_version` and
`training.randomness_version`. Additive migrations 1 -> 2 -> 3 -> 4 preserve all
pending settings and earlier contracts. Existing prepared YAML is checksummed in
its original bytes, migrated only in memory and never rewritten. Old run artifacts
cannot be blindly continued under a new fingerprint.

Still unresolved: exact three seeds/task seed, independent split preparation seed
and persisted split, normalization policy/constants for each dataset, `drop_last`,
all precision/backend settings, warmup factor, cosine minimum and cadence. Cross-seed
std convention and best-alpha tie reporting remain Step 8 decisions. No provisional
value has become a research default. See [configuration.md](configuration.md).
