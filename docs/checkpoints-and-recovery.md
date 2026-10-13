# Durable checkpoints, continuation and recovery

Step 5 is implemented in source only. No project imports/execution, model
construction, tests, compilation, linting, training/evaluation, installations,
downloads, artifact generation, experiments or jobs were performed. Runtime
correctness remains unverified. Step 6 connects the training launcher and
[manifest/controller](manifests-and-controller.md). This service performs no
signal registration, Slurm submission or requeue operation.
Step 7 now supplies explicit submission, scoped signal delivery and bounded safe
allocation continuation separately in [slurm-workflow.md](slurm-workflow.md).
This checkpoint service itself registers no signals or submits/requeues jobs.
The [approved robustness integration](weight-robustness-evaluation.md) updates the
current lifecycle: journal 2 completes training independently of test; legacy
journal 1 remains read-only compatible. Independent evaluation never advances
this journal or writes latest/best/final/optimizer state.

## Interfaces and scope

| Interface | Responsibility |
| --- | --- |
| `persistence.DurableTrainingPersistence(run)` | Concrete synchronous `TrainingPersistence` service bound to one active run lease |
| `commit_initial(run, InitialBoundary) -> CommitReceipt` | Durable clean epoch zero after component construction and stream initialization |
| `commit_epoch(run, EpochBoundary) -> CommitReceipt` | Latest, selected best/final, metrics, statistics and selection in one logical transaction |
| `publish_training_completion(run, context, final_commit) -> TrainingCompletionReceipt` | Full-budget training completion with immutable training CSVs/summary, no test |
| `TrainingFinalizationPlan` / `open_run(operation="finalize_training")` | CPU full-budget publication, no live model/optimizer/scheduler/scaler/RNG restoration |
| `publish_failure(run, context, failure)` | Failure/interruption transaction preserving committed continuation state |
| `checkpoint_journal.read_committed(paths, config, provenance)` | Read-only stored-state preflight from authoritative HEAD, never inferred logs/CSV counts |
| `continuation.ContinuationPlan(data=..., model=..., optimizer=..., scheduler=..., runtime=..., streams=...)` | Complete constructed-component preflight, repeated before/under run locks, then ordered restoration |
| `open_run(..., operation="continue"/"recover", continuation_plan=plan)` | Existing ownership/locks extended with mandatory full preflight before attempt/config writes |
| `plan.restore(run) -> ContinuationState` | Restore compatible complete latest into original model/optimizer bindings |
| `recovery.inspect_training_completion` / legacy name `inspect_completion` | Read-only training completion; legacy clean evidence is explicit, never robustness proof |
| `service.repair_publication()` | Explicit recover operation: rebuild aliases/CSVs/status and finish safe pruning |
| `service.reconcile_numerical_failure(reference)` | Explicitly commit a selected compatible terminal numerical-failure evidence record |
| `recovery.recovery_diagnostics`, `inspect_locks`, `release_confirmed_locks` | Evidence listing and conservative operator-confirmed lock recovery |
| `interruption.StopRequest.request(policy, reason)` | Request finish_epoch/discard_epoch without serializing inside a handler |

Scientific schema/fingerprint remains 4. Checkpoint format remains 1; journal
version 2 explicitly supports reading legacy 1. Storage layout does not change scientific identity. No
provisional scientific setting or hardware entitlement is selected by this step.

## File layout and authority

The three selected roots may reside on different filesystems. References name
their root (`metadata`, `checkpoints`, `results`), relative POSIX path and actual
SHA-256. `CheckpointReference` is relative to the selected checkpoint root.

```text
<metadata>/
  resolved_config.yaml, run_metadata.json, status.json
  attempts/<attempt>/resolved_config.yaml, metadata.json
  state/HEAD.json
  state/transactions/<transaction-id>.json
<checkpoints>/
  .run_owner.json
  generations/<training-commit-id>.latest.pt
  generations/<selected-training-commit-id>.best.pt
  generations/<epoch-200-training-commit-id>.final.pt
  generations/*.part-<unique-id>               # possible interrupted evidence
<results>/
  .run_owner.json
  records/<training-commit-id>.json            # epoch + clean statistics
  failures/<transaction-id>.json
  completions/<transaction-id>.test.json
  completions/<transaction-id>.summary.json
  completions/<transaction-id>.epochs.csv
  completions/<transaction-id>.layers.csv
  training_completions/<transaction-id>.summary.json
  training_completions/<transaction-id>.epochs.csv
  training_completions/<transaction-id>.layers.csv
  epochs.csv, layer_statistics.csv, summary.json, test.json, failure.json
<logs>/
  .run_owner.json, train.log
```

Every initial/epoch/training-completion/failure transaction has an immutable metadata JSON document with identity, parent
transaction reference, kind/status, training receipt, counters, best selection,
complete small record-reference list, artifact references, context/provenance and
creation time. HEAD names the current transaction and previous transaction.
The parent hash chain is an audit trail; older checkpoint tensors can be retired.

HEAD is the **single authoritative commit point**. All checkpoint/record files
are fully written, flushed/fsynced, hashed and validated before HEAD is atomically
replaced within its metadata filesystem. Checkpoint staging is restricted-loaded
and validated before rename. Immutable generation names cannot replace existing
content. New directory entries and relevant POSIX directories are fsynced too.

Publishing CSVs, status, metadata and top-level summary/test aliases is subsequent
derived work. Separate replacements across roots are **not** one atomic operation.
A reader follows HEAD and validates its immutable references, including immutable
training-completion CSV/summary content (or legacy combined CSV/test/summary). It never counts train.log lines, CSV rows,
filenames, guessed epochs or timestamps to determine progress. An incomplete
alias publication cannot undo a committed epoch/completion.

Each result repeats schema/specification/fingerprint/record versions, run ID/full
fingerprint, dataset/model/alpha/seed, data identities and separate provenance.
Per-epoch CSVs are rebuilt from committed records, with exactly one row per epoch
and one layer row per eligible tensor per epoch. Replaying unfinished work cannot
add uncommitted rows or duplicates. Epoch-zero views contain zero rows.

Training completion commits immutable complete CSV content and summary with
effective science, explicit training alpha, latest/best/final references, strict
validation selection, full committed records/statistics, time and actual publisher
provenance separate from training provenance. No test is required in this process.
Legacy completions retain their recorded clean test/timestamp unchanged. Independent
evaluation has its own namespace/slot/sweep HEADs and raw schemas in the evaluation guide.

## Checkpoint format and distinct roles

Pinned torch 2.5.1 writes a dictionary of tensors and built-in scalar/containers.
Loading always specifies `map_location="cpu", weights_only=True`. There is no
unrestricted pickle fallback or custom-class/NumPy allowlist. Python RNG state is
builtin tuples/integers; NumPy MT19937 is explicitly encoded as algorithm, 624
uint32 integer keys, position, Gaussian-cache flag/value. Restoration reconstructs
NumPy state only after validating those fields. Torch RNG states are CPU uint8
vectors. Owned CPU snapshots contain no gradients, Parameters, autograd graphs
or functional perturbation mappings.

All roles carry checkpoint/component versions, scientific configuration/identity,
actual data/source/split/normalization/preprocessing identities, architecture/
classifier/component metadata, parameter/buffer inventory with shapes/dtypes,
clean tensors, original budget, boundary/counters/selection, state-component
versions and separate source/software/attempt/creation provenance.

| Role | State and permitted use |
| --- | --- |
| latest | Full clean model and learned buffers; optimizer; original-budget scheduler; optional scaler; Python/NumPy/CPU/CUDA/noise RNG; data continuation; counters/best and committed record references. Sole restoration source. |
| best | Strict validation-selected clean model and learned buffers plus identity/selection metadata. No optimizer/scheduler continuation payload. Ties preserve the earlier file/reference. |
| final | Clean model and learned buffers after all original 200 epochs. No implicit continuation payload; never substituted for best test. |

The initial latest is a real completed-count-zero checkpoint: zero attempts,
examples, successful/skipped updates; no best/final; fresh optimizer/scaler and
scheduler position zero, initialized model/runtime/noise streams. The trainer
requests `commit_initial` after fresh construction and stream initialization,
before its first batch, carrying the receipt as an initial commit rather than
mislabeling the run as continuation. Recovery restores that exact initialized
state, including its noise state, instead of seeding a new sequence.

## Transactions, lost acknowledgements and retention

A completed-epoch boundary includes the full training epoch, scheduling, clean
statistics, full validation, strict best decision and complete owned snapshot.
`commit_epoch` verifies record/counter/LR/selection/statistic coverage, creates its
record, latest, best on improvement and final at 200, then publishes HEAD. A
returned receipt means durable success; the service never returns queued work.

| Interruption point | Outcome/recovery |
| --- | --- |
| During checkpoint/record staging | Old HEAD remains authoritative; preserve incomplete/unused evidence and replay from latest. |
| All files written, before HEAD | Old HEAD wins; candidate transactions are not promoted based on filenames or apparent newer epochs. |
| HEAD replaced, receipt not received | Read and validate new HEAD; its complete boundary wins even when the caller's acknowledged count is older. |
| CSV/status/summary alias replacement | Rebuild views from committed records under explicit recovery; do not retrain to match a stale CSV. |
| Completion committed, acknowledgement lost | Read-only inspection recognizes completion. Repair its views; never downgrade or rerun training/test because later diagnostics failed. |

Retention protects every checkpoint referenced by **current and previous HEAD
transactions**: at most two latest, two selected best, and one final across the
normal workflow (often fewer because transactions share references). Superseded
role files are removed only after HEAD is durable, when proven as committed
ancestors, unreferenced by those live generations and still matching their
recorded digest/owned generation filename. Parent-chain documents and small
epoch/statistic/failure/completion records remain for reporting and ownership
evidence; their historical checkpoint references are audit information, not live
restoration promises. No full optimizer/model checkpoint is kept for every epoch.

Partial staging, uncommitted candidates, unrelated files, linked paths and archives
are never guessed disposable or automatically pruned. Repeated failed staging can
require operator storage review; the committed-state bound is not a license to
delete ambiguous evidence. Cleanup failures stop the operation with its HEAD
preserved; recovery can retry safe cleanup. Source-side cost of hashing/loading,
CPU snapshots and derived CSV rebuilds is unmeasured.

## Preflight and complete restoration

Before fresh construction for continuation, `read_committed` validates stored
identity/configuration, content digests, formats, roles, clean finite tensors,
inventories, original budget, component state, records, selection and provenance.
Then construct the existing clean builder model, move it to the assigned device,
build native optimizer and scheduler with `initialize=False`, and create runtime
streams with `initialize=False`. No constructor restores a saved component.

Build a `ContinuationPlan` from those live objects. For unfinished training,
`open_run` requires this concrete plan for continue/recover and performs full preflight before attempt/
configuration changes, then repeats it while holding the existing storage-root
locks. Preflight compares actual constructed model/buffer inventory and component
versions; optimizer type/recipe/options/order, required state and saved LR;
scheduler descriptor/position and counters; scaler mode/state; actual prepared
data/continuation identities; runtime backend policy and RNG descriptors/states.
It validates current and necessary previous references and every committed record.
It mutates no model/optimizer/scheduler/global RNG or run-specific artifact.

For unfinished training, source-snapshot digest and full software mapping must remain compatible. There
is no automatic code/software migration. Allocation IDs, operational worker
settings, storage locations and visible CUDA index can change; they do not enter
scientific identity. Actual CUDA/native precision support must pass runtime policy
checks on the assigned device. No promise of bitwise equivalence across arbitrary
hardware/software is made.

Within the active continue context, `plan.restore(run)` rechecks the exact HEAD
and all compatibility before mutation. It copies clean values/buffers with
`assign=False` to preserve original bindings, loads native optimizer state, then
loads scheduler state against the restored LR. Native optimizer loading maps
parameter-shaped state to parameter devices and leaves noncapturable step scalars
on CPU; no blanket CUDA conversion is applied. Next restore scaler, counters,
best and data continuation. Restore all model/runtime/noise RNG state **last**,
after constructors and copies. The returned `ContinuationState` feeds the existing
trainer. A restoration exception requires discarding the partial live components;
there is no fresh-state fallback or resuming past a failed restoration.

`next_epoch_index == completed_epochs`. Replay an unfinished epoch from the last
committed boundary, including learned buffers, optimizer, schedule, scaler and
randomness. No mid-batch state or shortened remaining-epoch schedule is introduced.
At count 200, command continuation runs zero epochs and uses publication-only
TrainingFinalizationPlan; it requires no live training components or test sweep.
For full-budget publication repair, recover may use the same CPU plan. Historical
training provenance validates old evidence; the actual publisher is recorded
separately. This cannot authorize learning under changed code/software.
Independent best loading copies into its separate evaluation model and advances
no training journal. Legacy v1 evaluating transitions remain historical only.

## Completion, failures and stop requests

Training completion validates all 200 records/statistics, durable latest/best/final,
strict clean validation selection and immutable training summary/CSV content.
Epoch 200 alone is training_pending_completion. Legacy v1 completed additionally
validates its original full clean test; it establishes no positive-alpha evaluation
slots. Valid full-budget legacy boundaries are independently evaluation-eligible.
Robustness completion requires its own 28 successful raw slots and sweep authority.
`inspect_completion` reads committed completion without constructing a model,
changing files or acquiring writer locks. Partial/corrupt/incompatible existing
state raises rather than masquerading as a missing task. Concurrent mutation can
cause a conservative read failure; inspect again after the writer exits.
Step 6 removes derived-summary validation from the ownership gate: malformed
aliases cannot veto valid immutable HEAD completion or explicit repair. The
original configuration/root markers and immutable authoritative results still
must validate. Execution-failure evidence at the latest training boundary also
blocks automatic controller continuation; historical evidence is preserved.
`ContinuationPlan(..., allow_execution_failure=False)` repeats this automatic
dispatch gate before and under the original locks. Existing explicit continuation
callers retain the default True; terminal numerical evidence is always rejected.

Failure/interruption records preserve observed versus acknowledged versus actual
committed progress, location, reason/type/tensors, transaction outcome and attempt
history. Unavailable loss/accuracy is null. Failed/partial live state is never saved
as latest. `publish_failure` reads HEAD first: an already committed completion
dominates a lost acknowledgement or later diagnostic error. Numerical failures
during unfinished training remain terminal and are rejected by continuation, including valid numerical
failure evidence whose publication did not reach HEAD. Explicit recovery can
commit a selected compatible orphan numerical record; it cannot resume it as a
wall-time interruption. If failure storage itself fails, the caller's
`TrainingRunError` reports that publication failed; no durable failure claim is
made. Future orchestration must retain that exit classification.

`StopRequest.request("finish_epoch", reason)` lets the current epoch finish its
full boundary and commit before stopping. `discard_epoch` stops at a batch/evaluation
boundary or before snapshot publication, discarding unfinished live work. Requests
are checked at committed entry/epoch boundaries, between training/evaluation
batches, after training batches and before snapshot publication. No partial
epoch gets a completed label. A request arriving during a synchronous commit may
allow that already-safe transaction to finish; the next check stops further work.
Caught requests produce `interrupted` and preserve latest. Stop requests register
no signals and perform no serialization inside a handler. Discard takes precedence
over finish; callers do not reuse/clear an old request for a later allocation.

## Conservative recovery and filesystem assumptions

Use `recovery_diagnostics` to inspect locks, valid HEAD outcome and reserved
evidence. Missing initial HEAD, damaged ownership/configuration, invalid digests,
incompatible components or ambiguous state are rejected explicitly. A directory
with no completed epoch or summary is still protected. Missing initial authority
is not reconstructed from a promising staged file; operator review or explicitly
selected archived overwrite is required. No speculative rollback from corrupt
current state to a guessed earlier epoch exists.

For compatible committed publication repair, construct/preflight a plan and use
`open_run(operation="recover", continuation_plan=plan)`, then
`DurableTrainingPersistence(run).repair_publication()` inside the context.
Full-budget repair uses the CPU TrainingFinalizationPlan; unfinished training repair
retains the full live plan and conservative source/software policy.
Recovery records an attempt and keeps all ownership/locks; it restores no live
training components and cannot call the trainer. Continue computes remaining work;
recover repairs publication; completed inspection only reads; selected overwrite
continues to archive the old run.

Locks are never stolen from elapsed age or local PID checks. `inspect_locks`
returns exact path/digest/token/host/PID evidence. A later operator must establish
that **all recorded owners and allocations across nodes are inactive**, explicitly
confirm that fact and supply a reason to `release_confirmed_locks`. The interface
rechecks all selected lock paths/owners/bytes and renames locks beside their original
locations, retaining audit records. Damaged/unverifiable owners are rejected for
manual review. Partial lock release preserves evidence and requires reinspection;
it is never automatic continuation permission.

Durability assumes the later HPC filesystems implement cooperative exclusive
creation, same-filesystem atomic replacement and meaningful file/directory fsync.
All writers retain the existing active run lease/per-root locks; noncooperating
writers are outside that guarantee. POSIX directory fsync errors are surfaced.
Windows skips directory fsync and carries no equivalent crash-durability claim.
HPC filesystem behavior and all runtime paths still require later verification.
Recovery neither submits jobs nor consults account entitlements.

Still pending: scientific seed plan/task seed, independent split preparation seed
and identities, normalization, drop_last, precision/backend policy, warmup factor,
cosine minimum and cadence. Step 6 consumes these APIs for controller/manifest
work; Step 7 owns signal delivery, Slurm/account/resource policy and resubmission;
Step 8 owns aggregation conventions. No dataset/prepared artifacts were generated
or rewritten and no scientific fingerprint change was made for storage.
