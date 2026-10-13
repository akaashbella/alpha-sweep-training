# Independent weight robustness evaluation

The approved `weight_robustness_eval_v1` integration supersedes the former global
clean-only test rule. Clean validation still selects checkpoints. Training remains
the same 240 tasks, each with 200 epochs and the existing recipes and randomness.
This integration is source-only: no tests, imports/project execution, executable
verification, installation, downloads, preparation, artifact generation,
experiments or jobs were performed. Runtime correctness remains unverified.

## Protocol and the two alphas

`scientific.alpha`, legacy record `alpha`/`canonical_alpha`, and `__alpha-...` in
training run IDs mean **training alpha only**. `RunConfig.training_alpha` and
`identity.training_alpha` are explicit adapters. Schema/fingerprint version 4,
main-v1 declarations, training manifest format 1 and scientific hash bytes remain
unchanged. The frozen `scientific.evaluation` block means clean validation,
validation selection and the mandatory clean test baseline; it does not declare
the independent robustness protocol. Its old clean_weights field is retained for
wire compatibility, not interpreted as a ban on this approved sweep.

The separate `EvaluationProtocol` is built from central spec constants and the
resolved dataset evaluation preprocessing/normalization and precision/backend
policy. It fixes the evaluation grid, replicate policy, noise formula/scope,
full-pass lifetime, checkpoint selection, zero bypass, exact preservation, RNG
keys and version, test size/batch size, float32 master/noise/loss and no scaler.
`validate_protocol` rejects unknown fields/settings/versions by comparing the
entire canonical declaration. Its fingerprint is SHA-256 of compact sorted finite
UTF-8 JSON, independently of training identity. No paths/allocations/job grouping
enter this fingerprint. New protocol versions require an approved declaration;
there is no execution flag to change scientific evaluation settings.

The evaluation alpha grid is 0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00.
Alpha zero has one slot, replicate 0 and null perturbation seed. Each positive
alpha has replicates 1–3: 28 slots per best checkpoint, 6,720 across the 240-task
training grid. These are passes, not training runs or fixed Slurm job counts.
Every raw record retains training alpha/seed and evaluation alpha/replicate
separately. Test results never select checkpoints, training alpha or recipes.

## Separate lifecycle and legacy evidence

Training journal format 2 writes `training_pending_completion` at count 200 and
`training_completed` only after valid latest/best/final, all 200 epoch/clean-layer
records, strict validation selection, immutable complete CSVs and the training
summary are committed. `TrainingCompletionReceipt` acknowledges this authority.
`train_run` returns a training-only `TrainingResult`; it performs no test.

Training command dispatch at count 200 uses `TrainingFinalizationPlan` and
`open_run(operation="finalize_training")`: CPU stored-state validation repeated
under existing training locks, then `publish_training_completion`. It constructs
no model, loader, CUDA runtime, optimizer, scheduler, scaler or runtime streams;
there is no new epoch or reseeding. Actual publication-writer provenance is stored
separately from historical training provenance. The plan cannot authorize tensor
checkpoint writes or unfinished-training continuation. Ordinary training
continuation retains conservative actual source/software equality and full live
component preflight/restoration; evaluation does not globally weaken it.

Journal readers explicitly support versions 1 and 2; checkpoint/record format 1
remains readable. A version-1 `completed` transaction proves training and its
historical clean test, never the new 27 positive passes. Valid legacy full-budget
`trained_pending_test`, `evaluating` or post-training failure boundaries can bind
evaluation without retraining. Full-budget checkpoint eligibility and durable
training-completion publication are distinct reported fields. Historical failure
and clean evidence remain immutable; no robustness slots are fabricated.

`training_complete`, `training_finalization_pending`, `evaluation_eligible` and
`legacy_clean_test` replace ambiguous interpretation of generic completion.
Training coverage version 2 names counts `training_completed`, `training_failed`
and `training_unfinished`. Phase fields are `training_successfully_completed`
and `training_eligible`; pending robustness never forces more training or blocks
CIFAR-100 training merely because CIFAR-10 evaluation remains pending.

## Fixed realization, full data and preservation

For every eligible clean best-checkpoint tensor:

`W_eval = W_clean + evaluation_alpha * population_std(W_clean) * epsilon`

`evaluate_realization` uses the authoritative owned Conv2d/Linear inventory,
including stems, projections, depthwise/pointwise, SE and classifiers. Float32
population std uses correction=0 from the clean tensor; scale/noise detach and
are computed outside autocast. One private generator produces all independent
tensor noises once before the loop. The same functional parameter mapping remains
through the entire 10,000-image test pass. There is no per-batch noise resampling,
in-place add/subtract, bias/normalization/buffer perturbation, gradient or noise
accumulation across slots. The shared Gaussian offset helper preserves training's
existing per-mini-batch mathematics and draw order with a different caller lifetime.

Learned normalization buffers are used in eval mode, without reset/recomputation.
Positive forwards use private clean buffer copies and additionally verify those
copies remain unchanged. The clean model's original parameter/buffer bindings
remain intact. `CleanStateBaseline` owns CPU copies of **every** parameter and
buffer, including nonpersistent buffers and None bindings, with dtype/device/
requires-grad descriptors. Entry/exit use exact tensor comparisons and ownership
checks, including original module bindings. Prior module modes are restored individually.
On an invariant violation, restore original module/parameter/buffer bindings and
values, recheck the baseline and raise; a repaired failure cannot count as a
successful result. Unrecoverable device/structure errors discard the evaluation
model and prevent success publication. Exceptional and normal exits use the same
preservation guard. No such checks were executed during implementation.

Alpha zero delegates to `evaluate_clean`, without constructing a noise generator
or perturbation view; the entire-state guard still applies. Both paths share the
existing full-set metrics, precision and loss loop. `build_test_loader` validates
shared preparation and decodes sources read-only, constructing only the full
deterministic test loader, no training view/sampler/iterator. There is no download,
split/normalization regeneration, image augmentation/corruption/resize or subset.
`build_runtime(..., evaluation_only=True)` reuses explicit native CUDA/backend
policy with no GradScaler. Clean weights, noise/scales and cross-entropy remain
float32; forward autocast follows the resolved recorded policy across all slots.
Unresolved scientific decisions never become runtime defaults.

## RNG and checkpoint compatibility

The randomness contract is `sha256-best-checkpoint-realization-v1`. Encode sorted
compact finite UTF-8 JSON, ensure_ascii=False, with:

- randomness_version;
- best_checkpoint_identity = immutable generation ID and actual SHA-256 digest;
- protocol_version;
- canonical_evaluation_alpha;
- evaluation_replicate.

The first eight SHA-256 bytes interpreted little-endian yield the uint64 seed.
Positive seeds are checked for collisions and recorded individually. Clean seed
is null. A dedicated generator on the assigned device starts from that seed for
each pass. No training/model/noise RNG states are seeded, consumed or restored by
evaluation. The shared builder's isolated CPU construction context is retained.
Paths, workers, job IDs/order/grouping, batching and skipped completed slots are
not seed keys. Partial-pass restart uses the same checkpoint and seed. Reproducible
streams do not imply bitwise equality across arbitrary hardware/software.

`bind_evaluation_checkpoint` requires verified full-budget training journal,
latest/best/final and complete metrics/statistics. It binds the selected immutable
best reference/digest/selection and matches prepared data and model contracts.
Stored training provenance verifies stored evidence; it need not equal the new
evaluator's source digest. Actual evaluator provenance is collected separately.
Evaluation compatibility requires pinned component versions and full matching
torch/torchvision model metadata, complete parameter/buffer inventory and state
component metadata before `copy_clean_model(..., assign=False)`. It never loads
final in place of best or restores optimizer/scheduler/scaler/training RNG state.
Eligibility inspection is checkpoint eligibility, not a GPU capability probe;
actual prepared data and runtime are revalidated on execution.

## Independent durable storage and resumption

Default namespace:

```text
<execution.result_root or output_root>/evaluations/
  <training_run_id>/<training_fingerprint>/<best_sha256>/<protocol_fingerprint>/
    owner.json
    locks/evaluation-alpha-<beta>__replicate-<r>.lock
    slots/evaluation-alpha-<beta>__replicate-<r>/
      attempts/<attempt>.json
      records/<generation>.json
      HEAD.json
    sweep_completions/<generation>.json
    SWEEP_HEAD.json
```

`--evaluation-root` overrides the operational namespace base, outside scientific
identity. Linked/overlapping training managed paths are rejected. Evaluation never
holds the training writer lease throughout a pass and never writes its journal,
checkpoints, optimizer or status. It uses a brief namespace initialization lock,
then one cooperative exclusive slot lock. Different workers may publish different
slots. Same-slot concurrent writers are rejected; no age/PID lock stealing occurs.
Unowned partial namespaces and unknown/corrupt content remain evidence. Owned
unfinished slots preserve orphan/staging files and start a new attempt.

Immutable full result files are validated, fsynced and hashed before the slot
HEAD is atomically replaced. HEAD references current/previous records; records
carry ancestry. Only that authoritative reference proves a complete slot.
Uncommitted success files are not promoted. Failure evidence before HEAD remains
visible and requires explicit retry. No files are automatically deleted.
File/directory durability assumes the same POSIX HPC fsync/atomic-rename behavior
as training; Windows has no equivalent crash-durability claim.

Each successful result validates finite sample-weighted loss, integer correct
count, fraction accuracy, exactly 10,000 examples/79 batches and entire-state
preservation before publication. Partial/failed metrics are null with diagnostics.
Each pass remains an individual raw row; no averaging hides seeds/replicates.
An acknowledgement failure rereads authority while holding the slot lease;
committed success is not downgraded by subsequent diagnostics. Failure/interruption
never invalidates training or other slots. Failed slots need `--retry-failed`;
interrupted slots restart automatically at the same deterministic seed.

A separate immutable completion document and SWEEP_HEAD prove
`robustness_completed` only after validating all 28 successful slot references.
Successful raw slots with missing sweep publication are reported as publication
pending; another evaluation invocation skips their passes and reconciles the
completion marker. Settled failed sweeps remain distinct from successful coverage.
Sweep-lock contention with a validated compatible owner raises
`SweepFinalizationDeferred`; execution reports `completion_finalization_deferred`
and, while authority remains unverified, `sweep_finalization_deferred`. Deferral
alone contributes exit 0 and preserves successful selected-slot outcomes; existing
slot failures/interruption/rejection retain their exit classifications. Reinspect
or invoke evaluation again after the owner exits; no lock is stolen or removed.
Malformed/non-file/linked lock evidence, mismatched checkpoint/protocol bindings
and genuine I/O/integrity failures remain explicit errors, not contention.
Sweep lock acquisition stages a complete, fsynced owner record and creates the
exclusive `.sweep.lock` via a same-directory atomic hard link, so cooperating
readers cannot see a partly written owner. Only the acquiring invocation's own
staging file and token-verified acquired lock are cleaned up. Interrupted staging
evidence remains protected. Hard-link support and fsync durability are future HPC
filesystem requirements; no runtime capability or durability was verified here.

Coverage separates `completion_inspection_deferred` and its
`completion_deferred_reason` from genuine `completion_inspection_error`.
`completion_publication_pending` means all 28 raw slots are successful but sweep
authority has not been verified, including deferred or failed inspection.
Neither selected-slot success nor deferral proves `robustness_completed`: that
field requires verified SWEEP_HEAD and all 28 current successful references.
Lost-acknowledgement reconciliation still recognizes a verified committed sweep.
Genuine completion-inspection errors observed after contention retain exit 21;
they cannot be hidden by an earlier deferral diagnostic.
Read-only coverage separates stored statuses from locked/partial/incompatible/
unstable/inspection conditions. Changing owner/HEAD/lock tokens invalidate a read;
the report is not an atomic grid snapshot. Historical completion is distinguished
from compatibility with a requested current evaluator. Same-protocol slot reuse
requires matching evaluator source/software; protocol-version review is needed
for incompatible evaluator changes. Training-source equality is not that rule.

## Legacy clean reuse

Historical clean results are validated with the original completion/checkpoint
authority and exposed as candidates. `--reuse-legacy-clean` is an explicit
qualified import, not automatic coverage. It requires the exact same best digest,
full finite metrics/coverage, preprocessing/normalization, precision and compatible
torch/torchvision provenance, plus immutable exact entire-state preservation and
actual test-runtime evidence in the legacy summary. Standard v1 recorded neither
last requirement; its clean evidence is preserved and the independent clean slot
must run anew. No retrospective preservation/device fact is invented.

A qualified import records original test/summary references, provenance, runtime,
timestamp and test duration in `origin`, with new importer attempt/provenance
separate. It is never labeled freshly executed. Positive slots cannot import clean
results. Original historical files are not rewritten and no training is repeated.

## Commands and later interfaces

All commands below are implemented for later authorized HPC execution only:

```bash
alpha-sweep run --manifest "$MANIFEST" --run-id "$TRAINING_RUN_ID" \
  --base-dir "$SHARED_BASE" --source-root "$CHECKOUT" --operation auto
alpha-sweep evaluation-plan --manifest "$MANIFEST"
alpha-sweep evaluation-inspect --manifest "$MANIFEST" --run-id "$TRAINING_RUN_ID" \
  --base-dir "$SHARED_BASE"
alpha-sweep evaluate --manifest "$MANIFEST" --run-id "$TRAINING_RUN_ID" \
  --base-dir "$SHARED_BASE" --source-root "$CHECKOUT"
alpha-sweep evaluate --manifest "$MANIFEST" --run-id "$TRAINING_RUN_ID" \
  --base-dir "$SHARED_BASE" --source-root "$CHECKOUT" \
  --evaluation-alpha "$EVALUATION_ALPHA" --evaluation-replicate "$REPLICATE"
alpha-sweep evaluation-status --manifest "$MANIFEST" --base-dir "$SHARED_BASE"
```

Selectors reuse global training index, explicit phase/local index or exact training
run ID; evaluation alphas/replicates select one slot or a group for that checkpoint.
Omitting evaluation selectors processes all missing prescribed slots. Duplicate,
invalid/bool/type-conflicting selectors fail; clean alpha must select replicate 0.
Groups always filter the same prescribed set; there are no extra clean replicates.
Existing typed execution-only overrides apply; science cannot change through
execution. StopRequest discard stops between batches/before publication; finish
completes the current realization and its commit, then stops at the next boundary.
No signals/handler I/O, retries, jobs, arrays or scheduler dependencies are added.
The later Step 7 source now provides that operational layer separately, documented
in [slurm-workflow.md](slurm-workflow.md). Its runner passes StopRequest and an
immutable `expected_binding` to execute_evaluation; binding equality is repeated
before computation. One allocation processes one checkpoint's unfinished slots
sequentially, through the same private streams and completed-slot skip. No scientific
grid, seed, precision or realization lifetime is changed by scheduling.

Training outcomes now explicitly name `training_completed` and
`already_training_completed`, with lifecycle="training" and
ExitCode.TRAINING_COMPLETED. EvaluationOutcome uses lifecycle="evaluation" and
slot outcomes `evaluation_completed`, failures, interruption or rejection.
Exit 0 means the selected command succeeded, not generic whole-experiment
completion; 10 interrupted, 20 numerical failure, 21 execution failure, 22 rejected
inputs/ownership/inspection, 2 argparse usage. Coverage commands return 0 when a
report was collected even if rows are failed or missing. Inspect actual authorities
and lifecycle fields, not exit codes or allocation ending.

Step 7 consumes `evaluation_plan`, `prescribed_slots`, `select_evaluation_slots`,
`bind_evaluation_checkpoint`, `inspect_evaluation_eligibility`, `execute_evaluation`,
`collect_evaluation_coverage`, `collect_experiment_coverage`, EvaluationOutcome and
independent slot/sweep authorities. It owns CIFAR-10-first training dependencies,
evaluation job grouping, signal delivery, resubmission and workflow-wide concurrency
across training/evaluation. Logical index = training global index * 28 + slot index;
planning does not fabricate checkpoint references for untrained tasks.

Step 8 consumes version-1 individual `evaluation_realization` JSON: explicit
training dataset/architecture/run/fingerprint/alpha/seed; evaluation protocol
ID/version/fingerprint/alpha/canonical alpha/replicate/seed/RNG version; best role/
generation/digest/selection; data identities; `test_metrics` (loss/correct/examples/
batches/accuracy and units); preservation, runtime policy/effective backend;
training/evaluation provenance, attempt, duration, origin, failure and ancestry.
Join every expected slot and training identity. No mean/std, test-driven alpha
selection, heatmaps or plotting are implemented here.

Still pending: training seed plan, preparation seed/splits, normalization choices,
drop_last, precision/backend policy and scheduler endpoints/cadence. Account and
actual resources remain Step 7 inputs; aggregation conventions remain Step 8.
