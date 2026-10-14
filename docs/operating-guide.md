# End-to-end operating guide and execution handoff

All nine implementation steps are complete **in source only**. This guide is a
future operating sequence, not permission to start it. Separate explicit
authorization is required for environment setup, capability verification, data
preparation, artifact generation, reporting and Slurm/research execution. Nothing
in this sequence was executed during Step 9. Runtime correctness, launch readiness,
resource needs and experiment completion remain unverified.

The authoritative [scientific specification](scientific-specification.md) fixes
240 training tasks and 6,720 independent evaluation slots. Component guides below
define the detailed contracts; this guide orders their use. Shell variables denote
actual operator-approved inputs and canonical shared Linux paths. They are
placeholders, with no suggested seeds, account, normalization or scientific policy.
Do not paste these commands with unset variables. Preserve command output only to
new, operator-selected files outside protected research namespaces.

## Future authorized sequence

### 1. Resolve scientific inputs in a reviewed input set

On the local source-review workstation, review [configuration.md](configuration.md)
and the input register below. Approve three distinct training seeds and each
dataset's independent preparation seed/normalization policy. Approve `drop_last`,
precision/backend and scheduler cadence/endpoints. Hold recipes and choices fixed
across training alphas/seeds; preserve all native regularization and full sizes.
Checked-in `null` declarations remain valid declarations, not executable defaults.

After preparation, create separate, operator-owned full research input YAMLs from
each dataset's finalized `dataset_config.yaml`. Edit only unresolved approved
settings and necessary operational paths in these copies. Never edit the original
checksummed prepared YAML or overlay pending `configs/datasets` declarations over
finalized data. Architecture-specific approved scheduler inputs may live in a
reviewed model-declaration directory; the four fixed optimizer recipes still apply.
There is no CLI that implicitly resolves scientific decisions.

### 2. Confirm account, QOS and shared Linux storage

With the OSU operator/administrator, identify the actual research/class Slurm
account, allowed/default QOS, cluster/partition visibility and any constraints.
An ONID user name is not an account entitlement. Start an operator-owned policy
from [configs/workflow.yaml](../configs/workflow.yaml). A null QOS means no explicit
`--qos` request; its actual default/partition QOS still needs review.

Assign durable shared storage for the fixed checkout/environment, full manifest,
base, official data/preparation, metadata/output, checkpoints, results, evaluation,
workflow registry, application logs and Slurm stdout/stderr. Every later allocation
must see unchanged content at these paths. Do not rely on node-local scratch.
Operational roots must pass the existing overlap/link guards. Reporting needs a
separate shared root outside the base, checkout and every managed/protected store.

`policy.paths.split_file` must stay **null**: the full workflow inherits each
dataset's finalized manifest split. This is intentional inheritance, not a pending
input. Set both per-dataset `execution.storage.data_root` and `split_file` correctly
in the research input copies before manifest construction. If relocating data,
preserve its contents and preparation YAML/checksums, then rebuild/review the
manifest's operational document and linked phase views. The grid identity remains
scientific; changed full-document hashes require explicit ownership review, never
blind replacement of a manifest already bound to a registry.

### 3. Establish and freeze the compatible checkout/environment

Only after separate setup authorization, establish an existing compatible shared
Linux environment using the site's approved procedure. Preserve Python
`>=3.10,<3.13`, torch `2.5.1`, torchvision `0.20.1`, NumPy `1.26.4`, and PyYAML
`>=6.0`. CUDA wheel/driver selection remains an operator decision; no wheel is
chosen here. Optional headless figures require `reporting` (Matplotlib
`>=3.8,<3.11`). Collection/summaries/inspection do not require Matplotlib.

Set `environment.python_executable` to the absolute existing interpreter and, if
needed, an explicit plain activation script and versioned module names. Execute
from the selected checkout's installed editable package or `PYTHONPATH=<checkout>/src`.
The package initializer imports configuration/identity only. Five declared console
entry points map to actual `main` functions: `alpha-sweep-prepare`, `alpha-sweep`,
`alpha-sweep-workflow`, `alpha-sweep-allocation`, `alpha-sweep-report`.

Freeze source, research inputs, policy and environment before planning/computation.
Source inventory `checkout-python-and-declarations-v2` covers package Python,
pyproject, research/workflow declarations, named dataset/model declarations and the
scientific specification; arbitrary datasets/generated outputs are excluded.
Each new plan pins actual software plus version-1 external launch inputs: activation
script bytes, configured/resolved interpreter and bytes, and declared module names.
Rendering/submission/runner revalidate pins; Bash checks bytes/target **before**
activation using GNU `sha256sum`, `readlink` and `dirname`. Interpreter symlinks for
venvs are permitted and target-pinned; activation paths must have no linked ancestors.
These utilities must exist on submit and compute nodes. Module names and top-level
file pins do not freeze transitive environment files or module definitions: operators
must keep those immutable and review actual software/CUDA evidence.

Unfinished training requires identical source digest/full software mapping and
complete live-state preflight. This Step 9 source update changes that digest and
does **not** authorize migrating unfinished older runs. Continue them only with
their frozen compatible implementation/environment or an explicitly reviewed future
migration. Full-budget publication and historical readers validate stored training
provenance. Evaluation may use newer compatible model software without equality to
training source, but reusing a same-protocol slot requires matching its evaluator
source/software. Incompatible evaluator changes require protocol review; never edit
raw provenance. Historical reporting records its own source without requiring it
to equal historical research source.

Old version-1 plans/bundles lacking `launch_inputs` remain readable for workflow
ownership/reconciliation, but cannot render, submit or execute under this source.
Preserve uncertain old intents; resolve them before creating new reviewed work.
Legacy policies with a non-null global split remain parseable as historical state,
and current workflow inspection/reconciliation/recovery can use their original
sealed policy without research readiness. New planning/execution readiness rejects
them. Follow the historical-policy sequence below before corrected-policy adoption;
never substitute corrected values into an unresolved registry's historical evidence.

### 4. Obtain real runtime and operational capability evidence

On authorized allocated nodes, verify the actual Python/research packages, wheel,
driver/GPU architecture, native bfloat16 when selected, backend/determinism and
single-device allocation. Set approved `CUBLAS_WORKSPACE_CONFIG` **before Python**
when deterministic CUDA requires it. Do not assume source inspection established
CUDA support, precision correctness, memory fit, signal timing or performance.

On the submit node, obtain actual accounting visibility/format/retention and
submission-comment support. On shared storage, obtain actual exclusive-create,
atomic same-filesystem replace/rename, hard-link and file/directory fsync guarantees.
See the future verification section; there is no implemented verification CLI and
no verification recipe/configuration was generated during implementation. Research
computation belongs on allocated GPU nodes, never the login node.

### 5. Prepare the complete official CIFAR datasets

Use an authorized CPU allocation or other explicitly approved preparation resource;
decoding/hashing/normalization is substantial work, not login-node training. Follow
[datasets.md](datasets.md), choosing exactly one approved policy per dataset:

```bash
alpha-sweep-prepare --config "$PREPARATION_BASE_CONFIG" --dataset cifar10 \
  --data-root "$SHARED_CIFAR_ROOT" --split-seed "$APPROVED_CIFAR10_SPLIT_SEED" \
  --normalization derive_training
alpha-sweep-prepare --config "$PREPARATION_BASE_CONFIG" --dataset cifar100 \
  --data-root "$SHARED_CIFAR_ROOT" --split-seed "$APPROVED_CIFAR100_SPLIT_SEED" \
  --normalization derive_training
```

These commands apply only if training-partition derivation was approved. For an
approved explicit-constants policy, use the exact `--normalization explicit`,
`--mean`, `--std`, `--normalization-source` recipe in the dataset guide instead.
Without `--download`, official files must already exist and validate. Add it only
with explicit acquisition authorization. Preparation uses its own dataset lock,
persisted 45,000/5,000 split, normalization and last-published manifest; it never
uses `open_run`. Workers read these finalized artifacts without repair/download.
Now finish the two separate research input copies described in step 1.

### 6. Build and inspect the full manifest and phase views

On an authorized CPU/submit resource with sufficient storage/reader capacity:

```bash
alpha-sweep build --cifar10-config "$FINALIZED_CIFAR10_CONFIG" \
  --cifar100-config "$FINALIZED_CIFAR100_CONFIG" --models-dir "$REVIEWED_MODELS_DIR" \
  --seeds "$SEED_A" "$SEED_B" "$SEED_C" --base-dir "$SHARED_BASE" \
  --output "$FULL_MANIFEST" --phase1-output "$PHASE1_VIEW" --phase2-output "$PHASE2_VIEW"
alpha-sweep inspect --manifest "$FULL_MANIFEST" --phase-view "$PHASE1_VIEW"
alpha-sweep inspect --manifest "$FULL_MANIFEST" --phase-view "$PHASE2_VIEW"
alpha-sweep evaluation-plan --manifest "$FULL_MANIFEST"
```

Require all 240 ready tasks in original order, three approved seeds, each dataset's
shared preparation identities and exact linked 120-task phase views. Build performs
metadata/raw-checksum validation, not models/training/data iteration. Compatible
manifest reuse preserves bytes; differing valid outputs require explicit
`--replace-manifests`, which does not permit changing a registry's bound parent.
See [manifests-and-controller.md](manifests-and-controller.md).

### 7. Collect version-2 restriction evidence and finalize policy

On the submit node, under separate authorization to query Slurm:

```bash
alpha-sweep-workflow inspect-restrictions --policy "$POLICY" > "$NEW_RESTRICTION_REPORT"
```

Review the sealed report, exact arguments/field names/raw output, version and
`collection_complete`, association ancestors/user/cluster, default/selected/partition
QOS and flags/factors. Successful queries alone are insufficient. Missing/blank,
unsupported, denied or truncated evidence is explicit and never zero/unlimited.
Version 1/unknown reports cannot satisfy the submission gate. The documented
24.11 fields require actual installed-version/output compatibility evidence.
`GrpTRESMins`/`GrpWall` are cumulative usage; `GrpTRESRunMins` and user/account
run-minute limits bound simultaneous requested time. Interpret existing usage and
site enforcement explicitly, without automatic entitlement guesses.

Fill `effective_limits` with reviewed finite positive bounds for wall hours,
GPU/CPU/memory overlap, GPU/CPU/memory hours and `max_allocations`, plus the report
path and **actual file SHA-256** (distinct from its internal document seal).
The submission gate revalidates report bytes, version, query coverage, raw rows,
collection issues and account/QOS linkage. Review the dated published bounds
against current OSU policy; they are not verified current entitlements.
[slurm-workflow.md](slurm-workflow.md) gives the full evidence contract.

### 8. Plan and render one bounded wave, then explicitly submit

On the submit node using the frozen execution environment:

```bash
alpha-sweep-workflow plan --manifest "$FULL_MANIFEST" --policy "$POLICY" > "$NEW_WORK_PLAN"
alpha-sweep-workflow render --manifest "$FULL_MANIFEST" --policy "$POLICY" --plan "$NEW_WORK_PLAN"
```

Review eligibility, full source/software/launch pins, exact training indices,
actual evaluation checkpoint bindings, resource budgets and rendered mapping/Bash.
Use the returned `mapping` path as `$MAPPING`; do not invent paths or edit sealed
files. `render --work-id` accepts exact eligible unit keys from this plan. Planning
is read-only; rendering writes workflow-owned bundles/state but calls no `sbatch`.
Only after explicit launch authorization:

```bash
alpha-sweep-workflow submit --manifest "$FULL_MANIFEST" --policy "$POLICY" \
  --bundle "$MAPPING" --execute
```

Every supported submission uses the same registry/controller and durable intent
before `sbatch`. Keep one unresolved wave and the shared four-allocation planning
cap across datasets/lifecycles. Defaults (one GPU, eight CPUs, 32 GB, 12 hours)
are unmeasured requests; four such requests consume 48 requested GPU-hours.
Pending/unknown allocations reserve their full capacity. Never submit the rendered
script manually, create a second registry, or use an independent array/requeue
script. There are no supported legacy submission scripts in this checkout.

### 9. Run training allocations and advance verified continuation

Slurm launches the sealed batch; Bash preserves assigned GPU visibility, bounds
threads and execs the owning allocation runner. `alpha-sweep-allocation --mapping`
is the registered worker entry point, not a submit-node launcher or an independent
job recipe. It requires actual Slurm identity, durable intent and one owning claim.
One training task/process occupies one GPU allocation. Signals update StopRequest
only; clean boundary outcomes and compatible authority authorize bounded replay.

On the submit node:

```bash
alpha-sweep-workflow status --manifest "$FULL_MANIFEST" --policy "$POLICY"
alpha-sweep-workflow reconcile --manifest "$FULL_MANIFEST" --policy "$POLICY"
alpha-sweep-workflow advance --manifest "$FULL_MANIFEST" --policy "$POLICY"
```

`status` has no scheduler calls unless `--scheduler` is supplied. `reconcile` and
`advance` explicitly query/publish scheduler observations. `advance` without
`--submit` only reconciles/plans. Add `--submit` only when continued bounded
submission is expressly authorized. There is no daemon or blind requeue loop.
All 120 successful CIFAR-10 **training** completions gate CIFAR-100 training;
terminal failures do not satisfy this gate. Pending robustness never blocks it.
Manual `alpha-sweep run` also enforces this training gate; it has run ownership
but does not reserve shared workflow capacity or install allocation handlers.
Use it only for explicitly reviewed operator work on an allocated node after
managed owners have quiesced, not as a parallel production scheduling path.

### 10. Run independent missing evaluation work

Use the same plan/render/explicit-submit sequence. Select eligible evaluation
unit keys with `render --work-id` when desired. One checkpoint's prescribed
unfinished slots run sequentially in one GPU allocation. It binds verified
immutable best, never substitutes latest/final or restores optimizer/training RNG.
Zero has one clean slot; each positive alpha has three full 10,000-image passes
with one fixed network per pass. Successful compatible slots skip; an interrupted
pass restarts wholly with its original private seed. Evaluation failures leave
training authority untouched. See [weight-robustness-evaluation.md](weight-robustness-evaluation.md).

### 11. Inspect both lifecycles and resolve recovery conditions

Use workflow `status` for joined operational/scientific coverage and the decision
tables below. Direct readers, with the **same effective storage paths**, are:

```bash
alpha-sweep status --manifest "$FULL_MANIFEST" --base-dir "$SHARED_BASE" \
  --execution-overrides "$STORAGE_OVERRIDES"
alpha-sweep evaluation-status --manifest "$FULL_MANIFEST" --base-dir "$SHARED_BASE" \
  --execution-overrides "$STORAGE_OVERRIDES" --evaluation-root "$EVALUATION_ROOT"
```

`$STORAGE_OVERRIDES` is an operator-owned strict typed JSON object of existing
execution fields matching the effective policy stores. For a full grid, omit
`execution.storage.split_file` so the manifest's per-dataset paths remain intact.
Do not use a global split override. Readers may restricted-load CPU checkpoints
and all committed records; allow real memory/I/O capacity on an approved CPU node.
They acquire no research writer leases, register no signals and do no repair.

### 12. Collect, summarize, plot and inspect derived results

On an authorized CPU reporting allocation with adequate memory/disk and the
optional reporting extra for plotting:

```bash
alpha-sweep-report collect --manifest "$FULL_MANIFEST" --base-dir "$SHARED_BASE" \
  --source-root "$REPORTING_CHECKOUT" --execution-overrides "$STORAGE_OVERRIDES" \
  --evaluation-root "$EVALUATION_ROOT" --workflow-root "$WORKFLOW_ROOT" \
  --protected-root "$SLURM_LOG_ROOT" --output "$NEW_COLLECTION_DIR"
alpha-sweep-report summarize --collection-dir "$COLLECTION_DIR" \
  --source-root "$REPORTING_CHECKOUT" --output "$NEW_SUMMARY_DIR"
alpha-sweep-report plot --collection-dir "$COLLECTION_DIR" \
  --source-root "$REPORTING_CHECKOUT" --output "$NEW_FIGURE_DIR" --formats png pdf svg
alpha-sweep-report inspect --report-dir "$REPORT_DIR"
```

Choose distinct new derived directories outside base/source/managed roots; declare
additional protected roots with repeated `--protected-root`. A frozen collection
always represents 240 training/6,720 logical evaluation rows, including unavailable
rows and null unbound checkpoint/seeds. All individual observations remain visible.
Official summaries require complete sets, sample SD `ddof=1`, realization means
within checkpoint before cross-training-seed means/SD. Validation selection needs
all ten complete candidates and retains every exact co-winner. Pairing is descriptive
and differences use percentage points. Figures mask missing cells and retain
individual seeds/realizations. Partial reports do not authorize scientific repair.
[results-and-figures.md](results-and-figures.md) defines exports/figures in detail.

## Recovery decisions and evidence

Training `state/HEAD.json` is progress/completion authority. Evaluation slot HEADs
prove individual success; verified SWEEP_HEAD plus all 28 successes proves robustness
completion. Scheduler state only establishes allocation/reservation state. Derived
report COMPLETE establishes checksummed derived publication only. Exit 0 alone
proves none of these scientific lifecycles.

| Condition | Required evidence and next authorized action |
| --- | --- |
| Clean training interruption (exit 10) | A registered outcome with explicit stop and `safe_interruption=true`, compatible latest/full state, terminal scheduler proof and no residual locks permits managed advance within the allocation bound. Replay unfinished epoch; retain 200-epoch horizon. |
| Count 200, training publication pending | Valid full-budget latest/best/final and 200 records permit publication-only finalization. Managed plan or manual `run --operation auto` adds zero epochs and requires no test. `run --operation recover` repairs committed views. |
| Partial evaluation realization | Explicit clean stop/compatible binding and terminal ownership permit restart of the whole pass using the same seed. Partial metrics stay null; other successful slots skip. |
| All selected slots successful, sweep publication pending | Selected-slot success does not establish all-28 completion. Reinspect; when owners exit, managed evaluation finalizes missing authority without rerunning successful compatible passes. Genuine publication errors remain explicit. |
| SweepFinalizationDeferred | A validated cooperating sweep owner yields a clear deferral; contention alone is exit 0, not 21. Keep raw successes, wait for owner exit and reinspect. No lock removal or completion claim without verified authority. |
| Terminal numerical failure (exit 20) | Training cannot continue; managed numerical evaluation failures cannot be waived with retry-run-id. Preserve committed/orphan failure evidence; reconcile an exact unfinished-training numerical reference through `run --operation recover --numerical-reference`. No automatic retry/phase unlock. Direct evaluation's existing `--retry-failed` is an explicit operator override for failed slots, not managed automatic numerical continuation; it retains old failure evidence and requires separate review/authorization. |
| Execution failure/missing outcome (exit 21 or unclassified) | Establish cause, compatible authority and terminal/inactive ownership. Managed `plan/advance --retry-run-id --retry-reason` records an explicit decision; flags do not waive locks, science, compatibility or allocation bounds. Manual training `run --operation continue` likewise authorizes only execution retry. |
| Locked/corrupt/incompatible/unstable state (exit 22) | Preserve all evidence. Reinspect unstable reads after writers exit; obtain exact ownership evidence for lock recovery. Corruption/identity mismatch cannot be fixed by continuation, alias edits or optimistic rollback. Exact selected overwrite is a separately authorized archived replacement, not recovery. |
| Lost scientific acknowledgement | Read validated HEAD/SWEEP_HEAD before replay. Committed completion dominates later receipt/diagnostic errors. Recover derived views only; never replay work from stale CSVs. |

All manual `run/evaluate` actions require exact full-manifest selection, identical
effective storage overrides and explicit operator authorization on an appropriate
allocated node, with managed ownership quiescent. For publication repair:

```bash
alpha-sweep run --manifest "$FULL_MANIFEST" --run-id "$EXACT_RUN_ID" \
  --source-root "$FROZEN_COMPATIBLE_CHECKOUT" --base-dir "$SHARED_BASE" \
  --execution-overrides "$STORAGE_OVERRIDES" --operation recover
```

Unfinished recover still requires the full compatible live plan/GPU environment,
but never calls training. Count-200 recover uses the CPU publication plan. Adding
`--numerical-reference "$EXACT_NUMERICAL_REFERENCE_JSON"` selects an actual inspected
ArtifactReference for unfinished terminal evidence only, never a fabricated file.

| Operational condition | Required evidence and supported recovery |
| --- | --- |
| Lost submission acknowledgement | Durable uncertain intent reserves capacity. `resolve-submission --wave-id --job-id [--cluster]` validates exact scheduler comment/account/partition/array membership; a registered runner can also acknowledge it. Never blindly resubmit. |
| Proven no submission occurred | `resolve-submission --confirmed-not-submitted --evidence --reason` requires exact wave/comment/non-submission proof and no worker claims. Preserve intent and consumed ordinal; later planning is a new reviewed action. |
| Delayed/missing accounting | `reconcile` keeps the entire wave reserved. Empty queue/PID/age are insufficient. Complete matching terminal accounting for every array index, or explicit authoritative `resolve-terminal` proof, is required. |
| Hard termination/node failure/residual owners | Obtain cross-node scheduler/admin inactivity evidence for every owner. `inspect-locks` then `release-locks --confirmed-inactive-owners --reason` requires the exact unchanged report and archives locks with audit; it never marks science complete. Afterwards reconcile and make an explicit execution-retry/recovery decision. |
| Concurrent controllers | Compatible contention blocks a competing controller; workers wait only within the configured bound. Reinspect after release, never steal or start a second registry. |
| Exhausted allocation intents | Preserve exhaustion and all intent/outcome history. No automatic reset. Any operational policy revision needs explicit reviewed `adopt-policy --previous-policy --reason` after proven quiescence; this cannot relax scientific/source compatibility. |

Exact CLI recipes and evidence payloads/validation are in
[slurm-workflow.md](slurm-workflow.md) and
[checkpoints-and-recovery.md](checkpoints-and-recovery.md). Lock release never derives
authorization from local PIDs/age or removes another active owner's lock. Do not
delete partial/orphan evidence, edit sealed state, blind-resubmit, or restart training
because robustness is incomplete. Unsupported private scripts, manual `sbatch` and
outside requeues are beyond registry control and are not supported operating paths.

## Historical-policy inspection, recovery and subsequent adoption

This is a future, separately authorized operator sequence. No command was run
during this correction. `$ORIGINAL_POLICY` must contain the exact current policy
sealed in the existing registry, including its historical non-null `paths.split_file`.
Use its original registry root and the original full manifest. Do not edit that
policy to satisfy new research readiness; fingerprint mismatches still reject.
An absent, partial, corrupt, linked or unowned registry is not initialized or
reconstructed by this recovery path.

Recovery requires shared Linux access, the original registry and full-manifest
ownership/seals. Scheduler operations additionally need the original explicit
account/command configuration and actual scheduler access. They do not require
the historical activation/interpreter to still exist, launch pins, prepared-data
availability or readiness to train. Generic declaration validity still applies.

1. Inspect existing authority on the submit node:

   ```bash
   alpha-sweep-workflow status --manifest "$FULL_MANIFEST" --policy "$ORIGINAL_POLICY" --workflow-only
   alpha-sweep-workflow status --manifest "$FULL_MANIFEST" --policy "$ORIGINAL_POLICY" --workflow-only --scheduler
   ```

   `--workflow-only` explicitly suppresses scientific coverage. A non-null global
   split also makes plain workflow `status` automatically use this mode. Output
   has `scientific=null`, `scientific_coverage_collected=false` and a diagnostic;
   it makes no claim about dataset or training/evaluation coverage. Without
   `--scheduler`, no scheduler query occurs. A controller lock blocks status;
   use the exact lock-inspection path below if it remains after hard termination.

2. For an already acknowledged job, reconcile with the original policy:

   ```bash
   alpha-sweep-workflow reconcile --manifest "$FULL_MANIFEST" --policy "$ORIGINAL_POLICY"
   ```

   This writes a new sealed operational observation under the original policy,
   preserving old waves/bundles and their recorded scheduler identity. Terminal
   evidence requires complete exact array accounting and no residual allocation
   ownership; missing/unknown accounting retains the reservation.

3. For a lost acknowledgement, choose the evidence-supported branch:

   ```bash
   alpha-sweep-workflow resolve-submission --manifest "$FULL_MANIFEST" --policy "$ORIGINAL_POLICY" \
     --wave-id "$EXACT_WAVE_ID" --job-id "$EXACT_ARRAY_JOB_ID" --cluster "$ACTUAL_CLUSTER"
   ```

   Supply `--cluster` only when the actual cluster is known; otherwise omit it.
   The existing scontrol verification must match the original exact comment,
   account, partition, array root and bounded task selector. Afterwards reconcile
   with the original policy. If no scheduler identity can be verified, retain
   uncertainty; an empty queue never authorizes replacement.

   Only if actual authoritative evidence proves **no allocation was submitted**:

   ```bash
   alpha-sweep-workflow resolve-submission --manifest "$FULL_MANIFEST" --policy "$ORIGINAL_POLICY" \
     --wave-id "$EXACT_WAVE_ID" --confirmed-not-submitted \
     --evidence "$ACTUAL_NON_SUBMISSION_PROOF" --reason "$PROOF_REASON"
   ```

   The proof must retain the exact `wave_id`, `scheduler_comment` and
   `all_allocations_not_submitted_confirmed=true`, backed by operator/admin evidence.
   Existing worker claims or residual allocation locks prohibit this branch.
   The proof/intent remain immutable; this is not resubmission permission.

4. For residual ownership, inspect before any confirmed release:

   ```bash
   alpha-sweep-workflow inspect-locks --manifest "$FULL_MANIFEST" --policy "$ORIGINAL_POLICY" \
     --scope workflow > "$NEW_LOCK_EVIDENCE"
   alpha-sweep-workflow release-locks --manifest "$FULL_MANIFEST" --policy "$ORIGINAL_POLICY" \
     --evidence "$NEW_LOCK_EVIDENCE" --confirmed-inactive-owners --reason "$INACTIVITY_PROOF_REASON"
   ```

   These commands validate the original stored policy even when a controller lock
   blocks ordinary status. Require actual inactivity of **all** recorded owners
   across nodes, exact unchanged path/owner/digest evidence, and an audit reason.
   Release archives locks; it never deletes or steals active ownership. If relevant
   training/evaluation locks also remain, inspect the selected `--scope training`
   or `--scope evaluation --run-id "$EXACT_RUN_ID"` (both run scopes need `--run-id`)
   and release each exact report only after its own verified inactivity. Run-lock
   paths use original run storage only, excluding both global data_root and split
   overrides; evaluation scope also requires a verified best binding/original
   evaluation root. This is lock evidence, not scientific coverage or computation.

5. After ownership recovery, reconcile again. If acknowledged-job accounting
   cannot supply full terminal evidence, use the existing explicit proof path:

   ```bash
   alpha-sweep-workflow resolve-terminal --manifest "$FULL_MANIFEST" --policy "$ORIGINAL_POLICY" \
     --wave-id "$EXACT_WAVE_ID" --evidence "$ACTUAL_TERMINAL_PROOF" \
     --confirmed-terminal-inactive-owners --reason "$TERMINAL_PROOF_REASON"
   ```

   The proof binds the exact workflow/wave/comment and all recorded array indices,
   with actual terminal/inactive-owner evidence for each. Visible queue membership
   or residual allocation leases reject it. Unknown accounting alone is not proof.
   Reinspect workflow-only status: every wave must be conclusively terminal or
   explicitly proven not submitted, with submission/ownership ambiguity resolved.

6. Only then adopt the separately reviewed research-ready corrected policy:

   ```bash
   alpha-sweep-workflow adopt-policy --manifest "$FULL_MANIFEST" --policy "$REVIEWED_NEW_POLICY" \
     --previous-policy "$ORIGINAL_POLICY" --reason "$POLICY_REASON"
   ```

   The new policy must leave split_file null and retain the same registry location.
   Adoption verifies the previous policy fingerprint before writes and under the
   controller lease. It checks original per-wave scheduler policy/identities and
   preserved terminal/non-submission proofs, refuses unresolved submissions/owners,
   and preserves old mappings, policies, evidence and allocation ordinals. A corrected
   policy cannot authorize resolving old uncertainty in place of the original one.

7. Future new planning uses the adopted policy and new launch-pinned work. Old
   unpinned plans/bundles remain readable during this entire recovery sequence but
   cannot render, submit or run under current source. No recovery lease can reserve
   or submit new work. Unfinished-training source/software equality and independent
   evaluation compatibility remain unchanged; adoption is not a migration permit.

## Unresolved-input register

Nothing in this register has been populated during Step 9.

| Input / actual field or argument | Why needed / blocked future operation |
| --- | --- |
| `scientific.training_seeds`, build `--seeds` (three distinct uint32 values); task `scientific.training_seed` | Paired grid and initialization/data/model/noise streams; blocks manifest/run identity. Builder selects each task seed, not a new default. |
| Preparation `--split-seed` for each new dataset; `scientific.data.split.identity` | Independent persisted 45,000/5,000 content identity; blocks new preparation/finalized manifest/loaders until created and validated. |
| Preparation `--normalization`, optional `--mean/--std/--normalization-source`; `scientific.data.normalization.{mean,std,source}` | Approved explicit or training-only derivation; blocks preparation/manifest/computation. No common CIFAR constants inferred. |
| `scientific.training.drop_last` | Determines full training batch/sample counts; blocks research readiness/loaders/step horizon. |
| `scientific.precision.mode`, `float32_matmul_precision`, `allow_tf32`, `deterministic_algorithms`, `cudnn_deterministic`, `cudnn_benchmark` | Fixed explicit forward/backend policy; blocks research-ready manifest/runtime. Highest pairs with TF32 false; high/medium with true. Native device support remains unverified. |
| `scientific.training.scheduler_interval`, `warmup_start_factor`, `cosine_min_lr` | Epoch/successful-update indexing and approved endpoints; blocks research readiness/scheduler/manifest. No shortened horizon on continuation. |
| Policy `resources.account/qos/constraint`, `effective_limits` and inspection report path/SHA | Actual submitting entitlement/scope and version-2 reviewed evidence; missing account blocks operational readiness, missing evidence blocks submission. Null optional QOS/constraint still needs operator interpretation. |
| Policy `paths.*` except inherited null `split_file`; task data_root/split_file; all manifest/config/CLI paths | Durable shared stores and unambiguous full-grid linkage; blocks planning/render/run/loading. Do not globally override one split for both datasets. |
| Policy `environment.python_executable/activation_script/modules/variables`; fixed checkout/software/CUDA | Existing compatible immutable launch inputs and actual backend capability; blocks launch-pin collection/submission/runner/compute. Optional script/modules may be absent only if the approved environment needs neither. |
| File/directory fsync, atomic same-filesystem rename/replace, hard links, exclusive creation | Real durable journals/owner locks/sweep publication; requires filesystem/admin and later behavior evidence. No runtime guarantee from implementation. |
| Policy `commands.*`; actual Slurm 24.11 field compatibility, comments, individual array accounting and retention | Version-2 collection and exact terminal/submission reconciliation; blocks submission or capacity release when evidence is missing/denied/incompatible. |
| Real CPU/GPU RAM, warning lead, wall time, workers/native threads, continuation limit | Declared defaults are requests; need measured full-budget behavior before claiming capacity/performance/safe margins. No automatic tuning or budget reduction. |
| Optional Matplotlib/backend, report output paths, reader/renderer memory/runtime | Blocks plotting/publication where unavailable; current source does not establish practical cost of full checkpoint validation/large frozen collections/all figures. |

## Future verification evidence, requiring separate authorization

No tests, synthetic/smoke configurations or runtime verification artifacts were
created. The next phase needs a separately approved verification/execution scope;
this guide does not grant it. Evidence still required includes:

- Actual package installation/entry-point/import behavior on supported Python,
  pinned torch/vision/NumPy, chosen CUDA driver/GPU and optional headless renderer.
- Full official source decoding/checksum preparation and all split/normalization
  association checks, multi-worker reproducibility and persistent-worker epoch keys.
- All four actual CIFAR structures/eligible inventories, clean float32 ownership,
  functional gradients/buffer updates and exceptional state preservation.
- Real precision/AMP skip accounting, original scheduler trajectories/restoration,
  counters/strict best selection, complete epoch commits and count-200 finalization.
- Durable initial/epoch/failure/completion state under real I/O/interruption,
  lost acknowledgements, cross-root publication/retention, historical readers and
  conservative code/software rejection; fsync/atomic replace/hard-link behavior.
- Real independent 10,000-image fixed-network passes, exact all-state preservation,
  private seed restart, compatible completed-slot skip, concurrent/deferred sweep
  finalization and separate training/evaluation completion.
- Actual Slurm field/output/accounting/comment compatibility, intent/ack uncertainty,
  controller/allocation concurrency, signal delivery to owning PID, worker behavior,
  bounded safe continuation and residual-owner recovery without capacity leakage.
- Full historical/incomplete/complete reporting joins, finite export validation,
  two-stage summaries, validation-only exact co-winners, denominators/paired units,
  unavailable-cell masking and guarded PNG/vector publication/inspection.
- Measured full-run reader/snapshot/renderer memory, disk, duration, throughput and
  warning margins. Planning resources are not measurements or runtime estimates.

Source completion is the stopping point. Do not establish an environment, verify
executables, prepare data, produce manifests/bundles/reports or start jobs until the
user gives a separate explicit instruction for the relevant next phase.
