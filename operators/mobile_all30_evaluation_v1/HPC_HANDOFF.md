# Revised MobileNet: separate all-30 evaluation operator

This is an external operator handoff, revised November 4, 2026 from branch
`ops/mobile-all30-evaluation-v1`, commit
`0dafef07d0b379cf8c2f47eb6868db4319aaf8fe`. The user separately authorized local
CPU tests, test imports and syntax checks for this revision. Mocked checks ran;
no real checkpoint loads, evaluation, HPC operations, Slurm contact, dependency
installation or research reporting/plotting ran. No operational command below
has been run in this revision. Current changes are uncommitted and unpushed.

The bundle uses the frozen native evaluator at commit
`8010d07d9d1f0886ab1e180523f316623ef622a4`, source inventory
`a5726bac3f449455d36e2f53840f3e0fd77768a7ab2b989d2884da20f920005c`.
Native source, training artifacts, original epsilon-0.001 study, and historical
remaining-27 operator/plans remain untouched. No retraining, retuning, checkpoint
selection from test scores, commit or push is part of this handoff.

## External source and check inventory

| File | Responsibility |
|---|---|
| `launcher.py` | Fresh CPU inspection, bundle/plan ownership, full bindings, preview/submission, acknowledgment reconciliation, allocation claims, native evaluation dispatch, reservation settlement |
| `batch.sh` | Existing interpreter/environment, one owning Python PID, native signal delivery |
| `report.py` | All 30 training observations and 840 expected slots; native coverage and raw histories; complete-set summaries |
| `figures.py` | Six PNG/PDF plot pairs, including loss, individual seeds/realizations and paired clean-relative views |
| `HPC_HANDOFF.md` | Future commands and operating restrictions |
| `REVIEW.md` | Source review, evidence, checks still required and limitations |

Two check-support files live **outside** this six-file runtime inventory:

| Sibling directory `../mobile_all30_evaluation_checks/` | Responsibility |
|---|---|
| `run_cpu_checks.py` | Standard-library runner; syntax and Python 3.11 grammar checks |
| `test_operator.py` | Reporting regressions, native API wiring, mocked plotting and temporary operational publication |

This queue-scoping revision modifies `launcher.py`, `HPC_HANDOFF.md`, `REVIEW.md`,
and the sibling check file `../mobile_all30_evaluation_checks/test_operator.py`.
`batch.sh`, `report.py`, `figures.py` and `run_cpu_checks.py` retain their bytes.
The runtime inventory remains six files, with two separate support files. Tests do not enter
`BUNDLE_FILES` or the deployed runtime directory. Copying test files into that
directory correctly fails its strict inventory check. The branch contains an
older full repository snapshot: transfer only the external files and separate
check support, **never replace the frozen HPC native checkout**.

`BUNDLE_OWNER.json` is created exclusively **on HPC** by `own-bundle`, then pinned
along with all six source files and their absolute paths. Every actual import
from the external bundle is covered; no historical external helper is imported.
Changing any file after plan freezing rejects new execution. Revisions require a
separately reviewed new bundle/plan version; preserve unresolved intents first.
Do not copy this revision over an existing pinned runtime bundle or rewrite an
existing HPC plan. Transfer commands stop at existing destinations; adoption of
changed hashes into actual operational work remains separately authorized.
New execution requires a plan with an OS submitter pin. Older plans and intents
are preserved and rejected rather than silently given a new owner or new code.

## Fixed paths and scope

| Purpose | Absolute path |
|---|---|
| Frozen native checkout | `/nfs/hpc/share/bellaak/research/alpha-sweep/alpha-sweep-mobile-eps0316` |
| Manifest | `/nfs/hpc/share/bellaak/research/alpha-sweep/mobile-eps0316-v1/study.json` |
| Pilot plan, source of `base_dir` | `/nfs/hpc/share/bellaak/research/alpha-sweep/mobile-eps0316-v1/pilot.json` |
| Interpreter | `/nfs/hpc/share/bellaak/research/alpha-sweep/alpha-sweep-training/.venv/bin/python` |
| New external bundle | `/nfs/hpc/share/bellaak/research/alpha-sweep/mobile-all30-eval-launch-v1` |
| New operational root | `/nfs/hpc/share/bellaak/research/alpha-sweep/mobile-eps0316-evaluation-ops-v1` |
| Native evaluation root | `/nfs/hpc/share/bellaak/research/alpha-sweep/mobile-eps0316-evaluations-v1` |
| Separately copied export | `/nfs/hpc/share/bellaak/research/alpha-sweep/mobile-all30-eval-evidence-v1` |

The exported pilot records base directory
`/nfs/hpc/share/bellaak/research/alpha-sweep/main-v1`. The operator derives it from
the **actual** validated pilot, compares the export and pins it; it never derives
it from the study directory. Linked/overlapping output paths reject execution.

Indices 0–29 map to `(ALPHAS[index // 3], index % 3)` for
`0,.01,.02,.05,.1,.2,.3,.5,.75,1`. Each owns one validation-selected best
checkpoint and all native slots: clean replicate 0, nine positive alphas with
replicates 1–3. There are 30 checkpoints, 28 slots each, 840 full 10,000-image
passes. No accuracy threshold excludes a run. The full native binding, including
generation/digest, earliest validation selection, scientific identity, data,
protocol and original provenance, is forwarded as `expected_binding`.

## Transfer after separate authorization

These Windows PowerShell commands copy only the new external source and the
original review evidence. Set `$HpcHost` to your established OSU SSH target; this
handoff does not guess a login hostname. Stop if either destination exists.

```powershell
$HpcHost = Read-Host 'Established OSU SSH target (user@host)'
$TaskRoot = 'C:\Users\akaas\RESEARCH\alpha-sweep-training\.handoff-worktrees\mobile-all30-evaluation'
ssh $HpcHost 'test ! -e /nfs/hpc/share/bellaak/research/alpha-sweep/mobile-all30-eval-launch-v1'
if ($LASTEXITCODE -ne 0) { throw 'Bundle destination exists or SSH check failed' }
ssh $HpcHost 'test ! -e /nfs/hpc/share/bellaak/research/alpha-sweep/mobile-all30-eval-evidence-v1'
if ($LASTEXITCODE -ne 0) { throw 'Evidence destination exists or SSH check failed' }
scp -r "$TaskRoot\operators\mobile_all30_evaluation_v1" "${HpcHost}:/nfs/hpc/share/bellaak/research/alpha-sweep/mobile-all30-eval-launch-v1"
if ($LASTEXITCODE -ne 0) { throw 'Partial bundle transfer: preserve and inspect it' }
scp -r "$TaskRoot\handoff-evidence\mobile-all30-review" "${HpcHost}:/nfs/hpc/share/bellaak/research/alpha-sweep/mobile-all30-eval-evidence-v1"
if ($LASTEXITCODE -ne 0) { throw 'Partial evidence transfer: preserve and inspect it' }
```

Check transferred hashes against the local source inventory and the evidence's
`CHECKSUMS.json` before invoking copied code. Do not merge this operator into the
native checkout or overwrite/reuse the old six-file training operator.

## CPU: read actual inputs and inspect current HPC policy

Use the existing compatible environment on an approved CPU/controller host. No
model or CUDA runtime is constructed by these CPU commands. Actual checkpoint
binding can read substantial CPU checkpoint data and should not be repeated in
an unbounded loop on a login host.

```bash
set -euo pipefail
BASE=/nfs/hpc/share/bellaak/research/alpha-sweep
NATIVE="$BASE/alpha-sweep-mobile-eps0316"
STUDY="$BASE/mobile-eps0316-v1"
OP="$BASE/mobile-all30-eval-launch-v1"
EVIDENCE="$BASE/mobile-all30-eval-evidence-v1"
OPS="$BASE/mobile-eps0316-evaluation-ops-v1"
EVAL="$BASE/mobile-eps0316-evaluations-v1"
PY="$BASE/alpha-sweep-training/.venv/bin/python"
export PYTHONPATH="$NATIVE/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS VECLIB_MAXIMUM_THREADS
"$PY" -B "$OP/launcher.py" own-bundle
"$PY" -B "$OP/launcher.py" inspect-inputs --export-bindings "$EVIDENCE/data/evaluation_bindings.json"
```

`own-bundle` is a one-time exclusive write to the new bundle; `inspect-inputs` is
read-only. The latter reports mismatches and exits 22 if any exported binding
differs. There is no mismatch override. Resolve differences before freezing.

Read the actual native parsers at the frozen source. Their exact commands are:

```bash
"$PY" -B -m alpha_sweep_training.controller evaluation-plan --manifest "$STUDY/study.json"
"$PY" -B -m alpha_sweep_training.controller evaluation-inspect --manifest "$STUDY/study.json" \
  --global-index 0 --base-dir "$BASE/main-v1" --evaluation-root "$EVAL"
"$PY" -B -m alpha_sweep_training.controller evaluation-status --manifest "$STUDY/study.json" \
  --base-dir "$BASE/main-v1" --evaluation-root "$EVAL"
```

The pinned native `evaluation-plan` / `evaluation-status` implementation derives
its expected total from the actual manifest size: this profile has 30 tasks and
840 slots. The parser help still mentions the original 6,720-slot grid; it does
not change this study's size. Preserve per-checkpoint native 28-slot authorities.
The external wrapper uses the direct evaluation API for full binding and signal
handoff, rather than the native `controller evaluate` CLI.

Create a **new** inspection evidence directory. Replace `ACCOUNT` only after
checking your actual association; no historical account/node/QOS is an entitlement.

```bash
INSPECT="$BASE/mobile-all30-eval-inspection-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir "$INSPECT"
SUBMIT_UID=$(id -ru)
SUBMIT_USER=$(id -run)
sacctmgr -nP show assoc where user="$SUBMIT_USER" format=Cluster,Account,User,Partition,QOS,DefaultQOS > "$INSPECT/accounts.txt"
scontrol show partition dgxh >> "$INSPECT/accounts.txt"
sacctmgr -nP show qos format=Name,MaxWall,MaxTRESPU,GrpTRES >> "$INSPECT/accounts.txt"
sinfo -p dgxh -N -o '%N|%f|%G|%t|%m|%c' > "$INSPECT/nodes.txt"
scontrol show nodes >> "$INSPECT/nodes.txt"
unset SQUEUE_ACCOUNT SQUEUE_PARTITION SQUEUE_QOS SQUEUE_USERS SQUEUE_NAMES SQUEUE_STATES SQUEUE_FORMAT SQUEUE_FORMAT2 SLURM_CLUSTERS
id > "$INSPECT/active-work.txt"
squeue --local --all --array --states=all --user="$SUBMIT_UID" \
  --format='%i|%u|%U|%a|%P|%q|%T|%j|%R' >> "$INSPECT/active-work.txt"
df -h "$BASE" > "$INSPECT/storage.txt"
quota -s >> "$INSPECT/storage.txt"
```

Inspect account/QOS limits, shared project work under all users/accounts, node
features/GRES, accessible storage/quota and **evaluation-specific** wall-time
evidence. Write `walltime.txt` explaining the chosen request and its evidence.
No timing claim or default wall time is introduced here. If site commands or
accounting fields are unavailable, obtain administrator evidence and leave
execution blocked until the corresponding assumption is verified.

Write `$INSPECT/inspection.json` as strict JSON, replacing every angle-bracket
placeholder. Do not copy historical assertions without inspecting current work:

```json
{
  "inspected_at_utc": "<current UTC ISO timestamp>",
  "operator": "<actual operator>",
  "reason": "<account/QOS, GPU, storage and project reservation review>",
  "account": "<verified account>",
  "qos": "<verified QOS>",
  "partition": "dgxh",
  "node": "<verified full-H100 node>",
  "gres": "<verified one-GPU request, e.g. gpu:1>",
  "concurrency": 4,
  "other_project_reserved_tasks": 0,
  "all_prior_submissions_resolved": true,
  "exclusive_project_window": true,
  "wall_time": "<reviewed HH:MM:SS greater than 00:05:00>",
  "evidence_files": {
    "accounts": "<absolute inspection directory>/accounts.txt",
    "nodes": "<absolute inspection directory>/nodes.txt",
    "active_work": "<absolute inspection directory>/active-work.txt",
    "storage": "<absolute inspection directory>/storage.txt",
    "walltime": "<absolute inspection directory>/walltime.txt"
  }
}
```

Four is the approved project-wide maximum:
`concurrency + other_project_reserved_tasks <= 4`. Reservations are not increased
to absorb a shared account's backlog. The reported 1,453 EECS tasks owned by
unrelated users do not belong to this project's automatic queue count.

During plan creation, `submitter` is obtained from the kernel real/effective UID
and the POSIX account database (`pwd.getpwuid` and a reverse username lookup),
not from `USER` or `LOGNAME`. The plan pins `identity_version`, numeric `uid` and
canonical `username`. Different real/effective identities, failed lookups, or
UID/name disagreement reject creation. Submission/preflight, allocation claims,
acknowledgment reconciliation and settlement verify that actual OS identity
against the plan. Intents, acknowledgments, claims and settlement records retain
the same pin. A changed OS login or a same-name/different-UID login is rejected.

Automatic discovery uses `squeue --local --all --array --states=all --user=<UID>`
with **no account filter**, so the submitting user's work under every account
and partition on the configured local cluster is checked. Every pending array
element is a separate task. Known live/held/suspended/completing states count;
known terminal rows still visible briefly in `squeue` do not count as live.
Unknown states, malformed/compressed array output, duplicate jobs or unexpected
UID/name rows reject the inspection. The code clears all inherited `SQUEUE_*`
filters and `SLURM_CLUSTERS` before querying; the manual evidence command above
also clears common filter variables, and its full scope must be reviewed.
These option/field semantics follow the [official squeue manual](https://slurm.schedmd.com/squeue.html).

Every discovered live task of this user counts conservatively, even if it is
under a different account or belongs to unrelated work by that same user. Its
count must fit within `other_project_reserved_tasks`. Automatic discovery does
not infer project membership and does not discover collaborator-owned work,
other clusters, or submissions whose acknowledgment is still uncertain. Add
independently inspected evidence for those cases to `active-work.txt`; reserve
their full project capacity explicitly in `other_project_reserved_tasks`.
For example, one discovered owner task plus one collaborator reservation permits
at most two new concurrent tasks: `2 + 2 = 4`. The outside reservation stays two
even though only one task was automatically discovered.

All unresolved work must remain reserved. The separate unresolved-wave lock and
lost-acknowledgment rules still block another wave even when this user's queue is
empty; terminal queue rows alone never release those reservations. A project
reservation agreement must prevent other launchers/users/stages from adding
unreserved work during the window. Discovery is a cross-check, not a complete
project ledger or an enforcement mechanism over unrelated launchers.

Inspection must be no more than 15 minutes old at prepare/submission. Refresh to
a **new** directory before subsequent submissions. Combined limits are one GPU,
eight CPUs and 32 GB per task; maximum total resources are the per-task requests
times chosen concurrency, with each task's reviewed wall time. These are requested
resources, not measured needs or guaranteed QOS entitlements.

## CPU: freeze, validate, preview, then explicit submission

Freezing creates only the separate operational root, never an evaluation result.
The supplied export alone cannot freeze a plan; actual native binding runs first.

```bash
"$PY" -B "$OP/launcher.py" prepare --export-bindings "$EVIDENCE/data/evaluation_bindings.json" \
  --inspection "$INSPECT/inspection.json"
PLAN="$OPS/plan.json"
"$PY" -B "$OP/launcher.py" validate --plan "$PLAN"
"$PY" -B "$OP/launcher.py" submit --plan "$PLAN" --wave all30-first \
  --inspection "$INSPECT/inspection.json" --indices 0-29 \
  --reason 'Reviewed initial evaluation of all thirty selected checkpoints'
```

Preview performs no submission or intent write. Read the returned exact argv.
It includes `--array=0,1,...,29%N`, one GPU/process, `--signal=B:USR1@300`,
`--no-requeue`, exact environment paths, job name/comment, and separate logs.
Only after source/CPU-wiring verification and explicit operational authorization:

```bash
"$PY" -B "$OP/launcher.py" submit --plan "$PLAN" --wave all30-first \
  --inspection "$INSPECT/inspection.json" --indices 0-29 \
  --reason 'Operator-approved initial all-thirty evaluation' \
  --confirmed-project-window --execute
```

This is the **only** command that calls `sbatch`; it has not been invoked.
Submission acquires the controller lock, rejects every unresolved earlier wave,
rechecks native pins/checkpoint coverage/inspection/queue, and persists the exact
sealed intent **before** `sbatch --parsable`. Response and actual array ID are
exclusive separate records. Unknown acknowledgment keeps capacity reserved.

## GPU: allocation behavior

The batch shell `exec`s the owning Python process. Native
`allocation_signal_handlers(StopRequest())` are installed before allocation
validation; handlers only update stop state. Allocation identity and one
independent per-index claim must match the sealed intent. Runtime checks require
the recorded full H100 80GB HBM3, capability 9.0 and 85,009,367,040 bytes; MIG and
different devices reject. Existing Python 3.11.15, Torch 2.5.1+cu124,
TorchVision 0.20.1+cu124, NumPy 1.26.4, PyYAML 6.0.3, CUDA 12.4 and cuDNN 90100,
BF16 autocast, FP32 master/loss/noise and deterministic TF32-disabled backend
must match. No model, optimizer, scheduler or scaler is built by the wrapper;
its runtime preflight uses `evaluation_only=True`.

The native evaluator owns models/test loaders/leases/results. All-slot selectors
are `None`; default failed retry and legacy-clean reuse are false. Successes are
validated and skipped, partial passes restart from their original prescribed
realization, failures retain history, and native nonzero exits propagate. One
fixed native perturbation covers each entire official test pass. No controller
recovery or training invocation occurs on GPU.

## Monitor and reconcile without blind resubmission

Get the actual array ID from `waves/all30-first/ack.json`, parsed losslessly:

```bash
JOB=$("$PY" -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["array_job_id"])' \
  "$OPS/waves/all30-first/ack.json")
squeue --array -j "$JOB"
sacct -X -j "$JOB" --format=JobID%80,State%80,ExitCode,Elapsed,MaxRSS,AllocTRES -P
"$PY" -B -m alpha_sweep_training.controller evaluation-status --manifest "$STUDY/study.json" \
  --base-dir "$BASE/main-v1" --evaluation-root "$EVAL"
```

If acknowledgment was lost, inspect the preserved intent, unique job
name/comment, scheduler and accounting; do **not** call `submit` again. Once the
actual job has been independently identified, save read-only scheduler evidence
to a new file and use:

```bash
"$PY" -B "$OP/launcher.py" record-job --plan "$PLAN" --wave all30-first \
  --job-id "$ACTUAL_VERIFIED_ARRAY_ID" --reason 'Reconciled exact preserved submission intent' \
  --evidence "$ABSOLUTE_RECONCILIATION_FILE"
```

This also queries accounting for the pinned UID and requires the exact per-index
UID/username, job name, account, partition and plan-bound comment. It clears
inherited `SACCT_*` filters and `SLURM_CLUSTERS`. Accounting delay or a site that does not store
job comments blocks reconciliation; administrator assistance is needed. No
absence-of-queue inference frees reservations. The implementation has no command
to discard an unacknowledged intent or steal a lock. Partial transfer/ownership/
plan publication and stale controller/native locks remain evidence requiring
explicit cross-node ownership review; do not delete them to retry.

After all selected allocations have terminal accounting and verified inactive
owners across nodes, release only the scheduling reservation:

```bash
"$PY" -B "$OP/launcher.py" settle --plan "$PLAN" --wave all30-first \
  --confirmed-inactive-across-nodes --reason 'Verified exact array, all allocations and owners inactive'
```

Settlement queries the pinned user's live tasks across accounts and requires no
live member of the target array, every submitted index in terminal accounting
with matching UID/username, and explicit cross-node inactivity confirmation.
Unrelated live jobs of this user remain visible in the settlement snapshot but
do not prevent settling a different finished array; future submission still has
to reserve them. Malformed/missing accounting, wrong identities or unknown queue
states keep capacity reserved. Accounting fields follow the
[official sacct manual](https://slurm.schedmd.com/sacct.html).
Settlement does not prove research completion. If `sbatch` definitively rejected
without creating an array, this version deliberately leaves the wave blocked;
preserve its intent/response for an audited future no-job-resolution change.

For continuation, inspect native coverage and prior outcome/Slurm evidence first,
settle the prior wave, obtain fresh inspection, and preview a new named wave with
the **exact** indices needing work. Example selector syntax is `0,4-6`; replace
it with the actual reviewed list. Do not copy a list based on accuracy. New
submissions always require explicit `--execute`, reason and project reservation.

```bash
"$PY" -B "$OP/launcher.py" submit --plan "$PLAN" --wave continuation-01 \
  --inspection "$FRESH_INSPECTION_JSON" --indices "$REVIEWED_INDICES" \
  --reason 'Reviewed clean interruption and remaining native coverage'
```

Repeat that exact reviewed command with `--confirmed-project-window --execute`
only after explicit authorization. Failed-slot retries additionally require
`--retry-failed` and an explicit failure-specific reason, recorded in the new
intent. Numerical/execution failures are never automatically retried, and poor
accuracy is never a reason. Original successful slots are retained and skipped.

## CPU: all-30 reporting, then derived figures

After execution is quiescent, choose new output directories outside protected
stores. These commands write only separate derived outputs:

```bash
REPORT="$BASE/mobile-all30-eval-report-$(date -u +%Y%m%dT%H%M%SZ)"
"$PY" -B "$OP/report.py" --plan "$PLAN" --output-dir "$REPORT"
FIGURES="$BASE/mobile-all30-eval-figures-$(date -u +%Y%m%dT%H%M%SZ)"
"$PY" -B "$OP/figures.py" --report "$REPORT/report.json" --output-dir "$FIGURES"
```

`report.json` retains all 30 training rows and 840 evaluation rows. Evidence is
appended as read: actual bindings, native coverage, references, metrics, ancestry,
parsed raw records, attempts, orphan failures and unparsed evidence. A slot-local
failure is recorded on that slot and independent slots/sections continue.
`inspection_state` distinguishes `missing`, `unreadable`, `incompatible` binding
mismatches and `unstable` snapshots, with structured `inspection_errors` and
`checkpoint_errors`. Never-read rows retain null identities/seeds/metrics; earlier
observations survive later errors.

Reports add `verification_version: 1` and explicit verification booleans:

- `binding_verified`: the actual binding matches the plan and training authority
  remains verified.
- `observation_verified`: the slot was inspected without error within a verified
  binding and stable evaluation snapshot. Pending observations can be verified
  observations without a measurement.
- `measurement_verified`: the slot is a verified successful full-pass measurement.
- Training rows record `training_snapshot_verified` and
  `evaluation_snapshot_verified`. Robustness completion requires native published
  sweep authority **and all 28 verified measurements**.

A checkpoint-wide binding/stability/read failure invalidates observation and
measurement authority across all 28 slots without erasing raw evidence. Training
snapshot failure also invalidates validation-selection authority; evaluation-only
failure does not discard independently verified training evidence. A raw record's
`validated` field covers only its local parsing, not checkpoint/slot authority.
Status inside retained native coverage/results is an observation, not the report's
verified completion claim.

`slots.csv` is a convenience view; uint64 noise seeds must be read as integers,
never via spreadsheet/JavaScript floating-point conversion. Native HEADs remain
the authority. An allocation ending or diagnostic outcome never creates a success.
CSV numeric columns remain **raw**, accompanied by verification/state fields.
Raw numbers can be present with `measurement_verified: false`; analysis must
filter with all verification fields. JSON retains the full underlying evidence.

Positive realizations are averaged **within each training seed first**. Official
cross-seed summaries require all three seeds and use sample SD (`ddof=1`),
separately from within-checkpoint realization SD. Clean has one pass/seed and no
within-checkpoint SD. Validation selects training alpha only when all ten
three-seed candidates are complete, retaining exact mean ties in grid order.

Figures include ten training-alpha curves for test accuracy and loss; seed/
realization panels for both metrics; paired accuracy differences in percentage
points and retention relative to each checkpoint's own clean baseline. Missing
sets appear as gaps; all raw scores remain in the report. Formulas are explicit,
with null retention for a zero clean baseline. No AUC or new selection rule is
introduced. Final group presentations must show each graph type for every model,
label revised MobileNet epsilon 0.0316 and original other-model recipes, and avoid
calling population weight SD a flatness measurement. This stage does not evaluate
other architectures or produce a final all-model presentation.

Both aggregation and individual-point plotting require explicit verification.
Plots recompute summaries from verified raw rows rather than trusting cached
summaries. Older reports without `verification_version: 1` are rejected before
output creation; regenerate them with the revised read-only collector after
reporting is separately authorized. Unverified metrics enter no curve, error
bar, individual point or completion count.

## Executable CPU wiring checks

From this review worktree, use standard Python 3.11+:

```powershell
python -B operators/mobile_all30_evaluation_checks/run_cpu_checks.py
```

Actual final local result: **44 tests passed in 52.731 seconds** on Windows /
Python 3.13.12, including all 28 prior tests and 16 queue/identity/reservation/
settlement regressions. Syntax and Python 3.11 grammar checks passed for all five
Python files; `git diff --check` passed. No dependency installation, real checkpoint load,
Slurm contact, HPC operation or research computation occurred. POSIX behavior and
the real pinned GPU environment remain unverified here.

The runner requires only stdlib, the external bundle, and readable
`interruption.py` / `allocation_signals.py` helper source from this checkout.
It closes the native package namespace to actual research imports and mocks
Torch/CUDA/runtime/checkpoint/evaluator/Slurm boundaries. The real external
functions run, as do the stop/signal helpers with mocked signal registration;
no OS signal is sent. All records and mocked rendering outputs are temporary
fixtures. POSIX UID/account-database calls are mocked for portable tests; OS
identity on the actual HPC hosts has not been inspected here. Windows mocks POSIX directory-fsync; file writes, exclusive hard-link
publication and ownership checks remain real. Linux uses real directory-fsync.

For a future approved HPC CPU repeat, stage a new review directory with only:

```text
<new-review-directory>/
  mobile_all30_evaluation_v1/       # exactly the six runtime files
  mobile_all30_evaluation_checks/   # exactly the two check-support files
```

PowerShell transfer pattern, after separate transfer authorization, using the
established `$HpcHost` and `$TaskRoot` from the earlier example:

```powershell
$CpuReview = '/nfs/hpc/share/bellaak/research/alpha-sweep/mobile-all30-eval-cpu-review-20261104'
ssh $HpcHost "test ! -e $CpuReview"
if ($LASTEXITCODE -ne 0) { throw 'Review destination exists or SSH check failed' }
ssh $HpcHost "mkdir $CpuReview"
if ($LASTEXITCODE -ne 0) { throw 'Review directory creation failed' }
scp -r "$TaskRoot\operators\mobile_all30_evaluation_v1" "${HpcHost}:$CpuReview/"
if ($LASTEXITCODE -ne 0) { throw 'Partial external source transfer; preserve it' }
scp -r "$TaskRoot\operators\mobile_all30_evaluation_checks" "${HpcHost}:$CpuReview/"
if ($LASTEXITCODE -ne 0) { throw 'Partial CPU support transfer; preserve it' }
```

Exact future CPU command; no `own-bundle`, plan creation or installed research
package is required for the tests:

```bash
CPU_REVIEW=/nfs/hpc/share/bellaak/research/alpha-sweep/mobile-all30-eval-cpu-review-20261104
NATIVE=/nfs/hpc/share/bellaak/research/alpha-sweep/alpha-sweep-mobile-eps0316
PY=/nfs/hpc/share/bellaak/research/alpha-sweep/alpha-sweep-training/.venv/bin/python
"$PY" -B "$CPU_REVIEW/mobile_all30_evaluation_checks/run_cpu_checks.py" \
  --native-source-root "$NATIVE"
```

This reads only existing stop/signal helper source at the frozen checkout; it
does not install, check out, replace or alter that checkout. The review source
is never adopted as an existing operational plan. See `REVIEW.md` for actual
local results and remaining limitations. Historical native CUDA/CPU results in
the ZIP remain separate evidence.
