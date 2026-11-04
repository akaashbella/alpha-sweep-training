# MobileNet all-30 evaluation: v2 partition-filter handoff

This revision starts at `9edde981019201c114f0f5c6ed4f409042dd443b` on
`ops/mobile-all30-evaluation-v1`. The repository directory still has its historical
`mobile_all30_evaluation_v1` name; the revised operational version is
`mobile-all30-evaluation-operator-v2`. Deploy it to **new paths** only; do not
replace an already-owned v2 bundle.

Only local mocked CPU tests, syntax checks and scoped Git publication are
authorized in this task. No command below was run against HPC here. Actual
checkpoint loading, scheduler queries, submission, evaluation, reporting and
installation were not performed. Future operational commands require separate
authorization. The user reports all previous 58 checks passing on HPC, followed
by the real collector rejecting the deny-only partition output below. That pass
does not establish support for every real scheduler representation.

## Preserve the observed state

The user reported that preparation exhausted the old snapshot's freshness window:
it was 23.7 minutes old, no operational plan existed, the result JSON was empty,
and validation/submission preview never ran. The reported OS identity was
`bellaak`, UID 17907, with an empty user queue; all 30 bindings had matched the
export previously. These facts are not a new inspection or a reusable checkpoint
verification certificate.

Keep these exact existing paths and bytes:

| Purpose | Path under `/nfs/hpc/share/bellaak/research/alpha-sweep` |
|---|---|
| Already-owned v1 external bundle | `mobile-all30-eval-launch-v1` |
| Existing v2 deployment and CPU staging, if present | `mobile-all30-eval-launch-v2`, `mobile-all30-eval-cpu-review-v2` |
| Failed inspection, including empty `preparation-result.json` | `mobile-all30-eval-inspection.ZUxVLZ` |
| Every existing resource receipt, including failed collections | `mobile-all30-eval-resources-v2-*` |
| Historical v1 operational root, if any appears | `mobile-eps0316-evaluation-ops-v1` |
| Frozen native checkout | `alpha-sweep-mobile-eps0316` |
| Manifest/pilot/training artifacts | `mobile-eps0316-v1` |
| Original study and prepared data | `main-v1` |

New destinations:

| Purpose | Path under the same base |
|---|---|
| CPU review source staging | `mobile-all30-eval-cpu-review-v2-qos-filter` |
| New owned runtime bundle | `mobile-all30-eval-launch-v2-qos-filter` |
| Operational plan/registry, only if absent | `mobile-eps0316-evaluation-ops-v2/plan.json` |
| Each new resource snapshot | `mobile-all30-eval-resources-v2-<unique-token>` |

The native evaluation root remains `mobile-eps0316-evaluations-v1`; it is an
independent native result namespace, not an operational deployment version.
Native source stays at `8010d07d9d1f0886ab1e180523f316623ef622a4`, inventory
`a5726bac3f449455d36e2f53840f3e0fd77768a7ab2b989d2884da20f920005c`.
No scientific setting, checkpoint binding, training identity or protocol changed.

## Exact inventory

The runtime directory contains exactly six source files, plus its newly created
`BUNDLE_OWNER.json` after `own-bundle`:

- `launcher.py`: documented partition allow/deny selection, v2 phases, native
  verification, fresh resource collection, ownership, OS identity and submission.
- `batch.sh`: unchanged owning-Python `exec` and native allocation signal handoff.
- `report.py`: unchanged evidence-preserving 30/840 reporting and verification flags.
- `figures.py`: unchanged verified-only measurements and protected output paths.
- `HPC_HANDOFF.md`: this operating sequence.
- `REVIEW.md`: evidence, actual checks and remaining limitations.

CPU support remains outside the runtime inventory:
`operators/mobile_all30_evaluation_checks/run_cpu_checks.py` (unchanged) and
`test_operator.py` (extended). This revision changes four files: `launcher.py`,
both runtime Markdown files and `test_operator.py`. No new files.
Do not copy the older full repository snapshot over the pinned native checkout.

## Freshness and authority

`--resource-request` names an existing reviewed inspection-shaped JSON and its
evidence files. The old `--inspection` spelling remains an alias. Its resource
choices and manual reservation declarations are inputs; its timestamp is **not**
used as fresh scheduler evidence. It can be the preserved failed inspection after
the operator reviews/reconfirms its choices and project reservation agreement.
No timestamp or evidence file in that input is rewritten.

Preparation does the following in order:

1. Check actual OS identity, new bundle ownership, old-registry obstruction and
   explicit resource declarations.
2. Fully verify all 30 actual best checkpoints, native source/software/launch
   inputs, manifest and export bindings. No prior command's success is reused.
3. Re-read the request bytes to reject changes during the expensive phase.
4. Collect **new** association, QOS, partition, node, storage and user-queue
   evidence into an exclusively owned snapshot. Verify chosen eligibility and
   reservations, never choose a substitute account/QOS/node or retune resources.
5. Recheck current static source/launch/document bytes and external code, then
   verify freshness and publish only a new v2 operational plan.

The 900-second window starts at the earliest collection time, after lengthy
checkpoint checks. Collection records per-query start/end times, actual stdout,
file hashes, an ownership receipt and `COLLECTION.json`. The monotonic collection
duration must also fit the window. Fresh reads check ownership, evidence bytes,
receipt times and eligibility again. Merely changing a timestamp cannot make old
receipts fresh. A final age check follows metadata-byte validation so a slow read
cannot consume the freshness window unnoticed. Failed/incomplete snapshots remain
evidence; no deletion or automatic repair is performed.

Submission validates documents first, then holds the controller lease while it
checks all unresolved waves, performs full current source/checkpoint/coverage
verification and collects fresh resources. It does not reuse the preview snapshot
or a saved successful checkpoint inspection. User queue discovery runs after the
resource queries. Static source/launch checks and freshness checks follow; no
lengthy tensor operation occurs after collection. The exact intent is durable
before `sbatch`; freshness, identity and external bytes are checked again just
before invoking it. A delay or change after intent publication blocks submission
and leaves the intent reserved for audited reconciliation.

`account`, `qos`, `partition`, `node`, `gres` and `wall_time` must match the frozen
plan request. Explicit concurrency/outside-reservation declarations can be lowered
or updated in a separately reviewed request, always with
`concurrency + other_project_reserved_tasks <= 4`. Current discovered owner work
must fit that outside reservation; insufficiency rejects rather than increasing
it. All OS UID/name, expanded-array, lost-acknowledgment, no-requeue and native
slot-lease behavior remains in force.

Automatic eligibility checks cover resolved association/QOS membership,
partition allow/deny lists, node membership/state/configured CPU/memory/GRES and
partition/QOS wall-time limits. They do not prove available GPUs, user storage
quota, every site-specific QOS/TRES/group entitlement, or collaborator inactivity.
Unsupported/ambiguous output rejects for review; there is no bypass or install.
The operator must still approve current policy and outside-work reservations.
The submitting user's live tasks are queried across accounts on the configured
local cluster; collaborator-owned work, other clusters and uncertain submissions
need explicit independent reservations. Existing v1 operations evidence blocks
new work until cross-version ownership/capacity is separately reconciled.

### Partition access filters and the job QOS

For both account and QOS filters, an explicit `Allow*` list takes precedence over
the corresponding `Deny*` list; `ALL` permits any choice. With only `Deny*`
present, choices outside that list pass. These are Slurm's documented
[partition filter semantics](https://slurm.schedmd.com/slurm.conf.html).
The supplied OSU record has `AllowAccounts=ALL DenyQos=dgx2-plus QoS=dgxh`:
the QOS filter permits requested `normal` and rejects `dgx2-plus`. No site-specific
exception or resource-request change is applied.

Partition `QoS=dgxh` supplies a partition limit policy; it does not replace the
job's requested `--qos=normal`. See Slurm's
[partition QOS explanation](https://slurm.schedmd.com/qos.html). Association
membership for `normal` is still required. Empty, malformed, conflicting field
aliases and mixed wildcard lists reject. If both access-filter fields are absent,
the collector rejects incomplete evidence rather than inferring access from
plain `QoS`. Even a superseded deny field must be structurally valid.

Account membership checks use exact names; they do not resolve Slurm account
ancestry. Attached partition-QOS limits and effective TRES/group/OverPartQOS
policy are not fully resolved by this collector. These remain site-policy review
limits, not permission to adjust requests or weaken reservations.

Progress is flushed to stderr: named phase start, checkpoint index/run ID,
completion/failure and monotonic elapsed seconds. Stdout remains strict result
JSON. A checkpoint may still take time, but its current phase is visible.

## Local CPU checks

From this worktree:

```powershell
python -B operators/mobile_all30_evaluation_checks/run_cpu_checks.py
```

Only stdlib, external source and readable native stop/signal helper source are
needed. Native package imports, checkpoints, Torch/CUDA, scheduler commands and
renderer calls are mocked. All fixture outputs are temporary. OS identity and
POSIX directory-fsync are mocked on Windows; exclusive temporary file publication
is real. The runner checks all five Python files' syntax and Python 3.11 grammar.
Actual local result: **66 tests passed in 70.859 seconds** on Windows / Python
3.13.12, covering all previous 58 checks plus eight partition-filter regressions.
Syntax and Python 3.11 grammar checks passed for all five Python files; whitespace
checks passed. See `REVIEW.md` for the distinction from real HPC verification.

## Exact future continuation commands

First stage only the external source and tests to a **new** directory. In local
PowerShell, set the established OSU SSH target; do not guess or change login hosts:

```powershell
$HpcHost = Read-Host 'Established OSU SSH target (user@host)'
$TaskRoot = 'C:\Users\akaas\RESEARCH\alpha-sweep-training\.handoff-worktrees\mobile-all30-evaluation'
$Review = '/nfs/hpc/share/bellaak/research/alpha-sweep/mobile-all30-eval-cpu-review-v2-qos-filter'
ssh $HpcHost "test ! -e $Review"
if ($LASTEXITCODE -ne 0) { throw 'Review destination exists or SSH check failed' }
ssh $HpcHost "mkdir $Review"
if ($LASTEXITCODE -ne 0) { throw 'New review directory creation failed' }
scp -r "$TaskRoot\operators\mobile_all30_evaluation_v1" "${HpcHost}:$Review/"
if ($LASTEXITCODE -ne 0) { throw 'Partial external source transfer: preserve and inspect' }
scp -r "$TaskRoot\operators\mobile_all30_evaluation_checks" "${HpcHost}:$Review/"
if ($LASTEXITCODE -ne 0) { throw 'Partial check-support transfer: preserve and inspect' }
```

On an approved HPC CPU/controller host, after separate authorization:

```bash
set -euo pipefail
BASE=/nfs/hpc/share/bellaak/research/alpha-sweep
NATIVE="$BASE/alpha-sweep-mobile-eps0316"
PY="$BASE/alpha-sweep-training/.venv/bin/python"
REVIEW="$BASE/mobile-all30-eval-cpu-review-v2-qos-filter"
OP="$BASE/mobile-all30-eval-launch-v2-qos-filter"
OPS="$BASE/mobile-eps0316-evaluation-ops-v2"
FAILED="$BASE/mobile-all30-eval-inspection.ZUxVLZ"
REQUEST="$FAILED/inspection.json"
EVIDENCE="$BASE/mobile-all30-eval-evidence-v1"
EVAL="$BASE/mobile-eps0316-evaluations-v1"
test "$(id -ru)" = 17907
test "$(id -run)" = bellaak
test -s "$REQUEST"
test ! -e "$OP"
test ! -e "$OPS"
test ! -e "$BASE/mobile-eps0316-evaluation-ops-v1"
"$PY" -B "$REVIEW/mobile_all30_evaluation_checks/run_cpu_checks.py" --native-source-root "$NATIVE"
cp -R "$REVIEW/mobile_all30_evaluation_v1" "$OP"
export PYTHONPATH="$NATIVE/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS VECLIB_MAXIMUM_THREADS
"$PY" -B "$OP/launcher.py" own-bundle
cat "$REQUEST"
```

If any guarded destination already exists, stop and preserve it. An existing v2
plan pins the previous external bytes and must not be edited to accept this
bundle. Reconciliation or a separately reviewed new operational root is required
before proceeding; these commands never migrate or overwrite a plan.

Review all resource choices, wall-time justification and the current project
reservation agreement before continuing. If the original request is elsewhere,
locate the actual file read-only; never create one by overwriting the failed
evidence. If choices/declarations need changes, write and review a separate new
request file. Existing evidence/timestamps are not edited. The code reconfirms
live scheduler eligibility and owner work after each long verification.

```bash
NOTE=$(mktemp -d "$BASE/mobile-all30-v2-controller.XXXXXX")
"$PY" -B "$OP/launcher.py" prepare \
  --export-bindings "$EVIDENCE/data/evaluation_bindings.json" \
  --resource-request "$REQUEST" > "$NOTE/preparation-result.json"
PLAN="$OPS/plan.json"
"$PY" -B "$OP/launcher.py" validate --plan "$PLAN" > "$NOTE/validation-result.json"
"$PY" -B "$OP/launcher.py" submit --plan "$PLAN" --wave all30-first \
  --resource-request "$REQUEST" --indices 0-29 \
  --reason 'Reviewed all-thirty evaluation and explicit project reservations' \
  > "$NOTE/submission-preview.json"
```

Stderr progress stays visible while JSON is saved only to the new controller-note
directory. There is no need to retimestamp the request after a long validation:
preview and explicit submission each verify inputs and collect their own new
resource bytes. Preview creates a new resource snapshot but no wave intent or
job. Review its returned exact argv and evidence paths. A separate
`inspect-inputs` command is optional read-only diagnosis, not a cache for prepare.

Only after separate explicit operational approval, the actual submission command is:

```bash
"$PY" -B "$OP/launcher.py" submit --plan "$PLAN" --wave all30-first \
  --resource-request "$REQUEST" --indices 0-29 \
  --reason 'Operator-approved initial all-thirty evaluation with current reservation agreement' \
  --confirmed-project-window --execute > "$NOTE/submission-result.json"
```

That command was not run. It keeps `--signal=B:USR1@300`, `--no-requeue`, one
GPU/process per checkpoint and native `expected_binding` for all 28 slots.
Native source/evaluation math remains unchanged; clean replicate 0 and nine
positive evaluation alphas with replicates 1–3 produce 840 slots, including pilots.
No retraining, optimizer, scheduler, GradScaler, test-based selection or poor-score
exclusion is added. Preserve lossless Python JSON integer handling.

## Monitor, reconcile and continue later

Read the actual array ID from the v2 acknowledgment, not a guessed job number:

```bash
JOB=$("$PY" -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["array_job_id"])' \
  "$OPS/waves/all30-first/ack.json")
squeue --array -j "$JOB"
sacct -X -j "$JOB" --format=JobID%80,State%80,ExitCode,Elapsed,AllocTRES -P
"$PY" -B -m alpha_sweep_training.controller evaluation-status \
  --manifest "$BASE/mobile-eps0316-v1/study.json" --base-dir "$BASE/main-v1" --evaluation-root "$EVAL"
```

An absent acknowledgment does not authorize another submission. Preserve exact
intent/response/claims and match the unique name/comment against scheduler
evidence. Once independently resolved, use `record-job --plan "$PLAN" --wave
all30-first --job-id <actual-id> --reason <audited-reason> --evidence <absolute-file>`.
It requires owned, exact complete accounting linkage. A pre-sbatch freshness
failure after intent publication also remains blocked for audited no-job
resolution; do not delete it to retry. No automatic intent-abandonment exists.

Once all target allocations/owners are verified inactive across nodes and exact
accounting is terminal, `settle --plan "$PLAN" --wave all30-first
--confirmed-inactive-across-nodes --reason <audited-reason>` releases scheduling
reservation only. It never proves scientific completion. A later explicit named
wave uses exact reviewed indices, a reviewed resource request, fresh automatic
collection and the same full verification. Failed-slot retry additionally needs
`--retry-failed` and an audited failure-specific reason; poor accuracy is not one.

After quiescence and separate reporting authorization, `report.py --plan "$PLAN"
--output-dir <new-derived-root>` preserves every 30/840 observation and raw
history. `figures.py --report <that-root>/report.json --output-dir
<another-new-root>` uses verified measurements only. Output paths cannot overlap
native/training/evaluation/v1/failed-inspection stores. Raw CSV numbers can exist
with false verification flags; aggregation and plots require verified bindings,
observations and measurements. Realizations average within checkpoint before
three-seed sample SD. Validation alone selects training alpha from all ten complete
candidates; exact ties remain. Old reports without the verification marker are
rejected for plotting. All-model presentation/recipe labeling remains outside
this MobileNet operator task.
