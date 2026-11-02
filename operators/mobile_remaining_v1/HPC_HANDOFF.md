# Remaining 27 MobileNet runs: external operator v1

This is a source-only execution handoff prepared from the supplied
`mobile-pilot-review/CODEX_NEXT_STEPS.md`, `FINDINGS.md`, verification JSON,
original validation JSON and epoch/layer CSVs. No job, HPC command, package
installation, native mutation, commit, push or report generation was performed
for this handoff. CPU checks are **unrun**: no compatible local environment was
found. Local review checkout: `/private/tmp/alpha-sweep-mobile-remaining-review`,
branch `ops/mobile-remaining27-v1`, based on `8010d07`.

The uploaded review reports 600 finite epochs/38,400 finite tensor SD records,
three complete 200-epoch pilots, best epochs 179/174/189, and final validation
91.56%/91.46%/91.44%. It recommends proceeding with the frozen epsilon .0316
recipe. Transient validation dips remain evidence, not exclusions. These are
supplied derived observations: this session has no direct HPC access and has
not reloaded native checkpoints or obtained the actual thirty-task manifest.
The uploaded source/manifest identities are execution expectations, not locally
fabricated copies of missing HPC files. No official test score selects epsilon.

## Design and exact changed-file inventory

Only these six new external files are added to the isolated review checkout:

1. `operators/mobile_remaining_v1/launcher.py`: actual native/source/software/
   manifest/launch checks, explicit resource evidence, separate plan ownership,
   optional explicit submission with durable intent, allocation claims, native
   dispatch and read-only status.
2. `operators/mobile_remaining_v1/batch.sh`: original environment activation,
   native checkout `cd`/PYTHONPATH, audited thread/CUBLAS setup, Bash `exec` into
   the owning Python process.
3. `operators/mobile_remaining_v1/report.py`: separate all-30 historical reader,
   all epochs/layer SDs/failures, complete-set validation summaries, and all 840
   independent evaluation observations with preserved raw result histories.
4. `operators/mobile_remaining_v1/test_operator.py`: nine focused CPU checks
   with GPU, data/preparation, provenance, scheduler submission and writer doubles.
5. `operators/mobile_remaining_v1/run_cpu_checks.py`: compatible-environment gate
   and explicit CPU-only selection, no mixed GPU discovery or installations.
6. `operators/mobile_remaining_v1/HPC_HANDOFF.md`: this review and operator guide.

There are **no native package, configuration, scientific specification, original
pilot script or native test edits**. The external directory must be copied by
itself to a separate HPC directory. Do not deploy the entire development checkout
over the frozen native source. All six external file bytes/paths are independently
pinned in the new operational plan; they are outside native source inventory v2.

Frozen inputs checked by the launcher:

| Item | Required value |
| --- | --- |
| Native commit | `8010d07d9d1f0886ab1e180523f316623ef622a4` |
| Native source digest | `a5726bac3f449455d36e2f53840f3e0fd77768a7ab2b989d2884da20f920005c` |
| Profile | `cifar10-mobile-eps0316-v1` |
| Grid | `2c0689e4aa3d0b6734323642a89dd7de809337caabcbe358ab600b7287a16167` |
| Manifest document digest | `08555fb94e023849d37c0140833a21e0db12b02ce4ab382bc32d4202b76f6423` |
| Python / Torch / TorchVision | `3.11.15` / `2.5.1+cu124` / `0.20.1+cu124` |
| NumPy / PyYAML / CUDA / cuDNN | `1.26.4` / `6.0.3` / `12.4` / `90100` |

The actual `pilot.json` supplies the original base directory and launch pins.
Its file bytes and `study.json` file bytes are also pinned. Native `load_pilot`
and strict manifest readers remain authoritative for ownership/full Cartesian
validation. Existing pilot journal authority is rechecked before planning and
GPU dispatch. Uploads and array 22011623 never stand in for those reads. Native
Git HEAD must match and the native checkout must be clean. Private HPC patches
are not overwritten or waived; a mismatch stops preparation for explicit review.

| Global indices | Training alpha | Actual seed order |
| --- | ---: | --- |
| 3–5 | .01 | 0, 1, 2 |
| 6–8 | .02 | 0, 1, 2 |
| 9–11 | .05 | 0, 1, 2 |
| 12–14 | .10 | 0, 1, 2 |
| 15–17 | .20 | 0, 1, 2 |
| 18–20 | .30 | 0, 1, 2 |
| 21–23 | .50 | 0, 1, 2 |
| 24–26 | .75 | 0, 1, 2 |
| 27–29 | 1.00 | 0, 1, 2 |

Indices 0–2 and out-of-range values reject before any GPU dispatch. No index is
treated as a seed. All thirty native tasks/run IDs/fingerprints remain intact;
the original three pilots are neither selected nor cloned. The native optimizer,
200-epoch horizon, four data workers, clean/master weights, original RNG streams,
best/latest/final checkpoints, earliest ties and ContinuationPlan are unchanged.
No launcher/index-derived RNG keys, new scientific profile or thresholds exist.

## Later deployment and compatible CPU checks — unrun

After review, transfer only the six-file external directory to the chosen new
operator directory below. No transfer is executed here. These commands are for
the operator on HPC with the existing shared paths; they are not evidence that
the paths or resources have been inspected in this session.

```bash
set -euo pipefail
NATIVE=/nfs/hpc/share/bellaak/research/alpha-sweep/alpha-sweep-mobile-eps0316
STUDY=/nfs/hpc/share/bellaak/research/alpha-sweep/mobile-eps0316-v1
EXISTING_ENV=/nfs/hpc/share/bellaak/research/alpha-sweep/alpha-sweep-training/.venv
export OPERATOR_HOME=/nfs/hpc/share/bellaak/research/alpha-sweep/mobile-remaining27-launch-v1
export OPERATION_ROOT=/nfs/hpc/share/bellaak/research/alpha-sweep/mobile-eps0316-remaining27-ops-v1
source "$EXISTING_ENV/bin/activate"
cd "$NATIVE"
export PYTHONPATH="$NATIVE/src"
git status --short
git rev-parse HEAD
sha256sum "$STUDY/study.json" "$STUDY/pilot.json"
sha256sum "$OPERATOR_HOME/launcher.py" "$OPERATOR_HOME/report.py" \
  "$OPERATOR_HOME/batch.sh" "$OPERATOR_HOME/test_operator.py" \
  "$OPERATOR_HOME/run_cpu_checks.py" "$OPERATOR_HOME/HPC_HANDOFF.md"
"$EXISTING_ENV/bin/python" -B "$OPERATOR_HOME/run_cpu_checks.py" --native-source "$NATIVE"
```

Nine checks cover all 27 mappings, forbidden selectors, exact source/software/
manifest/launcher pins, storage/runtime declarations, combined capacity,
intent-before-sbatch and lost-ack blocking, actual native dispatch with mocked
CUDA/signals and single allocation claims, native auto continuation/completion/
failure decisions, CPU recovery separation, 30/840 missing-observation/tie summary
semantics, and full trajectory/layer/historical-failure retention. They are wiring
checks, not GPU, checkpoint or positive-alpha scientific verification. They reuse
existing native unit declarations only as
fixtures; no dataset is prepared and no native training model is constructed.

## Current evidence and explicit capacity choice — not collected here

Before freezing a plan, coordinate old/new work and pause other experiment
submission controllers as needed. This v1 operator path does not own the main-v1
managed registry and cannot enforce capacity against an independently submitting
controller. The operator must reserve a combined allowance of at most four
experiment tasks throughout this array. Inspect all relevant accounts/jobs,
including pending replacement intents and independent evaluation work. Unresolved
submission intent reserves capacity until actual scheduler/accounting evidence
resolves it; a missing process or old timestamp is not enough.

Capture actual current evidence in a new external directory, outside the study
and operator plan root. Add queries for other users/accounts if their experiment
work shares the intended allowance. These queries make no allocation request:

```bash
export INSPECTION_DIR=/nfs/hpc/share/bellaak/research/alpha-sweep/mobile-remaining27-inspection-$(date -u +%Y%m%dT%H%M%SZ)
mkdir "$INSPECTION_DIR"
sacctmgr show assoc where user="$USER" format=Cluster,Account,Partition,QOS > "$INSPECTION_DIR/accounts.txt"
scontrol show nodes > "$INSPECTION_DIR/nodes.txt"
squeue -u "$USER" -o '%.18i %.12P %.20j %.8u %.10T %.12a %.12q %.25R %.15b' > "$INSPECTION_DIR/active_work.txt"
sinfo -p dgxh -N -o '%N %t %G %c %m'
```

Inspect the files and current policies. Historical `eecs`/dgxh-2 full-H100
success is not a current entitlement/availability claim. dgxh-1 historically
exposed MIG slices. Do not guess a typed GPU label. The exact one-GPU GRES
request, node, account, QOS choice, other active experiment count and new
concurrency must be supplied after inspection. Twelve hours/eight CPUs/32 GB
remain unmeasured allocation requests, not a guarantee for nonzero alphas.

The following commands intentionally require operator-selected values; none is
populated with a guessed valid account/GPU/concurrency. An explicitly exported
empty VERIFIED_QOS means the operator verified that no QOS override is needed.
The concurrency must be 1–4 and its sum with other reserved/active experiment
slots must be at most four. Only set prior-submission resolution after checking
the relevant intent/accounting records across old/new work.

```bash
: "${VERIFIED_ACCOUNT:?Set from actual current account evidence}"
: "${VERIFIED_NODE:?Set the current single node with an actual full H100}"
: "${VERIFIED_GRES:?Set the exact one-GPU GRES request from current evidence}"
: "${VERIFIED_QOS?Explicitly export the verified QOS or an empty value}"
: "${VERIFIED_CONCURRENCY:?Choose after inspecting old/new work}"
: "${OTHER_EXPERIMENT_SLOTS:?Count reserved or active old/new experiment slots}"
: "${PRIOR_SUBMISSIONS_RESOLVED:?Export yes only after all ambiguity is resolved}"
: "${INSPECTION_REASON:?Describe actual resource review and capacity coordination}"
export VERIFIED_ACCOUNT VERIFIED_NODE VERIFIED_GRES VERIFIED_QOS VERIFIED_CONCURRENCY
export OTHER_EXPERIMENT_SLOTS PRIOR_SUBMISSIONS_RESOLVED INSPECTION_REASON
"$EXISTING_ENV/bin/python" -B - <<'PY'
from datetime import datetime, timezone
import json, os
from pathlib import Path
e = os.environ
assert e['PRIOR_SUBMISSIONS_RESOLVED'] == 'yes'
root = Path(e['INSPECTION_DIR'])
record = dict(inspected_at_utc=datetime.now(timezone.utc).isoformat(), operator=e['USER'],
    reason=e['INSPECTION_REASON'], account=e['VERIFIED_ACCOUNT'], qos=e['VERIFIED_QOS'] or None,
    node=e['VERIFIED_NODE'], gres=e['VERIFIED_GRES'], concurrency=int(e['VERIFIED_CONCURRENCY']),
    other_active_experiment_tasks=int(e['OTHER_EXPERIMENT_SLOTS']), all_prior_submissions_resolved=True,
    evidence_files={k: str(root / (k + '.txt')) for k in ('accounts', 'nodes', 'active_work')})
with (root / 'inspection.json').open('x') as f:
    json.dump(record, f, indent=2, allow_nan=False)
PY
```

These evidence files support an explicit operator attestation, not an automated
account/QOS parser or a scheduler-wide capacity lock. If resources/work change
before submission, take new evidence and freeze a new plan; do not rewrite pins.

## Freeze, inspect, and separately authorize submission

Declare the actual original main-v1 workflow/evaluation/protected namespaces.
Their paths are not included in this upload. Required variables below must come
from real storage evidence, not inferred placeholders. Add --protected-root for
other research/evaluation/report stores. Existing candidate source, study/data
and all thirty native artifact paths are protected automatically.

```bash
: "${ORIGINAL_WORKFLOW_ROOT:?Actual original managed registry namespace}"
: "${ORIGINAL_EVALUATION_ROOT:?Actual original independent evaluation namespace}"
"$EXISTING_ENV/bin/python" -B "$OPERATOR_HOME/launcher.py" prepare \
  --output-dir "$OPERATION_ROOT" --inspection "$INSPECTION_DIR/inspection.json" \
  --protected-root "$ORIGINAL_WORKFLOW_ROOT" --protected-root "$ORIGINAL_EVALUATION_ROOT"
export REMAINING_PLAN="$OPERATION_ROOT/plan.json"
"$EXISTING_ENV/bin/python" -B "$OPERATOR_HOME/launcher.py" inspect --plan "$REMAINING_PLAN"
"$EXISTING_ENV/bin/python" -B "$OPERATOR_HOME/launcher.py" status --plan "$REMAINING_PLAN"
```

The plan root must be new, with an existing persistent parent and no overlap
with protected storage, external source or evidence. It owns OWNER.json,
plan.json, allocation-logs/ and attempts/. A final plan is published only after
all checks. Partial namespaces remain evidence. `inspect` prints the exact
implemented sbatch argument list and shell preview; it never submits. The array
is **3–29%<the explicitly chosen concurrency>**. Do not run the pilot script or
modify pilot.json to obtain this selection. Freeze final external bytes before
preparing; a change in any of the six files rejects use of that plan.

Only after separate explicit submission approval and confirmation that this
same inspected evidence is still current, this implemented command submits once:

```bash
"$EXISTING_ENV/bin/python" -B "$OPERATOR_HOME/launcher.py" submit \
  --plan "$REMAINING_PLAN" --inspection "$INSPECTION_DIR/inspection.json" \
  --confirmed-current-inspection --execute
```

This writes exclusive `submission-intent.json` containing the exact sbatch argv,
plan seal, resource/evidence pins and time **before** calling sbatch. It writes
the response and acknowledged array ID separately. A second invocation cannot
replace the intent or blindly resubmit, even after a timeout/nonzero return.
The shell execs the owning Python PID; `B:USR1@300` uses native discard-epoch
StopRequest behavior and --no-requeue. Workers require the matching intent,
actual allocation/account/QOS/node, one assigned full H100 (not MIG), exact
source/software/manifest/activation/interpreter/operator bytes, thread environment,
torch intraop 8/inter-op 112 and CUBLAS :4096:8. They use the original bfloat16 /
FP32 / deterministic / TF32-off scientific settings. An external actual-array
claim resolves pre-ack worker starts to one array ID; each selected index has one
exclusive allocation claim in the plan. Raw native writers retain their leases.

If acknowledgement is lost, retain all intent/response/claim files and query
current scheduler/accounting. Do not infer absence from a client exception or
start another array. For an actual array ID found through exact evidence:

```bash
: "${ACTUAL_ARRAY_ID:?Exact ID proved by submission/accounting/worker evidence}"
: "${RECONCILIATION_REASON:?Record the exact supporting evidence paths/findings}"
squeue -j "$ACTUAL_ARRAY_ID"
sacct -j "$ACTUAL_ARRAY_ID" --format=JobID,JobName,Account,Partition,State,ExitCode,Elapsed,NodeList
"$EXISTING_ENV/bin/python" -B "$OPERATOR_HOME/launcher.py" record-job \
  --plan "$REMAINING_PLAN" --job-id "$ACTUAL_ARRAY_ID" --reason "$RECONCILIATION_REASON"
```

An existing acknowledged ID is immutable; do not call record-job again to replace
it. Contradictory worker/accounting IDs reject. If conclusive site evidence proves
no submission occurred, preserve that written reconciliation with the old intent;
only a separately approved new plan/root may authorize another attempt. No
automatic reconciliation or replacement controller is implemented here.

## Status, logs, interruption, and CPU publication recovery

```bash
"$EXISTING_ENV/bin/python" -B "$OPERATOR_HOME/launcher.py" status --plan "$REMAINING_PLAN"
ls -l "$OPERATION_ROOT/allocation-logs" "$OPERATION_ROOT/attempts"
: "${ACTUAL_ARRAY_ID:?Actual acknowledged or reconciled array ID}"
tail -n 80 "$OPERATION_ROOT/allocation-logs/${ACTUAL_ARRAY_ID}_3.out"
tail -n 80 "$OPERATION_ROOT/allocation-logs/${ACTUAL_ARRAY_ID}_3.err"
```

Native HEAD/journal coverage proves training lifecycle, not Slurm completion.
`auto` keeps completed runs unchanged, continues compatible clean interruptions
with original optimizer/schedule/data/RNG state, blocks terminal numerical and
execution failures, and performs full-budget finalization without another epoch.
Positive-alpha finite poor outcomes and failures are retained; no accuracy cutoff,
seed replacement, early stop, new hyperparameter or automatic retry is added.
An execution failure needs a separate audited decision; this v1 launcher has no
continue override. Further explicit allocations for clean interruptions require
prior array/accounting resolution and a new owned operational plan, never an
in-place pin rewrite or Slurm requeue. This minimal v1 plan fixes indices 3–29;
native auto skips any already-complete members if a new plan is later approved.

GPU `run` has no recovery selector. At verified native count 200, the existing
CPU controller handles publication repair separately with no StopRequest:

```bash
: "${RECOVERY_INDEX:?Exact candidate index 3 through 29 with verified count-200 authority}"
"$EXISTING_ENV/bin/python" -B -m alpha_sweep_training.controller run \
  --manifest "$STUDY/study.json" --global-index "$RECOVERY_INDEX" \
  --source-root "$NATIVE" --base-dir "$("$EXISTING_ENV/bin/python" -B -c \
    'import json,os; print(json.load(open(os.environ["REMAINING_PLAN"]))["native"]["base_dir"])')" \
  --operation recover
```

`recover` repairs committed publication views. If count 200 is committed and the
training completion transaction is still pending, the same native command with
`--operation auto` selects TrainingFinalizationPlan/publication only. These CPU
guarantees do not apply to unfinished-run recovery. Never pass StopRequest to
recover or load best over latest optimizer state.

## Complete all-30 report after quiescence — not generated here

The native pilot report intentionally remains three-member and the main-v1
240/6720 collector remains unchanged. The external report joins **all thirty**
stored tasks to validated historical native journal/receipts, preserving every
epoch/layer SD, final/best/last-committed metrics, selected epoch, raw provenance,
committed/orphan failure evidence and missing/locked/corrupt/unstable conditions.
It never takes a research writer lease or repairs aliases. Completion is separate
from health, whose assessment remains unreviewed without new thresholds.

All 840 logical evaluation observations are retained: unbound checkpoints/seeds
are null; bound results retain native slot coverage, result/failure ancestry,
orphan evidence, raw evaluator provenance and independent SWEEP_HEAD authority.
Missing slots never authorize retraining. Evaluation inspection does not change
validation selection or training-completion evidence. Complete three-seed means
use sample SD (ddof=1), paired differences preserve each seed, and best training
alpha is selected only with all ten complete candidates, retaining exact ties in
grid order. Official test outcomes are not used for recipe/validation selection.

After the actual new runs finish and writers are quiescent, identify the actual
candidate evaluation root (even if still unexecuted) and a new separate derived
output directory. Required variables are deliberately not guessed from the upload:

```bash
: "${CANDIDATE_EVALUATION_ROOT:?Actual separate candidate evaluation namespace}"
: "${NEW_ALL30_REPORT:?New derived directory outside all protected stores/source/plan/evidence}"
"$EXISTING_ENV/bin/python" -B "$OPERATOR_HOME/report.py" \
  --plan "$REMAINING_PLAN" --evaluation-root "$CANDIDATE_EVALUATION_ROOT" \
  --output-dir "$NEW_ALL30_REPORT"
```

The output OWNER.json and finite/checksummed report.json are derived, never
scientific completion authority. Each task is read stably; the full report is
not an atomic multi-run snapshot. Missing evaluation remains explicit pending
coverage rather than zero-valued scores. Later independent robustness uses the
original native evaluate CLI/28-slot protocol, retained best checkpoints and its
own leases/retries; this operator does not launch evaluation.

## Review and limits

Validation here consists of source, branch/worktree, CLI/parser and diff review;
the original native package/checkouts/pilot artifacts are unchanged. No compatible
local Python/Torch/TorchVision environment or HPC filesystem is available, so the
nine external CPU tests, actual source/manifest/checkpoint checks, resource
inspection, operational freezing, submission, training and report generation
remain **unrun**. The supplied pilot findings are evidence for proceeding with
the frozen study, not runtime certification of this external launcher or proof
that nonzero alphas will perform well. Review, explicit later verification and
submission authority remain necessary before using these commands.

The supplied validation.original.json raw SHA-256 was checked locally and matches
`b5caee184b745b9952f545b07fb8cfba8a24e0bd85b106b0cb48f1c01aec464e`.
The CSVs have 600 epoch rows and 38,400 layer rows, excluding headers. These are
file-consistency observations, not independent checkpoint or numerical replay.
