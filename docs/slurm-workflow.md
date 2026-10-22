# Production OSU dgxh workflow

Step 7 is implemented in source only. None of the commands below was executed.
During the original Step 7/9 work, no tests, project imports/execution, compilation/linting, environment installation,
downloads/preparation, models, research manifests, workflow artifacts, rendered
bundles, training/evaluation/results, experiments or Slurm/HPC commands/jobs were
created or run during implementation. Runtime correctness is unverified. All nine
steps are now complete in source; [operating-guide.md](operating-guide.md) orders
future operation and recovery. Stop at source handoff; execution needs a separate
explicit instruction.
The later queued-array update includes focused local operational unit checks;
these do not establish live Slurm or research runtime correctness. Its migration
and verification scope is documented in [queued-slurm-arrays.md](queued-slurm-arrays.md).

## Operational inputs and policy

`configs/workflow.yaml` is an operational version-2 declaration, separate from
RunConfig/scientific schema 4. `load_policy` strictly validates unknown fields,
types, paths, resources, signals, environment, command names, continuation budget
and reviewed restrictions. Planning does not choose unresolved science. The
existing full research manifest must already contain approved seeds, split/
normalization, batch handling, precision/backend and scheduler settings.
Version 1 remains readable without migration or changes to its sealed bytes.
The [queued-array guide](queued-slurm-arrays.md) describes the v2 contract,
focused unit checks, site-change preservation and explicit post-quiescence migration.

Supply an actual Slurm account, canonical absolute shared Linux paths for the
fixed checkout, full manifest, persistent base, data/split, run metadata/output,
checkpoints, results, evaluation, workflow registry and logs. No account is inferred
from an ONID username. Nodes=1, GPUs=1 and partition=dgxh are fixed. Account/QOS,
CPU/RAM, wall time, global cap, workers, threads, optional GPU feature constraint,
environment activation/modules and warning lead are operational. No individual
node or H100/H200 feature is selected by default. Operational directories cannot
overlap protected run directories or each other. Paths cannot contain control
characters, traversal or Slurm '%' filename expansions. `paths.split_file` stays
null to inherit each dataset's finalized manifest split; a non-null legacy global
override is rejected at operational readiness. Historical sealed policies still
validate as declarations. Relocation requires correctly paired data_root/split_file
in each separate research input before manifest build, preserving prepared YAML.

Defaults are unmeasured requests: eight CPUs, 32G memory, twelve hours, up to 40
queued tasks with at most four running across the workflow, and 600 seconds warning.
`resources.max_queued` controls wave size; `max_concurrent` controls the Slurm
running throttle. Forty tasks render `--array=0-39%4`. Four loader workers and
four native threads fit the CPU request; each worker uses one torch thread.
Native BLAS/OpenMP pools are bounded and torch interop uses one thread. This is
pool configuration, not a measured throughput or CPU-affinity claim. Operational
continuation defaults to twenty allocation intents per work identity, including
resolved failed submissions, and a 120-second bounded worker wait for a compatible
controller. These are configurable bounded policies, not extra training epochs.

The published OSU dgxh policy was consulted on **2026-10-13**: two-day maximum
wall time; eight GPUs, 224 CPUs and 2000 GB; combined allowances of 48 GPU-hours,
1536 CPU-hours and 12000 GB-hours. They are published bounds, not verified account
entitlements. Version-2 `resource_budget` checks simultaneous overlap at the
enforced throttle, including capacity reserved by pending/unknown work; four
one-GPU twelve-hour requests use all 48 running requested GPU-hours. Forty queued
tasks request 480 whole-wave GPU-hours, reported separately for cumulative quota
review. It also applies tighter reviewed account/QOS running and submission bounds.
See [OSU Slurm policy](https://it.engineering.oregonstate.edu/hpc/slurm-howto) and
[cluster description](https://it.engineering.oregonstate.edu/hpc/about-cluster).

`inspect-restrictions` is a future explicit, read-only Slurm operation. Its sealed
**restriction report version 2** preserves each query's identifier, command,
arguments, ordered field names, raw output, string-valued rows, blank-field locations
and error. It collects tool versions, `scontrol` partition/configuration, selected
account associations with inherited limits, all visible associations with
`WOPLimits` to expose where limits are set, and QOS records. Association `ID`,
`ParentID`, `ParentName`, cluster/account/user/partition and QOS/default-QOS columns
support scope and ancestor review; missing ancestor rows are collection errors.

The field contract uses documented Slurm **24.11** LIST/SHOW names, including
`MaxTRESPJ`, `MaxTRESPN`, `MaxTRESMinsPJ`, `GrpTRESRunMins`, `GrpJobs`,
`GrpSubmitJobs`, `MaxJobs` and `MaxSubmitJobs` on associations. QOS adds
`MaxTRESPU`, `MaxTRESPA`, `MaxTRESRunMinsPU/PA`, `MaxJobsPU/PA`,
`MaxSubmitJobsPU/PA`, group counts and applicable age-accrual counts. QOS `Flags`,
`LimitFactor`, `UsageFactor`, `UsageThreshold`, minimum TRES and preemption settings
are retained. The installed OSU version is **not verified now**; recorded client
versions and controller configuration must be reviewed for actual compatibility.
Unsupported fields or denied queries remain errors; there is no reduced-field
fallback that silently declares sufficient coverage. See the
[versioned Slurm manual](https://raw.githubusercontent.com/SchedMD/slurm/slurm-24-11-0-1/doc/man/man1/sacctmgr.1).

`GrpTRESMins` and `GrpWall` concern cumulative accounted usage, with enforcement,
decay/reset and QOS flags affecting interpretation. They are not interchangeable
with `GrpTRESRunMins` or `MaxTRESRunMinsPU/PA`, which constrain concurrent running
jobs' requested time. `MaxTRESMinsPJ` constrains a single job. Plain `GrpTRES` and
per-job/node/user/account TRES fields limit resource quantities. Do not translate
a cumulative quota into the workflow's overlap budget without reviewing usage and
enforcement. Blank strings remain explicit unresolved values; they are never
converted to zero or unlimited. Missing columns/rows, empty version/configuration
output, suspect display truncation, absent selected account/QOS or missing ancestor
evidence make `collection_complete` false even if commands exited successfully.

Collection completeness is not an entitlement decision. An operator must review
inherited association scopes, actual submitting user and cluster, selected/default
and partition QOS, flags/factors, per-job and simultaneous GPU/CPU/memory/time
limits, job/submit counts, existing usage and accounting configuration. Private or
site-specific evidence unavailable to these queries needs administrator review;
there is no automatic entitlement guess.

Fill `effective_limits` with finite positive tighter bounds (`max_wall_hours`,
`max_gpus`, `max_cpus`, `max_memory_gb`, `gpu_hours`, `cpu_hours`,
`memory_gb_hours`, `max_allocations`) plus `inspection_report_path` and the actual
file SHA-256. In v2, `max_allocations` bounds running jobs. Additionally supply
reviewed `max_submitted_tasks` (all pending plus running tasks), `max_array_size`
and `max_array_tasks`. Queued task count is checked against all three. Submission
also requires complete recorded MaxArraySize/SchedulerParameters evidence and
checks the reviewed array caps against it, using the documented max_array_tasks
default only when complete parameter evidence omits that option. Missing evidence
is an error, not unlimited capacity. Fractional time/memory bounds are supported. Submission verifies the
report digest, version-2 contract, exact query/field coverage, raw-output linkage,
recomputed collection issues and account/QOS/partition linkage. Version-1 reports
and unknown report versions cannot authorize new submissions: recollect version-2
evidence, review it and update the report digest/bounds explicitly (using policy
adoption after quiescence where a registry already exists). No automatic evidence
migration or workflow reset occurs. Missing
account/environment/paths/evidence produces actionable future errors. Reports do
not prove driver, hardware, filesystem or package capability. Actual Slurm policy
still controls admission, including unrelated jobs this registry cannot govern.
See [Slurm resource-limit rules](https://slurm.schedmd.com/resource_limits.html).

## Work and scheduling

`build_work_plan` validates the full parent manifest and actual execution source/
software. It reuses stored training coverage, checkpoint binding and evaluation
slot/sweep coverage. Training remains 240 tasks/two 120-task dataset views, with
original indices, identities and 200 epochs. Phase-2 training computation requires verified
successful completion of **all 120** phase-1 training tasks. Terminal failures
never satisfy that gate. Pending robustness never blocks CIFAR-100 training.
Publication-only finalization of already verified count-200 authority is exempt
from the computation gate in planner, runner and manual executor; it adds no epochs.

One training work item is one exact manifest task. One evaluation work item is
one verified immutable best checkpoint/protocol, processing all prescribed
unfinished slots sequentially through `execute_evaluation`. There remain 28
logical slots/checkpoint and 6,720 total, not 6,720 obligatory jobs. Training and
evaluation are separate allocations. A wave also excludes simultaneous training/
finalization and evaluation of the same training run. Evaluation eligibility
includes existing valid count-200 authority with training publication pending;
there is no retraining to obtain missing results.

Default planning order is eligible training in original global order, then
eligible evaluation in original global order. Rendering greedily selects at most
the queued-wave cap (40 by default); explicit `--work-id` selection can choose another eligible bounded
group. This allows evaluation of trained checkpoints while later training is
pending, subject to the same authority and cap. Only one unresolved wave may
exist, across both datasets and both lifecycles. No new wave is submitted until
every prior allocation is conclusively terminal and its allocation lease is gone.
Partial-wave completion never makes a free submission slot. Pending, configuring,
running, suspended, completing, requeued and unfamiliar/uncertain states retain
the wave's full queued reservation and possible running throttle. Slurm fills free
running slots from that wave automatically. No new wave is allowed merely because
some elements have finished. If later accounting loss/outside requeue makes multiple old
waves uncertain, the registry preserves every uncertainty and blocks all new waves;
it does not discard observations to make the state fit its normal one-wave policy.
The normal policy sacrifices throughput for auditable continuation and shared
concurrency. It controls cooperating managed
commands only; outside submissions/requeues remain outside its authority.

Every work item seals lifecycle, run/fingerprint, original global/phase indices,
exact selector, operation/retry decision, allocation ordinal and initial training
transaction. Evaluation additionally carries the actual best generation/SHA,
protocol identity and full binding. No placeholder digest is created for untrained
tasks. Plans and bundles seal full manifest grid/document digest, policy snapshot/
fingerprint and actual planned source/software. The deterministic execution pin
contains the imported-source inventory/digest, software versions and version-1
external `launch_inputs` (activation script bytes, configured/resolved interpreter
path/bytes and declared modules); process IDs,
timestamps and allocation metadata belong to actual execution provenance. Workers reject unknown/altered
membership, source/software, policies, selectors and checkpoint bindings. Seeds,
training fingerprints and scientific state never include operational fields.
Source inventory v2 includes the checked-in workflow declaration. Existing work
items/registry/outcomes stay version 1; v2 policies produce plan/bundle version 2
and separate queued/running budget and reservation fields. Legacy v1 plans/bundles
retain their original policy-specific shape/semantics. Historical pins without `launch_inputs` remain readable
for ownership/reconciliation but cannot render, submit or execute under this source.
New work must be replanned after any uncertain old intent is resolved; existing
bundle bytes/registry history are never rewritten. This source change cannot
migrate unfinished training or incompatible same-protocol evaluator slots.

## Durable workflow authority

Use exactly one configured shared registry for the experiment. Do not evade it by
choosing another root. All managed submission entry points call `submit_bundle`
under the concrete `WorkflowLease`; the Slurm submission backend also requires
that live authority and exact durable intent. A registry cannot police independent
scripts or other users' jobs. Its version-1 owner links the full manifest. Current
policy is in sealed state snapshots and changes only through explicit `adopt-policy`
after all allocations are conclusively terminal and leases gone. Changing policy
does not relax unfinished-training source/software checks or reset attempt counts.
Adoption requires `--previous-policy` containing the exact currently sealed policy
and `--policy` containing the reviewed corrected policy at the same registry root.
Original fingerprint equality is checked before writes and under the controller;
per-wave scheduler evidence uses original sealed policy values.

Existing-registry inspection/reconciliation/recovery uses
`require_recovery_ready`, separate from research `require_ready`. A historical
non-null global split does not block workflow-only `status`, `reconcile`,
`resolve-submission`, `resolve-terminal` or exact lock inspection/recovery. These
paths validate existing ownership/seals and the unchanged selected policy; they
cannot initialize replacement authority, reserve work or submit through a recovery
lease. `status --workflow-only` skips scientific coverage and requires no research
environment readiness. Plain status under a non-null global split does the same,
returning scientific=null and an explicit coverage-not-collected diagnostic. New
planning/rendering/submission/allocation execution still reject that split policy.
Run-lock recovery projects only original run-storage paths, never global data/split
overrides. See the [historical-policy recovery sequence](operating-guide.md#historical-policy-inspection-recovery-and-subsequent-adoption).

```text
workflow_root/
  owner.json
  .controller.lock
  HEAD.json                       # digest-referenced current/previous snapshots
  states/<generation>.json        # immutable policy, waves, claims, observations
  bundles/<wave>/mapping.json     # immutable index -> exact work item
  bundles/<wave>/batch.sh         # digest-checked actual launch source
  allocations/<wave>/<index>.lock
  outcomes/<wave>/<index>/<claim>.json
  diagnostics/<wave>/<index>/<attempt>.json
  resolutions/<generation>.json
```

Atomic hard-link creation publishes complete controller/allocation lock owners.
Controllers fail on contention; allocation workers may wait for a compatible
controller up to their explicit bound. No age/PID inference, lock stealing or
unbounded wait occurs. File/directory fsync and same-filesystem atomic replacement/
hard-link support are future shared-filesystem requirements, not verified facts.

`submit_bundle` refreshes scheduler observations and scientific eligibility under
the shared controller lock. It verifies immutable sources, resource overlap and
actual restriction evidence, creates Slurm output/error directories **before**
sbatch, then durably records `submission_uncertain` intent and exact argument
vector. Only then does it call `sbatch --parsable`. Returned array ID/cluster is
committed. Any error/timeout/crash before acknowledgement retains uncertainty;
empty queue output never proves non-submission. The token is an exact Slurm
comment, not a loose job-name match. No shell interpolates subprocess arguments.

Use `resolve-submission --job-id` with an operator-obtained exact array ID; it
checks `scontrol` comment/account/partition/array linkage. A valid allocation
runner may also establish the exact recorded array identity after lost submission
acknowledgement, under the same registry and mapping checks. Explicit documented
non-submission resolution requires administrator/operator evidence, confirmation
that no allocation was submitted, no existing worker claims and an audit reason.
It preserves the old intent and cannot silently submit it again. No cancellation
or requeue operation is implemented. See [sbatch](https://slurm.schedmd.com/sbatch.html).

`reconcile` queries `squeue --array` and `sacct --allocations --duplicates` using
recorded exact IDs and cluster. Accounting must retain submission comments and
individual array records. Only absent queue entries plus one complete, matching,
terminal accounting record for every expected index prove a wave terminal.
Accounting delays, missing comments, duplicate requeue records, unfamiliar states,
lost identities and query failures remain uncertain. Past waves are rechecked,
so accounting retention/configuration must support the experiment's lifetime;
missing historical proof blocks advancement until authoritative operator recovery.
`resolve-terminal` accepts explicit actual administrator/accounting proof for
every exact array index and confirmation that all owners are inactive, plus a
reason. It refuses any currently visible queue allocation or residual allocation
lease. This preserves an immutable terminal resolution when accounting history
is unavailable; it never infers terminal state from absence, age or timeout.
It releases scheduling reservation only, never marks science completed or repairs
missing/failed outcomes. Later outside requeues are outside that explicit proof
and this workflow's supported operations.
The published status is a bounded cooperative observation, not an atomic cluster
snapshot. See [squeue](https://slurm.schedmd.com/squeue.html) and
[sacct](https://slurm.schedmd.com/sacct.html).

## Batch runner and signals

`batch_source` is the production Bash template. It uses an existing explicit
activation script/modules and absolute Python executable, a fixed checkout and
PYTHONPATH, bounded thread variables and original Slurm-assigned GPU visibility.
Before activation, it verifies pinned launch bytes and configured interpreter
target using GNU sha256sum/readlink/dirname, rejecting changed or linked activation
inputs explicitly. These utilities are future node prerequisites. Interpreter
symlinks (for venvs) are target-pinned; runtime also revalidates pins before claim.
Module definitions/transitive activation files and environment must be frozen by
the operator; module names alone cannot verify them. It never clones/pulls,
installs, downloads or prepares data. Existing Python
3.10-3.12, torch 2.5.1, torchvision 0.20.1 and NumPy 1.26.4 requirements remain.
Choose the compatible CUDA build/driver and environment during later approved
setup; this step supplies no new wheel/version/precision default.

The signal path is concrete: `sbatch --signal=B:USR1@600` targets the batch PID;
after environment setup Bash **execs** the owning Python process at the same PID.
There is no background/detached research child or srun forwarding assumption.
Python installs handlers before heavy component imports, then applies the approved
warning policy. USR1 defaults to discard_epoch; TERM/INT always request discard.
Handlers only update `StopRequest`. They do no logging, I/O, subprocess/scheduler,
CUDA, checkpoint or lock operation. Prior handlers are restored on exit. DataLoader
worker initialization ignores USR1/INT and keeps normal child TERM behavior; no
worker inherits ownership of the parent's warning/stop request. Slurm's B warning
does not signal those workers. See [Slurm signal semantics](https://slurm.schedmd.com/sbatch.html)
and [Python signal handlers](https://docs.python.org/3/library/signal.html).

`run_allocation` validates registered mapping/script, actual account/array/one-node/
one-task/CPU/GPU identity and configured interpreter. It rejects external restart
counts and duplicate work claims. It holds one independent allocation lease, not
the workflow controller during research. It repeats fixed source/software and
phase/boundary checks, then delegates exactly once to `execute_task` or
`execute_evaluation(expected_binding=...)`. Assigned device is process-local
cuda:0; no physical GPU index is selected. The actual allocation environment and
task outcomes remain separate from scientific identity. Each immutable structured
outcome is published independently of Slurm stdout and joined by reconciliation,
even when controller acknowledgement or later cleanup failed. Registered runner
startup/cleanup failures retain separate sealed diagnostic files and require
explicit review, never safe-interruption inference. Invalid/unregistered mappings
cannot authorize those writes. Shell/environment/dependency failure before the
Python runner reaches valid registry ownership can leave a missing outcome; Slurm
stderr/accounting must then be inspected and no automatic replacement is allowed.

Stop requests prevent further work after the current documented boundary. Training
replays the unfinished epoch from committed latest; evaluation discards partial
aggregates and restarts the same entire realization/seed. Optional warning
finish_epoch finishes the current epoch or full realization, with no guarantee
that remaining wall time suffices. Signals can wait for Python/CUDA work to return.
SIGKILL, node loss, shell/environment startup failure, child TERM, filesystem
failure or an insufficient warning margin can prevent clean publication and leave
locks. Those cases require authoritative inspection and explicit recovery, not
an automatic timeout/node-failure retry. No mid-batch checkpoint, shortened budget,
partial success metric or schedule reset is introduced.

## Continuation and explicit recovery

`advance` is one-shot: reconcile allocations, plan eligible work and optionally
render/submit a bounded wave only with `--submit`. No daemon or research process
is required on a submit node. Completed scientific work is excluded. Count-200
training uses existing CPU publication-only finalization, with zero more epochs.
All-success raw evaluation with missing sweep publication uses the existing
completed-slot skip and sweep API, never repeats the 28 passes.

Automatic replacement requires an explicit recorded clean interruption and
compatible durable authority with no unsafe locks/failure/submission ambiguity.
Not-started stop requests can retry only when ownership remains missing or safe.
Numerical failures are terminal in the managed workflow. Execution failures,
unclassified/missing allocation outcomes and adoption of existing unfinished
work need `--retry-run-id` plus an audit reason; science/compatibility/locks and
attempt bounds still apply. Numeric failure cannot be waived through this flag.
Each reserved intent consumes the per-unit bounded allocation ordinal. Exhaustion
is reported explicitly; no infinite resubmission or resetting epoch/LR/seed occurs.

`SweepFinalizationDeferred` remains a deferral, not an execution failure solely
from contention. While a sweep/slot owner remains, planning reports it and schedules
no overlapping replacement. Once it exits, a verified completed sweep skips, or
the missing marker can finalize in a later bounded allocation with no repeated
pass. Genuine integrity/I/O failures retain their explicit classification.

`inspect-locks` returns exact paths/digests/owners for workflow/allocation, training,
or a selected verified evaluation checkpoint namespace. `release-locks` requires
that complete unchanged report, explicit confirmation that **all** recorded
owners/allocations across nodes are inactive and an audit reason. Training scope
delegates to Step 5's existing recovery API. Other scopes use the same evidence,
audit-before-rename and preserved archive policy. PID/age alone never authorizes
release. Damaged/unowned storage remains manual evidence; partial initial state
is not reconstructed from promising staging files. After recovery, run explicit
scientific publication repair and workflow reconciliation as appropriate.

## Future commands and Step 8 interfaces

All commands are for later authorized operation with a finalized full manifest and
the approved operational policy. Shell variables denote real operator-supplied
shared paths/account selections, never scientific defaults or reduced examples.

```bash
alpha-sweep-workflow inspect-restrictions --policy "$POLICY" > "$RESTRICTION_REPORT"
# Review actual restrictions and fill effective_limits/report digest before submit.
alpha-sweep-workflow plan --manifest "$FULL_MANIFEST" --policy "$POLICY" > "$WORK_PLAN"
alpha-sweep-workflow render --manifest "$FULL_MANIFEST" --policy "$POLICY" --plan "$WORK_PLAN"
alpha-sweep-workflow submit --manifest "$FULL_MANIFEST" --policy "$POLICY" --bundle "$MAPPING" --execute
alpha-sweep-workflow status --manifest "$FULL_MANIFEST" --policy "$POLICY" --scheduler
alpha-sweep-workflow reconcile --manifest "$FULL_MANIFEST" --policy "$POLICY"
alpha-sweep-workflow advance --manifest "$FULL_MANIFEST" --policy "$POLICY"
alpha-sweep-workflow advance --manifest "$FULL_MANIFEST" --policy "$POLICY" --submit
alpha-sweep-workflow advance --manifest "$FULL_MANIFEST" --policy "$POLICY" \
  --retry-run-id "$EXACT_RUN_ID" --retry-reason "$REVIEW_REASON" --submit
alpha-sweep-workflow resolve-submission --manifest "$FULL_MANIFEST" --policy "$POLICY" \
  --wave-id "$EXACT_WAVE_ID" --job-id "$EXACT_ARRAY_JOB_ID"
alpha-sweep-workflow resolve-terminal --manifest "$FULL_MANIFEST" --policy "$POLICY" \
  --wave-id "$EXACT_WAVE_ID" --evidence "$ACTUAL_TERMINAL_PROOF" \
  --confirmed-terminal-inactive-owners --reason "$TERMINAL_PROOF_REASON"
alpha-sweep-workflow inspect-locks --manifest "$FULL_MANIFEST" --policy "$POLICY" \
  --scope training --run-id "$EXACT_RUN_ID" > "$LOCK_EVIDENCE"
alpha-sweep-workflow release-locks --manifest "$FULL_MANIFEST" --policy "$POLICY" \
  --evidence "$LOCK_EVIDENCE" --confirmed-inactive-owners --reason "$INACTIVITY_PROOF_REASON"
alpha-sweep-workflow adopt-policy --manifest "$FULL_MANIFEST" --policy "$REVIEWED_NEW_POLICY" \
  --previous-policy "$ORIGINAL_POLICY" --reason "$POLICY_REASON"
```

Plan/render never call sbatch. Status adds scheduler queries only with --scheduler;
`status --workflow-only` is the explicit historical registry-only reader.
reconcile/advance explicitly query and publish workflow observations. Exit 0 means
the operation/report succeeded, not scientific completion. Allocation exits retain
0/10/20/21/22 for selected work success/interruption/numerical/execution/rejection.
Submission uncertainty/controller/configuration errors exit 22 and reserve capacity.
No supported submission command cancels, steals, or automatically requeues jobs.

Step 8/final integration consume `OperationalPolicy`, `resource_budget`, sealed
version-1/2 work plans and wave index mappings, version-1 `WorkflowLease` registry snapshots,
exact scheduler observations, version-1 allocation outcomes (mapping/claim/scheduler
IDs, task outcome, exit, safe interruption, stop policy, actual provenance and
diagnostic), plus the existing full training/evaluation scientific coverage and
individual realization schemas. Workflow statuses describe allocations; only
training HEAD and evaluation slot/SWEEP_HEAD authorities establish science.
Aggregation, mean/std, test-driven selection and plotting are not implemented here.

Step 8 now supplies [reporting/figures](results-and-figures.md) separately.
`alpha-sweep-report` never queries Slurm or mutates the workflow. Declare workflow
and additional operational roots so derived outputs cannot overlap them. Reporting
validates historical journal/slot evidence independently of allocation success,
with all expected rows and explicit deferred/failure coverage. Test-driven selection
remains prohibited.
