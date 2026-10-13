# OSU dgxh execution contract

Step 7 implements this contract in source, without execution or measurements.
The exact commands, operational schema, registry, signals and operator recovery
are in [slurm-workflow.md](slurm-workflow.md). No jobs, Slurm queries or preliminary
experiments were performed during implementation; runtime correctness is unverified.

Target OSU's `dgxh` partition. Run **one GPU and one training process** for each
independent dataset–architecture–alpha–training-seed combination. There are 120
CIFAR-10 runs followed by 120 CIFAR-100 runs, 240 total. No distributed/multi-GPU
recipe is implied. Parallel task execution must not change the scientific run.

| Operational setting | Proposed starting request |
| --- | --- |
| GPUs per task | 1 |
| CPUs per task | 8 |
| Host RAM per task | 32 GB |
| Wall time per allocation | 12 hours |
| Concurrent tasks across the workflow | Up to 4 |
| Advance warning | 600 seconds, discard by default |
| Account | Explicitly supplied for the actual allocation; unresolved |

These are planning defaults, not measured requirements or guaranteed account
entitlements. Account, CPU/RAM requests, wall time, and concurrency are
configured in separate version-1 `configs/workflow.yaml` and applied as typed
execution-only overrides; one GPU and `dgxh` express this main workflow's
fixed execution design. Do not claim currently available cluster/account limits
without verifying them during the future authorized operational phase.

Research runs may span multiple allocations. Store data/split artifacts,
resolved configuration, metadata, checkpoints, results, and logs on storage
accessible from subsequent nodes. Paths must not assume node-local scratch
persists between jobs. Allocation IDs, submission times, resource requests,
wall time, and storage locations do not alter the scientific fingerprint.
Record them in per-attempt provenance.

Each allocation consumes one manifest-selected task. Continuing an interrupted
run requires explicit compatible continuation and a durable latest checkpoint.
Maintain the original total budget of 200 epochs, warmup/cosine indexing,
optimizer/scaler, best-validation record, and RNG states. Do not reset a schedule
or treat the next allocation as a new research seed. After training, continue
independent evaluation from immutable best rather than retrain. Training command
count-200 continuation only reconciles training completion; robustness has its
own allocations/slot resumption and never restores training optimizer/RNG state.

Step 7 uses one persistent **workflow-wide** registry/controller and at most one
unresolved bounded wave across both datasets and lifecycles. Never independently
give each array four slots. Immutable array mappings and shared status ensure
CIFAR-10 is processed first, followed by CIFAR-100. Submission/requeue logic must
also cooperate with per-storage-root run locks, validate aggregate requested
GPU/CPU/memory-time against dated published and reviewed account/QOS limits,
and leave explicit interrupted/failure status when an allocation stops.

An allocation ending proves neither lifecycle complete. Training completion
requires the full budget, valid latest/best/final, clean validation selection,
epoch/weight statistics and durable training summary. Robustness completion
requires all 28 successful independent full-test result slots and sweep authority.
Numerical failure must remain visible to
controller and aggregation. See [the shared contracts](implementation-plan.md).

Step 5's [checkpoint/recovery APIs](checkpoints-and-recovery.md) are implemented
without execution. The Step 6 launcher/controller uses full continuation
preflight, read-only committed completion inspection and the active run lease.
The production allocation runner installs scoped StopRequest handlers. Bash execs
the owning Python PID, making B:USR1 delivery explicit; handlers perform no I/O,
logging, serialization, subprocess, CUDA or lock work. Stale locks require explicit confirmation of inactive owners
across nodes, never local PID/age guesses. Numerical failure is terminal, and
lost completion acknowledgement must be reconciled before any rerun/resubmission.
Step 6's [manifest/controller guide](manifests-and-controller.md) documents
`alpha-sweep` commands and `load_manifest`, `load_phase`, `select_task`,
`inspect_task`, `collect_coverage`, `dispatch`, `execute_task`, TaskOutcome/ExitCode.
Each task command handles one process/GPU; phase scheduling is a Step 7 responsibility.
`alpha-sweep evaluate` handles one checkpoint and one or a selected group of
evaluation slots; 6,720 logical slots do not imply 6,720 jobs. The training grid
stays 240 tasks/two 120-task phases. `evaluation-plan`, `evaluation-inspect` and
`evaluation-status` expose linked planning, eligibility and complete coverage.
See [weight-robustness-evaluation.md](weight-robustness-evaluation.md) for APIs.
Coverage includes every seed and a conservative phase-2 eligibility flag requiring
all phase-1 training tasks durably complete. Pending robustness slots do not make
a checkpoint untrained or force another training allocation. Settled terminal numerical failures remain
distinct and do not silently unlock the next phase. Automatic candidates exclude
execution/numerical failures and unsafe inspection conditions.

Use a fixed actually imported checkout and persistent base/storage across
allocations. The execution collector records actual dependencies/source and
allowlisted allocation environment without Slurm commands; the runtime records
actual assigned-device/backend details. Status probes no GPU and validates saved
authority using historical provenance. It is a read-only set of individual
observations, not an atomic grid snapshot. Retain unsuccessful CLI outcomes when
failure publication was unavailable. Exit codes and ended allocations alone cannot
prove completion. Step 7 implements render-only scripts, explicit guarded submission,
one-shot reconciliation/advancement and bounded clean-interruption continuation;
no daemon, automatic requeue or numerical-failure retry is implied. Account/resource
evidence, accounting and shared-filesystem behavior remain future operational checks.
OSU's [published dgxh policy](https://it.engineering.oregonstate.edu/hpc/slurm-howto),
consulted 2026-10-13, caps aggregate GPU-time at 48 GPU-hours; four one-GPU twelve-hour
requests consume that allowance. Unknown submissions/scheduler state retain the
entire wave's reservation. Training ordering is CIFAR-10 first; pending evaluation
does not block otherwise eligible CIFAR-100 training. See the operating guide.
