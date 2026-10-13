# OSU dgxh execution contract

This is a design for Step 7, not a submission implementation or a measurement of
resource requirements. No jobs or preliminary experiments are submitted during
the nine implementation steps.

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
| Account | Explicitly supplied for the actual allocation; unresolved |

These are planning defaults, not measured requirements or guaranteed account
entitlements. Account, CPU/RAM requests, wall time, and concurrency are
configurable in `execution.slurm`; one GPU and `dgxh` express this main workflow's
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
pending clean test evaluation from the best checkpoint rather than retrain.

Step 7 must coordinate a **workflow-wide** concurrency limit; independently
setting both dataset arrays to four tasks could accidentally launch eight.
Define the dataset ordering dependency and shared task status handling so
CIFAR-10 is processed first, followed by CIFAR-100. Submission/requeue logic must
also cooperate with per-storage-root run locks, handle actual allocation limits,
and leave explicit interrupted/failure status when an allocation stops.

The allocation ending does not mean scientific completion. Completion requires
the complete training budget, durable best/final checkpoints, clean evaluation
of each run's best model on the full official test set, epoch/weight statistics,
and a valid final result summary. Numerical failure must remain visible to
controller and aggregation. See [the shared contracts](implementation-plan.md).
