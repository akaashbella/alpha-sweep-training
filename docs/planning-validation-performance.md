# Pre-submission validation performance

This correction targets repeated validation during `plan`, `submit` and
`advance --submit`. It retains 40 queued tasks and the running throttle of four;
policy versions, resource checks, mappings and registry formats do not change.
No policy migration is required for this correction itself.

## Authority and freshness

Each planning operation owns one in-memory `PlanningInspection`. `advance` shares
it with its subsequent submission preflight; separate commands start empty.
The first observation still performs restricted checkpoint loading, content hashes,
current/previous journal validation, epoch/layer validation, immutable completion
CSV byte comparison, prepared-data/ownership checks and continuation compatibility.
Validated observations and compact journal metadata can then be reused. The cache
retains neither tensors nor a persistent index.

Reuse requires a fresh recursive snapshot of metadata, checkpoint, result and log
roots and their sibling writer locks. Snapshots include every descendant (including
orphan failure evidence), directory membership, type, device/inode, size, nanosecond
modification/change times and ancestor identities. HEADs, config, run/namespace
owners and lock files are also content-hashed on every snapshot. Replacements,
deletion, new evidence, changed owner/config/HEAD or a new lock invalidate reuse.
Linked paths retain existing rejection/diagnostic rules; snapshots never follow
child links or release locks.

POSIX immutable artifacts use change-time metadata to avoid rereading checkpoint
and CSV contents on every reuse. This requires coherent filesystem metadata; it
is not a reader lease or atomic grid snapshot. Privileged timestamp forgery or
broken filesystem coherence is outside that observation contract. Windows hashes
all regular files conservatively because its creation-time field cannot serve as
POSIX change time. Windows therefore saves reconstruction/deserialization CPU work
while retaining substantial read overhead. Every new command starts with actual
content validation.

Within one journal read, identical checkpoints shared by current/previous
transactions are validated once, with file freshness checks before reuse. Temporary
tensors are released before retaining compact planning evidence. `inspect_task`
no longer calls `inspect_completion` after the journal has already validated
completion. Planning evaluation bindings use already validated best metadata;
actual evaluation still loads the full immutable best payload.

Historical and actual-source eligibility remain distinct. Keys include effective
configuration, prepared identities and relevant source/software compatibility.
Inspection timestamps do not affect compatibility; complete supplied provenance
is still validated, and persisted provenance is never rewritten. Evaluation
coverage has its own binding/provenance/root key and snapshots its namespace,
locks, raw slots and sweep authority.

Submission still reconciles scheduler/registry ownership, rebuilds eligibility
against current state, checks selected sealed work and launch/source/software pins,
verifies restriction evidence and compares registered mapping/script content.
Immediately before durable reservation, it freshly checks **all** retained evidence,
including CIFAR-10 observations supporting the CIFAR-100 gate. A change rejects the
attempt before intent or sbatch; replan after writers exit. Mutation during
validation also rejects reuse. Completion never derives from existence, aliases
or scheduler outcomes. Independent observations do not lock the scientific grid.

Progress goes only to stderr: reconciliation, training task index/run ID,
immutable completion CSV verification, evaluation eligibility, reuse counts,
final freshness guard and durable-intent/submission boundary. JSON stdout remains
machine-readable. Counts describe cache operations across observation/journal/
coverage categories, not completed research tasks.

## Focused verification and measured limits

The user explicitly authorized these isolated tests and synthetic benchmark after
the source-only implementation phase. No model, torch, dataset, training,
evaluation, checkpoint creation, scheduler command or HPC execution was involved.
With `src` on `PYTHONPATH`:

```text
python -B -m unittest discover -s tests -p test_planning_inspection.py -v
python -B benchmarks/planning_validation.py --tasks 4 --passes 3
```

Tests cover exact legacy/new UTF-8 CSV bytes and serialization counts; quoted,
newline, Unicode and null fields; headers; context changes; nonfinite rejection;
operation isolation; production observation-cache wiring; source/software keys;
malformed provenance; changed/deleted/replaced artifacts; same-size edits with
restored mtime; new HEAD/owner/config/failure evidence; locks; mutation during
validation; final freshness rejection; and stderr isolation. The deep scientific
reader is mocked only at the observation wiring boundary. These tests do not
certify checkpoint formats or real submission. The symlink case is skipped if
Windows denies symlink creation.

Final local discovery (`python -B -m unittest discover -s tests -v`) ran 25 cases:
24 passed and the real-symlink case was skipped for missing Windows privileges.
This includes the 12 unchanged queue-policy/legacy-wave regression checks. The
mocked linked-ancestor traversal check passed independently of those privileges.

Measured on Windows 11, Python 3.13.12 (standard-library verification only;
supported research Python remains 3.10–3.12):

| Synthetic workload | Before | After | Result |
| --- | ---: | ---: | --- |
| CSV: 200 epochs, 64 layers/epoch, 96 provenance inventory entries | 3.233 s | 1.776 s | 1.82× faster |
| Identity serializations for that CSV | 12,800 | 200 | Once per context/render |
| Four runs, three repeated passes, three 2 MiB binaries and 200 JSON records/run, including full CSV reconstruction and final guard | 38.032 s | 8.881 s | 4.28× faster; equivalent observations |
| Deep validations / CSV reconstructions | 12 / 12 | 4 / 4 | Eight unchanged reads reused |
| Deep-validator read bytes | 75,656,952 | 25,218,984 | Excludes snapshot hashing/physical I/O |
| Same filesystem workload without CSV reconstruction | 0.704 s | 1.954 s | Slower on Windows; conservative hashing dominates |

Both CSV renderers produced exactly 159,151,994 bytes with SHA-256
`2699f64d808e859748005ec8bc1bb00aa5073a3f7c82bb5f695ef3979094d101`.
The benchmark preserves the old per-row renderer as its equivalence oracle and
compares repeated deep reads against reuse plus the final guard. It removes its
temporary synthetic files and writes nothing to scientific namespaces.

These are single local wall-clock measurements, not HPC throughput estimates.
Read counters measure the synthetic validator, not `/proc`, OS cache misses or
storage traffic. The synthetic reader hashes/parses bytes and renders CSV; it
does not deserialize PyTorch checkpoints or exercise live submission, registry
recovery, all 240 tasks or concurrent cross-node writers. POSIX metadata latency,
large checkpoints, shared-filesystem coherence and end-to-end HPC improvement
remain unmeasured. Cold scans still deeply validate all selected authority.

## Safe HPC transfer

Transfer source through a separate staged checkout/three-way merge, preserving the
active checkout, policies, registry history, bundles, results and raw checkpoints.
This patch does **not** edit `work_render.py` or `slurm_client.py`. Retain and review
HPC email notifications, full-queue squeue with exact job-ID filtering, and blank
sacct Comment identity fallback using exact recorded SubmitLine arguments. Do not
replace those files wholesale with workstation copies. Private HPC patches were
not available here; their merged behavior remains unverified.

The [existing transfer and quiescence procedure](queued-slurm-arrays.md) still
applies. This source update changes the inventory digest. Unfinished training
requires its original frozen source/software: keep that checkout for compatible
continuation and finish it before changing execution source. Never rewrite
checkpoint provenance or bypass compatibility for a performance fix. Completed
historical runs remain inspectable. Do not rewrite old bundles; fresh work needs
actual launch/source pins. Any later checkout-path policy change uses explicit
quiescent adoption. No transfer, active HPC change, Slurm query, submission or
scientific operation was performed.
