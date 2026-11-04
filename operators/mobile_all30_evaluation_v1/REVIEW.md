# Source review and evidence boundary

Revised November 4, 2026 on Windows. Worktree:
`C:\Users\akaas\RESEARCH\alpha-sweep-training\.handoff-worktrees\mobile-all30-evaluation`;
branch `ops/mobile-all30-evaluation-v1`, HEAD
`0dafef07d0b379cf8c2f47eb6868db4319aaf8fe`. The prior reporting correction and
CPU checks were committed/pushed in the previous authorized task. This queue-scoping revision authorizes
CPU tests, test-related imports and syntax checks, and remains uncommitted and
unpushed. It authorizes no HPC operations, installs or research execution.

The native commit named by the ZIP was absent from local Git during initial
preparation; relevant APIs were read through GitHub at that exact pinned commit.
Frozen native commit `8010d07d9d1f0886ab1e180523f316623ef622a4`, scientific
settings, manifests, checkpoints, results, historical operators and HPC plans
remain untouched. Tests read existing local StopRequest/signal helper source
under mocked native imports and mocked signal registration, never native compute.

## Evidence read

The complete supplied ZIP was copied under
`handoff-evidence/mobile-all30-review/`. Its 27 declared file hashes all matched
`CHECKSUMS.json` using read-only .NET ZIP streams and SHA-256. The checksum manifest
has no self-checksum; matching it does not independently authenticate the author.
The original ZIP and its contents were preserved. The nested tar's member names,
review Markdown, verification metadata, CSV schemas/sample rows, analysis source,
pilot plan, historical operational plan and exported full-binding structure were
inspected. The supplied analysis was **not** executed. All 6,000 epoch and
384,000 tensor-row claims remain the supplied review's results; this session did
not independently recompute those analyses or inspect HPC tensor/dataset bytes.

The supplied review records 30 full-budget training runs and 0/840 successful
evaluation slots at export time. It recommends retaining every run, including
high-alpha poor scores, and evaluating every validation-selected best checkpoint.
It reports that training alpha zero has the highest mean best validation
accuracy; alpha .05 is approximately .02 percentage points lower. These are
historical validation observations, not new test results or evidence of current
HPC state. Transient instability remains visible; weight spread is not flatness.

Pinned native interfaces read:

- [Evaluation dispatch](https://github.com/akaashbella/alpha-sweep-training/blob/8010d07d9d1f0886ab1e180523f316623ef622a4/src/alpha_sweep_training/evaluation_execution.py): direct API, full `expected_binding`, native outcomes and defaults.
- [Checkpoint binding](https://github.com/akaashbella/alpha-sweep-training/blob/8010d07d9d1f0886ab1e180523f316623ef622a4/src/alpha_sweep_training/evaluation_inputs.py): actual best/full-budget CPU authority; metadata-only reporting.
- [Manifest reader](https://github.com/akaashbella/alpha-sweep-training/blob/8010d07d9d1f0886ab1e180523f316623ef622a4/src/alpha_sweep_training/manifests.py): separate 30-task profile and exact order.
- [Signal API](https://github.com/akaashbella/alpha-sweep-training/blob/8010d07d9d1f0886ab1e180523f316623ef622a4/src/alpha_sweep_training/allocation_signals.py): owning PID and native stop policies.
- [Coverage](https://github.com/akaashbella/alpha-sweep-training/blob/8010d07d9d1f0886ab1e180523f316623ef622a4/src/alpha_sweep_training/evaluation_coverage.py), [storage/reader](https://github.com/akaashbella/alpha-sweep-training/blob/8010d07d9d1f0886ab1e180523f316623ef622a4/src/alpha_sweep_training/evaluation_persistence.py): slot/sweep authority and raw record ancestry.
- [Provenance](https://github.com/akaashbella/alpha-sweep-training/blob/8010d07d9d1f0886ab1e180523f316623ef622a4/src/alpha_sweep_training/execution_provenance.py), [pilot reader](https://github.com/akaashbella/alpha-sweep-training/blob/8010d07d9d1f0886ab1e180523f316623ef622a4/src/alpha_sweep_training/mobile_candidate.py), [CLI](https://github.com/akaashbella/alpha-sweep-training/blob/8010d07d9d1f0886ab1e180523f316623ef622a4/src/alpha_sweep_training/controller.py): source inventory, configured/resolved interpreter/activation, base directory and actual options.

Historical external
[launcher](https://github.com/akaashbella/alpha-sweep-training/blob/daa83e8959d72b6d11fe3954f9ef3d01fef62316/operators/mobile_remaining_v1/launcher.py)
and [report](https://github.com/akaashbella/alpha-sweep-training/blob/daa83e8959d72b6d11fe3954f9ef3d01fef62316/operators/mobile_remaining_v1/report.py)
were inspected. This bundle follows useful exclusive publication and historical
reader patterns, but imports neither file. The old six-file ownership/inventory
is not used to authorize new code. Local existing native source and operating/
evaluation contracts were also read. No scientific constants were changed.

## Implementation review

The runtime inventory remains exactly the original six files, listed in
`HPC_HANDOFF.md`. This revision changes only:

- `operators/mobile_all30_evaluation_v1/launcher.py`
- `operators/mobile_all30_evaluation_v1/HPC_HANDOFF.md`
- `operators/mobile_all30_evaluation_v1/REVIEW.md`
- `operators/mobile_all30_evaluation_checks/test_operator.py`

No new files. CPU support remains outside the six-file runtime inventory.
`run_cpu_checks.py`, `batch.sh`, `report.py`, `figures.py` and all native source
retain their bytes. Existing
untracked `handoff-evidence/` remains untouched; no new scientific artifacts exist.
New operational output namespaces have exclusive ownership, per-file/path pins,
full best bindings, finite sealed JSON and POSIX durable exclusive publication.
Missing summary/status aliases do not authorize replacement. Staging and
ambiguous evidence survive failures. No stale lock is inferred from PID/age.

The wrapper's only computation dispatch is native `execute_evaluation`. It
constructs no training components. All-slot selectors are `None`; the full
inspected binding is passed unchanged. Native StopRequest/signal contexts and
native exit codes remain visible. GPU properties and effective native runtime
are compared before dispatch. Evaluation and derived reporting use distinct
namespaces; reporting never acquires training/evaluation writer leases or repairs
scientific storage. Operational diagnostic files are not completion authority.

At most one unresolved submission wave exists in the fixed study operations
root. Intents precede `sbatch`, acknowledgments are separate, workers independently
claim exact index/array identities, and only explicit audited commands can submit
another settled wave or retry failed slots. The project reservation agreement
and fresh current evidence cover nonparticipating legacy stages; the operator
cannot force unrelated launchers to honor its lock. Queue discovery now checks
only the pinned submitting user's jobs across accounts, while manual reservations
cover project work beyond that scope. Scheduler settlement never proves scientific success.

All 840 expected observations remain in reporting, including unbound reads and
null metrics, native failures, orphan evidence and histories. Summary means require
complete prescribed realization/seed sets. Individual passes remain prominent;
within-checkpoint realization SD and between-training-seed SD are separate.
Plot generators are source only; no research report/figure was generated.

## Reporting correction

The old collector held slot evidence in a temporary list until all 28 reads and
final stability checks succeeded. Any later error replaced those observations
with empty unbound rows. The revised collector appends evidence in place and
records slot-section errors while continuing independent inspections. Both
initial and returned final native coverage snapshots remain visible, even when
they disagree. Actual bindings are retained before plan comparison.

The report keeps all 30 training/840 evaluation rows, full raw metrics and
references, ancestry, records, attempts, orphan failures and unparsed evidence.
Separate flags gate authority: `binding_verified`, `observation_verified`, and
`measurement_verified`, plus checkpoint training/evaluation snapshot flags.
Missing, unreadable, incompatible and unstable inspections are explicit states;
structured errors retain their phase and diagnostics.

A local read/validation error invalidates that slot only when checkpoint-level
stability is otherwise verified. A checkpoint-wide failure invalidates all 28
measurement/observation flags without removing evidence. Training instability
also invalidates validation selection. Native sweep publication alone cannot
produce a report completion claim after an inspection failure; all 28 verified
measurements must also be present. A parsed record's `validated` flag is not a
replacement for those broader checks.

Aggregation excludes unverified rows from means, SDs, paired baselines and
validation selection. Individual-point summary entries mask invalid metrics,
while full raw values stay in evaluation rows and the flagged raw CSV. Plotting
recomputes summaries from explicit raw verification flags, ignores cached
measurements, and checks individual-point flags. Old reports without the new
`verification_version: 1` marker fail closed before plot output creation.

## OS identity and queue correction

The user reported a shared `eecs` backlog of 1,453 tasks. Previously the entire
account was counted against at most three outside reservations. No live HPC
inspection was performed in this task; the reported count is task context, not a
new measurement. The correction changes only external operational policy.

New plans pin the real/effective OS UID and a verified UID/name account-database
round trip. Neither inherited `USER`/`LOGNAME` nor the free-text inspection
`operator` field authorizes execution. Submission, allocation, reconciliation and
settlement verify the actual OS identity. The pin is carried through intent,
acknowledgment, observed-array, claim and settlement records. Older plans/intents
without it reject new execution; none are rewritten, adopted or repaired.

The user query has no account filter, expands arrays, includes all partitions
and states on the configured local cluster, and checks returned UID and username.
Known live/held/suspended/completing states count separately per array element;
known terminal records are excluded from the live count. Unknown states, malformed
or compressed IDs, duplicates, query errors and identity disagreement fail closed.
Inherited `SQUEUE_*` filters and `SLURM_CLUSTERS` are removed from the query
environment. Accounting queries similarly clear `SACCT_*` filters, select the
pinned UID and validate owner fields as well as exact array mapping.

`concurrency + other_project_reserved_tasks <= 4` is unchanged. Every discovered
live task of the submitter across accounts must fit the explicit outside-work
reservation. Collaborator-owned tasks, other clusters and ambiguous submissions
still require independently reviewed reservations and a project-wide exclusive
window; discovery cannot infer those automatically. Unrelated EECS users no
longer enter this count. Unrelated work of the submitting user still counts
conservatively. The unresolved-wave and lost-acknowledgment rules remain
independent of an empty queue or a terminal state, and settlement still needs
complete terminal accounting and explicit cross-node owner inactivity.

References checked without contacting HPC: [squeue options/fields](https://slurm.schedmd.com/squeue.html)
and [sacct owner/accounting fields](https://slurm.schedmd.com/sacct.html).

## Local verification

The executable command, from the review worktree, is:

```powershell
python -B operators/mobile_all30_evaluation_checks/run_cpu_checks.py
```

Environment: Windows, `C:\Python313\python.exe`, Python **3.13.12**. The runner
uses only stdlib and local readable source. It performs in-memory syntax checks
and Python 3.11 grammar parsing for all five Python runtime/support files, then
runs the mocked unittest suite. It writes no bytecode. The final run passed
**44 tests in 52.731 seconds**: all **28 existing tests** plus **16 new queue,
identity, reservation, reconciliation and settlement checks**. Syntax and Python
3.11 grammar checks passed for **five Python files**. `git diff --check` passed.
No pinned Torch/TorchVision installation was required. These are local Windows
results, not HPC results; no POSIX host identity or live Slurm state was inspected.

During the earlier reporting task, the first run had 20 passes and one failure: permission-denied binding reads were
mislabelled `incompatible`. The fix gives I/O exception types precedence, so
permission errors are `unreadable` and binding mismatches remain `incompatible`.
The suite was then expanded for plotting calls, read-only evidence, post-preflight
binding/runtime changes and final-read errors. No unrun check is counted as passed.

| Area | Exercised behavior |
|---|---|
| Reporting regression | Late HEAD/ancestry/validation/final-read errors retain prior evidence, attempts, orphans and unparsed files; independent slots continue; all 30/840 rows persist |
| Snapshot authority | Evidence/training/coverage stability failures invalidate affected flags and completion; actual mismatched bindings remain visible |
| Consumers | Missing explicit flags fail closed; unverified rows disappear from official summaries and actual mocked scatter calls; cached summaries cannot bypass gating |
| Statistics | Within-checkpoint realization means, three-seed means/sample SD, complete validation candidates and grid-ordered exact ties |
| Mapping/dispatch | All 30 actual wrapper workers, pilots included; independent prescribed 28-slot fixture; invalid selectors reject; full lossless `expected_binding`, roots, all-slot selectors, retry/legacy defaults |
| Scope | Native training/optimizer/scheduler/scaler/model/checkpoint-loader/training-loader boundaries are forbidden guards; no research data is loaded |
| Pins | Native commit/digest/dirty-source rejection; modified external source rejection; changed checkpoint binding during plan validation and again after preflight; effective runtime mismatch rejection |
| Integer JSON | `2**64 - 59` survives real exclusive publication/read/seal and is forwarded unchanged |
| Signals/outcomes | Real source StopRequest/signal context with mocked registration forwards the same stop request and preserves native exit codes 10/20/21/22 |
| Submission | Mock sbatch observes an already durable exact intent; acknowledgment persisted; timeout leaves unresolved capacity and blocks repeated/new waves without another sbatch |
| Ownership/concurrency | Real temporary exclusive owner/publication, controller contention, no-write preview, excess submitting-user work rejection and explicit-only failed retry |
| Read-only reporting | Raw temporary scientific-evidence bytes unchanged; collector cannot call publication/ownership functions |
| Shared-account scope | 1,453 unrelated EECS user tasks do not consume reservations; submitting-user work under multiple accounts counts; expanded pending array elements each count |
| OS identity | Actual kernel/account-database pin despite misleading environment; changed UID/name, effective UID, failed lookup, reverse mapping and unpinned legacy plan rejection |
| Four-task limit | Outside collaborator reservations retained; combined totals above four reject; user queue in excess of reserved capacity rejects before any intent |
| Queue parsing | Inherited filters cleared; unknown states, malformed/compressed/duplicate jobs and returned UID/name mismatch reject; terminal rows do not count as live |
| Settlement/reconciliation | User-scoped discovery across accounts; exact owned accounting; active target arrays, missing/malformed/nonterminal/wrong-owner evidence retain reservations; unrelated owner jobs do not stop another array's valid settlement |

These are external wiring/regression tests, not tests of native GPU numerics,
native checkpoint parsing, scheduler behavior or model evaluation. Fake native
dependencies intentionally prevent real research computation. Matplotlib is
mocked; only its call inputs are checked, not actual renderer output. Windows
mocks UID/account-database calls and directory-fsync because the production
primitives are POSIX-only; actual
temporary file writes, hard-link exclusion and ownership remain exercised.

The exact future HPC CPU repeat is documented in `HPC_HANDOFF.md` using
`run_cpu_checks.py --native-source-root <existing-frozen-checkout>`, with the
same suite and real POSIX directory-fsync on that host. No HPC command was run
here. No dependency blocker remains for these pure mocked local checks.

Automatic discovery remains local-cluster/user-scoped. Collaborator-owned work,
other clusters and uncertain submissions need explicit independently reviewed
reservations; unrelated work owned by the submitter counts conservatively.
Unsupported heterogeneous job displays, unknown states or unavailable ownership
fields block execution for review rather than being treated as zero tasks.
The suite mocks OS identity and scheduler responses and cannot establish actual
HPC NSS/UID consistency, scheduler visibility, account entitlement or deployment
compatibility. Existing plans without the new pin are not migrated or rewritten.

## Material limitations and blocked operations

Actual HPC connectivity/state, current account/QOS limits, prepared data, all 30
checkpoint files, interpreter/activation bytes and allocation runtime have not
been inspected here. No operational plan has been frozen. No timing measurement
supports a wall-time default. Actual Matplotlib rendering was not exercised or
installed. Fresh resource evidence is reviewed by the operator;
free-text administrator output is not automatically interpreted as entitlement.

This deliberately conservative first version does not clear an intent with no
confirmed job, repair partial ownership/plan publication, delete locks, cancel
jobs or automatically resubmit. Missing accounting/comment evidence blocks
reconciliation. Such cases require an audited separately implemented recovery
with exact cross-node evidence. Existing scientific artifacts stay unchanged.

Full-input validation reads all 30 best checkpoints on CPU before each worker;
this favors conservative equality checks but may be expensive on shared storage.
No performance claim was measured. The final all-model thesis presentation is
outside this MobileNet-only implementation; matching graph types and explicit
recipe labels remain required when that presentation is later requested.

The older full repository snapshot on this branch must not replace the pinned
HPC checkout. Only six external runtime files and two separate CPU support files
are transferred to a new review/deployment directory. Changed runtime hashes
must not be retrofitted into existing pinned plans or unresolved submissions.
