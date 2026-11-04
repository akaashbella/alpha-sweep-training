# v2 partition filters, evidence and verification review

Worktree: `C:\Users\akaas\RESEARCH\alpha-sweep-training\.handoff-worktrees\mobile-all30-evaluation`.
Branch: `ops/mobile-all30-evaluation-v1`. Starting commit:
`9edde981019201c114f0f5c6ed4f409042dd443b`. This task authorizes scoped code,
mocked CPU tests/syntax checks and commit/push after checks. It does not authorize
HPC operations, real checkpoint loading, research evaluation, training, installs
or submission. None of those operations was performed here.

## Reported failure and preserved evidence

The user reported a 23.7-minute-old inspection at
`/nfs/hpc/share/bellaak/research/alpha-sweep/mobile-all30-eval-inspection.ZUxVLZ`,
an empty preparation result, no operational plan, and no validation/preview.
They also reported the original 44 checks passing on HPC, actual identity
`bellaak`/17907, an empty owner queue and previously matching 30 best bindings.
These are supplied observations, not a new HPC inspection or current integrity
proof. v1 deployment ownership and failed evidence are never repaired or changed.

For this revision, the user reports all 58 prior checks passing on HPC, followed
by real resource collection failing because `AllowQos` was required. Their supplied
partition record has `AllowAccounts=ALL DenyQos=dgx2-plus QoS=dgxh`, no
`AllowQos`, and a user association supporting requested `normal`. That exact
representation is now a CPU fixture; no live OSU query was run here. Every
existing v1/v2 deployment, resource receipt and HPC plan is preserved.

The original ZIP remains untracked under `handoff-evidence/mobile-all30-review/`.
Its 27 declared hashes were checked in initial preparation; its analysis and
scientific figures were not rerun. The original review's 6,000 epoch/384,000 tensor
statistics and historical 0/840 slot count remain exported evidence, not new
measurements. Population weight SD is not a flatness measurement.

Frozen native commit `8010d07d9d1f0886ab1e180523f316623ef622a4`, its source
inventory `a5726bac3f449455d36e2f53840f3e0fd77768a7ab2b989d2884da20f920005c`,
scientific settings, manifests, checkpoint identities, original results and
historical operators stay unchanged. Source APIs were originally inspected at
the pinned GitHub commit; this revision changes only the external operator.

## Exact changes

Four modified files, no new files:

- `operators/mobile_all30_evaluation_v1/launcher.py`
- `operators/mobile_all30_evaluation_v1/HPC_HANDOFF.md`
- `operators/mobile_all30_evaluation_v1/REVIEW.md`
- `operators/mobile_all30_evaluation_checks/test_operator.py`

`batch.sh`, `report.py`, `figures.py`, `run_cpu_checks.py` and native source retain
their bytes.
The runtime inventory is still six files; both CPU support files remain outside
it. The historical repository folder name does not name the deployment version.
The handoff names a fresh `v2-qos-filter` deployment/staging pair; existing bundles
and receipts are protected. The v2 plan path must still be absent before prepare.
If any old operational registry exists, new work blocks for explicit cross-version
capacity reconciliation. No old ownership claim is adopted or widened.

## Final behavior

Partition filters now follow Slurm's documented
[allow/deny precedence](https://slurm.schedmd.com/slurm.conf.html): an explicit
allow list (including `ALL`) controls membership even when a deny list is also
present. Without an allow list, a deny list excludes only its named choices.
The same helper selects account filters. The supplied deny-only OSU record thus
permits `normal` and rejects `dgx2-plus`; no OSU-specific exception is used.
Malformed supplied lists, duplicate aliases, mixed wildcards and absent filter
evidence still reject. Structurally invalid deny evidence rejects even when an
allow list would otherwise control membership.

Attached partition `QoS=dgxh` remains separate from requested job QOS `normal`,
as described in Slurm's [QOS documentation](https://slurm.schedmd.com/qos.html).
It does not grant association membership or change `--qos`. The new integration
check exercises real prepare/collection/argv functions and verifies this boundary.
No freshness, identity, reservation, binding or intent ordering logic changed.

`prepare` reads an old reviewed inspection-shaped request as declarations, runs
full actual source/software/launch/manifest/checkpoint checks, then collects new
scheduler/resource bytes. It never changes a timestamp on the old snapshot.
Original request bytes are re-read after verification to detect edits. This
avoids requiring a manually collected snapshot to survive a lengthy CPU phase.

The collector publishes a unique owned v2 snapshot. Association/QOS/partition/node
commands run after the long work; disk capacity is read anew and the user queue
is queried last. Raw query receipts include actual argv/stdout/start/end times.
Per-file hashes, owner, inspection and `COLLECTION.json` establish the snapshot's
operational linkage. Freshness begins at collection start and remains 900 seconds,
with bounded monotonic collection duration, receipt-time consistency and another
age check after metadata validation. Changing timestamp/seal without new resource
receipts cannot make the old snapshot fresh.

Chosen account/QOS/partition/node/GRES/wall-time remain explicit and match the
plan. Resolved association, partition permissions, QOS/partition time limits,
node state/membership/configured CPU/memory/GRES and reservation insufficiency
reject rather than silently changing settings. The four-task project ceiling,
OS UID/name pins, cross-account expanded owner queue, collaborator reservations
and unresolved-wave/lost-acknowledgment safeguards remain intact.

Submission acquires its controller lease before full binding/coverage checks.
It collects resources only after those checks, revalidates current static
source/software/launch/document/external bytes, persists the exact intent, then
rechecks freshness and identity immediately before sbatch. A late failure leaves
the intent reserved with no automatic resubmission. Preview collects a new owned
resource snapshot but creates no intent/job. Explicit submission repeats actual
verification and collection, never trusts the preview as a cache.

There is no persistent successful checkpoint-inspection cache. Each prepare,
validate, preview, submission and allocation continues to use current native
readers/full binding comparisons. Small static byte checks after collection do
not deserialize tensors or waive those earlier validations. The native evaluator
rebinds again before computation with the exact `expected_binding`. Full tensor
verification remains potentially slow; the fix changes ordering, not its scope.

Phases and each checkpoint print flushed stderr progress and monotonic elapsed
seconds. Stdout remains result JSON. Derived figures also reject output placement
under the old v1 bundle, registry and failed inspection, without changing plots'
measurement/aggregation rules. Reporting keeps all 30/840 observations with raw
evidence separate from explicit verified-measurement authority.

## Executable local verification

```powershell
python -B operators/mobile_all30_evaluation_checks/run_cpu_checks.py
```

The runner uses stdlib and local readable helper source. Five Python files get
syntax and Python 3.11 grammar checks. Tests call real external functions with
closed mocked native imports, fake OS identity, fake scheduler boundaries, a
controllable UTC/monotonic clock and temporary directories. No real scheduler
query, checkpoint load or model construction can escape the mocks. On Windows,
POSIX directory-fsync is mocked; exclusive file publication is real. Rendering
is mocked and no research reports/figures are produced.

The final run passed **66 tests in 70.859 seconds** on Windows / Python 3.13.12:
all previous **58 checks** plus **eight partition-filter regressions**. Syntax and
Python 3.11 grammar checks passed for all **five Python files**; whitespace checks
passed. This is local mocked verification, not a new HPC result. No unrun HPC
check is claimed here.

| Area | Behavioral evidence |
|---|---|
| Partition filters | Exact supplied deny-only record; allowed `normal`/denied `dgx2-plus`; explicit allow lists/ALL; allow-over-deny precedence; equivalent account cases; malformed/ambiguous evidence rejection |
| QOS roles | Fresh prepare/receipt/argv path keeps requested `normal` and association checks despite attached partition `QoS=dgxh`; request bytes remain unchanged |
| Long prepare | Clock advances 30 minutes during actual wrapper/native-adapter verification; collection happens afterward with new receipts; old request unchanged |
| Long submit | Full checks run under controller ownership before new resource/queue capture; one actual intent precedes mocked sbatch |
| Freshness | Old generated snapshots, copied/retimestamped receipts, collection exceeding 15 minutes, post-intent delay and final validation I/O delay all reject |
| Choices/identity | Changed association/QOS/partition/node eligibility, changed explicit account, OS identity change and insufficient current reservations reject without retuning |
| Preservation | Owned v1 bundle, failed inspection and empty result retain exact bytes; old registry evidence blocks; derived-output paths cannot enter old stores |
| Changes after collection | Mutated evidence bytes block submission, preserve the durable intent and do not call sbatch |
| Existing 44 | 30/840 reporting regression, verified-only summaries/plots, all indices/pilots/28-slot delegation, full binding/roots/flags, forbidden training components, source/code/binding/runtime pins, uint64 JSON, stop/exit propagation, ownership, queue scoping, intent/acknowledgment/reservations and settlement |

## Remaining HPC verification and limits

`HPC_HANDOFF.md` gives exact external-only staging, CPU checks, v2 ownership,
prepare/validate/preview, monitoring and continuation commands. The revised parser
has not been run against OSU here. Actual `sacctmgr` resolved-QOS output and `scontrol`
fields, OS/NSS identity consistency, permissions, quotas, node availability,
Slurm/accounting behavior and POSIX durability remain unverified by these mocks.
Unsupported or ambiguous site output fails closed for review; no parser override
or package installation is supplied. All prior scientific environment/hardware
requirements still apply to later authorized evaluation.

Automatic eligibility is not a proof of every account/QOS/TRES/group entitlement,
available GPUs or storage quota. Account ancestry is not resolved by exact-name
checks. Effective attached partition-QOS/TRES/group/OverPartQOS limits still need
site-policy review; a partition QOS is not substituted for the job QOS.
Outside-user/cluster/uncertain project work needs
independent explicit reservations and an agreement against unreserved launches.
The collector does not infer collaborator inactivity from an empty owner queue.
An old v1 registry or a new partial/uncertain intent is not automatically cleared.

There is still no cancellation, lock stealing, uncertain-intent deletion, partial
plan repair, automatic resubmission or numerical-failure retry. Staging/failed
snapshot evidence remains for audited review. A future rerun after failure must
not overwrite any existing deployment or resource receipt, the old failed
inspection or a new partially published v2 plan/intent. The native checkout is
never replaced by this branch's older repository snapshot.
