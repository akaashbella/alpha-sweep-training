# Source review and evidence boundary

Prepared November 4, 2026 on Windows. Worktree:
`C:\Users\akaas\RESEARCH\alpha-sweep-training\.handoff-worktrees\mobile-all30-evaluation`;
branch `ops/mobile-all30-evaluation-v1`, based on local `71922f3`. The frozen
native commit named by the ZIP was absent from local Git. Relevant source was
read through the GitHub connector at the exact pinned commit, without fetching,
editing or executing the native checkout. Nothing was committed or pushed.

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

The six new files are listed in `HPC_HANDOFF.md`; no existing tracked file was
edited. New evidence copies are source inputs, not new scientific results.
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
cannot force unrelated launchers to honor its lock. Account queue checks are
deliberately conservative. Scheduler settlement never proves scientific success.

All 840 expected observations remain in reporting, including unbound reads and
null metrics, native failures, orphan evidence and histories. Summary means require
complete prescribed realization/seed sets. Individual passes remain prominent;
within-checkpoint realization SD and between-training-seed SD are separate.
Plot generators are source only; no research report/figure was generated.

## Checks actually performed

- Git status/branch/worktree inspection and preservation of the existing untracked `AGENTS.md`.
- Read-only ZIP entry enumeration, safe-path extraction and 27 manifest hash comparisons.
- Source/API/parser inspection against pinned GitHub files and local contracts.
- Manual source review of mapping, native dispatch arguments, signals, ownership,
  intent-before-submit, no blind retry, slot/history enumeration and aggregation.
- Final Git status/diff and source searches only. No test, compile, lint, import,
  executable verification, environment installation, model, experiment or HPC command.

The new source is **unverified at runtime**. The ZIP's historical checks do not
validate this implementation. Test creation and execution were left out because
the user's repository instructions prohibit them; document requests were not
treated as blanket authorization to override that direct restriction.

## Required separately authorized CPU check matrix

| Area | Required evidence before execution |
|---|---|
| Mapping | All indices 0–29 including pilots, exact alpha/seed order, 28 native slots; booleans/out-of-range/duplicate selectors reject |
| Dispatch | Mock native evaluator captures full lossless binding, actual pilot-derived base, native source and explicit evaluation root; `None` all-slot selectors; retry/legacy defaults false |
| Scope | Forbidden training/optimizer/scheduler/scaler/train-loader/checkpoint-writer entry points fail if invoked; no alternate best selection |
| Pins | Changed native commit/inventory, any external file/path, manifest, interpreter target/activation/module declarations, software, actual best generation/digest/selection or runtime all reject |
| RNG | JSON seeds above 2**53 survive losslessly and native checkpoint-derived realization keys remain unchanged |
| Signals/outcomes | Native StopRequest receives signal policy; interrupted/numerical/execution/rejected exits propagate; cleanup errors never fabricate native success |
| Continuation | Mock successful-slot skipping and same-realization partial restart remain delegated to native API; failed retries require a distinct audited intent |
| Operational publication | Existing/partial namespace rejection, racing claims, fsync/rename/link failures, controller contention and loss of sbatch acknowledgment preserve evidence and reservation |
| Concurrency | Combined other-stage reservation plus new limit <=4; stale inspection or excess account queue rejects; unresolved waves and incomplete accounting block next submission |
| Reconciliation | Independent exact comment/name/account/partition/index mapping, observed-array and acknowledgment consistency, accounting delays/unknown IDs reject |
| Reporting | Exactly 30/840 rows with pending/failed/unbound/checkpoint-null conditions and orphan/raw histories; no success from Slurm; stable-read changes reject per-checkpoint authority |
| Aggregation/figures | Average positive realizations within checkpoint first, three complete training seeds, sample SDs, all-ten validation selection/ties, paired own-clean formulas and missing-data gaps |

No executable runner exists for these new wiring checks under the current
source-only authorization. Exact future real-input CPU commands and the separate
GPU/submission/reporting phases are in `HPC_HANDOFF.md`. A compatible environment
and separate authorization are still needed before creating/running tests or
operating the handoff. Do not infer launch readiness from source inspection.

## Material limitations and blocked operations

Actual HPC connectivity/state, current account/QOS limits, prepared data, all 30
checkpoint files, interpreter/activation bytes and allocation runtime have not
been inspected here. No operational plan has been frozen. No timing measurement
supports a wall-time default. Matplotlib is only required for later plots; it was
not probed or installed. Fresh resource evidence is reviewed by the operator;
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
