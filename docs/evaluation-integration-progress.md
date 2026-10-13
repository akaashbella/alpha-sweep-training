# Evaluation integration source progress

Baseline: clean working tree at reviewed Step 6 head
`95081affb5a2914c18557b5a83ede5d0efa1e739`.

Implemented source: independent protocol/slots/hash seeds; training-only journal
version 2 and typed completion; legacy v1 reading; CPU count-200 finalization;
test-only data/runtime; shared full evaluation loop with fixed full-network
temporary weights and exact entire-state preservation; pinned best binding;
independent namespace ownership/slot locks/atomic result and sweep authority;
resumable groups, skips, failure evidence, stop requests, optional qualified
legacy import; linked logical plan and complete lifecycle coverage; controller
subcommands. Training schema/fingerprint 4 and the 240-task grid remain unchanged.

Documentation and source/Git-diff review are complete. Traced source paths:

| Case | Source invariant inspected (not executed) |
| --- | --- |
| Training preservation | Same 240-task Cartesian product, defaults/hash bytes, model/optimizer/scheduler/data streams; shared offset preserves training draw order; clean strict validation selection and separate final |
| Count 200 | TrainingFinalizationPlan validates historical full-budget authority before/under locks; no more epochs or test; actual publisher provenance separate; checkpoint writes rejected |
| Clean baseline | Slot 0/replicate 0/null seed delegates ordinary clean evaluator under whole-state guard; no perturbation generator or duplicate clean slots |
| Positive realization | Entire owned Conv/Linear inventory, correction=0 float32 scales and fixed private temporary tensors allocated once before full-test loop |
| Replicates/restart | Seed binds actual best generation/SHA/protocol/canonical alpha/replicate; no path/order/worker keys or training streams; collision guard; restart whole interrupted pass |
| Exceptional preservation | Exact owned all-parameter/buffer/module bindings and values; restore/recheck then raise, prior flags restored; failed invariant cannot publish success |
| Partial pass/stop | Between-batch discard or finish-current-pass; null failed metrics and explicit diagnostics; no handler registration or mid-pass recovery |
| Completed skip/unknown ack | Validate authority before compute; repeat under slot lease; committed slot/sweep success dominates lost acknowledgement/cleanup diagnostics |
| Legacy | Version-1 training/clean evidence validated and immutable; full-budget eligible; standard legacy lacks exact preservation/runtime proof and needs independent clean execution, never retraining |
| Failed evaluation | Independent record/HEAD only, no training lease/state writes; raw failure and unacknowledged evidence retained, explicit failed retry, discard failed data/model/runtime cache |
| Concurrent publication | Namespace initialization and per-slot exclusive locks, repeated ownership/failure gate, immutable digest validation then atomic HEAD; no lock stealing; separate all-28 sweep authority |
| Planning/coverage | Training manifest remains 240/two 120-task views; linked 6,720 logical slots with no guessed best digest; explicit training phase eligibility and complete raw slot coverage |

Changed files: six new evaluation modules (protocol, state, inputs, persistence,
coverage, execution), shared evaluation/runtime/data/perturbation, training/service/
contracts and checkpoint/journal/continuation/recovery/artifact guards, configuration/
identity/spec adapters, manifest/controller/task execution/coverage, AGENTS/README
and scientific/configuration/dataset/training/checkpoint/controller/HPC/plan guides.
Older Step 5/6 progress notes remain historical and link to the superseding contract.

Git: main at reviewed Step 6 head, 33 modified tracked files and eight new untracked
source/documentation files; no staged changes, commits or deletions. No existing
user edits were present at the initial inspection. Runtime and HPC filesystem/
hardware compatibility remain unverified. Stale evaluation locks and damaged
unowned storage require explicit operator review; no automatic repair is implied.
Scientific seeds/split/normalization/drop_last/precision/scheduler choices remain
pending with actionable execution-time gates. Step 7 receives planning, eligibility,
execution, StopRequest and completion/coverage APIs; Step 8 receives individual
version-1 raw realization schemas. Slurm/signals/orchestration/aggregation/plots
remain outside this integration. No source requirement is intentionally deferred.

No tests, project execution/imports, executable verification, installations,
downloads, preparation/artifact generation, experiments or jobs were performed.
Scientific decisions remain pending; no provisional values have been promoted.
