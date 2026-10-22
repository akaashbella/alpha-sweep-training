# Step 9 final source integration

## Later planning-validation performance correction

The [performance correction and measurements](planning-validation-performance.md)
address repeated journal/checkpoint/completion inspection and per-layer provenance
serialization. Planning now reuses stable observations and tensor-free journal
metadata only within one operation, snapshots relevant artifacts/ownership before
reuse, and performs a final fresh guard before durable submission intent. Actual
eligibility, scheduler reconciliation, source/software/launch pins, scientific
completion and the CIFAR-100 training gate remain authoritative. CSV bytes are
unchanged; progress is stderr-only. Existing queue/throttle policy and histories
remain unchanged.

Unlike the historical source-only Step 9 work below, the latest request explicitly
authorized focused equivalence/invalidation tests and a synthetic benchmark. Local
CSV reconstruction improved 1.82×; combined synthetic repeated planning improved
4.28× (38.032 s to 8.881 s). Windows pure-read performance regressed because safe
snapshot hashing dominates; POSIX/HPC end-to-end improvement remains unmeasured.
No torch/models/data/research, packages, Slurm commands/jobs or HPC changes were
run. The operating guide links the actual measurements, test boundaries and safe
staged transfer retaining private email/scheduler patches and frozen continuation
source. Historical Git inventories below describe earlier turns.

Final focused discovery ran 25 cases: 24 passed, one real-symlink case skipped for
Windows privileges; the mocked linked-ancestor check and existing queue/legacy-wave
checks passed. This correction leaves nine tracked files modified and adds four
implementation/documentation/verification files; the pre-existing untracked
`AGENTS.md` is preserved. Nothing was staged, committed, pushed or transferred.

The original Step 9 record below is historical. Its recovery-readability claims
are corrected by the historical-policy follow-up at the end of this note; the
original Git inventory describes that implementation turn, not the current tree.

Reviewed baseline and actual local HEAD: `main`,
`68c0dbfba0f7596db00355a7332507e8a5e1e38c` (the supplied last reviewed commit).
The initial working tree was clean; no later user edits were present, reset or
discarded. Existing implementations were reused. Source/configuration/documentation
and Git were inspected only; no path in the evidence table was executed.

## Concrete corrections

1. The global workflow `split_file` override incorrectly selected one file for
   both datasets. Policy retains its version-1 historical declaration shape but
   execution inherits the dataset-specific finalized manifest split: null is
   intentional and excluded from pending paths/global overrides. A non-null legacy
   global split produces an actionable execution-readiness error. Historical policy
   state remains parseable; any policy adoption is explicit after quiescence.
2. Direct `execute_task` could compute CIFAR-100 before the managed all-120
   CIFAR-10 training completion gate. New/continue/overwrite computation now checks
   the same historical training coverage before construction and again under its
   run lease. Robustness is absent from the gate. Recovery/completed skip and
   count-200 publication-only finalization add no epochs and remain independent.
3. Planner/runner applied the phase gate to count-200 publication-only work.
   Already verified full-budget authority is now exempt in both, matching direct
   dispatch without permitting any additional training.
4. Batch seals pinned activation/interpreter **paths**, but not their external
   bytes/target. Plans now seal version-1 `launch_inputs`; render, write, submit and
   runner repeat actual-file compatibility. Bash checks interpreter target and
   launch bytes before activation. Activation inputs/ancestors must be plain;
   interpreter symlinks are explicitly resolved and target-pinned. Actual allocation
   provenance retains the launch pin. Source inventory v2 includes the checked-in
   workflow declaration as well as all prior executable/declaration inputs.
5. Stale source/doc status and the plan's acknowledgement-only durable-progress
   wording were corrected. HEAD remains authoritative after a lost acknowledgement.
   Historical progress notes are labeled historical, retaining their original
   stopping points and Git inventories without retroactive execution claims.

The scientific specification was updated only in its implementation/handoff prose.
Scientific schema/fingerprint 4, spec constants and identity encoder, recipes,
training/data/model/noise semantics, durable authority, checkpoint bindings and
evaluation seeds are unchanged. The executable source digest deliberately changes:
unfinished old work still fails conservative source/software compatibility. No
global migration or relaxed compatibility gate was introduced.

Work plan/bundle/registry schema versions remain 1, with additive nested launch
provenance. Historical work lacking the launch pin remains readable by existing
ownership/reconciliation validators but is explicitly rejected for new rendering,
submission and execution. Preserve uncertain intents and old sealed bytes; resolve
ownership/scheduler state before replanning. Restrictions stay report version 2;
older/missing-coverage reports cannot authorize submission. No existing artifacts
were generated, resealed, migrated, repaired, removed or modified.

## Static evidence matrix

| Contract / source path traced | Evidence and producer/consumer boundary reviewed |
| --- | --- |
| Strict declarations/readiness | [config.py](../src/alpha_sweep_training/config.py): RunConfig validation/unresolved_settings/require_research_ready, load_config/from_declaration; [spec.py](../src/alpha_sweep_training/spec.py) constants and fixed recipes, unchanged. Pending inputs remain explicit. |
| Preparation and full manifest | [prepare.py](../src/alpha_sweep_training/prepare.py): parser/prepare_dataset; [prepared.py](../src/alpha_sweep_training/prepared.py): read_prepared_bundle/load_prepared_dataset/config association; [manifests.py](../src/alpha_sweep_training/manifests.py): build_manifest/manifest_from_dict/load_phase/task_config. Prepared bases precede architecture/task settings; exact 240 Cartesian tasks/linked phases and immutable data identity checks. |
| Data streams | [data.py](../src/alpha_sweep_training/data.py): EpochSampler/TrainSampleKey/training and evaluation views, build_loaders/build_test_loader; local epoch/sample keys exclude architecture/alpha and workers never prepare/repair. Fixed official counts/transforms remain unchanged. |
| Scratch model/weight/optimizer ownership | [models.py](../src/alpha_sweep_training/models.py): build_model/_adapt/_cpu_initialization; [weights.py](../src/alpha_sweep_training/weights.py): owned all-Conv/Linear discovery; [optimizers.py](../src/alpha_sweep_training/optimizers.py): fixed single-group native recipes on clean final-device parameters. All four CIFAR adaptations/native regularization retained. |
| Fresh one-task execution | [task_execution.py](../src/alpha_sweep_training/task_execution.py): execute_task/dispatch before construction and inside open_run; [artifacts.py](../src/alpha_sweep_training/artifacts.py): new/overwrite/active ownership; [training.py](../src/alpha_sweep_training/training.py): commit_initial before batches. No fresh fallback for existing state. |
| Per-batch perturbation and precision | [perturbation.py](../src/alpha_sweep_training/perturbation.py): training_forward/gaussian_weight_offset; [runtime.py](../src/alpha_sweep_training/runtime.py) explicit CUDA/autocast/backend; [randomness.py](../src/alpha_sweep_training/randomness.py) separate fresh/restored RuntimeStreams. Functional tensors retain gradients to float32 clean parameters; correction=0 detached scales/noise, original BN buffers, zero bypass. |
| Update/epoch counters and strict best | [training.py](../src/alpha_sweep_training/training.py): _backward_and_update, epoch loop, clean validation, snapshot; [checkpoint_records.py](../src/alpha_sweep_training/checkpoint_records.py): validate_epoch. Observed native optimizer calls distinguish AMP skips; successful-update cadence, full coverage, statistics and strict correct-count improvement agree. |
| Original scheduler/restoration | [scheduler.py](../src/alpha_sweep_training/scheduler.py): completed-unit equation/state validation/saved_schedule_rates; [continuation.py](../src/alpha_sweep_training/continuation.py): full preflight/restore. initialize=False, optimizer then scheduler/scaler/data, RNG last; original horizon and separate counts. |
| Durable commits and lost acknowledgements | [persistence.py](../src/alpha_sweep_training/persistence.py): _commit/commit_epoch/publish_failure; [checkpoint_journal.py](../src/alpha_sweep_training/checkpoint_journal.py): read_committed/receipt and record/role validation; [checkpoints.py](../src/alpha_sweep_training/checkpoints.py) restricted CPU payloads. Immutable files precede HEAD; committed completion dominates receipt/diagnostic loss. |
| Interruption/continuation/numerical failure | [task_execution.py](../src/alpha_sweep_training/task_execution.py), [continuation.py](../src/alpha_sweep_training/continuation.py), [interruption.py](../src/alpha_sweep_training/interruption.py), [recovery.py](../src/alpha_sweep_training/recovery.py): original latest restoration/replay; orphan/committed numerical evidence terminal, execution retry explicit, publication repair separate. |
| Count-200/legacy boundary | TrainingFinalizationPlan, execute_task early finalization, DurableTrainingPersistence.publish_training_completion, journal v1/v2 readers: stored provenance and full budget before/under ownership; no builders/extra epochs/test sweep; legacy clean evidence does not fabricate robustness. |
| Independent best binding | [evaluation_inputs.py](../src/alpha_sweep_training/evaluation_inputs.py): bind_evaluation_checkpoint; [evaluation_protocol.py](../src/alpha_sweep_training/evaluation_protocol.py): prescribed_slots/perturbation_seed. Actual immutable best generation/digest/data/protocol; approved private keys, 28 slots/checkpoint, no training restore. |
| Fixed full-pass network/preservation | [evaluation.py](../src/alpha_sweep_training/evaluation.py): evaluate_realization/_evaluate_split; [evaluation_state.py](../src/alpha_sweep_training/evaluation_state.py): CleanStateBaseline.realization/verify/restore. Once-per-pass functional weights; full 10,000-image counts, clean zero path; exact parameter/buffer/binding/mode preservation on exceptions. |
| Slot skip/failure/deferred sweep | [evaluation_execution.py](../src/alpha_sweep_training/evaluation_execution.py): execute_evaluation; [evaluation_persistence.py](../src/alpha_sweep_training/evaluation_persistence.py): leases/publish_robustness_completion/SweepFinalizationDeferred; [evaluation_coverage.py](../src/alpha_sweep_training/evaluation_coverage.py): historical compatible observations/verified sweep. Null partial metrics, immutable ancestry, success after lost ack, contention exit 0 and pending/deferred/error distinctions; only all-28 authority claims completion. |
| Phase gating and work selection | [work_plans.py](../src/alpha_sweep_training/work_plans.py): build_work_plan/candidate/validate_item; [coverage.py](../src/alpha_sweep_training/coverage.py): collect_coverage/inspect_task. Phase computation requires 120 successful training completions; terminal failure never unlocks phase 2; pending evaluation independent. |
| Packaging/provenance/launch | [pyproject.toml](../pyproject.toml), [__init__.py](../src/alpha_sweep_training/__init__.py), [execution_provenance.py](../src/alpha_sweep_training/execution_provenance.py): actual imported checkout, source inventory v2, actual software and launch pins; five CLI mains/parser arguments matched to documented commands. Paths/allocation/PIDs/timestamps stay outside scientific identity. |
| Render/explicit submission | [work_render.py](../src/alpha_sweep_training/work_render.py): batch_source/render_work/write_bundle; [hpc_workflow.py](../src/alpha_sweep_training/hpc_workflow.py): submit_bundle; [slurm_client.py](../src/alpha_sweep_training/slurm_client.py): guarded submit/validate_restriction_report. Seal/index/policy/checkpoint/launch linkage, report v2 coverage, refreshed eligibility and intent before sbatch, one shared cap. |
| Signal/owning allocation | [allocation_runner.py](../src/alpha_sweep_training/allocation_runner.py): run_allocation/_run_owned_allocation/allocated_identity; [allocation_signals.py](../src/alpha_sweep_training/allocation_signals.py): scoped attribute-only StopRequest handlers and worker hook. Bash exec preserves owning PID/GPU visibility; one claim and one sequential task/checkpoint per allocation. |
| Concurrent controllers/uncertain scheduler | [workflow_registry.py](../src/alpha_sweep_training/workflow_registry.py): open_workflow/controller_lock/reserve/claim_allocation/reconcile; SlurmClient.observe_wave and hpc_workflow resolve-submission/terminal/locks. One unresolved bounded wave, exact intent/IDs, unknown accounting/residual locks retain capacity; verified safe-stop continuation only. |
| Historical complete/incomplete collection | [results.py](../src/alpha_sweep_training/results.py): collect_results/_collect_evaluation; [reporting_contract.py](../src/alpha_sweep_training/reporting_contract.py): validate_collection/reporting_provenance. All 240/6720 expected rows; null unbound refs/seeds, stable brackets, explicit conditions, saved provenance and evaluation inspection isolated from validation eligibility. |
| Summaries/selection/pairing | [aggregation.py](../src/alpha_sweep_training/aggregation.py): summarize_results/validation_selection/_signature/_paired. Complete sets only, sample ddof=1, realization→checkpoint→seed means, integer validation denominators/exact co-winners, descriptive percentage-point pairs. |
| Figures and derived authority | [plotting.py](../src/alpha_sweep_training/plotting.py): plot_results; [report_publication.py](../src/alpha_sweep_training/report_publication.py): open_report/publish/load/inspect. Lazy Figure/CanvasAgg, individual observations/masked unavailable cells, separate guarded output outside base/managed/source, final checksums and explicit partial failure; derived completion never establishes science. |
| Read-only boundaries | Controller inspect/select/status/evaluation-inspect/status, workflow status without --scheduler, historical collectors/validators: no research computation/repair, signal registration, Slurm calls or research writer leases. Planning records actual versions/launch bytes but submits nothing; rendering writes operational state and is separately classified. |

These traces include fresh/initialized continuation, clean interruption, count-200,
completed compatible skip, exceptional preservation, lost acknowledgement, deferred
finalization/genuine errors, explicit numerical/execution failure, uncertain submission,
concurrent controllers and wholly unavailable/partial/complete reporting. They are
static evidence, not proof of runtime behavior.

## Packaging and remaining gates

Python `>=3.10,<3.13`, torch `2.5.1`, torchvision `0.20.1`, NumPy `1.26.4`, PyYAML
`>=6.0` and optional reporting Matplotlib `>=3.8,<3.11` declarations are unchanged.
Console scripts and their actual mains are present; imports requiring research
compute remain in their component paths. No older standalone scripts/templates,
inactive tests or alternate submission entry points were found in the inspected
layout. Direct manual run/evaluate commands are explicitly distinguished from the
registered worker/supported production submission path in the operating guide.

No unresolved repairable source blocker was identified in this review. Actual
installation/import/rendering, GPU/software compatibility, accounting version/output,
ownership races/signals, durable shared filesystem behavior, large-reader/renderer
memory/runtime and numerical behavior remain unverified. Module definitions and
activation transitive files require operator freezing beyond top-level launch pins.

Pending inputs remain training seeds; preparation seeds/persisted splits;
normalization policy/constants/source; drop_last; precision/backend; scheduler
cadence/warmup factor/minimum LR; actual account/QOS/restriction review/shared paths/
fixed environment/CUDA; fsync/atomic rename/hard-link guarantees; accounting visibility
and full production reader/runtime/renderer evidence. The exact fields, blocked
operations and separate future verification handoff are in
[operating-guide.md](operating-guide.md).

## Final source-only handoff

All nine implementation steps are complete in source, including the approved
independent evaluation integration. The end-to-end guide orders twelve future
operations and supplies recovery decision tables without executing them.
README, AGENTS, implementation plan and affected guides reflect the current scope;
historical notes remain historical.

Final Step 9 working-tree inventory: 23 modified tracked files and two new
untracked documentation files (`operating-guide.md`, this note), nothing staged;
branch/HEAD remain the reviewed `main` commit above. Nothing committed or pushed.
Changes comprise eight source files, the operational workflow declaration, and
documentation/status integration. No research scientific defaults were populated.

No tests, project execution/imports, executable verification, compilation/linting,
installations, environment setup, downloads/preparation, models/training/evaluation,
actual manifests/submission bundles/workflow state, aggregates/reports/figures,
synthetic examples/experiment artifacts, Slurm commands or HPC jobs were performed.
Runtime correctness remains unverified. Stop here; the future operating sequence
requires a separate explicit instruction and has not begun.

## Historical-policy inspection/recovery source correction

Correction baseline: clean `main` at
`4035a6121738f0dc97ce86fa1d949350fa021c5f`. No subsequent user edits were present;
no reset/discard was performed. The earlier Step 9 change retained declaration
parsing for old global-split policies, but the CLI, registry root guard and writer
preflight still required research readiness. Consequently old uncertain submission
authority could not be resolved before adoption. The original static review missed
this integration defect; this follow-up corrects the actual supported CLI path.

Changed source:

- `hpc_policy.py`: `require_recovery_ready` requires shared Linux/original registry
  inputs, plus the original account for scheduler operations; it does not require
  execution environment/data readiness or remove the original global split field.
  `recovery_storage_overrides` projects only run-storage paths for lock evidence,
  excluding both global data and split overrides.
- `workflow_registry.py`: explicit existing-registry `recovery_only` access preserves
  ownership/full-manifest/sealed-state/bundle validation and original policy
  fingerprint equality before writes and under the controller. It cannot initialize
  authority or adopt a different policy. Recovery leases cannot reserve work;
  reconciliation validates its client policy. Adoption requires an explicitly
  selected `previous_policy` matching stored authority at the same registry root.
- `hpc_workflow.py`: old-policy status/reconcile/submission/terminal/lock recovery
  use those prerequisites. `status --workflow-only` reports no scientific coverage;
  non-null global-split status automatically does the same with explicit null/false/
  diagnostic fields. Run-lock evidence never feeds that global split into scientific
  computation or coverage. Non-submission proof rejects worker claims/residual
  allocation locks. `adopt-policy --previous-policy` refuses unknown submission
  identity, validates original terminal/non-submission proof and queries original
  per-wave scheduler policy rather than substituting corrected policy values.
- `slurm_client.py`: the submission backend explicitly rejects recovery leases
  and retains research/submission readiness, including unsafe global-split rejection.

Existing work/bundle/registry/lock-report schemas and scientific identity bytes are
unchanged. Sealed old policy values, wave mappings, scheduler comments/IDs, proofs,
allocation ordinals and old unpinned plans remain readable and preserved. New
planning/rendering/submission/advance and allocation execution still require the
corrected research-ready policy and current launch pins; old unpinned work cannot
be executed through recovery. No source/software continuation or evaluation gate,
training state, checkpoint binding or seed was relaxed or rewritten.

Static traces inspected, without executing them:

| Case | Actual current source decision |
| --- | --- |
| Old global-split status | CLI selects recovery-only registry read; `_read_state` validates original owner/seals/bundles and `read_workflow` retains stored fingerprint equality. `scientific=null`, coverage-collected=false and explicit diagnostic; no data/coverage collector. `--workflow-only` also supports otherwise non-ready historical environments. |
| Corrected values supplied for old status/recovery | Original stored-policy equality rejects them. Existing ownership is validated before recovery lock writes and again under the lease; no replacement registry is initialized. |
| Acknowledged-job reconciliation | Recovery-only open with original policy; exact registered scheduler identity/comment/account/index checks and residual allocation ownership retain reservation when terminal proof is incomplete. New operational snapshot retains the original policy, never runs research. |
| Lost acknowledgement with an exact job ID | `resolve-submission --job-id [--cluster]` uses original policy/commands and existing scontrol membership verification before recording acknowledgement. No sbatch or replacement work. |
| Explicit non-submission proof | Exact wave/comment/confirmed absence, audit reason, no worker claims and no residual allocation locks; immutable proof precedes original-policy state update. Absence/PID/age alone never resolves it. |
| Residual controller/allocation/run ownership | `inspect-locks` validates original stored policy even when ordinary status is blocked by a controller lock. Exact unchanged owner/path/digest evidence plus all-owner inactivity confirmation is required for audit/archive release. Selected run scopes use storage-only overrides and verified evaluation binding where needed. |
| Corrected-policy adoption | Required --previous-policy matches original stored fingerprint before writes and under controller; same registry root. Every submitted wave must have verified terminal evidence under original wave policy or matching immutable terminal proof; non-submitted waves must retain matching explicit proof with no acknowledgement/claims. Unknown submissions and residual allocation locks reject adoption. Old histories/ordinals persist. |
| Historical missing launch pins / new execution | `_read_state`/validate_bundle retain old read compatibility. Recovery lease reservation/submission is rejected; research readiness and current renderer/submitter/runner launch-pin checks still apply to new execution. |

The [operating guide](operating-guide.md#historical-policy-inspection-recovery-and-subsequent-adoption)
now documents the actual sequence: original-policy workflow-only status; original
acknowledged-job reconciliation or exact-ID/non-submission resolution; verified
owner recovery and terminal evidence; then explicit corrected-policy adoption with
both policy inputs, followed only by separately authorized fresh planning. The
[Slurm guide](slurm-workflow.md) describes the new prerequisites/status fields and
required adoption argument. These are future instructions; no commands were run.

Correction Git inventory: seven modified tracked files (four source modules and
three guides/progress documents), no untracked or staged files; branch/HEAD remain
the correction baseline above. Nothing committed or pushed. Scientific and
operational inputs remain unresolved; actual recovery, scheduler/output behavior,
filesystem durability and runtime correctness remain unverified.

Only source/documentation and Git inspection/editing occurred. No tests, project
execution/imports, compilation/linting/executable verification, packages, artifacts,
Slurm commands/submissions or HPC jobs were created or run. Stop after source work;
runtime verification or operation still requires separate explicit authorization.
