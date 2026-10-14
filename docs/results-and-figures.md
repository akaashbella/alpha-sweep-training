# Production results and scientific figures

Step 8 is implemented in source only. No code, tests, imports, compilation/linting,
installations, research/reporting artifacts, figures or Slurm/HPC operations were
run. Runtime correctness remains unverified. Stop before Step 9 final integration.

## Readers and frozen observations

`results.collect_results(manifest, *, base_dir, source_root, overrides=None,
evaluation_root=None, workflow_root=None, extra_protected=())` accepts the validated
**full** 240-task manifest. Phase views are not independent reporting inputs.
Execution-only dotted overrides reuse `task_config`; science cannot be overridden.
The returned finite, sealed version-1 `result_collection` embeds the original full
manifest, grid/document identities, reporting conventions, observation start/end,
collector source inventory/software and protected storage roots.

Training reads reuse `inspect_task`, `read_committed`, restricted CPU checkpoint
validation, epoch/context/statistic validators and completion inspection. Evaluation
reads reuse verified full-budget best binding, coverage and `read_slot` validation.
No model, optimizer, data iterator, evaluation pass, stream restoration, scheduler
query or repair is triggered. Training/evaluation/workflow writer locks are never
acquired. Current reporting source is recorded separately, never substituted for
historical training/evaluator provenance. Old research is not invalidated merely
because reporting code was added; unfinished-training continuation remains governed
by its original conservative source/software gate.

Each expected task retains a training row even when entirely unexecuted. There are
exactly 240 training and 6,720 evaluation rows in original manifest/slot order.
Canonical alpha strings supply joins; numeric alphas supply axes. No approximate
floating-point matching or successful-file glob determines membership.

Training authority/locks bracket each task. All selected checkpoint slot/owner/
sweep tokens bracket its evaluation read. A changing bracket discards its accepted
metrics and records an unstable observation. Other inspection failures are explicit;
one unsafe task is not fabricated as successful. These are independent stable
reads, **not an atomic whole-grid snapshot**. Evaluation inspection/availability
cannot discard valid training validation observations or influence selection;
only a changing/unsafe training read invalidates its training metrics. Frozen exports are validated on load,
including original manifest membership, committed epoch/receipt/selection history,
raw protocol/binding/seed/metrics and captured sweep references. They describe the
collected historical observation, not current mutable storage or permission to run.

`condition` and `stored_status` are separate fields. Numerical/execution failures,
interruptions, orphan failure evidence, missing/locked/partial/corrupt/incompatible/
unstable observations, completion pending/deferred and inspection errors remain
visible. Unexpected root entries, namespaces, slots and unacknowledged records are
diagnostics with `accepted=false`; they never silently join expected results.

Verified independent successes remain usable with other slots missing or sweep
publication deferred. Robustness completion requires captured, verified SWEEP_HEAD
completion and all 28 matching successful raw references. Full-budget best may be
evaluated/reported while training-completion publication is pending; it is not an
eligible completed validation-selection seed until training completion validates.
Legacy clean test/summary evidence has its own `legacy_clean` field. It never fills
an independent clean slot or creates duplicate clean replicates. Previously explicit
qualified imports retain their actual `origin=legacy_import` and original evidence.

## Version-1 schemas and exports

The collection contains the following distinct arrays/documents:

| Field | Content |
| --- | --- |
| `training` | Identity/global/phase indices, dataset/architecture/training alpha/canonical alpha/seed, complete original TaskObservation, science/prepared identities, paths, best-validation metrics, snapshot and legacy evidence |
| `training[].snapshot` | Original transaction/reference, checkpoint receipt, historical provenance/component metadata, each committed epoch reference/document and immutable completion/failure documents; null for unsafe/missing observations |
| `evaluation` | Identity/logical index, prescribed slot, requested protocol/document/fingerprint, condition/status, actual binding/checkpoint/seed or null, namespace, raw reference/digest, whole raw record, success/failure/orphans, diagnostic and separate robustness-completed flag |
| `training[].evaluation_coverage` | Existing independent coverage plus captured immutable sweep document when verified; pending/deferred/error fields remain separate |
| `epochs`, `layer_statistics` | Ordered run/epoch/reference indices into the single embedded epoch documents, avoiding repeated per-layer provenance inventories |
| `accepted_source_references` | Actual training transactions/checkpoints/epochs/completions/metric files and independent raw/sweep files, with path, SHA-256 and role |
| `diagnostics` | Unselected, unsafe or unexpected artifact inspection evidence; no success inference |

Unbound expected slots contain the requested protocol, but **actual checkpoint
identity, digest, seed, raw reference and raw record remain null**. No placeholder
binding is invented. Each accepted raw evaluation retains the existing complete
`evaluation_realization` schema: training/protocol identities, both alpha roles,
replicate/private seed, immutable best selection/data, full 10,000-image metrics
and units, whole-state preservation, runtime/effective backend, historical
training/evaluation provenance, origin/attempt/duration/failure/ancestry.

`report_publication.publish_collection(collection, *, output_dir, source_root)`
exports `collection.json`, flattened `training.csv`/`evaluation.csv`, streamed
`epochs.csv`/`layer_statistics.csv`, `coverage.json` and `diagnostics.json`.
Training CSV retains identities, observation/progress, best accuracy/loss/selection,
checkpoint receipt, provenance/failures/legacy evidence. Evaluation CSV retains
both alpha roles, slot/seed/binding/ref, metrics/counts/units, runtime/preservation/
provenance/origin and unavailable diagnostics. Large epoch histories occur once in
JSON rather than one huge CSV cell for every training row. Layer CSV expands actual
qualified names, module types, shapes, element counts, population std and committed
epoch/reference metadata. No cross-architecture layer equivalence or sensitivity
analysis is asserted. Nested CSV cells are finite JSON; unavailable cells are empty.

## Statistics, validation selection and pairing

`aggregation.summarize_results(collection)` revalidates the frozen collection and
returns a sealed version-1 `result_summaries` linked to its collection digest.
Accuracy remains a fraction; figures label percent. Differences are
`100 * (target_accuracy - baseline_accuracy)` **percentage points**. Means are not
rounded before comparison. Supporting SD is sample SD, `ddof=1`, null for n<2.
The perturbation/layer population std remains `correction=0` unchanged.

An official summary requires **all prescribed observations**. Missing/failed values
stay null with actual eligible/expected counts, individual rows and exclusion
reasons; they are not zeros. No descriptive partial mean is provided. A successful
raw value can remain visible even when compatibility excludes it from an official
summary. There are no hypothesis tests, CIs, significance claims, pooled-nine-run
uncertainty, smoothing, curve fits or added robustness scores.

For each of 80 dataset/architecture/training-alpha groups, `training_groups` keeps
three individual seeds and distinct validation accuracy, clean accuracy and clean
loss summaries. Validation eligibility requires verified full-budget **completed**
training. Clean test eligibility is verified independent clean-slot success, so
pending training summary publication does not hide a valid test observation.

`checkpoint_robustness` has 2,400 checkpoint/evaluation-alpha groups. At alpha zero,
the one clean replicate 0 supplies the checkpoint mean and SD=null. At positive
alpha, all three prescribed realizations must succeed compatibly before computing
a checkpoint mean/sample SD. `robustness_groups` has 800 cells: it summarizes these
checkpoint means across the three training seeds, with cross-seed sample SD
separate from `within_checkpoint_sample_sds`. It never pools nine realizations as
nine independent training runs. Compatibility checks data (excluding the deliberately
distinct training seed), requested protocol, effective settings and historical
torch/torchvision versions; missing or incompatible inputs cannot supply a complete
summary. The existing frozen manifest already fixes recipes across alphas/seeds.

`validation_selection(training_groups)` reads only identity/validation fields.
All ten candidate alphas must have complete three-seed completed-training validation
sets to expose an official winner. Otherwise selection is unavailable with missing
candidates/exclusions; complete candidates can have an explicitly provisional
ranking. Validation means use summed integer correct counts over the fixed summed
5,000-image denominators, avoiding ordering-dependent arithmetic for exact ties.
Every exact unrounded maximum is retained as a co-winner in approved grid order.
No smallest-alpha or test-result tie-break is used. This is distinct from the
unchanged checkpoint policy, which retains the earlier epoch on equal validation.
Every alpha/seed's test results remain reported regardless of selection.

`clean_seed_pairs` contains 240 alpha-zero comparisons paired by dataset,
architecture and training seed; training settings except alpha must match. It keeps
absolute accuracies, full-test denominators, references, pair eligibility/reasons
and percentage-point differences. `paired_training_groups` requires all three
pairs. `noise_realization_pairs` retains 6,480 positive-realization differences from
the same checkpoint's independent clean result, with denominators/references and
eligibility. Checkpoint mean differences require all three realizations before
cross-seed aggregation. These are descriptive comparisons, not significance tests.

## Figures and derived publication

`plotting.plot_results(collection, *, output_dir, source_root, collection_dir=None,
formats=("png", "pdf", "svg"), dpi=200)` validates/rederives summaries and lazily uses
Matplotlib Figure/CanvasAgg. The optional `reporting` dependency extra declares
Matplotlib separately; pinned torch/torchvision/NumPy versions are unchanged. No
package or CUDA build was installed/selected during implementation.

Implemented families:

1. Individual seed best-validation accuracy versus training alpha and supporting
   complete mean ± cross-seed sample SD.
2. Independent clean-test accuracy versus training alpha, with all seed observations.
3. Paired clean-test percentage-point differences from training alpha zero.
4. Robustness versus evaluation alpha for every training alpha: individual
   realizations, seed-specific checkpoint means and complete cross-seed mean ± SD.
5. Numeric training/evaluation-alpha heatmaps for every dataset/architecture,
   both complete two-stage aggregate and each individual training seed.
6. Every run's training/clean-validation accuracy and loss learning curves from
   committed epochs, labeled 1–200, including incomplete/no-accepted-epoch states.

Axes use actual numeric alphas, including zero; no logarithmic exclusion. Overlap
uses deterministic color/shape/size rather than pretending changed alpha positions.
Replicate/seed roles, best-checkpoint policy, units, uncertainty convention and
contributing counts are labeled. Incomplete heatmap cells are masked gray/crossed,
never interpolated or plotted as zero. Curves do not pad/smooth/interpolate or connect
across missing epochs. CSV coverage tables expose each missing, failed, interrupted
or unsafe observation. `figure_index.json` records family filters/counts/positions
and conventions. PNG plus at least one PDF/SVG vector format are required.

Outputs occupy a new exclusive `derived-reporting-v1` namespace with sealed
`OWNER.json`, fresh publisher source/software/options, checksummed output references
and a last-published sealed `COMPLETE.json`. Existing directories, links/junctions,
or overlap with declared scientific/data/preparation/evaluation/workflow/log/source
storage are rejected. The storage base is conservatively protected too: choose a
separate reporting root **outside the base and managed stores**. Declare workflow
root even before its creation; supply any additional operational roots explicitly.
No report may overwrite/recover an earlier report silently; choose a new output
directory. Figure/CSV/write failure preserves partial evidence and publishes no
confirmed completion. A failed final-marker acknowledgement is retained as
`UNCONFIRMED_COMPLETE.json` when possible; failure evidence overrides completeness.
No research files or locks are edited or automatically removed.

Derived publication completeness means its declared files passed checksum/ownership
validation. It **never means scientific experiment completion**. Inspection separates
these statuses and includes scientific coverage. Filesystem hard termination or I/O
failure can leave partial ownership/staging and needs explicit inspection/new output.
No runtime durability or plotting capability was verified here.

## Future production commands and Step 9 interfaces

All commands below are source implementations for later authorized use only. Paths
and full manifest/settings must be actual approved inputs, not reduced examples.

```bash
alpha-sweep-report collect --manifest "$FULL_MANIFEST" --base-dir "$SHARED_BASE" \
  --source-root "$REPORTING_CHECKOUT" --execution-overrides "$STORAGE_OVERRIDES" \
  --evaluation-root "$EVALUATION_ROOT" --workflow-root "$WORKFLOW_ROOT" \
  --protected-root "$SLURM_LOG_ROOT" --output "$NEW_COLLECTION_DIR"
alpha-sweep-report summarize --collection-dir "$COLLECTION_DIR" \
  --source-root "$REPORTING_CHECKOUT" --output "$NEW_SUMMARY_DIR"
alpha-sweep-report plot --collection-dir "$COLLECTION_DIR" \
  --source-root "$REPORTING_CHECKOUT" --output "$NEW_FIGURE_DIR" --formats png pdf svg
alpha-sweep-report inspect --report-dir "$REPORT_DIR"
```

Individual storage flags and strict execution-override JSON reuse the existing
controller interface. Collection is separate from summarization and plotting;
neither calls Slurm or any research execution/repair entry point. Inspection exits
0 for a collected report even with incomplete science/publication; invalid input,
unsafe output or failed production operation exits 22 with an explicit diagnostic.

Step 9 can consume `collect_results`, `validate_collection`, `summarize_results`,
validation-only selection, collection/summaries publishers, `load_collection`,
`inspect_report` and `plot_results`, plus the reporting conventions, seals and
per-file references. No final integration or actual results/figures were generated.
