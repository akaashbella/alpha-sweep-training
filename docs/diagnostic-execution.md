# Isolated diagnostic and read-only export handoff

These commands are prepared, not run. Implementation authorization does not start
the execution stage. No Slurm submission, CIFAR-100 run, main-v1 repair or recipe
replacement is included. Use reviewed absolute paths and an existing approved
interpreter/environment. The diagnostic checkout must be separate from the frozen
checkout used by old allocations/continuations; adding tools changes its full
source inventory. Preserve private HPC email/scheduler patches during staging.
Later GPU commands belong in a separately authorized diagnostic allocation/process,
never alongside an active managed research worker on its claimed GPU. Match the
original launcher/thread environment and keep it fixed across arms; the audit
records current intraop/interop counts and thread environment, without guessing
an unavailable original policy or account entitlement.

## First locate evidence without running project code

On the filesystem where the report actually resides, use the user-supplied paths
as quoted values, not Windows-local substitutes:

```bash
tar --list --file "$REPORT_ARCHIVE"
find "$REPORT_ROOT/collection" "$REPORT_ROOT/summaries" -maxdepth 1 -type f -print
```

The supplied archive name contains parentheses; keep it quoted. Do not reconstruct
truncated `collection.json` or feed it into authority validation. Request
`collection/training.csv`, `epochs.csv`, `layer_statistics.csv`, `coverage.json`,
`diagnostics.json` and `summaries/validation_selections.json`, `summaries.json`,
`training_groups.csv`, `coverage.json`, plus available OWNER/COMPLETE checksum
records. Keep the original files unchanged. Evaluation CSVs may be retained for
completeness, but do not use test robustness scores to choose diagnostic candidates.

On HPC, set `READ_ROOT` to a narrow operator-approved project storage root, then
perform read-only discovery:

```bash
find "$READ_ROOT" -type f -name run_metadata.json -path '*cifar10__mobilenetv3_large__alpha-0__seed-*' -print
find "$READ_ROOT" -type f \( -name '*manifest*.json' -o -name '*workflow*.yaml' -o -name resolved_config.yaml \) -print
rg -n '"run_id"|"scientific_fingerprint"|"source_sha256"' --glob run_metadata.json "$READ_ROOT"
```

Inspect candidate full manifests for `kind: full`, all 240 tasks, three actual seed
integers and their seals. A dataset-preparation manifest or a scheduler mapping is
not the full run manifest. Map report seed labels 2/1 to their numeric seed/run IDs;
do not assume those labels mean integers 2/1. Read the exact corresponding saved
`resolved_config.yaml`, `run_metadata.json`, `state/HEAD.json` and its current and
previous immutable transactions. Do not use current research.yaml as a substitute.

For each control/failing run obtain:

| Evidence | Required purpose |
| --- | --- |
| Original resolved config, provenance inventory, full manifest and actual storage roots | Effective recipe, seed mapping, source bytes, immutable scientific identity and output protection |
| Current/previous HEAD transactions and all their epoch/failure/completion references | Complete committed trajectory, selected epoch, exact best/previous binding and failure evidence |
| Referenced latest/best/final checkpoint files plus SHA-256 | Native optimizer/scaler/RNG/data/scheduler state and clean model/buffers; full-budget selection |
| Actual epoch-zero checkpoint, if retained | Original initialization/RNG proof; initial files may have been pruned |
| Any retained checkpoints around the first divergent epoch | Localize optimizer/BN changes; absent/pruned state cannot be reconstructed from final values |
| Prepared manifest, split and normalization descriptors, raw-data content identities | Exact training/validation reconstruction without downloads or preparation |
| Recorded hardware, Python/Torch/TorchVision/NumPy/CUDA/cuDNN, precision/backend/launch environment | Matched comparison and original validation recheck |

For a candidate epoch-zero transaction, read-only `rg -l '"kind": "initial"'
against that run's transaction directory locates its metadata. Follow its exact
checkpoint reference, not an inferred `initial.pt` filename. If it no longer
exists, report it unavailable and use only explicitly labeled seeded reconstruction.

## Later authorized CPU audit and exact export list

Diagnostic CLI commands below use `--execute`; do not run them as part of source
review. Additional `--protected-root` arguments must include every external
evaluation/workflow/research namespace. Output must be a new directory outside
the source, original storage, data, selected inputs and all supplied roots.

```bash
PYTHONPATH="$DIAGNOSTIC_CHECKOUT/src" "$PYTHON" -B -m alpha_sweep_training.diagnostics audit-run \
  --execute --saved-config "$SAVED_SEED2_ALPHA0_CONFIG" --base-dir "$ORIGINAL_BASE" \
  --source-root "$DIAGNOSTIC_CHECKOUT" --output-dir "$NEW_SEED2_AUDIT" \
  --protected-root "$EVALUATION_ROOT" --protected-root "$WORKFLOW_ROOT"
```

Repeat for the actual seed-1 control with a different config and new output path.
Optional `--initial-reference "$INITIAL_REFERENCE_JSON"` takes exactly
`{"relative_path":"generations/<actual>.latest.pt","sha256":"<actual sha256>"}`.
Audit restricted-loads saved CPU state, validates historical authority against its
own saved provenance, compares actual current source/software separately, and
reports native optimizer options/state summaries, original schedule/data/runtime,
epoch metrics and best binding. It checks the selected effective config against
the archived resolved declaration, including execution/data paths, rather than
accepting scientific-hash equality alone. It never opens a research writer lease or repairs
aliases. Writer locks reject inspection; no lock is released here.

The full source digest normally changes when diagnostic tooling is added. Every
original core computation file must still match, with unchanged software. Other
recorded-file differences (including the added CLI entry in pyproject.toml or
historical auxiliary HPC patches) require an explicit review object:

```json
{
  "pyproject.toml": {
    "saved": "<exact saved SHA-256 from audit.json>",
    "actual": "<exact actual SHA-256 from audit.json>",
    "reason": "Reviewed CLI registration only; original recipe and computational source unchanged"
  }
}
```

Review every path in `source_comparison.recorded_file_differences`; include exactly
that set with actual hashes/reasons. An empty object is accepted only if there are
no recorded-file differences. Core changes and software mismatches cannot be
waived. This applies only to separately identified diagnostics, never production
continuation. Do not copy the example reason onto an unreviewed diff.

`audit.json` includes actual paths/SHA-256s; `export-files.txt` lists the exact
current/previous dependency closure and optional supplied initial file. From a
quiescent run, create a **new** external archive, reading original paths only:

```bash
tar --create --file "$NEW_EXTERNAL_AUDIT_ARCHIVE" --verbatim-files-from --files-from "$NEW_SEED2_AUDIT/export-files.txt"
sha256sum "$NEW_EXTERNAL_AUDIT_ARCHIVE"
```

Preserve the audit JSON alongside it and verify every archived file against its
listed hash after transfer. Tar strips leading slashes by default; this is not a
relocation of the archived config. The CLI reads the actual original run paths on
HPC; it does not guess mappings for transferred archives. Copy manifests, source
diffs and prepared descriptors by their separately verified exact paths as needed;
never overwrite originals or silently rewrite archived bindings. These exports
write only the new external audit/archive, never the research namespace.

## Later authorized CSV health report

Create a reviewed threshold JSON outside the original report. Required fields are
`near_chance_margin`, `extreme_loss`, `early_best_epochs` (1-based),
`weak_final_accuracy`, and `seed_accuracy_range`. Accuracy thresholds are fractions.
For example, a descriptive margin .02 and loss threshold 1e6 can reproduce the
audit's near-chance/extreme-loss categories; these are explicit display heuristics,
not research defaults, failure statuses, exclusion rules or early stopping.
Supply the other thresholds explicitly after review.

```bash
PYTHONPATH="$DIAGNOSTIC_CHECKOUT/src" "$PYTHON" -B -m alpha_sweep_training.diagnostics health \
  --execute --report-root "$REPORT_ROOT" --health-policy "$HEALTH_POLICY_JSON" \
  --source-root "$DIAGNOSTIC_CHECKOUT" --output-dir "$NEW_HEALTH_OUTPUT" \
  --protected-root "$ORIGINAL_RUN_ROOT" --protected-root "$WORKFLOW_ROOT" --protected-root "$EVALUATION_ROOT"
```

This joins all 120 CIFAR-10 identities, checks flat/embedded epoch metrics and
earliest-best selection, retains every individual trajectory/seed, streams layer
medians, and hashes the input CSVs before/after analysis. It never opens
collection.json. Summaries are retained as separate evidence, not used to tune
health flags or select alpha. CSV completion booleans remain `reported_*`;
`training_completion_verified` and `evaluation_completion_verified` stay null.
No official means or completion claims are synthesized without journal authority.
`median_weight_sd_last_to_first_recorded_epoch_ratio` divides the last recorded
epoch's post-training median by the first recorded epoch's post-training median.
The adjacent `median_weight_sd_ratio_numerator_epoch_label` and
`median_weight_sd_ratio_denominator_epoch_label` are explicit 1-based labels.
The denominator is not initialization; missing/zero denominator evidence stays null.
A complete sealed collection can instead be passed with `--collection`; a
truncated collection is rejected, never silently migrated. Health reporting never
excludes weak seeds from the original results or touches official aggregation.

The report also retains exported provenance/checkpoint receipts and compares
recorded core file digests with the actual diagnostic checkout when those inventory
cells exist. Missing inventories stay unknown; matching report digests still do
not prove actual checkpoint state or independently reproduce the reported collapse.

## Authorized CPU checks and separate GPU reference stage

The review authorizes only the focused CPU suite in an existing compatible
environment. Run this explicit entry point, not discovery of the mixed CPU/GPU
file:

```bash
cd "$DIAGNOSTIC_CHECKOUT"
PYTHONPATH="$DIAGNOSTIC_CHECKOUT/src" "$PYTHON" -B tests/run_diagnostic_cpu_checks.py
```

The entry point checks Python >=3.10,<3.13, Torch 2.5.1, TorchVision 0.20.1,
NumPy 1.26.4 and PyYAML >=6 before any project/Torch imports. It selects only
`CpuContracts`, `DiagnosticContracts` and `DiagnosticRunnerWiring`, hides CUDA
devices, and explicitly disables `ALPHA_SWEEP_CUDA_TESTS`. It installs nothing.
The runner/recheck integration checks use tiny CPU tensors and mocked native-model,
data, runtime, RNG, checkpoint I/O and full-evaluator boundaries. Native CPU
RMSprop/scheduling/metrics/observation/state-preservation logic would be exercised
when this suite runs; a passing wiring check would not verify MobileNet, original
checkpoint validation counts, CUDA/bfloat16 behavior or original-stream replay.

Actual October 30 review attempt: `C:\Python313\python.exe -B
tests/run_diagnostic_cpu_checks.py` with this checkout's `src` on PYTHONPATH
reported **CPU CHECKS BLOCKED** before project imports/tests. Available Python
3.13.12, Torch 2.9.1 and TorchVision 0.24.1 fail the required versions (NumPy
1.26.4/PyYAML 6.0.2 are installed). No compatible interpreter was found through
the Python launcher, PATH or the checked workspace/common environment locations.
No tests passed or failed; none started. No installation or GPU fallback occurred.

GPU execution is still a separately authorized stage, with a separate opt-in:

The new suites are opt-in so ordinary operational-test discovery does not launch
GPU computations. In a separately approved assigned-GPU process, with pinned
Torch 2.5.1/TorchVision 0.20.1 and no installations:

```bash
cd "$DIAGNOSTIC_CHECKOUT"
ALPHA_SWEEP_DIAGNOSTIC_TESTS=1 ALPHA_SWEEP_CUDA_TESTS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH="$DIAGNOSTIC_CHECKOUT/src" \
  "$PYTHON" -B -m unittest discover -s tests -p 'test_computation_contracts.py' -v
ALPHA_SWEEP_DIAGNOSTIC_TESTS=1 PYTHONPATH="$DIAGNOSTIC_CHECKOUT/src" \
  "$PYTHON" -B -m unittest discover -s tests -p 'test_diagnostic_contracts.py' -v
```

Once explicitly enabled, missing CUDA/software/required environment is an error,
not a successful skipped CUDA verification. These are small isolated module
references, not reduced CIFAR experiments or a certification of all native model
builders. Model-construction/full-evaluation-loader boundaries are mocked
explicitly; the real perturbation, fixed-offset, baseline preservation, native
optimizer, scheduler, accumulator and observer arithmetic is exercised. Positive
gradients are compared using the **same held detached offsets**, not finite
differences that accidentally differentiate a recomputed detached scale.

## Later authorized original best validation recheck

```bash
PYTHONPATH="$DIAGNOSTIC_CHECKOUT/src" "$PYTHON" -B -m alpha_sweep_training.diagnostics recheck-validation \
  --execute --saved-config "$SAVED_SEED2_ALPHA0_CONFIG" --base-dir "$ORIGINAL_BASE" \
  --source-root "$DIAGNOSTIC_CHECKOUT" --output-dir "$NEW_VALIDATION_RECHECK" \
  --source-review "$REVIEWED_AUXILIARY_SOURCE_JSON" \
  --protected-root "$EVALUATION_ROOT" --protected-root "$WORKFLOW_ROOT"
```

This loads verified immutable best into a separate model and evaluates the full
original 5,000-image validation set. It checks original source/software, GPU,
backend and prepared identities. It reports exact correct/example agreement and
loss difference; it cannot publish training/evaluation completion or update the
checkpoint. Resolve mismatches before interpreting candidate differences. Perform
the same check for the seed-1 control. No test loader is iterated.

## Later authorized controlled training diagnostics

First inspect a short bounded trace in a new diagnostic output. This proposed
four-batch, one-epoch-prefix protocol limits expensive observations and selects
only one activation module (verify its exact name against the native model).
It is not a research default or execution approval:

```json
{
  "protocol_version": 1,
  "kind": "optimization_diagnostic_v1",
  "base_scientific_fingerprint": "<exact fingerprint from this seed's audit>",
  "arm": "baseline",
  "epsilon": 0.001,
  "learning_rate": 0.01,
  "precision_mode": "amp_bfloat16",
  "epochs": 1,
  "max_batches_per_epoch": 4,
  "trace_every": 1,
  "activation_modules": ["features.0.0"],
  "initial_checkpoint": null,
  "source_difference_review": {}
}
```

The prefix still delegates the full original clean validation pass, with validation
batch tracing disabled. It does not claim a full training epoch, and is not a
baseline reproduction. After inspecting its output, prepare a **different**
protocol/output for tracing-disabled baseline reproduction. Keep the other fields
and initial state identical; explicitly replace these four fields:

```json
{
  "epochs": 20,
  "max_batches_per_epoch": null,
  "trace_every": 0,
  "activation_modules": []
}
```

This is a protocol edit fragment, not a standalone executable protocol. The
proposed full-data 20-epoch budget spans the reported epoch-17 divergence, while
retaining the original 200-epoch LR horizon. Review the budget before separate
execution; no command here automatically launches both stages. Use a new output
directory for each repetition and for the seed-1 control. Do not use an all-layer,
every-batch 20-epoch trace as the starting example. Any later trace near a suspected
divergence needs an explicit bounded observation plan, not an unrestricted replay.

Verify those baseline values against saved evidence, not the template. If an
epoch-zero checkpoint is supplied, put its exact relative path/SHA in
`initial_checkpoint`; both arms must use the same state. Null explicitly means
seeded reconstruction without proof of original initialization. Match the initial
state digests, data identity/order descriptors and RNG stream descriptors across
arms. Numeric seeds come from each original saved config, never these examples.
Fill `source_difference_review` with the exact reviewed auxiliary-difference object
above; do not leave it empty when packaging/auxiliary files differ. It is part of
the diagnostic identity. Added files and full actual provenance are also retained.

Prepare separate immutable protocols/outputs for these controlled comparisons:

| Arm | Sole allowed change from the original saved baseline |
| --- | --- |
| `baseline` | None; establish repeatability on seed 2 and control seed 1 first |
| `epsilon` | Explicit candidate epsilon .0316, original LR/precision |
| `lower_lr` | Explicit reviewed positive LR below .01, original epsilon/precision |
| `float32` | Precision mode only; original RMSprop settings and all backend flags |

The float32 arm preserves the original TF32/backend policy; do not describe it as
an IEEE-only comparison if the recorded policy allows TF32. One-factor validation
rejects combined epsilon/LR/precision changes. The main-v1 fixed-recipe validators
remain strict; the diagnostic creates a private native RMSprop, records/validates
all effective options, and never falsifies the production optimizer descriptor.
All arms use the original scratch builder/data streams and full 200-epoch schedule
horizon, not a shortened cosine horizon. A truncated-batch diagnostic is limited
to one original-epoch prefix; it never advances epoch cadence across omitted
batches or claims a full epoch. Full baseline diagnosis should use null truncation.

```bash
PYTHONPATH="$DIAGNOSTIC_CHECKOUT/src" "$PYTHON" -B -m alpha_sweep_training.diagnostics train \
  --execute --saved-config "$SAVED_SEED2_ALPHA0_CONFIG" --base-dir "$ORIGINAL_BASE" \
  --protocol "$REVIEWED_PROTOCOL_JSON" --source-root "$DIAGNOSTIC_CHECKOUT" \
  --output-dir "$NEW_DIAGNOSTIC_OUTPUT" --protected-root "$EVALUATION_ROOT" --protected-root "$WORKFLOW_ROOT"
```

Tracing is expensive and isolated: `trace_every: 0` with empty activation modules
disables it; positive cadence samples deterministically, exact names restrict
hooks, and `['*']` selects Conv/Linear/BN outputs. Records include prediction
histograms, all-value logit/loss quantiles, gradients/parameters, measured native
update-to-weight ratios, RMSprop square-average/momentum and BN buffers/affines.
Training and clean-validation rows have distinct split/boundary labels. Validation
uses the original full evaluator through a delegating runtime and read-only root
hook, not a second model pass. Its detached loss observation precedes the real
finite-loss guard; a forward observation does not certify successful batch loss.
No extra model forward/backward, random subsampling, BN update or optimizer
decision is introduced by observations. Detached unreduced CE is labeled as a
diagnostic duplicate; native mean CE alone drives gradients. Unsampled batches
retain no trace snapshots. Inspect matching baseline traces before asserting
noninterference for an actual GPU/precision policy; the prepared reference tests
have not been run here.

Outputs are only diagnostic JSON/JSONL; there is no research HEAD, summary alias,
tensor checkpoint writer, publication callback, scheduler submission or resume.
`diagnostic_budget_finished` never means main-v1 completion, even at 200 diagnostic
epochs. Failures preserve partial diagnostic evidence; reruns require new paths.
`failure` records `origin`, `stage`, zero-based `epoch_index`/`batch_index`, typed
exception/cause information and structured `numerical_failure` stage/reason/tensor
names. A null index means no such boundary was active/known, not batch or epoch
zero. Model-computation, data/setup, observation and output failures stay distinct;
activation hooks retain their observation origin even through evaluator wrappers.
`secondary_failures` and `observation_errors` preserve failures while recording the
primary problem. Final-publication failure propagates to the CLI with the prior
failure retained, even when the diagnostic JSON cannot be written. CLI failure
publication errors remain explicit on stderr; JSON stdout remains clean.
The new directory is claimed exclusively and its immutable owner token is checked
on each file/trace write; lost/linked ownership rejects writes. Neither missing
final JSON nor a stale owner authorizes replacement or automatic resume.
Do not change BN buffers in archived checkpoints or introduce recalibration without
a separately reviewed clone/training-only diagnostic protocol.

Only after evidence supports a stable baseline across all three seeds and full
budget should a new production scientific identity be proposed for the affected
whole 10-alpha × 3-seed architecture grid. No production recipe/rerun is authorized
by this handoff. Existing main-v1 data and the original frozen source remain intact.
