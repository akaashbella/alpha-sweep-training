# Stage 1: cross-entropy loss geometry from existing reports

This package analyzes **already-produced, completed reporting-v1 summaries
publications**. It does not change the experiment, load neural-network checkpoints,
read live training/evaluation storage, generate perturbations, or run training or
evaluation. It adds no command to the original CLI. All package code, documentation
and software tests are confined to this directory; `analysis` is a Python namespace
package. No dependency or packaging changes are required.

No completed scientific report was supplied for this implementation. No scientific
analysis has been executed. Synthetic fixtures exist only in `tests/` and temporary
unit-test directories, and must never be used to obtain scientific findings.

## Motivation and local derivation

At a **fixed clean best-validation checkpoint**, let `W_l` denote each eligible
owned Conv2d/Linear weight tensor and `sigma_l = population_std(W_l)` with
correction zero over its entire tensor. The existing evaluation protocol samples
independent standard Gaussian tensors `z_l` and holds

```text
delta_l = alpha_eval * sigma_l * z_l
W_eval,l = W_l + delta_l
```

fixed throughout one full 10,000-image test pass. Other parameters and learned
buffers stay at their clean values, using evaluation mode. Stage 1 consumes that
experiment's reported cross-entropy means; it samples no noise itself.

For the empirical test loss `L`, the formal second-order Taylor expansion is

```text
L(W + delta) = L(W) + grad L(W)^T delta + (1/2) delta^T H(W) delta + remainder
E[delta] = 0
Cov(delta) = alpha_eval^2 * blockdiag(sigma_l^2 I_l)
E[L(W + delta)] - L(W)
    approximately (alpha_eval^2 / 2) * sum_l sigma_l^2 Tr(H_ll(W)).
```

This uses a local expansion with finite derivatives, a fixed clean-tensor noise
scale, independent centered Gaussian offsets, and negligible expected remainder.
It does **not** require the trained point to be stationary: the first-order term
vanishes in expectation, but not necessarily for three sampled realizations.
Gaussian support is unbounded; locality depends on the actual perturbation scales
and remainder, not just a small nominal alpha. CNN nonlinearities and finite
precision introduce further limitations to a smooth Taylor interpretation.
Off-diagonal blocks contribute zero to this expectation because noise covariance
between tensors is zero. We never differentiate the scale with respect to `W`:
this expansion concerns additive perturbations at the fixed baseline.

We fit an **empirical local loss-sensitivity coefficient** `c`, with `2c` as an
empirical curvature proxy. Stage 1 does not explicitly estimate a Hessian, its
trace, or its blocks. A good fit is not proof of actual local Hessian curvature or
out-of-sample predictive validity. Negative differences and coefficients remain
negative; an indefinite Hessian, realization noise, nonlocality, or other effects
can prevent a positive fitted response.

## Data contract and validation

Supply the **directory** of a completed `kind=summaries` publication containing
`OWNER.json`, `COMPLETE.json`, `summaries.json`, and the other files declared by the
existing publisher. A bare JSON file, collection publication, figure publication,
partial directory, or unknown reporting version is rejected.

The actual schema comes from:

- [aggregation.py](../../src/alpha_sweep_training/aggregation.py):
  `checkpoint_robustness`, `training_groups`, `robustness_groups`, `clean_seed_pairs`.
- [reporting_contract.py](../../src/alpha_sweep_training/reporting_contract.py):
  version 1, document seals and exact conventions.
- [report_publication.py](../../src/alpha_sweep_training/report_publication.py):
  owner/completion/file references and checksums.
- [results-and-figures.md](../../docs/results-and-figures.md) and
  [weight-robustness-evaluation.md](../../docs/weight-robustness-evaluation.md).

The loader reuses the original read-only `verify_seal`, strict JSON reader,
`canonical_alpha` and central grid constants. It validates owner/completion seals,
version/kind/roles, every published file checksum, safe relative paths, summary
seal and owner/summary grid/manifest/collection linkage. Failure or unconfirmed
completion evidence rejects the publication. It checks all 80 training groups,
240 checkpoint identities, 2,400 checkpoint/alpha records, 800 robustness groups,
the repeated ordered three-seed plan, logical realization indices, official counts,
means/sample SDs, exclusions, canonical alphas and consistent checkpoint bindings.
Missing expected records are malformed input; scientific absence is represented
by the publisher's explicit unavailable observations, not by deleting rows.

There is no original standalone summary validator. The original `inspect_report`
eagerly imports checkpoint I/O that imports PyTorch. This package implements the
equivalent publication-level read-only file checks plus the summary-specific
checks above using the original lightweight validators; it never imports
checkpoint I/O or calls `validate_collection`/`collect_results`. It validates the
frozen producer's decisions, not current mutable scientific storage. It does not
reaggregate raw evaluation files. Checksums establish integrity, not authenticity
or independent proof that the original science was correct.

Input checksums are rechecked after loading and before publishing completion.
The input publication is opened for reading only. No repair, lock acquisition,
dataset access or traversal of raw references occurs. Source metadata retains the
whole original owner/completion documents (including publisher provenance),
summary document and byte digests, original collection digest, manifest document
digest, and grid identity. Recipe revisions remain in run IDs, which are retained
everywhere; detailed historical recipes remain in the linked original collection.

### Completeness and compatibility

- Alpha zero uses its one prescribed clean replicate 0.
- A positive-alpha checkpoint mean requires all three valid compatible
  realizations. Incomplete sets never produce partial means.
- Each positive mean is paired only to that **same checkpoint's** official clean
  result. Best generation/digest and compatibility signatures must agree; the
  producer's `noise_change_pp` pairing eligibility must also be complete.
- A missing clean baseline excludes every difference and fit for that checkpoint;
  independently available absolute metrics remain visible.
- Original `robustness_groups` compatibility is an additional gate on cross-seed
  summaries. Incompatible settings do not become compatible just because each
  checkpoint has three successes.
- Training comparisons additionally require the original `clean_seed_pairs`
  eligibility, which checks training settings except alpha. All nine positive
  training alphas are compared with zero using the same training seed.
- `coverage.json` retains all 2,400 analytical observations and all 6,720 prescribed
  individual slots, observed/accepted values, references and exclusions, plus the
  original coverage document and separate fit exclusions. A derived publication
  can be complete while its scientific coverage is incomplete. Sweep completion
  remains the original verified field; no new completion claim is inferred.

## Exact calculations and replication units

All loss values are mean cross-entropy in **nats per example**. Accuracy in input
and absolute CSV columns is a fraction; explicit percent columns are also saved.

```text
Delta L_r(a) = official_checkpoint_mean_loss_r(a) - official_clean_loss_r(0)
Delta Accuracy_r(a) = 100 * (official_mean_accuracy_r(a) - clean_accuracy_r(0))
Accuracy degradation_r(a) = -Delta Accuracy_r(a)  # percentage points
```

For each checkpoint and each window, unweighted OLS through the origin uses
`x_i = alpha_eval,i^2`, `y_i = Delta L_i`:

```text
c = sum_i x_i y_i / sum_i x_i^2
empirical_curvature_proxy_2c = 2*c
prediction_i = c*x_i
residual_i = y_i - prediction_i
RSS = sum_i residual_i^2
RMSE = sqrt(RSS / n)
R_squared = 1 - RSS / sum_i (y_i - mean(y))^2
```

The primary window is **0.01, 0.02, 0.05**. The secondary window additionally uses
**0.10**. Every point in the selected window is required. Zero is an anchor and is
not counted as a fitting point; no larger alpha enters either local fit. A missing
point excludes the fit with its valid count and reason rather than changing its
window. `R_squared` is the **centered** definition even though the model has no
intercept. It can be negative, is never clamped, and is null with an explicit status
if observed fitting values are constant (including all zero). RMSE uses `n`, not a
degrees-of-freedom correction. There are no positivity constraints.

Three perturbation realizations are **within-checkpoint replicates**. The existing
report averages them first. Three training seeds are **independent training
replicates**. Fit each seed separately, then summarize checkpoint measurements
using complete compatible three-seed means and sample SD (`ddof=1`). No pooling
of nine realizations as nine training experiments occurs. Seed SD is descriptive,
not a confidence interval or total uncertainty estimate. Different checkpoints'
three-realization means have sampling error. All deltas for a checkpoint share its
clean baseline; the OLS fit does not model this covariance or estimate Hessian
uncertainty. With three or four fitting points, fit diagnostics deserve caution.

For every alpha in the full grid, both local fits generate predictions and signed
and absolute residuals, retaining missing observations as null. Positive alphas
outside each fit's window are labeled extrapolation. No relative error is defined
near zero loss increases, and no formal breakdown threshold is asserted.

Training comparisons preserve every target seed and compare both `Delta L` and
`2c` to clean training, then report complete three-seed paired summaries. A negative
paired difference indicates lower loss sensitivity, not automatically better
classification. `relationships.csv` describes Pearson correlations of loss change
or each local proxy with **accuracy degradation**, separately for dataset,
architecture and evaluation alpha; it also includes strata at each training alpha.
Separate compatibility-signature cohorts prevent correlations from pooling
different data/protocol/runtime/software settings, even when individual checkpoint
means are valid. Cohort counts and full-scope expected counts are both retained.
The across-training-alpha strata have 30 expected checkpoints, the within-alpha
strata three. Available-pair counts and undefined-constant/insufficient-pair
statuses are explicit. These conditional descriptive correlations may use fewer
available pairs; they are **not** official three-seed means. They carry no p-values,
confidence intervals, causal claims or independence claim from pooling repeated
evaluation alphas. The findings tables show all training alphas, absolute accuracy,
both proxies and paired differences; raw seed rows remain primary evidence.

## Running locally

Use an environment with the existing repository dependencies available. A normal
installed checkout works; the module entry point also locates this checkout's
`src/` so installation is not required. For scientific use, use the repository's
declared Python **3.10–3.12** and the existing optional `reporting` extra
(`matplotlib>=3.8,<3.11`). Stage 1 adds no third-party dependency. Numerical analysis
uses the standard library. Do not install or change the research environment as an
implicit part of running this package.

From the repository root, for example in PowerShell (replace paths with your own):

```powershell
python -B -m analysis.loss_geometry_stage1 `
  --report-dir 'D:\reports\summaries-v1-completed' `
  --output 'D:\stage1-results\loss-geometry-001' `
  --protected-root 'D:\research-storage' `
  --protected-root 'D:\workflow-storage' `
  --protected-root 'D:\reports' `
  --dpi 300
```

`--output` must not exist, even if empty. It must be disjoint in **both** directions
from the input directory, checkout, and every declared protected root. Linked
paths/junctions are rejected. The summary format does not embed the full
collection's protected-root inventory, so `--protected-root` is required: repeat it
for **all** training/checkpoint/evaluation/data/workflow/log storage bases and
original derived-reporting bases, including external stores. This declaration is
the operator's responsibility; the package cannot discover missing roots without
accessing other artifacts. Output roots must be separate from these bases. Merely
having a different final directory name is insufficient. No overwrite/resume
mechanism is provided.

`--dpi` accepts 72–1200 and defaults to 300. PNG and PDF are both mandatory.
There is no switch to relax completeness, change fitting windows, inject raw files,
load models, regenerate reporting, or select favorable training alphas.
Exit 0 means analytical output publication succeeded, not that all research data
exist. Invalid input, unsafe paths or execution failures exit 2 and give a diagnostic.

## Running in an authorized HPC CPU allocation

Do this only after explicit authorization and after a CPU allocation and an
appropriate environment already exist. No scheduler submission is implemented or
required by this package; use the institution's approved allocation procedure.
No GPU is needed. Copy/use the completed summary publication read-only and keep
the output outside the protected research and reporting roots.

```bash
cd "$CHECKOUT"
python -B -m analysis.loss_geometry_stage1 \
  --report-dir "$SUMMARIES_DIR" \
  --output "$STAGE1_NEW_OUTPUT" \
  --protected-root "$RESEARCH_STORAGE_BASE" \
  --protected-root "$WORKFLOW_STORAGE_BASE" \
  --protected-root "$ORIGINAL_REPORTS_BASE" \
  --dpi 300
```

Declare further protected roots if storage extends outside these bases. Running
this command is an explicit analysis action; implementing the package does not
authorize running it on research data or accessing live storage.

### If the original summary report is missing

Do not silently collect or summarize. The existing commands for a later explicitly
authorized reporting phase are:

```bash
# If a completed collection publication already exists:
alpha-sweep-report summarize --collection-dir "$COLLECTION_DIR" \
  --source-root "$CHECKOUT" --output "$NEW_SUMMARIES_DIR"

# Only if collection is also missing; this separately authorized operation reads
# live authorities and checkpoints, and is NOT part of Stage 1:
alpha-sweep-report collect --manifest "$FULL_MANIFEST" \
  --base-dir "$SHARED_BASE" --source-root "$CHECKOUT" \
  --workflow-root "$WORKFLOW_ROOT" --output "$NEW_COLLECTION_DIR"
# Then summarize NEW_COLLECTION_DIR with the first command.
```

For non-default scientific storage, the collect command must also use the actual
existing execution overrides (`--execution-overrides "$OVERRIDES_JSON"`), actual
`--evaluation-root "$EVALUATION_ROOT"` when used, and repeated
`--protected-root "$EXTRA_OPERATIONAL_ROOT"` for other operational roots. Do not
invent locations, configurations, missing realizations or completion evidence.
The full 240-task manifest, not a phase manifest, is required by original reporting.
Stage 1 never invokes any of these commands.

## Outputs

```text
<new output>/
  analysis_metadata.json
  loss_sensitivity.csv
  quadratic_fits.csv
  group_summaries.csv
  predictions_residuals.csv
  paired_training_comparisons.csv
  paired_training_summaries.csv
  relationships.csv
  coverage.json
  figure_index.json
  findings.md
  figures/
    figure1_<dataset>_<architecture>.png/.pdf
    figure2_<dataset>_<architecture>_local.png/.pdf
    figure2_<dataset>_<architecture>_full.png/.pdf
    figure3_<dataset>_<architecture>.png/.pdf
    figure4_<dataset>_<architecture>.png/.pdf
    figure5_<dataset>_<architecture>.png/.pdf
    figure6_<dataset>_<architecture>.png/.pdf
  COMPLETE.json
```

The sensitivity table contains 2,400 rows; fits 480 rows; predictions/residuals
4,800 rows including clean anchors. Nested provenance/individual/fitted-point CSV
cells use finite JSON. Unavailable numeric cells are empty, not zero.
Metadata records options, versions, code hashes, equations, limitations and input
identities. Analytical rows are deterministic; creation timestamps are operational
metadata, and rendering bytes can depend on Matplotlib/platform versions.

The six figure families are produced separately for each dataset and architecture:

1. `Delta L` against evaluation alpha: full and local views, every training alpha,
   individual seeds and complete cross-seed mean +/- sample SD.
2. `Delta L` against evaluation alpha squared: all ten training alphas in separate
   panels, individual observations and primary/secondary seed fits; solid/dashed
   fitted portions and dotted/dash-dot extrapolation. Separate local/full files
   keep small-alpha behavior visible. Primary fit range is shaded.
3. Empirical `2c` against training alpha: all individual seeds and complete mean/SD,
   with separate primary and secondary panels and shared scale.
4. Observed against predicted `Delta L`: equality line; fitting points and larger
   alphas have different markers; shared observed/predicted scale.
5. Accuracy change against loss change: individual checkpoint means, all training
   and positive evaluation alphas. Marker size tracks evaluation alpha.
6. Paired differences in loss response from clean training: all nine positive
   training alphas and the full evaluation grid, individual paired seeds and
   complete mean/SD. No perturbation-training alpha is cherry-picked.

Axes are linear and preserve zero and negative values. Comparisons within panels
use shared scales where needed; dataset/architecture files use their own observed
scales and labels. Missing points interrupt curves. Missing-data families have
explicit index entries with reasons and no invented plots. A complete input with
all usable observations yields 56 figure documents, each exported in PNG and PDF.
No smoothing, artificial scientific observations, uncertainty pooling, CI, or
formal breakdown detection is added.

An exception preserves partial outputs and `ANALYSIS_FAILURE.json` rather than
deleting evidence. `COMPLETE.json` is written last and checksums all other outputs;
it is an analytical-output marker, never a scientific completion authority.
Do not reuse a failed directory; select a new output. No crash-durability claim is
made for this standalone analysis writer. Consumers must reject failure evidence
and verify the completion references before relying on partial file collections.

## Software verification

Only the tests below are authorized by the Stage 1 implementation request. They
use synthetic mathematical fixtures and temporary reporting-shaped publications,
not actual research data. No original test suite is modified or run.

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
python -B -m unittest discover -s analysis/loss_geometry_stage1/tests -t . -v
```

```bash
PYTHONPATH=src python -B -m unittest discover \
  -s analysis/loss_geometry_stage1/tests -t . -v
```

Tests cover baseline identity, units, quadratic/negative/zero coefficients, both
windows, centered unclamped R-squared, residuals/RMSE, missing baselines and
realizations, full expected coverage, duplicates/mismatched identities, two-stage
seed aggregation, cross-seed incompatibility, document/publication corruption and
versions, deterministic results, path guards, input byte preservation, exclusive
outputs and failure evidence. Where Matplotlib is already installed, export tests
render all six families for one synthetic dataset/architecture into temporary
software-test storage. They are skipped explicitly if Matplotlib is unavailable;
no packages are installed. Symlink tests can skip if the OS denies symlink creation.
Use the final implementation handoff for the actual executed test counts and
environment limitations; source review alone is not runtime verification.

## Limitations and later layer-wise Hessian work

Loss robustness does not guarantee accuracy robustness. A poorly classifying or
nearly constant network may have a small loss response. Absolute clean and perturbed
accuracy, all seeds, missing data and baseline loss must accompany sensitivity
comparisons. Architecture comparisons are descriptive: architectures also use
different optimization recipes, and the CIFAR-100 MobileNet epsilon revision is
an additional recorded recipe consideration. Neither architecture nor noise
training can be assigned sole causal responsibility from these descriptions.

Future, separately approved analysis could independently estimate eligible-block
traces/Hessian-vector quantities and compare their weighted sum with `2c` in a
verified local regime. This package preserves checkpoint generation/digest, run
IDs, seed, both alphas, the reporting collection link, windows and residuals to
support that investigation. It implements no Hessian-vector products, layer-wise
perturbations, new experiments, model loading or later-stage machinery.
