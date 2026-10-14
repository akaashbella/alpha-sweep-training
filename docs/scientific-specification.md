# Main experiment: main-v1

This is the authoritative scientific specification. Schema version 4 and
`src/alpha_sweep_training/spec.py` implement its current configuration contract.
Changes to the scientific specification require explicit agreement and a
versioned update; execution resource changes do not.

## Research question and perturbation

How does training-time Gaussian weight perturbation strength affect clean
classification performance and fixed-network weight robustness across CNN architectures?
The training specification/schema remain main-v1/version 4; the approved independent
evaluation addition is versioned separately as `weight_robustness_eval_v1`.

For every eligible current clean weight tensor, before every mini-batch:

`W_perturbed = W_clean + alpha * population_std(W_clean) * epsilon`

- Compute the standard deviation over the **entire** current clean tensor,
  using `correction=0`, before introducing any noise.
- Detach both the scale and sampled noise from autograd. Sample fresh independent
  standard-normal noise for each eligible tensor and mini-batch.
- Perturb every `Conv2d.weight` and `Linear.weight` simultaneously. Include stems,
  projections, depthwise/pointwise convolutions, SE components, and classifiers
  where present; do not filter by position or tensor dimensions.
- Do not directly perturb biases, normalization parameters, or buffers.
  Unperturbed trainable parameters learn normally. BatchNorm running statistics
  retain normal training updates.
- Evaluate gradients at perturbed weights, then update clean/master parameters.
  Temporary noise must never accumulate in stored weights, optimizer state, or
  checkpoints. Restoration must also be safe on exceptions.
- Alpha is one fixed global scalar for the run. Alpha zero bypasses perturbation
  entirely, including noise sampling. Validation uses clean weights. Test uses
  the mandatory clean baseline and the separate approved robustness sweep.

Step 4 implements this contract with a functional temporary view connected to
original float32 parameters and original buffer references. Statistics/noise
are float32 outside autocast. Configuration records the fixed
`functional-fp32-training-v1` engine and `sha256-runtime-streams-v1` randomness
contracts. Their source-only implementation, gradient ownership, alpha-zero
bypass and precision policy are in [the training guide](training-and-perturbation.md).

## Approved grid and data

| Dimension | Values |
| --- | --- |
| Dataset order | CIFAR-10, then CIFAR-100 |
| Architectures | resnet50, mobilenetv3_large, efficientnet_b0, convnext_tiny |
| Global alpha | 0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00 |
| Training seeds | Three distinct integers, pending; reused across alphas |
| Counts | 120 runs per dataset; 240 total |

Use native 32×32 RGB images. Persist one fixed 45,000-training/5,000-validation
split from the official 50,000-image training set for each dataset; use all
10,000 official test images. The split is independent of the training seed and
shared across architectures, alphas, and seeds. Never use reduced subsets for a
research configuration. Test data do not inform split or normalization decisions.

Training transforms, in order: random crop with padding 4, random horizontal
flip, conversion to CPU float32 channel-first tensors using uint8/255, and
dataset-specific normalization. Padding is constant zero on all four sides;
crop offsets are uniform over 0–8 inclusive; flip probability is 0.5.
Validation/test use deterministic tensor conversion and normalization without
random augmentation or resizing. Final normalization
constants or their derivation policy remain pending. Step 2 must persist the
actual finite constants and their source/derivation identity before execution.

Step 2's dataset implementation exists, with execution deferred. Shared
preparation requires an explicit independent split seed for a new unstratified
PCG64 permutation and an explicit normalization policy (supplied RGB constants
with source, or training-partition population moments). No permanent split seed
or normalization policy has been chosen. The implementation is documented in
[the dataset guide](datasets.md); training workers only read finalized artifacts.

## Models

Train from scratch with no pretrained weights. Output 10 classes for CIFAR-10,
100 for CIFAR-100. Preserve the architecture except for these adaptations:

| Architecture | CIFAR adaptation |
| --- | --- |
| ResNet50 | 3×3 stem, stride 1, padding 1; initial max pool replaced with Identity |
| MobileNetV3-Large | Initial stem stride 1 |
| EfficientNet-B0 | Initial stem stride 1 |
| ConvNeXt-Tiny | 3×3 stem, stride 1, padding 1 |

Keep remaining structure and native regularization (including dropout and
stochastic depth where present). Step 3 implements builders in source only,
with isolated CPU float32 initialization independent of alpha, and explicit
native initialization of replacement stems. ConvNeXt's internal 7×7 depthwise
convolutions and later downsampling remain unchanged. Classifiers output logits.
See [the model and optimization guide](models-and-optimization.md).

## Training and optimization

200 epochs, batch size 128, cross-entropy, five-epoch linear warmup followed by
cosine decay. Do not stop a poorly performing run early. No added MixUp, CutMix,
label smoothing, gradient clipping, or EMA. Hold the architecture's entire
recipe fixed across alphas and seeds. Exact handling of an incomplete final
training batch, precision/backend policy, and unspecified scheduler details
remain explicit decisions in [the configuration guide](configuration.md).

| Architecture | Fixed optimizer recipe |
| --- | --- |
| ResNet50 | SGD; LR 0.1; momentum 0.9; weight decay 0.0005 |
| MobileNetV3-Large | RMSprop; LR 0.01; momentum 0.9; weight decay 0.00001; rmsprop_alpha 0.9; epsilon 0.001 |
| EfficientNet-B0 | Same specified RMSprop recipe |
| ConvNeXt-Tiny | AdamW; LR 0.001; weight decay 0.05; betas (0.9, 0.999) |

`scientific.alpha` always means perturbation strength.
`scientific.optimizer.rmsprop_alpha` always means RMSprop's smoothing parameter.
The optimizer never reads the perturbation alpha as its own smoothing factor.
Unspecified optimizer arguments use the selected implementation's native
defaults, whose software version is recorded separately. Step 3 documents the
arguments passed and effective native defaults in the model/optimization guide.
All clean trainable parameters, including biases and normalization, occupy one
group with the declared weight decay. Recipes have not been empirically validated
in this rebuilt repository.

The implemented versioned scheduler uses completed cadence units n. For epoch
cadence T=200 and W=5; for optimizer-step cadence both multiply by the actual
loader training-batch count. For n<W, lr(n)=L*[f+(1-f)*n/W]; for W<=n<=T,
lr(n)=m+(L-m)*[1+cos(pi*(n-W)/(T-W))]/2. The next unit uses lr(n) before
completion advances n. Thus lr(0)=L*f, lr(W)=L, and lr(T)=m after T units;
the last training unit uses lr(T-1). The factor f, minimum m and cadence remain
pending. Skipped AMP updates do not advance optimizer-step position. Continuation
preserves the original horizon and separate completed-epoch/successful-update
counts; see the guide for restoration order. None of this source has been run.

Step 4's shared engine observes actual native optimizer calls to distinguish
float16 GradScaler skips from successful updates. Bfloat16 uses native support
without a scaler; float32 disables autocast. All modes preserve float32 master
parameters and use the same configured forward precision for clean evaluation.
Runtime/model RNGs and private weight-noise RNGs are independent of initialization,
data streams and alpha; fresh streams seed once, continuation restores them.
Unresolved precision/backend decisions remain unresolved.

## Evaluation, reporting, and failures

Save best-validation and final checkpoints. Select best by highest clean
validation accuracy; replace it only on strict improvement so exact ties retain
the earlier epoch. Every training alpha and training seed gets an independent
best-checkpoint test sweep: evaluation alpha 0 once, and each positive alpha in
the approved ten-value grid three times (replicates 1–3). Each realization holds
one complete perturbed network fixed for all 10,000 test images. That is 28 result
slots/checkpoint, 6,720 total, without multiplying the 240 training tasks.
Choose best training alpha using mean
validation accuracy across all three seeds, never test accuracy.

Prominently display each seed's validation and test result. Mean/std are
supporting summaries, not substitutes for individual runs. Record per-epoch
metrics and clean layer-weight statistics, including population standard
deviation. Step 8 reporting uses sample SD (`ddof=1`, null for n<2), with complete
prescribed seed/realization sets required for official summaries. Positive-noise
realizations are averaged within each checkpoint before means/sample SD across
training seeds; these two SD levels remain separate. Global training-alpha selection
requires all ten complete three-seed completed-training validation sets. Retain
all exact unrounded mean ties as co-winners in approved grid order; incomplete grids
have no official winner. Test results do not affect eligibility or tie-breaking.
Accuracy is a fraction; differences are percentage points. Paired comparisons are
descriptive only. See [results-and-figures.md](results-and-figures.md) for implemented
raw/derived schemas, completeness and individual-first figures.

Numerical failures are explicit terminal status records with
reason, epoch/step, and the affected metric/tensor where available. JSON numeric
fields with unavailable or nonfinite results are `null` accompanied by failure
metadata, never NaN/Infinity tokens, fabricated scores, or successful completion.
No silent dropping of failed seeds from a supposedly complete three-seed summary.
Clean layer statistics are recorded after each epoch's updates, before full clean
validation, including alpha zero. Finite float16 forward/loss with scaler-managed
gradient overflow records a skipped update; it introduces no skip threshold or
extra training. Invalid clean state and nonfinite forward/loss are terminal
numerical failures. Metrics come from the actual training forward and use
sample-weighted loss, integer correct counts and fraction accuracy.

Latest, best and final remain distinct. Training completes after all 200 epochs,
valid latest/best/final, strict clean validation selection, metrics/statistics and
durable training summary. It does not require test in that allocation. Independent
evaluation binds immutable best and has its own guarded result slots and complete
all-success sweep authority; evaluation failure never invalidates training.

For evaluation, use `W_eval = W_clean + evaluation_alpha * population_std(W_clean) * epsilon`.
Recompute correction=0 population std from each clean best tensor, sample detached
float32 independent noise once per owned Conv2d/Linear tensor, and retain the same
temporary values for the entire test pass. Biases/normalization/buffers are not
perturbed; eval mode uses learned buffers. Alpha zero bypasses noise. Exact owned
whole-state comparisons protect all parameters/buffers/bindings/prior modes on
normal and exceptional exits. A failed invariant cannot publish success.

Independent versioned hash seeds bind best generation/digest, protocol version,
canonical evaluation alpha and replicate; no paths/jobs/workers/order enter them.
The explicit evaluation precision/backend policy matches the resolved configuration;
float32 weights/noise/scales/loss are preserved. No optimizer/scheduler/scaler or
training RNG restoration is required. See the authoritative
[evaluation contract](weight-robustness-evaluation.md) for exact declaration,
encoding, preservation, persistence, legacy compatibility and raw schemas.

Frozen schema-4 `scientific.alpha` and legacy run/record alpha names mean training
alpha only. Frozen evaluation fields describe clean validation/selection and the
baseline; new protocol identity is separate. Changing evaluation never silently
changes training fingerprints. Legacy completed clean-test evidence establishes
training and that historical clean result, not 27 positive results.

Quantization, layer-wise perturbation experiments, additional datasets and fine
sweeps remain outside scope. The fixed full-network evaluation weight-noise sweep
is explicitly in scope; image corruptions/adversarial attacks are not added.

## Development workflow

All nine steps are implementation-only. Read/edit source and inspect Git status
and diffs; do not run tests, project code, compilation, linting, training,
evaluation, benchmarks, synthetic/smoke experiments, or preliminary notebooks.
Do not install packages, download data, or submit jobs. Existing inactive test
files need not be maintained. Runtime verification is deferred until the full
implementation is complete on OSU HPC. Source inspection does not establish
runtime correctness. All nine steps are now complete in source; the
[operating guide](operating-guide.md) records the unresolved-input and verification
handoff. Later setup, verification and execution require a separate explicit
instruction. Step 9 changes operational integration/provenance only, preserving
schema-4 scientific hashes, training semantics and independent evaluation seeds.
