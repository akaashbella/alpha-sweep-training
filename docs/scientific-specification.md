# Main experiment: main-v1

This is the authoritative scientific specification. Schema version 2 and
`src/alpha_sweep_training/spec.py` implement its current configuration contract.
Changes to the scientific specification require explicit agreement and a
versioned update; execution resource changes do not.

## Research question and perturbation

How does training-time Gaussian weight perturbation strength affect final clean
classification performance across CNN architectures?

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
  entirely, including noise sampling. Validation and test use clean weights.

Step 4 implements this contract; there is no perturbation/training implementation
in Step 1. The configuration records the contract rather than selecting a
particular future autograd implementation.

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
stochastic depth where present). Step 3 implements builders; all architecture
names are already valid configuration declarations.

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
defaults, whose software version is recorded separately; Step 3 must document
the arguments actually passed and avoid silently adding a different recipe.

## Evaluation, reporting, and failures

Save best-validation and final checkpoints. Select best by highest clean
validation accuracy; replace it only on strict improvement so exact ties retain
the earlier epoch. Every alpha and training seed gets full clean test evaluation
using its own best-validation checkpoint. Choose best alpha using mean
validation accuracy across all three seeds, never test accuracy.

Prominently display each seed's validation and test result. Mean/std are
supporting summaries, not substitutes for individual runs. Record per-epoch
metrics and clean layer-weight statistics, including population standard
deviation. Reporting must state its standard-deviation convention explicitly;
the cross-seed summary convention and best-alpha tie handling are to be finalized
for Step 8, without concealing ties or missing/failed seeds.

Numerical failures are explicit terminal or recoverable status records with
reason, epoch/step, and the affected metric/tensor where available. JSON numeric
fields with unavailable or nonfinite results are `null` accompanied by failure
metadata, never NaN/Infinity tokens, fabricated scores, or successful completion.
No silent dropping of failed seeds from a supposedly complete three-seed summary.

Latest, best, and final checkpoints have distinct roles. Epoch 200 or the end of
a Slurm allocation does not complete a run. The full prescribed clean test
evaluation and final result artifacts are required; see [the checkpoint and
result contracts](implementation-plan.md).

Quantization, layer-wise perturbation experiments, evaluation-time noise,
additional datasets, and fine sweeps are outside this main implementation.

## Development workflow

All nine steps are implementation-only. Read/edit source and inspect Git status
and diffs; do not run tests, project code, compilation, linting, training,
evaluation, benchmarks, synthetic/smoke experiments, or preliminary notebooks.
Do not install packages, download data, or submit jobs. Existing inactive test
files need not be maintained. Runtime verification is deferred until the full
implementation is complete on OSU HPC. Source inspection does not establish
runtime correctness.
