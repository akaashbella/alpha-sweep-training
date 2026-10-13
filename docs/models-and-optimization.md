# CIFAR models, clean optimizers, and original-budget scheduling

Step 3 is implemented in source only. Models and optimizers have not been
constructed; forwards, parameter counts, scheduler trajectories, and continuation
have not been executed or runtime-verified. Verification belongs to OSU HPC
after all nine steps. Step 4's shared training is documented in
[training-and-perturbation.md](training-and-perturbation.md); checkpoint-file
persistence and recovery are implemented in Step 5's
[checkpoint guide](checkpoints-and-recovery.md), also without execution.

## Production interfaces and readiness

| Interface | Result and component requirements |
| --- | --- |
| `models.build_model(config)` | CPU float32 `nn.Module`; requires a validated `RunConfig` and selected `scientific.training_seed` |
| `models.describe_model(model)` | `ModelMetadata` with `to_dict()`; observations of the actual constructed model without a forward |
| `weights.discover_eligible_weights(model)` | Deterministically sorted tuple of `EligibleWeight` records including live clean parameter references |
| `optimizers.build_optimizer(model, config)` | Native optimizer bound to the clean model, after final device placement; fixed recipe and one group |
| `optimizers.describe_optimizer(optimizer)` | `OptimizerMetadata` with `to_dict()`; original recipe/base LR and actual effective options/current LR |
| `optimizers.validate_optimizer(optimizer, config=None)` | Read-only recipe, identity, ownership, group and trainability guard |
| `scheduler.build_scheduler(optimizer, config, steps_per_epoch, initialize=True)` | `WarmupCosineScheduler`; explicit factor, minimum and cadence required |
| Scheduler properties / `metadata()` | `interval`, `completed_units`, `total_units`, `warmup_units`, `base_learning_rates`, `current_learning_rates`; serializable metadata |
| Scheduler `state_dict()`, `validate_state_dict(state)`, `load_state_dict(state)` | In-memory continuation state and compatibility checks; no files |
| Scheduler `require_binding(optimizer, config, steps_per_epoch)` | Step 4 preflight of exact optimizer ownership, original descriptor and installed rates; no LR mutation |
| Scheduler `learning_rates_at(position)` / `saved_schedule_rates(...)` | Step 5 pure saved-LR/descriptor/counter checks, using the same scheduler equation without initialization |

Construction does not call whole-run research readiness. Models do not require
datasets, normalization, scheduler decisions, precision policy or Slurm settings.
Optimizers require a supported constructed model and compatible fixed recipe.
Schedulers require their own decisions and the actual loader count for step
cadence. The launcher must still call `require_research_ready()` before research
execution. Importing configuration or the package initializer does not import
the new compute modules. Runtime construction checks the pinned torch 2.5.1 and
torchvision 0.20.1 releases; local CUDA wheel suffixes are permitted and recorded.

## Exact model adaptations

All constructors receive `weights=None`, `num_classes=10` or `100`, and
`progress=False`. No pretrained download or optional architecture variant is
enabled. Classifiers are created by their native constructors and produce logits;
no softmax is added.

| Architecture | Only changed structure | Replacement initialization |
| --- | --- | --- |
| ResNet50 | `conv1`: 3->64 channels, bias=False, 7x7/stride2/pad3 becomes 3x3/stride1/pad1; initial max pool becomes Identity | Kaiming normal, fan_out, ReLU |
| MobileNetV3-Large | `features.0.0` stride becomes (1,1); kernel3/pad1, 3->16 channels and bias=False retained | Existing initialized weight retained |
| EfficientNet-B0 | `features.0.0` stride becomes (1,1); kernel3/pad1, 3->32 channels and bias=False retained | Existing initialized weight retained |
| ConvNeXt-Tiny | `features.0.0`: 3->96 channels, bias=True, 4x4/stride4/pad0 becomes 3x3/stride1/pad1 | Truncated normal std=0.02; bias zero |

Guards inspect native model, stem, normalization, activation and classifier
types/settings before adaptation. No forward is used for inspection. Unaffected
weights are never reinitialized. ConvNeXt's surrounding LayerNorm2d, internal
7x7 depthwise convolutions, later downsampling, native layer scale (1e-6), and
internal linear layers remain intact. All other native structure is preserved.

Pinned native regularization includes MobileNet/EfficientNet classifier dropout
p=0.2 (inplace), EfficientNet's distributed stochastic depth using maximum
constructor probability 0.2, and ConvNeXt-Tiny's distributed stochastic depth
using maximum 0.1. Actual per-module dropout and stochastic-depth settings are
recorded by `describe_model`. ResNet has neither added dropout nor stochastic
depth. These settings follow the pinned [MobileNet source](https://github.com/pytorch/vision/blob/v0.20.1/torchvision/models/mobilenetv3.py),
[EfficientNet source](https://github.com/pytorch/vision/blob/v0.20.1/torchvision/models/efficientnet.py),
and [ConvNeXt source](https://github.com/pytorch/vision/blob/v0.20.1/torchvision/models/convnext.py).
The new convolution policies follow the pinned [ResNet source](https://github.com/pytorch/vision/blob/v0.20.1/torchvision/models/resnet.py)
and ConvNeXt source above.

## Initialization RNG ownership

`sha256-isolated-cpu-float32-v1` hashes UTF-8 JSON with sorted keys, compact
separators and ensure_ascii=True. Its exact fields are `version`,
`stream="model_initialization"`, `dataset`, `architecture`, and `training_seed`.
The first eight SHA-256 bytes interpreted little-endian give an unsigned 64-bit
CPU generator seed. Alpha, split identity, operational paths and worker settings
are absent. Identical dataset/architecture/selected-seed declarations consume
identical initialization randomness across alphas.

Construction saves the caller's CPU RNG state and default dtype, seeds a local
CPU generator, temporarily installs its state into the CPU default generator,
and constructs under a CPU device context with float32 default dtype. Native
constructors and replacement-stem initialization use this isolated stream.
The CPU state and dtype are restored in `finally`, and the device context restores
the caller's device selection. `torch.manual_seed`, accelerator seeding, and
CUDA initialization are not used. The [pinned RNG source](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/random.py)
distinguishes CPU-only get/set state from global multi-device seeding.

A module lock serializes these builders. Because CPU RNG/default settings are
process-global, the future launcher must construct serially before training;
unrelated concurrent RNG/default-setting mutations are unsupported. Initialization
does not seed future dropout, model-runtime or perturbation streams. Step 4 gives
those streams explicit ownership; Step 5 restores their states last. No
training seed fallback or three-seed list is supplied.

Model metadata records contract versions, dataset/classes, selected and derived
seeds, actual software versions, adaptations, and actual total/trainable parameter
counts. Counts use the model's parameters at future runtime. Classifier inspection
is structural only and does not assert runtime output-shape verification.

## Eligible clean weights

`conv2d-linear-owned-weights-v1` discovers every owned weight `Parameter` on
`isinstance(module, (nn.Conv2d, nn.Linear))`, sorted by qualified name. Records
include owning qualified module/type, shape, element count, requires_grad, and
the live parameter; `to_dict()` omits the live reference. There are no dimension,
size, position, naming-fragment, or architecture-list filters. Stems, projections,
depthwise/pointwise, SE, internal linear and classifier weights are included.

Biases, normalization parameters, buffers and other parameter types are excluded
from this inventory without being frozen. Duplicate parameter objects/owners,
shared backing storage, meta parameters and parametrized unowned weights are
explicitly rejected. All approved builders own independent dense parameters.
Discovery reads structure only: no statistics, noise, updates or mode changes.
Step 4's perturbation and clean layer-statistics code must reuse this helper and
retain these exact clean/master parameter objects.

## Fixed optimizers and effective native defaults

`single-group-native-recipe-v1` uses all clean trainable parameters once, sorted
by qualified name, in one group. It rejects frozen/replaced/aliased parameters
and later group/recipe changes. Move the model to its final device before
optimizer construction. Biases and normalization parameters receive the same
declared weight decay as every other parameter; there are no exclusions or
differential learning rates. No configuration field is modified.

| Architecture | Explicit constructor settings | Other pinned native defaults |
| --- | --- | --- |
| ResNet50 | SGD lr=0.1, momentum=0.9, weight_decay=0.0005 | dampening=0, nesterov=False, maximize=False, foreach=None, differentiable=False, fused=None |
| MobileNetV3-Large / EfficientNet-B0 | RMSprop lr=0.01, momentum=0.9, weight_decay=0.00001, alpha=0.9, eps=0.001 | centered=False, capturable=False, foreach=None, maximize=False, differentiable=False |
| ConvNeXt-Tiny | AdamW lr=0.001, weight_decay=0.05, betas=(0.9,0.999) | eps=1e-8, amsgrad=False, maximize=False, foreach=None, capturable=False, differentiable=False, fused=None |

The defaults are read from pinned PyTorch [SGD](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/optim/sgd.py),
[RMSprop](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/optim/rmsprop.py),
and [AdamW](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/optim/adamw.py)
source. Unspecified flags are not passed by the builder. None execution flags
permit native selection; metadata does not claim that a fused/foreach path
actually ran. RMSprop's alpha comes only from `optimizer.rmsprop_alpha`, never
`scientific.alpha`. Metadata reports effective group options separately from
scientific settings and retains the original configured base LR even after the
scheduler changes current group LR. The recipes have not been empirically
validated in this rebuilt repository.

## Schedule and call order

`completed-units-linear-cosine-v1` uses completed cadence units n and the original
base LR L. Factor f, minimum m and interval are required explicit decisions.
For epoch cadence T=200 and W=5. For optimizer_step cadence, pass the actual
`len(loaders.train)` (also `loaders.metadata.counts.training_batches`):
T=200*steps_per_epoch and W=5*steps_per_epoch. Never substitute remaining epochs
or a guessed batch count. A supplied count is a positive integer and must agree
with the fixed data/batch contract (either full-set count while drop_last is
pending, exactly the declared count when resolved). Epoch
state canonicalizes steps_per_epoch to null because it does not define its LR
trajectory; data continuation independently validates batch policy.

For 0 <= n < W:

`lr(n) = L * [f + (1-f) * n/W]`

For W <= n <= T:

`lr(n) = m + (L-m) * [1 + cos(pi * (n-W)/(T-W))]/2`

Fresh construction validates the optimizer is fresh and at original base LR,
then installs lr(0)=L*f. Zero is supported. The next training unit uses lr(n),
then `step()` after completion installs lr(n+1). The warmup boundary n=W reaches
L; completing all T units reaches m. The final training unit uses lr(T-1), while
m is the boundary value after training. Advancing beyond T is an error.

Epoch cadence holds LR constant throughout the epoch and advances once after
that completed training epoch. Optimizer-step cadence advances once after each
actual successful optimizer update. AMP-skipped updates must not call `step()`;
attempted batches and successful updates are separate counters. There is no
inference between completed epochs and updates. If skips occur, reaching epoch
200 need not reach n=T or m; do not invent extra updates or extend training to
force the endpoint. Log the actual position/LR and preserve both counters.

## Step 5 continuation order

1. Validate the checkpoint's run identity, original budget, provenance, role,
   data identities, model structure, parameter order, recipe and state availability.
   Build the clean model and place it on its final device. Rebuild loaders from
   unchanged prepared artifacts, obtaining their actual training batch count.
2. Build the optimizer on these clean parameters. Construct the scheduler with
   `initialize=False` **before** loading optimizer state. This mode does not
   change optimizer LR and cannot step or serialize a current position until
   scheduler restoration succeeds.
3. Preflight scheduler state with `validate_state_dict`. It checks the exact
   descriptor and position without changing LR; it can run before optimizer
   restoration. Step 5 separately validates saved optimizer groups/options/LRs
   and full-state compatibility before restoration or run-artifact changes.
4. Restore clean model/buffers and optimizer state. Then call scheduler
   `load_state_dict` last among these three. It rechecks descriptor and group
   compatibility, checks restored optimizer LR against lr(n), and only then
   installs the validated position and rates. It never initializes n=0 as a
   restoration side effect. Do not call `step()` to catch up a saved position.
5. Restore scaler, completed-epoch and successful-update counters, data
   continuation and separate model/noise RNG states before further iteration.
   Epoch cadence position must match completed_epochs; step cadence position
   must match successful optimizer updates, including skipped-update history.

Scheduler state identifies format/version, architecture, cadence, original
epoch/warmup budgets, relevant steps_per_epoch, original base LR list, explicit
factor/minimum and completed_units. Unknown fields, wrong types, changed horizons,
groups/options/base LRs and out-of-range positions are errors. State is independent
of experiment alpha. Scheduler compatibility is one part of full continuation;
checkpoint files, atomic persistence, recovery and run-lock use are implemented
in Step 5, as documented in the checkpoint guide.

## Remaining decisions and scope

The exact three training seeds, split preparation seed, normalization policy and
constants for each dataset, drop_last, precision/backend policy, warmup factor,
cosine minimum and scheduler cadence remain unresolved. Pending component
settings produce actionable construction errors. Aggregation's standard-deviation
convention and best-alpha tie policy remain Step 8 decisions.

Schema/fingerprint version 4 retains fixed model builder, initialization and
scheduler fields and adds training engine/runtime randomness fields. Explicit
migrations 1 -> 2 -> 3 -> 4 add contracts without resolving decisions. Existing
checksummed schema-2/3 prepared YAML is migrated in memory and never rewritten.
Use prepared dataset_config.yaml as the base,
then the appropriate model declaration and task overrides. Do not reapply a
dataset declaration containing null split/normalization over finalized values.
Operational paths and model selection do not change dataset artifact identities.

Step 4 implements functional perturbation, runtime RNG/precision setup, shared
training, clean evaluation, metrics, layer statistics and explicit failures.
It retains the original optimizer/model ownership and scheduling contract above.
Step 5 provides the concrete persistence service, full preflight/restoration and
durable initial/epoch boundaries. Step 6's
[controller](manifests-and-controller.md) now connects one selected task using these
builders, explicit fresh/continuation initialization and actual loader counts.
Step 7 now supplies source-only Slurm/signals through
[slurm-workflow.md](slurm-workflow.md). Reporting and HPC runtime checks remain later work.
