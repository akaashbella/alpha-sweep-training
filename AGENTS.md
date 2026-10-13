# alpha-sweep-training implementation instructions

The authoritative scientific specification is [docs/scientific-specification.md](docs/scientific-specification.md).
The shared contracts and ordered remaining steps are in [docs/implementation-plan.md](docs/implementation-plan.md).
Configuration semantics and unresolved decisions are in [docs/configuration.md](docs/configuration.md).
Implemented dataset/preparation contracts are in [docs/datasets.md](docs/datasets.md).
Implemented model/optimizer/scheduler contracts are in [docs/models-and-optimization.md](docs/models-and-optimization.md).
Implemented training/perturbation/evaluation and Step 5 service contracts are in [docs/training-and-perturbation.md](docs/training-and-perturbation.md).
Implemented durable checkpoint/continuation/recovery contracts are in [docs/checkpoints-and-recovery.md](docs/checkpoints-and-recovery.md).
Implemented manifest/controller/provenance/coverage contracts are in [docs/manifests-and-controller.md](docs/manifests-and-controller.md).

## Implementation-only workflow

These instructions replace any earlier development requirement to run tests or
preliminary experiments. Inspect the working tree first, preserve existing edits,
and reuse useful source. During all nine implementation steps:

- Do not run tests, training, evaluation, benchmarks, synthetic experiments, or smoke runs.
- Do not create tests, smoke configurations, reduced-data experiments, or demonstration notebooks.
- Do not install packages, download datasets, submit HPC jobs, or execute project code.
- Do not generate dataset artifacts now. Preparation commands are implemented
  for later authorized HPC execution; production artifact checks are source work.
- Do not run compilation, linting, or executable verification commands.
- Read source, search files, inspect Git status/diffs, and edit files only.
- Existing tests/smoke files may remain inactive; do not expand or maintain them.
- Runtime verification happens on OSU HPC after all nine implementation steps.
- Source inspection does not establish runtime correctness.

Do not stop implementation over unresolved scientific decisions. Represent them
explicitly, implement actionable execution-time errors, and never promote a
provisional value into a research default. Keep declarations valid independently
of whether a model or runtime component has been implemented.

## Fixed scientific scope

Research question: how does training-time Gaussian weight perturbation strength
affect final clean classification performance across CNN architectures?

Use `W_perturbed = W_clean + alpha * population_std(W_clean) * epsilon`.
Recompute population standard deviation (`correction=0`) from each entire current
clean tensor before every mini-batch. Detach scale/noise; draw independent fresh
standard-normal noise per eligible tensor per mini-batch. Perturb all `Conv2d`
and `Linear` weights simultaneously, including stems, projections, depthwise,
pointwise, SE, and classifier weights. Do not directly perturb biases,
normalization parameters, or buffers. Unperturbed parameters learn normally;
BatchNorm running statistics update normally. Compute gradients at perturbed
weights and update clean/master weights without accumulated noise. Alpha is one
fixed global scalar per run; zero bypasses perturbation. Validate and test clean
weights.

The grid is CIFAR-10 first, then CIFAR-100; `resnet50`, `mobilenetv3_large`,
`efficientnet_b0`, `convnext_tiny`; alpha values
`0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00`; three training
seeds reused across alphas. There are 120 runs per dataset, 240 total.

Native 32×32 input; fixed 45,000/5,000 official-training split independent of
training seed and shared across architectures/alphas/seeds; full official 10,000
test images. Train transforms: crop with padding 4, horizontal flip, dataset
normalization. Validation/test transforms are deterministic. Train from scratch;
apply only the specified CIFAR stems/classifiers and preserve remaining structure.

Every run uses 200 epochs, batch size 128, cross-entropy, five-epoch linear warmup
then cosine decay, fixed architecture-specific optimizer recipes from the
specification, and native architecture regularization. No poor-performance early
stopping, MixUp, CutMix, label smoothing, gradient clipping, or EMA.

Save best-validation and final checkpoints. Highest validation accuracy selects
best; exact ties keep the earlier epoch. Test every alpha/seed's best checkpoint
with clean weights. Select best alpha using mean validation accuracy across
seeds. Prominently preserve individual results; mean/std support them. Record
epoch metrics, clean layer-weight statistics, and explicit numerical failures.

Quantization, layer-wise perturbation experiments, evaluation-time noise, extra
datasets, and fine sweeps are outside the main scope. Do not add them implicitly.

## Shared implementation rules

- Scientific schema and constants live in `src/alpha_sweep_training/spec.py`;
  strict loading/readiness checks live in `config.py`. Keep documentation aligned.
- Keep storage, allocation settings, and provenance outside scientific hashes.
- Guard every run-specific write against existing incomplete and completed runs.
  Missing `summary.json` never authorizes replacement. Writers hold `open_run`'s
  context throughout their use of the run.
- New, compatible continuation, explicit publication recovery, read-only completion
  inspection and selected overwrite are different operations. Overwrite archives old directories.
- Latest checkpoint means continuation; best means validation selection; final
  means the full training budget. Allocation end/epoch 200 is not completion:
  prescribed clean test evaluation and final result artifacts are also required.
- Implement only the requested step. Document later interfaces; do not add empty
  modules that imply future functionality already exists.
- Shared preparation uses its own dataset lock and final manifest, never
  `open_run`. Loaders must be read-only: no download, split generation,
  normalization derivation, or automatic repair. Normalization and the split
  preparation seed have no scientific defaults.
- The fixed data contract uses constant-zero padding 4, random 32×32 crops,
  horizontal flips with probability 0.5, CPU float32 CHW tensors on [0,1], and
  normalization. Evaluation has no random augmentation or resize. Versioned
  local order/augmentation streams exclude architecture/alpha and carry epoch
  in sampler keys, including for persistent workers.
- Models initialize serially in an isolated CPU float32 RNG context independent
  of alpha. Use the authoritative owned Conv2d/Linear inventory in `weights.py`.
  Move clean models to the final device before optimizer construction; preserve
  one fixed parameter group including bias/normalization weight decay.
- Scheduler position counts completed cadence units, with the full original
  horizon. AMP-skipped updates do not advance optimizer-step cadence. Restore
  with `initialize=False`, optimizer state first and scheduler state afterward;
  completed epochs and successful optimizer updates remain separate counters.
- Step 4 uses functional temporary tensors connected to original float32 clean
  parameters, detached float32 scales/noise, and original learned buffers. Alpha
  zero takes the ordinary forward and consumes no noise. Never checkpoint a
  perturbed view or reset BatchNorm for clean evaluation.
- Runtime model/noise streams exclude alpha and are seeded once or restored;
  artifact callbacks preserve them. Honor explicit CUDA/precision/backend policy.
  Managed float16 AMP skips use observed native optimizer calls, never return values.
- Step 5 implements TrainingPersistence's commit_initial, commit_epoch, load_best,
  publish_completion and publish_failure. Owned CPU snapshots are read-only;
  acknowledgements mean durable transactions. CSV/logs never determine resume
  position. Load persisted best into a separate evaluation model; never overwrite
  latest with best weights and final optimizer state.
- HEAD in the metadata root is the sole durable progress point. Immutable files
  are validated before HEAD; CSV/status/summary aliases are derived views. Protect
  current/previous references before pruning proven superseded generations.
  Preserve staging, ambiguous evidence, unrelated files and archives.
- Continue/recover requires full ContinuationPlan preflight before attempt/config
  writes and repeated under existing locks. Numerical failure remains terminal;
  committed completion dominates lost acknowledgement/diagnostic errors.
  Restore original bindings and native optimizer state, scheduler/scaler/data,
  then model/runtime/noise RNG last. Never reseed a recovered initial boundary.
- StopRequest requests finish/discard without signal registration or handler I/O.
  Stale locks require exact ownership evidence and explicit confirmation that all
  recorded owners/allocations are inactive across nodes. Never infer this from
  local PID or age. Step 6 connects the controller; Step 7 owns Slurm/signals.
- Manifest construction uses prepared dataset bases, then architecture declarations
  and typed task settings; never overlay raw pending dataset declarations. Validate
  the exact full Cartesian product and linked phase views. Keep grid identity
  separate from full-document integrity and permit execution-only task overrides.
- One-task execution uses stored-state preflight before compute, original builders,
  full ContinuationPlan and the existing lease. Auto dispatch blocks numerical and
  execution failures; explicit continue authorizes only execution failure retry.
  Publication recovery and exact selected archival overwrite remain separate.
- Coverage joins every expected identity to journal authority. Keep stored status
  separate from locks/partial/incompatible/corrupt/unstable/inspection conditions;
  never infer completion from derived summaries or an allocation ending.
  Source-only Step 6 work stops before Slurm submission, signal registration,
  requeue/resubmission and cluster concurrency orchestration.

## HPC constraints

Target OSU `dgxh`: one GPU and one training process per independent
dataset–architecture–alpha–seed run. Continue across allocations without resetting
the 200-epoch budget or LR schedule. Storage must be accessible from later nodes.
Account, resource requests, wall time, and concurrency are configurable. Planning
defaults: one GPU, eight CPUs, 32 GB host RAM, 12 hours per allocation, up to four
concurrent tasks across the workflow. These are unmeasured requests, not account
entitlements. Do not submit jobs during implementation.
