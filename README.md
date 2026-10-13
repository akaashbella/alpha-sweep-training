# alpha-sweep-training

Research implementation for how **training-time Gaussian weight perturbation**
affects clean performance and independently measured fixed-network weight
robustness across CNN architectures.

Steps 1–6 implement the scientific specification, strict configuration,
scientific run identity, safe artifact initialization, shared CIFAR preparation,
reproducible full-set loaders, all four scratch CIFAR model builders, eligible
weight discovery, fixed optimizers, the original-budget warmup/cosine scheduler,
shared training, functional weight perturbation, clean evaluation, runtime RNG/
precision policy, metrics, numerical-failure handling, restricted durable
checkpoints, complete continuation and coordinated publication/recovery,
strict full-grid manifests, one-task execution and read-only coverage.
The approved post-Step-6 integration separates training completion from independent
`weight_robustness_eval_v1`: one clean and 27 perturbed full-test realizations per
best checkpoint, 6,720 logical slots across the unchanged 240 training tasks.
These implementations **have not been executed or runtime-verified**.
Slurm submission/signal orchestration and reporting remain later steps.
Production commands are implemented for later authorized HPC use only.

The main grid contains 240 research runs: CIFAR-10 then CIFAR-100, four
architectures, ten alphas, three training seeds; each run uses 200 epochs and
batch size 128. See [the authoritative scientific specification](docs/scientific-specification.md).

Read [AGENTS.md](AGENTS.md) before editing. All nine implementation steps use an
implementation-only workflow: no tests, package installation, code execution,
datasets, preliminary experiments, compilation/linting, or job submission.
Runtime verification is deferred to OSU HPC after implementation. Source review
is not evidence of runtime correctness.

The [configuration guide](docs/configuration.md) describes schema version 4,
layering, unresolved settings, fingerprints, and artifact operations.
[configs/research.yaml](configs/research.yaml) is a full-budget development
declaration with explicit pending decisions; it cannot launch a research run.
The [dataset guide](docs/datasets.md) documents the implemented
`alpha-sweep-prepare` / `python -m alpha_sweep_training.prepare` entry point,
explicit preparation seeds/normalization policies, identities, read-only loaders,
and completed-epoch reproducibility. Commands are for later HPC execution only.
PyYAML, PyTorch/TorchVision, and NumPy are declared; no packages were installed.

The [model and optimization guide](docs/models-and-optimization.md) documents
the exact stems, isolated CPU float32 initialization, complete owned-weight
inventory, native optimizer defaults, scheduler indexing, and the in-memory
continuation contract. Scheduler endpoints/cadence and the training seed plan
remain explicit decisions.

The [training and perturbation guide](docs/training-and-perturbation.md) documents
clean parameter ownership, temporary noisy forwards, BatchNorm, independent RNGs,
precision/AMP skips, full clean validation/test, finite records, and the exact
required persistence interfaces. Epoch 200 remains `training_pending_completion`
until the training-only summary and required training artifacts are durably
published. Test evaluation has a separate namespace and completion authority.

The [checkpoint and recovery guide](docs/checkpoints-and-recovery.md) documents
the concrete persistence service, durable initial boundary, authoritative HEAD,
bounded role retention, preflight/restoration, stop requests and conservative
recovery. The [progress note](docs/step5-progress.md) records source-only completion.

The [manifest/controller guide](docs/manifests-and-controller.md) documents
`alpha-sweep build`, `inspect`, `select`, `run` and `status`: finalized input
layering, exact 240-task identities/two 120-task views, safe one-task dispatch,
execution provenance, coverage, outcomes and the Step 7 handoff. No manifests
were generated. Scientific settings in the development declaration remain pending.
The [Step 6 progress note](docs/step6-progress.md) records source review only.
The [robustness evaluation guide](docs/weight-robustness-evaluation.md) is the current
lifecycle/compatibility contract, with `evaluation-plan`, `evaluation-inspect`,
`evaluate` and `evaluation-status` commands. It documents fixed full-pass noise,
independent seeds, exact whole-state preservation, resumable slot publication,
legacy handling and Step 7/8 interfaces. The
[integration progress note](docs/evaluation-integration-progress.md) records source work.

The [implementation plan](docs/implementation-plan.md) defines shared interfaces,
checkpoint/result metadata, completion criteria, and remaining Steps 7–9. The
[HPC design](docs/hpc-design.md) preserves OSU `dgxh` execution requirements.
