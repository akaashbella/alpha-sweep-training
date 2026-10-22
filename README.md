# alpha-sweep-training

Research implementation for how **training-time Gaussian weight perturbation**
affects clean performance and independently measured fixed-network weight
robustness across CNN architectures.

Steps 1–9 implement the scientific specification, strict configuration,
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
Research computation **has not been runtime-verified**. Later focused operational
queue-policy checks do not establish HPC/research runtime correctness.
Step 7 adds production Slurm planning/submission, signals and durable workflow
tracking in source. Step 8 adds authoritative results, complete prescribed summaries,
validation-only alpha selection and headless scientific figures. Step 9 completes
source integration and the [end-to-end operating guide](docs/operating-guide.md),
including unresolved-input and future-verification handoffs. Stop here: completing
source does not authorize environment setup, verification or execution. Those
require a separate explicit instruction.
Production commands are implemented for later authorized HPC use only.

The [results/figures guide](docs/results-and-figures.md) documents `alpha-sweep-report`
collect/summarize/plot/inspect, frozen 240-task/6720-slot coverage, sample SD,
co-winners, paired comparisons, output ownership and the optional reporting extra.
See [Step 9 progress](docs/step9-progress.md) for the final static evidence matrix
and corrections; earlier progress notes remain historical.

The main grid contains 240 research runs: CIFAR-10 then CIFAR-100, four
architectures, ten alphas, three training seeds; each run uses 200 epochs and
batch size 128. See [the authoritative scientific specification](docs/scientific-specification.md).

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
checkpoint/result metadata, completion criteria, and all nine completed source steps. The
[HPC design](docs/hpc-design.md) preserves OSU `dgxh` execution requirements.
The [Slurm operating guide](docs/slurm-workflow.md) documents strict separate
[operational declarations](configs/workflow.yaml), `alpha-sweep-workflow` planning,
rendering, explicit submission, status/reconciliation/advancement and recovery.
One shared registry limits both lifecycles to a single bounded wave; immutable
mapping/intent, exact scheduler IDs and scientific authority remain separate.
The [queued-array update](docs/queued-slurm-arrays.md) supports 40 queued tasks
with a separate four-task running throttle, reviewed submission/array limits,
unchanged legacy-wave reading and explicit safe transfer/migration to HPC. The
active HPC checkout and its private email/scheduler changes are not modified here.
The [Step 7 progress note](docs/step7-progress.md) records source review only.
