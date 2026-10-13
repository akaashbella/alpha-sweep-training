# alpha-sweep-training

Research implementation for how **training-time Gaussian weight perturbation**
affects final **clean classification performance** across CNN architectures.

Steps 1–2 implement the scientific specification, strict configuration,
scientific run identity, safe artifact initialization, shared CIFAR preparation,
and reproducible full-set loaders. Dataset implementation **has not been executed
or runtime-verified**. Model builders, training, checkpoint resume, sweep
execution, Slurm submission, and reporting are not yet implemented. There is no
training entry point.

The main grid contains 240 research runs: CIFAR-10 then CIFAR-100, four
architectures, ten alphas, three training seeds; each run uses 200 epochs and
batch size 128. See [the authoritative scientific specification](docs/scientific-specification.md).

Read [AGENTS.md](AGENTS.md) before editing. All nine implementation steps use an
implementation-only workflow: no tests, package installation, code execution,
datasets, preliminary experiments, compilation/linting, or job submission.
Runtime verification is deferred to OSU HPC after implementation. Source review
is not evidence of runtime correctness.

The [configuration guide](docs/configuration.md) describes schema version 2,
layering, unresolved settings, fingerprints, and artifact operations.
[configs/research.yaml](configs/research.yaml) is a full-budget development
declaration with explicit pending decisions; it cannot launch a research run.
The [dataset guide](docs/datasets.md) documents the implemented
`alpha-sweep-prepare` / `python -m alpha_sweep_training.prepare` entry point,
explicit preparation seeds/normalization policies, identities, read-only loaders,
and completed-epoch reproducibility. Commands are for later HPC execution only.
PyYAML, PyTorch/TorchVision, and NumPy are declared; no packages were installed.

The [implementation plan](docs/implementation-plan.md) defines shared interfaces,
checkpoint/result metadata, completion criteria, and Steps 2–9. The
[HPC design](docs/hpc-design.md) preserves OSU `dgxh` execution requirements.
