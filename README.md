# alpha-sweep-training

Research implementation for how **training-time Gaussian weight perturbation**
affects final **clean classification performance** across CNN architectures.

Step 1 establishes the scientific specification, strict configuration,
scientific run identity, safe artifact initialization, and contracts for later
steps. The repository initially contained no source files or commits. Dataset
preparation, model builders, training, resume, sweep execution, Slurm submission,
and reporting are not yet implemented. There is no training entry point.

The main grid contains 240 research runs: CIFAR-10 then CIFAR-100, four
architectures, ten alphas, three training seeds; each run uses 200 epochs and
batch size 128. See [the authoritative scientific specification](docs/scientific-specification.md).

Read [AGENTS.md](AGENTS.md) before editing. All nine implementation steps use an
implementation-only workflow: no tests, package installation, code execution,
datasets, preliminary experiments, compilation/linting, or job submission.
Runtime verification is deferred to OSU HPC after implementation. Source review
is not evidence of runtime correctness.

The [configuration guide](docs/configuration.md) describes schema version 1,
layering, unresolved settings, fingerprints, and artifact operations.
[configs/research.yaml](configs/research.yaml) is a full-budget development
declaration with explicit pending decisions; it cannot launch a research run.
PyYAML is declared for future YAML I/O; no packages were installed. Future
compute dependencies will be declared alongside their implementation.

The [implementation plan](docs/implementation-plan.md) defines shared interfaces,
checkpoint/result metadata, completion criteria, and Steps 2–9. The
[HPC design](docs/hpc-design.md) preserves OSU `dgxh` execution requirements.
