# Alpha Sweep Training

Infrastructure and experiment code for studying architecture-specific
sensitivity to training-time Gaussian weight perturbations. The authoritative
scientific specification is [`AGENTS.md`](AGENTS.md).

Current implementation status:

- deterministic CIFAR-10/CIFAR-100 data infrastructure;
- clean CIFAR-adapted ResNet50 baseline;
- clean training, validation, checkpointing, and test evaluation;
- OSU CoE HPC diagnostics and Slurm smoke jobs;
- no Gaussian weight perturbation or alpha sweep yet.

See [`docs/hpc.md`](docs/hpc.md) for environment setup and smoke-test commands.

Run ordinary unit tests from the repository root:

```bash
python -m unittest discover -s tests -v
```

The configuration in `configs/base.yaml` retains the 200-epoch research
recipe but deliberately contains unresolved seed and normalization values.
`configs/smoke/cifar10_resnet50.yaml` is an infrastructure-only configuration
and must not be treated as research data.
