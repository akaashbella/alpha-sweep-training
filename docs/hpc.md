# OSU College of Engineering HPC setup

These instructions target the OSU College of Engineering Slurm cluster. The
cluster is heterogeneous, so confirm your account, partition, Python module,
and compatible CUDA-enabled PyTorch wheel before research runs. Official OSU
references:

- [Getting started](https://it.engineering.oregonstate.edu/hpc/getting-started)
- [Slurm HOWTO](https://it.engineering.oregonstate.edu/hpc/slurm-howto)
- [HPC FAQ](https://it.engineering.oregonstate.edu/hpc/faqs)

## Clone or update the repository

Connect to an OSU CoE submit host using your own ONID and clone the repository:

```bash
ssh <ONID>@submit-a.hpc.engr.oregonstate.edu
git clone <REPOSITORY_URL>
cd alpha-sweep-training
```

For an existing clone:

```bash
cd /path/to/alpha-sweep-training
git pull --ff-only
```

Do not run training directly on a submit host. Use Slurm to obtain a compute
node.

## Create and activate the Python environment

Inspect the modules actually available on the cluster and load a supported
Python version. The module name is intentionally not hard-coded:

```bash
module avail python conda
module load python/<VERSION_AVAILABLE_ON_THE_CLUSTER>
python --version
```

Choose a persistent environment location. OSU recommends HPC share storage for
user-managed environments; verify the correct location for your account:

```bash
export ALPHA_SWEEP_VENV=/path/on/hpc-share/alpha-sweep-venv
python -m venv "$ALPHA_SWEEP_VENV"
source "$ALPHA_SWEEP_VENV/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The dependency versions are pinned or bounded in `requirements.txt`. PyTorch must be a
CUDA-enabled build compatible with the allocated node's NVIDIA driver. If the
initial installation is CPU-only or incompatible, select the matching Linux
CUDA command from the [official PyTorch installer](https://pytorch.org/get-started/locally/),
reinstall `torch==2.9.1` and `torchvision==0.24.1`, and rerun the diagnostic
below. Do not proceed by silently accepting CPU fallback.

The Slurm scripts expect `ALPHA_SWEEP_VENV` to be exported in the environment
passed to `sbatch`. Add the export to your shell startup file or set it before
every submission.

## Storage locations

All storage roots can be set independently. For example:

```bash
export ALPHA_SWEEP_DATA_ROOT=/scratch/or/share/path/cifar
export ALPHA_SWEEP_CHECKPOINT_ROOT=/scratch/or/share/path/checkpoints/smoke
export ALPHA_SWEEP_RESULTS_ROOT=/scratch/or/share/path/results/smoke
export ALPHA_SWEEP_LOG_ROOT=/scratch/or/share/path/logs/smoke
```

If unset, the training smoke job uses the ignored `data`, `checkpoints/smoke`,
`results/raw/smoke`, and `logs/smoke` directories in the repository. HPC scratch
is not necessarily backed up; copy important completed artifacts to persistent
storage according to OSU policy.

## Verify a GPU interactively

Set the Slurm values assigned to your account. They are not repository
defaults:

```bash
export OSU_SLURM_ACCOUNT=<YOUR_SLURM_ACCOUNT>
export OSU_SLURM_PARTITION=<YOUR_GPU_PARTITION>
srun --account="$OSU_SLURM_ACCOUNT" \
  --partition="$OSU_SLURM_PARTITION" \
  --gres=gpu:1 --cpus-per-task=2 --mem=8G --time=00:10:00 --pty bash
```

Inside the allocated shell:

```bash
cd /path/to/alpha-sweep-training
source "$ALPHA_SWEEP_VENV/bin/activate"
nvidia-smi
python scripts/check_hpc_env.py --require-gpu
```

The diagnostic exits nonzero if CUDA is not visible to PyTorch. It reports
Python/PyTorch versions, the PyTorch CUDA runtime, devices, hostname, working
directory, and relevant Slurm variables.

## Submit the environment smoke test

Slurm opens stdout/stderr before executing the job, so create the log directory
first. The exact submission command is:

```bash
mkdir -p logs/slurm
sbatch --account="$OSU_SLURM_ACCOUNT" \
  --partition="$OSU_SLURM_PARTITION" \
  slurm/gpu_smoke_test.sbatch
```

If your permitted partition does not require an explicit account or partition,
omit the corresponding command-line option only after confirming that with
`sacctmgr`, `sinfo`, or CoE support.

## Submit the clean ResNet50 smoke test

Ensure the environment and optional storage variables are exported, then run:

```bash
mkdir -p logs/slurm
sbatch --account="$OSU_SLURM_ACCOUNT" \
  --partition="$OSU_SLURM_PARTITION" \
  slurm/resnet50_smoke_test.sbatch
```

This runs two epochs on small fixed subsets. It downloads CIFAR-10 when absent,
so pre-stage the dataset under `ALPHA_SWEEP_DATA_ROOT` if compute nodes cannot
reach the internet. It is explicitly not a research run.

## Monitor jobs and inspect logs

```bash
squeue -u "$USER"
sacct -j <JOB_ID> --format=JobID,JobName,Partition,State,ExitCode,Elapsed
tail -f logs/slurm/alpha-env-smoke-<JOB_ID>.out
tail -f logs/slurm/alpha-resnet-smoke-<JOB_ID>.out
```

Slurm stdout/stderr appear under `logs/slurm`. The training script additionally
writes a run-specific `train.log` under `ALPHA_SWEEP_LOG_ROOT`.

Successful clean smoke output includes:

```text
$ALPHA_SWEEP_CHECKPOINT_ROOT/infrastructure-smoke__cifar10__resnet50__seed-0/
  best.pt
  final.pt

$ALPHA_SWEEP_RESULTS_ROOT/infrastructure-smoke__cifar10__resnet50__seed-0/
  resolved_config.yaml
  epochs.csv
  layer_statistics.csv
  summary.json

$ALPHA_SWEEP_LOG_ROOT/infrastructure-smoke__cifar10__resnet50__seed-0/
  train.log
```

Repeating a completed run fails rather than overwriting it. Use the training
CLI's `--overwrite` option only when intentionally replacing smoke-test output;
the supplied Slurm script does not enable it.
