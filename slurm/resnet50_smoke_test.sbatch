#!/usr/bin/env bash
#SBATCH --job-name=alpha-resnet-smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"

: "${ALPHA_SWEEP_VENV:?Set ALPHA_SWEEP_VENV to the project virtual environment path}"
source "${ALPHA_SWEEP_VENV}/bin/activate"

DATA_ROOT="${ALPHA_SWEEP_DATA_ROOT:-${PROJECT_ROOT}/data}"
CHECKPOINT_ROOT="${ALPHA_SWEEP_CHECKPOINT_ROOT:-${PROJECT_ROOT}/checkpoints/smoke}"
RESULTS_ROOT="${ALPHA_SWEEP_RESULTS_ROOT:-${PROJECT_ROOT}/results/raw/smoke}"
TRAIN_LOG_ROOT="${ALPHA_SWEEP_LOG_ROOT:-${PROJECT_ROOT}/logs/smoke}"

mkdir -p "${DATA_ROOT}" "${CHECKPOINT_ROOT}" "${RESULTS_ROOT}" "${TRAIN_LOG_ROOT}"

python scripts/check_hpc_env.py --require-gpu
python scripts/train.py \
  --config configs/smoke/cifar10_resnet50.yaml \
  --require-cuda \
  --data-root "${DATA_ROOT}" \
  --checkpoint-root "${CHECKPOINT_ROOT}" \
  --results-root "${RESULTS_ROOT}" \
  --logs-root "${TRAIN_LOG_ROOT}"

