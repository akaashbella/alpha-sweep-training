#!/usr/bin/env bash
#SBATCH --job-name=alpha-env-smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"

: "${ALPHA_SWEEP_VENV:?Set ALPHA_SWEEP_VENV to the project virtual environment path}"
source "${ALPHA_SWEEP_VENV}/bin/activate"

python scripts/check_hpc_env.py --require-gpu

