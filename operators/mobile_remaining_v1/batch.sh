#!/bin/bash
set -euo pipefail
: "${OPERATOR_HOME:?Absolute frozen external operator directory required}"
: "${REMAINING_PLAN:?Absolute separately owned remaining-27 plan required}"
EXISTING_ENV=/nfs/hpc/share/bellaak/research/alpha-sweep/alpha-sweep-training/.venv
NATIVE_SOURCE=/nfs/hpc/share/bellaak/research/alpha-sweep/alpha-sweep-mobile-eps0316
source "$EXISTING_ENV/bin/activate"
cd "$NATIVE_SOURCE"
export PYTHONPATH="$NATIVE_SOURCE/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS VECLIB_MAXIMUM_THREADS
# This is a native global task index, never an actual seed or pilot selector.
exec "$EXISTING_ENV/bin/python" -B "$OPERATOR_HOME/launcher.py" run \
  --plan "$REMAINING_PLAN" --global-index "${SLURM_ARRAY_TASK_ID:?Require the explicit remaining-27 array index}"
