#!/bin/bash
set -euo pipefail
: "${EVAL_OPERATOR_HOME:?}" "${EVAL_PLAN:?}" "${EVAL_WAVE:?}" "${SLURM_ARRAY_TASK_ID:?}"
# The batch PID becomes the Python owner; B:USR1 reaches native StopRequest.
# No module/dependency changes. The worker verifies the original activation pin.
source /nfs/hpc/share/bellaak/research/alpha-sweep/alpha-sweep-training/.venv/bin/activate
export PYTHONPATH=/nfs/hpc/share/bellaak/research/alpha-sweep/alpha-sweep-mobile-eps0316/src
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS VECLIB_MAXIMUM_THREADS
exec /nfs/hpc/share/bellaak/research/alpha-sweep/alpha-sweep-training/.venv/bin/python -B \
  "$EVAL_OPERATOR_HOME/launcher.py" run --plan "$EVAL_PLAN" --wave "$EVAL_WAVE" \
  --global-index "$SLURM_ARRAY_TASK_ID"
