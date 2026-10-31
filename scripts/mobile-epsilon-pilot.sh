#!/bin/bash
# Explicit later submission only. Resource/account/GPU options are supplied by
# the operator after current resource inspection, never inferred here.
# Submit with --signal=B:USR1@300 --no-requeue --array=0-2%3 and one task/GPU.
set -euo pipefail
: "${CANDIDATE_CHECKOUT:?Set the frozen candidate checkout}"
: "${CANDIDATE_PLAN:?Set the absolute candidate pilot.json path}"
EXISTING_ENV=/nfs/hpc/share/bellaak/research/alpha-sweep/alpha-sweep-training/.venv
source "$EXISTING_ENV/bin/activate"
cd "$CANDIDATE_CHECKOUT"
export PYTHONPATH="$CANDIDATE_CHECKOUT/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS VECLIB_MAXIMUM_THREADS
case "${SLURM_ARRAY_TASK_ID:?Require an explicit pilot array task}" in
  0|1|2) ;;
  *) exit 22 ;;
esac
exec "$EXISTING_ENV/bin/python" -B -m alpha_sweep_training.mobile_candidate run \
  --plan "$CANDIDATE_PLAN" --seed "$SLURM_ARRAY_TASK_ID" --operation "${PILOT_OPERATION:-auto}"
