#!/usr/bin/env bash
set -euo pipefail
prefix="${SUPERMIMIC_ENV_PREFIX:-${HOME}/miniconda3/envs/supermimic-official}"
if [[ ! -x "$prefix/bin/python" || ! -x "$prefix/bin/uv" ]]; then
  echo "Missing environment: $prefix. Run bash reproduction/bootstrap.sh first." >&2
  exit 1
fi
exec env CONDA_PREFIX="$prefix" \
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}" \
  TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-8}" \
  MUJOCO_GL="${MUJOCO_GL:-egl}" WANDB_MODE="${WANDB_MODE:-offline}" \
  WANDB_SILENT="${WANDB_SILENT:-true}" \
  "$prefix/bin/uv" run --no-project --active "$@"
