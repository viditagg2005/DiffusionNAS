#!/usr/bin/env bash
# Shared setup for CPU-only jobs (no GPU requested): repo root, .venv-saea, one BLAS thread per worker.
set -euo pipefail
REPO_ROOT="${DIFFUSIONNAS_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "${REPO_ROOT}"
if [[ ! -f .venv-saea/bin/activate ]]; then
  echo "Missing ${REPO_ROOT}/.venv-saea; run: bash scripts/setup_saea_env.sh" >&2
  exit 2
fi
source .venv-saea/bin/activate
export PYTHONPATH="${REPO_ROOT}/src" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
WORKERS="${SLURM_CPUS_PER_TASK:-$(nproc)}"
echo "checkout ${REPO_ROOT}; job ${SLURM_JOB_ID:-none}; node ${SLURMD_NODENAME:-$(hostname)}; workers ${WORKERS}; $(lscpu | grep 'Model name' | sed 's/  */ /g')"
