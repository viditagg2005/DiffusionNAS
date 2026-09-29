#!/usr/bin/env bash
# Shared setup sourced by every SLURM job. Do not submit this file directly.

set -euo pipefail

if [[ -z "${REPO_ROOT:-}" ]]; then
  echo "REPO_ROOT must be set by the calling SLURM script" >&2
  exit 2
fi
if [[ ! -f "${REPO_ROOT}/pyproject.toml" || ! -f "${REPO_ROOT}/scripts/slurm/job_env.sh" ]]; then
  echo "Cannot find the DiffusionNAS checkout at: ${REPO_ROOT}" >&2
  echo "Submit from the repository root, or use:" >&2
  echo "  sbatch --export=ALL,DIFFUSIONNAS_ROOT=/absolute/path/to/DiffusionNAS JOB.sbatch" >&2
  exit 2
fi

REPO_ROOT="$(cd -- "${REPO_ROOT}" && pwd)"
if [[ ! -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
  echo "Missing ${REPO_ROOT}/.venv; run scripts/setup_cluster.sh first" >&2
  exit 2
fi

cd "${REPO_ROOT}"
source .venv/bin/activate

echo "DiffusionNAS checkout: ${REPO_ROOT}"
echo "Slurm job: ${SLURM_JOB_ID:-not submitted via Slurm}; node: ${SLURMD_NODENAME:-$(hostname)}"

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-${SCRATCH:-${REPO_ROOT}/.cache}/huggingface}"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "${HF_HOME}"

if [[ "${DIFFUSIONNAS_REQUIRE_GPU:-1}" == "1" ]]; then
  python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable in this job"; print(torch.cuda.get_device_name(0))'
fi

