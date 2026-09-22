#!/usr/bin/env bash
# Shared setup sourced by every SLURM job. Do not submit this file directly.

set -euo pipefail

if [[ -z "${REPO_ROOT:-}" ]]; then
  echo "REPO_ROOT must be set by the calling SLURM script" >&2
  exit 2
fi
if [[ ! -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
  echo "Missing ${REPO_ROOT}/.venv; run scripts/setup_cluster.sh first" >&2
  exit 2
fi

cd "${REPO_ROOT}"
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-${SCRATCH:-${REPO_ROOT}/.cache}/huggingface}"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "${HF_HOME}"

python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable in this job"; print(torch.cuda.get_device_name(0))'

