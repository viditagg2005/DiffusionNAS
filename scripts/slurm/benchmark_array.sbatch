#!/usr/bin/env bash
#SBATCH --job-name=diffnas-benchmark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-2%1

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/scripts/slurm/job_env.sh"

CONFIGS=(
  "${REPO_ROOT}/configs/generated/calibration/baseline_sd15.json"
  "${REPO_ROOT}/configs/generated/calibration/autodiffusion_sd15.json"
  "${REPO_ROOT}/configs/generated/calibration/deepcache_sd15.json"
)

nvidia-smi
diffusionnas run "${CONFIGS[${SLURM_ARRAY_TASK_ID}]}"

