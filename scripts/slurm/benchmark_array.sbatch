#!/usr/bin/env bash
#SBATCH --job-name=diffnas-benchmark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

REPO_ROOT="${DIFFUSIONNAS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "${REPO_ROOT}/scripts/slurm/job_env.sh"

CONFIGS=(
  "${REPO_ROOT}/configs/generated/calibration/baseline_sd15.json"
  "${REPO_ROOT}/configs/generated/calibration/autodiffusion_sd15.json"
  "${REPO_ROOT}/configs/generated/calibration/deepcache_sd15.json"
)

nvidia-smi
for config in "${CONFIGS[@]}"; do
  [[ -f "${config}" ]] || { echo "Missing calibration config: ${config}" >&2; exit 2; }
  diffusionnas validate "${config}"
done

# Keep all three timings on one allocated GPU; array tasks may land on different models.
for config in "${CONFIGS[@]}"; do
  echo "Starting calibration benchmark: ${config}"
  diffusionnas run "${config}"
done
echo "All three calibration benchmarks completed."

