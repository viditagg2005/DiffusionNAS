#!/usr/bin/env bash
#SBATCH --job-name=diffnas-controls
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

REPO_ROOT="${DIFFUSIONNAS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "${REPO_ROOT}/scripts/slurm/job_env.sh"

CONFIG_DIR="${CONTROLS_CONFIG_DIR:-${REPO_ROOT}/configs/generated/controls}"
mapfile -t CONFIGS < <(find "${CONFIG_DIR}" -maxdepth 1 -name '*.json' | sort)
(( ${#CONFIGS[@]} > 0 )) || { echo "No control configs in ${CONFIG_DIR}; run scripts/prepare_controls.py" >&2; exit 2; }

nvidia-smi
for config in "${CONFIGS[@]}"; do
  diffusionnas validate "${config}"
done

# One allocation for every policy, including re-measured references, so latency is comparable.
for config in "${CONFIGS[@]}"; do
  echo "Starting control benchmark: ${config}"
  diffusionnas run "${config}"
done

diffusionnas score \
  --runs-root "${REPO_ROOT}/artifacts/controls/runs" \
  --clip-model openai/clip-vit-large-patch14 \
  --device cuda \
  --batch-size "${CLIP_BATCH_SIZE:-8}"
diffusionnas pareto \
  --runs-root "${REPO_ROOT}/artifacts/controls/runs" \
  --output "${REPO_ROOT}/artifacts/controls/pareto_front.json"
echo "Control benchmark, scoring and Pareto construction completed."
