#!/usr/bin/env bash
#SBATCH --job-name=diffnas-holdout
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/scripts/slurm/job_env.sh"

HOLDOUT_PROMPTS="${HOLDOUT_PROMPTS:-${REPO_ROOT}/data/prompts/holdout.json}"
HOLDOUT_SEEDS="${HOLDOUT_SEEDS:-101,202}"

diffusionnas holdout \
  --front "${REPO_ROOT}/artifacts/calibration/pareto_front.json" \
  --prompts "${HOLDOUT_PROMPTS}" \
  --seeds "${HOLDOUT_SEEDS}" \
  --output "${REPO_ROOT}/artifacts/holdout" \
  --clip-model openai/clip-vit-large-patch14 \
  --device cuda \
  --batch-size "${CLIP_BATCH_SIZE:-32}"

