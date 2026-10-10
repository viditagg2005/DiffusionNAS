#!/usr/bin/env bash
#SBATCH --job-name=diffnas-validation
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

FINALISTS="${REPO_ROOT}/artifacts/finalists/finalists.json"
# Lock finalists once; never re-select after validation results exist.
if [[ ! -f "${FINALISTS}" ]]; then
  python scripts/select_finalists.py \
    --runs-root "${REPO_ROOT}/artifacts/controls/runs" \
    --runs-root "${REPO_ROOT}/artifacts/search_v2" \
    --output "${FINALISTS}"
fi

diffusionnas holdout \
  --front "${FINALISTS}" \
  --prompts "${REPO_ROOT}/data/prompts/validation.json" \
  --seeds "${VALIDATION_SEEDS:-303,404}" \
  --warmup-runs 10 \
  --output "${REPO_ROOT}/artifacts/validation" \
  --clip-model openai/clip-vit-large-patch14 \
  --device cuda \
  --batch-size "${CLIP_BATCH_SIZE:-8}"
