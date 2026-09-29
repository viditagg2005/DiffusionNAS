#!/usr/bin/env bash
#SBATCH --job-name=diffnas-score
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=04:00:00

set -euo pipefail

REPO_ROOT="${DIFFUSIONNAS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "${REPO_ROOT}/scripts/slurm/job_env.sh"

diffusionnas score \
  --runs-root "${REPO_ROOT}/artifacts/calibration/runs" \
  --clip-model openai/clip-vit-large-patch14 \
  --device cuda \
  --batch-size "${CLIP_BATCH_SIZE:-32}"

