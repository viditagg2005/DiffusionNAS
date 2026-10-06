#!/usr/bin/env bash
#SBATCH --job-name=diffnas-ecad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=23:50:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

REPO_ROOT="${DIFFUSIONNAS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "${REPO_ROOT}/scripts/slurm/job_env.sh"

python -m diffusionnas.search_cli ecad \
  --config "${REPO_ROOT}/configs/generated/search/ecad_sd15.json" \
  --output "${REPO_ROOT}/artifacts/search/ecad" \
  --clip-model openai/clip-vit-large-patch14 \
  --device cuda \
  --batch-size "${CLIP_BATCH_SIZE:-8}" \
  --max-walltime-seconds "${SEARCH_MAX_WALLTIME_SECONDS:-82800}"
