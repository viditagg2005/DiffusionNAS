#!/usr/bin/env bash
#SBATCH --job-name=diffnas-random
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=47:30:00

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/scripts/slurm/job_env.sh"

python -m diffusionnas.search_cli random \
  --config "${REPO_ROOT}/configs/generated/search/random_search_sd15.json" \
  --output "${REPO_ROOT}/artifacts/search/random" \
  --clip-model openai/clip-vit-large-patch14 \
  --device cuda \
  --batch-size "${CLIP_BATCH_SIZE:-32}"

