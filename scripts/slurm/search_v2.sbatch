#!/usr/bin/env bash
#SBATCH --job-name=diffnas-search-v2
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

CONFIG_DIR="${SEARCH_V2_CONFIG_DIR:-${REPO_ROOT}/configs/generated/search_v2}"
mapfile -t CONFIGS < <(find "${CONFIG_DIR}" -maxdepth 1 -name '*.json' | sort)
(( ${#CONFIGS[@]} > 0 )) || { echo "No search configs in ${CONFIG_DIR}; run scripts/prepare_search_v2.py" >&2; exit 2; }

nvidia-smi
# One shared deadline for all searches; completed searches replay from their checkpoints on resubmission.
DEADLINE=$(( $(date +%s) + ${SEARCH_MAX_WALLTIME_SECONDS:-82800} ))
for config in "${CONFIGS[@]}"; do
  output_root=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_root"])' "${config}")
  remaining=$(( DEADLINE - $(date +%s) ))
  echo "Starting search: ${config} -> ${output_root} (${remaining}s left)"
  status=0
  python -m diffusionnas.search_cli run "${config}" \
    --output "${REPO_ROOT}/${output_root}" \
    --clip-model openai/clip-vit-large-patch14 \
    --device cuda \
    --batch-size "${CLIP_BATCH_SIZE:-8}" \
    --max-walltime-seconds "$(( remaining > 0 ? remaining : 0 ))" || status=$?
  if (( status == 75 )); then
    echo "Paused at the wall-time budget; resubmit this script to resume." >&2
    exit 75
  fi
  (( status == 0 )) || exit "${status}"
done
echo "All v2 searches completed."
