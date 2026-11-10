#!/usr/bin/env bash
#SBATCH --job-name=diffnas-nas
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=23:50:00
#SBATCH --output=slurm-%j.out
# One optimizer seed per job: reference run, then random and NSGA-II (150 unique candidates each).
# Submit with: sbatch --export=ALL,DIFFUSIONNAS_ROOT="$PWD",NAS_SEED=1 scripts/slurm/nas_search.sbatch
set -euo pipefail

REPO_ROOT="${DIFFUSIONNAS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "${REPO_ROOT}/scripts/slurm/job_env.sh"
SEED="${NAS_SEED:?set NAS_SEED to 1, 2 or 3}"
OUT="${REPO_ROOT}/artifacts/nas/seed${SEED}"
CONFIGS="${REPO_ROOT}/configs/generated/nas"
DEADLINE=$(( $(date +%s) + ${SEARCH_MAX_WALLTIME_SECONDS:-84600} ))
nvidia-smi --query-gpu=name,memory.total --format=csv

# 1. Reference in this allocation: fidelity anchor and latency anchor for speedups.
REF_ROOT="${OUT}/reference"
REF_RUN=$(ls -d "${REF_ROOT}"/runs/*/ 2>/dev/null | head -1 || true)
if [[ -z "${REF_RUN}" || ! -f "${REF_RUN}/quality_measurements.jsonl" ]]; then
  rm -rf "${REF_ROOT}" && mkdir -p "${REF_ROOT}"
  python - "${CONFIGS}/reference.json" "${REF_ROOT}/runs" "${REF_ROOT}/reference_config.json" <<'PY'
import json, sys
config = json.load(open(sys.argv[1])); config["output_root"] = sys.argv[2]
json.dump(config, open(sys.argv[3], "w"), indent=2)
PY
  diffusionnas run "${REF_ROOT}/reference_config.json"
  REF_RUN=$(ls -d "${REF_ROOT}"/runs/*/ | head -1)
  python scripts/evaluate_runs.py --runs-root "${REF_ROOT}/runs" --reference-run "${REF_RUN}"
fi
echo "reference run: ${REF_RUN}"

# 2. Both optimizers; alternate their order by seed so slow drift does not favour one.
if (( SEED % 2 == 1 )); then ORDER=(random nsga2); else ORDER=(nsga2 random); fi
for method in "${ORDER[@]}"; do
  remaining=$(( DEADLINE - $(date +%s) ))
  status=0
  python -m diffusionnas.nas_cli --config "${CONFIGS}/seed${SEED}_${method}.json" --output "${OUT}/${method}" \
    --reference-run "${REF_RUN}" --max-walltime-seconds "$(( remaining > 0 ? remaining : 0 ))" || status=$?
  if (( status == 75 )); then echo "Paused at wall-time; resubmit the same command to resume." >&2; exit 75; fi
  (( status == 0 )) || exit "${status}"
done
echo "seed ${SEED} done"
