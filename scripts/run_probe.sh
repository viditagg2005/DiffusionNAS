#!/usr/bin/env bash
# Single-unit sensitivity probe: ~45-60 min on one RTX 6000 Ada. Run inside an interactive allocation:
#   srun --gres=gpu:1 --cpus-per-task=8 --mem=48G --time=02:00:00 --pty bash scripts/run_probe.sh
set -euo pipefail

REPO_ROOT="${DIFFUSIONNAS_ROOT:-$(pwd)}"
source "${REPO_ROOT}/scripts/slurm/job_env.sh"

CONFIG_DIR="${REPO_ROOT}/configs/generated/probe"
mapfile -t CONFIGS < <(find "${CONFIG_DIR}" -maxdepth 1 -name '*.json' | sort)
nvidia-smi --query-gpu=name,memory.total --format=csv
for config in "${CONFIGS[@]}"; do diffusionnas validate "${config}"; done

# Skip configs that already have a succeeded run, so an interrupted probe can be resumed.
for config in "${CONFIGS[@]}"; do
  label=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["label"])' "${config}")
  if grep -ls "\"label\": \"${label}\"" "${REPO_ROOT}"/artifacts/probe/runs/*/config.json 2>/dev/null \
      | xargs -r -n1 dirname | xargs -r -I{} grep -l '"status": "succeeded"' {}/manifest.json | grep -q .; then
    echo "skip ${label} (already succeeded)"; continue
  fi
  echo "probe: ${label}"
  diffusionnas run "${config}"
done

python scripts/evaluate_runs.py --runs-root "${REPO_ROOT}/artifacts/probe/runs"
python scripts/summarize_probe.py --runs-root "${REPO_ROOT}/artifacts/probe/runs" --output "${REPO_ROOT}/artifacts/probe/probe_summary.json"
echo "Probe done. Copy back: artifacts/probe/probe_summary.json and artifacts/probe/probe_summary.csv"
