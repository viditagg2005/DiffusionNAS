#!/usr/bin/env bash
# Real calibration evaluation of the surrogate-benchmark confirmation points (~30-40 min, RTX 6000 Ada).
#   srun --gres=gpu:1 --cpus-per-task=8 --mem=48G --time=02:00:00 --pty bash scripts/run_confirmation.sh
set -euo pipefail
REPO_ROOT="${DIFFUSIONNAS_ROOT:-$(pwd)}"
source "${REPO_ROOT}/scripts/slurm/job_env.sh"
CONFIG_DIR="${REPO_ROOT}/configs/generated/confirmation"
RUNS="${REPO_ROOT}/artifacts/confirmation/runs"
nvidia-smi --query-gpu=name,memory.total --format=csv
for config in $(ls "${CONFIG_DIR}"/*.json | sort); do
  label=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["label"])' "${config}")
  if python - "${RUNS}" "${label}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
hit = root.exists() and any(json.load(open(d / "config.json")).get("label") == sys.argv[2]
          and json.load(open(d / "manifest.json")).get("status") == "succeeded" for d in root.glob("*") if (d / "config.json").exists())
sys.exit(0 if hit else 1)
PY
  then echo "skip ${label}"; continue; fi
  echo "confirmation: ${label}"
  diffusionnas run "${config}"
done
REF=$(python - "${RUNS}" <<'PY'
import json, sys
from pathlib import Path
print(next(d for d in sorted(Path(sys.argv[1]).glob("*")) if json.load(open(d / "config.json")).get("label") == "reference"))
PY
)
python scripts/evaluate_runs.py --runs-root "${RUNS}" --reference-run "${REF}"
echo "Done. Copy back artifacts/confirmation (json, jsonl)."
