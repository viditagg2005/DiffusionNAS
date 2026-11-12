#!/usr/bin/env bash
# C1 confirmatory test (~60-75 min on one RTX 6000 Ada). Run inside an interactive allocation:
#   srun --gres=gpu:1 --cpus-per-task=8 --mem=48G --time=02:00:00 --pty bash scripts/run_c1.sh
set -euo pipefail
REPO_ROOT="${DIFFUSIONNAS_ROOT:-$(pwd)}"
source "${REPO_ROOT}/scripts/slurm/job_env.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv
RUNS="${REPO_ROOT}/artifacts/c1/runs"
for config in $(ls "${REPO_ROOT}"/configs/generated/c1/*.json | sort); do
  label=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["label"])' "${config}")
  if python - "${RUNS}" "${label}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
hit = root.exists() and any((d / "config.json").exists() and json.load(open(d / "config.json")).get("label") == sys.argv[2]
          and json.load(open(d / "manifest.json")).get("status") == "succeeded" for d in root.glob("*"))
sys.exit(0 if hit else 1)
PY
  then echo "skip ${label}"; continue; fi
  echo "C1: ${label}"
  diffusionnas run "${config}"
done
REF=$(python - "${RUNS}" <<'PY'
import json, sys
from pathlib import Path
print(next(d for d in sorted(Path(sys.argv[1]).glob("*")) if json.load(open(d / "config.json")).get("label") == "reference"))
PY
)
python scripts/evaluate_runs.py --runs-root "${RUNS}" --reference-run "${REF}" --distribution --real-dir "${REPO_ROOT}/data/coco_real" \
  2>&1 | tee "${REPO_ROOT}/artifacts/c1/score.log" || echo "WARNING: KID/FID failed; CLIP and fidelity are in place"
echo "C1 done. Copy back artifacts/c1 (json, jsonl, npy, log)."
