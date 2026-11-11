#!/usr/bin/env bash
#SBATCH --job-name=diffnas-validation-v3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
# All validation runs in ONE allocation (latency comparable), in the committed order, then scoring.
# Prerequisite on a login node: python scripts/fetch_coco_real.py  (real images for KID/FID vs real)
set -euo pipefail

REPO_ROOT="${DIFFUSIONNAS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "${REPO_ROOT}/scripts/slurm/job_env.sh"
CONFIG_DIR="${REPO_ROOT}/configs/generated/validation_v3"
RUNS="${REPO_ROOT}/artifacts/validation_v3/runs"
REAL="${REPO_ROOT}/data/coco_real"
[[ -f "${REAL}/manifest.json" ]] || { echo "Run scripts/fetch_coco_real.py on a login node first" >&2; exit 2; }
nvidia-smi --query-gpu=name,memory.total --format=csv

mapfile -t FILES < <(python -c 'import json,sys; [print(o["file"]) for o in json.load(open(sys.argv[1]))]' "${CONFIG_DIR}/order.json")
for file in "${FILES[@]}"; do diffusionnas validate "${CONFIG_DIR}/${file}"; done
for file in "${FILES[@]}"; do
  label=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["label"])' "${CONFIG_DIR}/${file}")
  done_already=$(python - "${RUNS}" "${label}" <<'PY'
import json, sys
from pathlib import Path
root, label = Path(sys.argv[1]), sys.argv[2]
hits = [d for d in root.glob("*") if (d / "config.json").exists()
        and json.load(open(d / "config.json")).get("label") == label
        and json.load(open(d / "manifest.json")).get("status") == "succeeded"] if root.exists() else []
print("yes" if hits else "no")
PY
)
  if [[ "${done_already}" == "yes" ]]; then echo "skip ${label}"; continue; fi
  echo "validation run: ${label}"
  diffusionnas run "${CONFIG_DIR}/${file}"
done

REF_A=$(python - "${RUNS}" <<'PY'
import json, sys
from pathlib import Path
print(next(d for d in sorted(Path(sys.argv[1]).glob("*")) if json.load(open(d / "config.json")).get("label") == "ref_a"))
PY
)
python scripts/evaluate_runs.py --runs-root "${RUNS}" --reference-run "${REF_A}" --distribution --real-dir "${REAL}"
echo "Validation done. Copy back artifacts/validation_v3 (json, jsonl, npy) and data/coco_real/manifest.json."
