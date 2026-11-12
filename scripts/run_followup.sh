#!/usr/bin/env bash
# Follow-up GPU session (~60-90 min on one RTX 6000 Ada). Run inside an interactive allocation:
#   srun --gres=gpu:1 --cpus-per-task=8 --mem=48G --time=02:00:00 --pty bash scripts/run_followup.sh
# 1. KID/FID for the validation runs (skipped per run if already present),
# 2. real calibration runs of the surrogate-benchmark confirmation points,
# 3. post-hoc BK-SDM controls (10-step DDIM, fp16 and int8) on the validation split.
# Every step is resumable: rerun the same command after an interruption.
set -euo pipefail
REPO_ROOT="${DIFFUSIONNAS_ROOT:-$(pwd)}"
source "${REPO_ROOT}/scripts/slurm/job_env.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv

label_done() {  # $1 runs dir, $2 label -> exit 0 if a succeeded run with that label exists
  python - "$1" "$2" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
hit = root.exists() and any((d / "config.json").exists() and json.load(open(d / "config.json")).get("label") == sys.argv[2]
          and json.load(open(d / "manifest.json")).get("status") == "succeeded" for d in root.glob("*"))
sys.exit(0 if hit else 1)
PY
}
run_dir_for() {  # $1 runs dir, $2 label -> path of the succeeded run
  python - "$1" "$2" <<'PY'
import json, sys
from pathlib import Path
print(next(d for d in sorted(Path(sys.argv[1]).glob("*")) if (d / "config.json").exists()
           and json.load(open(d / "config.json")).get("label") == sys.argv[2]))
PY
}
run_configs() {  # $1 config dir, $2 runs dir
  for config in $(ls "$1"/*.json | sort); do
    label=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["label"])' "${config}")
    if label_done "$2" "${label}"; then echo "skip ${label}"; continue; fi
    echo "run ${label}"
    diffusionnas run "${config}"
  done
}

echo "== 1/3 validation KID/FID"
VAL="${REPO_ROOT}/artifacts/validation_v3/runs"
python scripts/evaluate_runs.py --runs-root "${VAL}" --reference-run "$(run_dir_for "${VAL}" ref_a)" \
  --distribution --real-dir "${REPO_ROOT}/data/coco_real" 2>&1 | tee "${REPO_ROOT}/artifacts/validation_v3/rescore.log"

echo "== 2/3 confirmation runs (calibration)"
CONF="${REPO_ROOT}/artifacts/confirmation/runs"
run_configs "${REPO_ROOT}/configs/generated/confirmation" "${CONF}"
python scripts/evaluate_runs.py --runs-root "${CONF}" --reference-run "$(run_dir_for "${CONF}" reference)"

echo "== 3/3 post-hoc BK-SDM controls (validation)"
POST="${REPO_ROOT}/artifacts/posthoc_bksdm/runs"
run_configs "${REPO_ROOT}/configs/generated/posthoc_bksdm" "${POST}"
python scripts/evaluate_runs.py --runs-root "${POST}" --reference-run "$(run_dir_for "${POST}" posthoc_ref)" \
  --distribution --real-dir "${REPO_ROOT}/data/coco_real"
echo "Follow-up done. Copy back artifacts/{validation_v3,confirmation,posthoc_bksdm} (json, jsonl, npy, log)."
