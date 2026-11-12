#!/usr/bin/env bash
# Recompute KID/FID for the validation and post-hoc BK-SDM runs (~15-20 min; cached Inception features are reused).
#   srun --gres=gpu:1 --cpus-per-task=8 --mem=48G --time=01:00:00 --pty bash scripts/rescore_distribution.sh
set -euo pipefail
REPO_ROOT="${DIFFUSIONNAS_ROOT:-$(pwd)}"
source "${REPO_ROOT}/scripts/slurm/job_env.sh"
python -c "import scipy; print('scipy', scipy.__version__)"
ref_for() {
  python - "$1" "$2" <<'PY'
import json, sys
from pathlib import Path
print(next(d for d in sorted(Path(sys.argv[1]).glob("*")) if (d / "config.json").exists()
           and json.load(open(d / "config.json")).get("label") == sys.argv[2]))
PY
}
for pair in "validation_v3:ref_a" "posthoc_bksdm:posthoc_ref"; do
  dir="${pair%%:*}"; ref="${pair##*:}"; runs="${REPO_ROOT}/artifacts/${dir}/runs"
  echo "== ${dir}"
  python scripts/evaluate_runs.py --runs-root "${runs}" --reference-run "$(ref_for "${runs}" "${ref}")" \
    --distribution --real-dir "${REPO_ROOT}/data/coco_real" 2>&1 | tee "${REPO_ROOT}/artifacts/${dir}/rescore.log"
done
echo "Done. Copy back artifacts/{validation_v3,posthoc_bksdm} (json, jsonl, npy, log)."
