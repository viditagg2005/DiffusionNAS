#!/usr/bin/env bash
# Run every numbered config in a directory in order (skipping labels that already succeeded), then score CLIP and
# fidelity against the succeeded ref_a run; optional KID/FID vs real COCO last (its failure does not undo CLIP/fidelity).
#   bash scripts/run_config_dir.sh CONFIG_DIR RUNS_DIR [--distribution]
set -euo pipefail
REPO_ROOT="${DIFFUSIONNAS_ROOT:-$(pwd)}"
source "${REPO_ROOT}/scripts/slurm/job_env.sh"
CONFIG_DIR="$1"; RUNS="$2"; DIST="${3:-}"
nvidia-smi --query-gpu=name,memory.total --format=csv
for config in $(ls "${CONFIG_DIR}"/[0-9]*.json | sort); do
  label=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["label"])' "${config}")
  if python - "${RUNS}" "${label}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
hit = root.exists() and any((d / "config.json").exists() and (d / "manifest.json").exists()
          and json.load(open(d / "config.json")).get("label") == sys.argv[2]
          and json.load(open(d / "manifest.json")).get("status") == "succeeded" for d in root.glob("*"))
sys.exit(0 if hit else 1)
PY
  then echo "skip ${label}"; continue; fi
  echo "run: ${label}"
  diffusionnas run "${config}"
done
REF=$(python - "${RUNS}" <<'PY'
import json, sys
from pathlib import Path
print(next(d for d in sorted(Path(sys.argv[1]).glob("*")) if (d / "manifest.json").exists()
           and json.load(open(d / "config.json")).get("label") == "ref_a"
           and json.load(open(d / "manifest.json")).get("status") == "succeeded"))
PY
)
python scripts/evaluate_runs.py --runs-root "${RUNS}" --reference-run "${REF}" 2>&1 | tee "${RUNS}/../score.log"
if [[ "${DIST}" == "--distribution" ]]; then
  python scripts/evaluate_runs.py --runs-root "${RUNS}" --reference-run "${REF}" --distribution --real-dir "${REPO_ROOT}/data/coco_real" \
    2>&1 | tee "${RUNS}/../score_distribution.log" || echo "WARNING: KID/FID failed; CLIP and fidelity are in place"
fi
echo "done: ${CONFIG_DIR}"
