#!/usr/bin/env bash
# C4 stage B: re-measure the screened configurations on calibration2 (~1-1.5 h on one RTX 6000 Ada). Interactive:
#   srun --gres=gpu:1 --cpus-per-task=8 --mem=48G --time=02:30:00 --pty bash scripts/run_c4_remeasure.sh
set -euo pipefail
REPO_ROOT="${DIFFUSIONNAS_ROOT:-$(pwd)}"
exec bash "${REPO_ROOT}/scripts/run_config_dir.sh" "${REPO_ROOT}/configs/generated/c4/remeasure" "${REPO_ROOT}/artifacts/c4/remeasure/runs"
