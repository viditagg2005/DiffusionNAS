#!/usr/bin/env bash
#SBATCH --job-name=diffnas-pareto
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail

REPO_ROOT="${DIFFUSIONNAS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
DIFFUSIONNAS_REQUIRE_GPU=0
source "${REPO_ROOT}/scripts/slurm/job_env.sh"

diffusionnas pareto \
  --runs-root "${REPO_ROOT}/artifacts/calibration/runs" \
  --output "${REPO_ROOT}/artifacts/calibration/pareto_front.json"

