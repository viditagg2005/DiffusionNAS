#!/usr/bin/env bash
#SBATCH --job-name=c4_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=07:00:00
#SBATCH --output=slurm-%j.out
# C4 test: every NFS and naive set on test3 in one allocation (~3-4 h). Submit from the repo root after c4_select.py:
#   sbatch --export=ALL,DIFFUSIONNAS_ROOT="$PWD" scripts/slurm/c4_test.sbatch
set -euo pipefail
REPO_ROOT="${DIFFUSIONNAS_ROOT:-${SLURM_SUBMIT_DIR}}"
export DIFFUSIONNAS_ROOT="${REPO_ROOT}"
bash "${REPO_ROOT}/scripts/run_config_dir.sh" "${REPO_ROOT}/configs/generated/c4/test" "${REPO_ROOT}/artifacts/c4/test/runs" --distribution
