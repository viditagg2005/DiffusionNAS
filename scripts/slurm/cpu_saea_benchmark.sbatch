#!/usr/bin/env bash
#SBATCH --job-name=saea_bench
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=23:50:00
#SBATCH --output=slurm-%j.out
# SAEA benchmark shard (configs/nas/saea_benchmark.json). CPU only, no GPU. Resumable: resubmit the same shard
# if it hits the time limit. 32 cores (per-user cap). Submit from the repo root, e.g.:
#   sbatch --export=ALL,DIFFUSIONNAS_ROOT="$PWD",SETTING=main scripts/slurm/cpu_saea_benchmark.sbatch
source "${DIFFUSIONNAS_ROOT:-${SLURM_SUBMIT_DIR}}/scripts/slurm/cpu_job.sh"
python scripts/saea_benchmark.py --setting "${SETTING:-main}" --workers "${WORKERS}" --shard "${SHARD:-0}" --num-shards "${SHARDS:-1}"
