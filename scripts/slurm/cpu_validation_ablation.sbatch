#!/usr/bin/env bash
#SBATCH --job-name=lhfid_val_abl
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.out
# LHFiD reimplementation check against the LHFiD paper (configs/nas/lhfid_validation.json) and the SA-LHFiD
# ablations (configs/nas/sa_ablation.json). CPU only, no GPU; both resumable. Submit from the repo root:
#   sbatch --export=ALL,DIFFUSIONNAS_ROOT="$PWD" scripts/slurm/cpu_validation_ablation.sbatch
source "${DIFFUSIONNAS_ROOT:-${SLURM_SUBMIT_DIR}}/scripts/slurm/cpu_job.sh"
python scripts/sa_ablation.py --workers "${WORKERS}"
python scripts/lhfid_validation.py --workers "${WORKERS}"
