#!/usr/bin/env bash
#SBATCH --job-name=actgen-eval
#SBATCH --partition=RTX3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=runs/slurm/actgen-eval-%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
: "${CAMPAIGN:?Set campaign directory}"
cd /home/scnu202438025446/pslg-nilm
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
exec /home/scnu202438025446/miniconda3/envs/nilmformer-cu118-v1/bin/python -m nilm_experiments.generation_lab.evaluate --campaign "$CAMPAIGN"
