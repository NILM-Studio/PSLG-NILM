#!/usr/bin/env bash
#SBATCH --job-name=actgen-tests
#SBATCH --partition=RTX3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=runs/slurm/actgen-tests-%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
cd /home/scnu202438025446/pslg-nilm
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
exec /home/scnu202438025446/miniconda3/envs/nilmformer-cu118-v1/bin/python -m nilm_experiments.generation_lab.models
