#!/usr/bin/env bash
#SBATCH --job-name=actgen-sample
#SBATCH --partition=RTX3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=runs/slurm/actgen-sample-%A_%a.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
: "${SLURM_ARRAY_TASK_ID:?Submit as array}"
: "${CAMPAIGN:?Set campaign directory}"
cd /home/scnu202438025446/pslg-nilm
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
exec /home/scnu202438025446/miniconda3/envs/nilmformer-cu118-v1/bin/python -m nilm_experiments.generation_lab.sample --campaign "$CAMPAIGN" --index "$SLURM_ARRAY_TASK_ID"
