#!/usr/bin/env bash
#SBATCH --job-name=actgen-train
#SBATCH --partition=RTX3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=runs/slurm/actgen-train-%A_%a.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
: "${SLURM_ARRAY_TASK_ID:?Submit as array}"
: "${CAMPAIGN:?Set campaign directory}"
: "${PHASE:?Set phase}"
cd /home/scnu202438025446/pslg-nilm
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
exec /home/scnu202438025446/miniconda3/envs/nilmformer-cu118-v1/bin/python -m nilm_experiments.generation_lab.train --campaign "$CAMPAIGN" --gap "${GAP_SECONDS:-18}" --phase "$PHASE" --index "$SLURM_ARRAY_TASK_ID"
