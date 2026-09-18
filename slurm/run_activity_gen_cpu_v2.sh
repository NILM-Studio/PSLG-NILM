#!/usr/bin/env bash
#SBATCH --job-name=actgen-gate
#SBATCH --partition=RTX3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=runs/slurm/actgen-gate-%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
: "${CAMPAIGN:?Set campaign directory}"
: "${STAGE:?Set stage}"
cd /home/scnu202438025446/pslg-nilm
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
PY=/home/scnu202438025446/miniconda3/envs/nilmformer-cu118-v1/bin/python
if [[ "$STAGE" == plan ]]; then
  exec "$PY" -m nilm_experiments.generation_lab.plan --campaign "$CAMPAIGN" --gap "${GAP_SECONDS:-18}"
else
  exec "$PY" -m nilm_experiments.generation_lab.gate --campaign "$CAMPAIGN" --gap "${GAP_SECONDS:-18}" --stage "$STAGE"
fi
