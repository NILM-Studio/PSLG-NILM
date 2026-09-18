#!/usr/bin/env bash
#SBATCH --job-name=actgen-dict
#SBATCH --partition=RTX3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=runs/slurm/actgen-dict-%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
: "${CAMPAIGN:?Set campaign directory}"
cd /home/scnu202438025446/pslg-nilm
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TF_FORCE_GPU_ALLOW_GROWTH=true
TF_ENV=/home/scnu202438025446/miniconda3/envs/nilm-discovery-tf-v1
for lib in "$TF_ENV"/lib/python3.12/site-packages/nvidia/*/lib; do
  export LD_LIBRARY_PATH="$lib:${LD_LIBRARY_PATH:-}"
done
exec "$TF_ENV/bin/python" -m nilm_experiments.generation_lab.dictionary --campaign "$CAMPAIGN" --gap "${GAP_SECONDS:-18}"
