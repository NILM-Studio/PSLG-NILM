#!/usr/bin/env bash
#SBATCH --job-name=nilm-seq-audit
#SBATCH --partition=RTX3090
#SBATCH --nodelist=h104-slurm-a
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:15:00
#SBATCH --output=runs/slurm/nilm-seq-audit-%j.log
set -euo pipefail
cd /home/scnu202438025446/pslg-nilm
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
/home/scnu202438025446/miniconda3/envs/nilm-discovery-tf-v1/bin/python \
  scripts/audit_nilm_sequence_run.py --run-id "${RUN_ID:?Set RUN_ID to the source workflow run}"
