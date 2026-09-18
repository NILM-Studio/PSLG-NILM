#!/usr/bin/env bash
#SBATCH --job-name=nilm-seq-verify
#SBATCH --partition=RTX3090
#SBATCH --nodelist=h104-slurm-a
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=runs/slurm/nilm-seq-verify-%j.log
set -euo pipefail
cd /home/scnu202438025446/pslg-nilm
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NILM_TORCH_PYTHON=/home/scnu202438025446/miniconda3/envs/nilmformer-cu118-v1/bin/python
PY=/home/scnu202438025446/miniconda3/envs/nilm-discovery-tf-v1/bin/python
"$PY" -m unittest discover -s tests -p test_nilm_workflow_integration.py -v
"$PY" -m unittest discover -s tests -p test_nilm_sequence_labels.py -v
"$PY" -m unittest discover -s tests -p test_state_merge.py -v
"$PY" -m unittest discover -s tests -p test_m1_framework.py -v
