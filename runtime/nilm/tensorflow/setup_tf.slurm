#!/usr/bin/env bash
#SBATCH --job-name=nilm-tf-setup
#SBATCH --partition=RTX3090
#SBATCH --nodelist=h104-slurm-a
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=runs/slurm/tf-setup-%j.log
set -euo pipefail
cd /home/scnu202438025446/pslg-nilm/nilm_experiments
CONDA_EXE=/home/scnu202438025446/miniconda3/bin/conda
if ! "$CONDA_EXE" run -n nilm-discovery-tf-v1 python --version >/dev/null 2>&1; then
  "$CONDA_EXE" create -y --offline --override-channels -c conda-forge -n nilm-discovery-tf-v1 python=3.12 pip=25.0
fi
"$CONDA_EXE" run --no-capture-output -n nilm-discovery-tf-v1 python -m pip install --timeout 120 --retries 8 -r runtime/tensorflow/requirements-tf.txt
"$CONDA_EXE" run --no-capture-output -n nilm-discovery-tf-v1 python -m pip check
"$CONDA_EXE" run -n nilm-discovery-tf-v1 python -m pip freeze > ../runs/slurm/tf-freeze-${SLURM_JOB_ID}.txt
touch ../runs/slurm/TF_ENV_READY-${SLURM_JOB_ID}

