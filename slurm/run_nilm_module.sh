#!/usr/bin/env bash
#SBATCH --job-name=nilm-module
#SBATCH --partition=RTX3090
#SBATCH --nodelist=h104-slurm-a
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=runs/slurm/nilm-module-%x-%j.log
set -euo pipefail

cd /home/scnu202438025446/pslg-nilm
: "${COMMAND:?set COMMAND=data|labels|train|select|evaluate|report}"
CONFIG="${CONFIG:-config/config_nilm_sequence_pilot.yaml}"
RUN_ID="${RUN_ID:?set RUN_ID to an existing or new run id}"
TORCH_ENV="${TORCH_ENV:-/home/scnu202438025446/miniconda3/envs/nilmformer-cu118-v1}"

export PYTHONPATH="$PWD:$PWD/nilm_experiments"
exec "$TORCH_ENV/bin/python" -m nilm_lab.standalone "$COMMAND" \
  --config "$CONFIG" --run-id "$RUN_ID"
