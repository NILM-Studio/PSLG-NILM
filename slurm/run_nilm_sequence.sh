#!/usr/bin/env bash
#SBATCH --job-name=nilm-sequence
#SBATCH --partition=RTX3090
#SBATCH --nodelist=h104-slurm-a
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=runs/slurm/nilm-sequence-%j.log
set -euo pipefail
cd /home/scnu202438025446/pslg-nilm
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TF_FORCE_GPU_ALLOW_GROWTH=true
TF_ENV=/home/scnu202438025446/miniconda3/envs/nilm-discovery-tf-v1
TORCH_ENV=/home/scnu202438025446/miniconda3/envs/nilmformer-cu118-v1
CONFIG="${CONFIG:-config/config_nilm_sequence_pilot.yaml}"
RUN_ID="${RUN_ID:-nilm_sequence_${SLURM_JOB_ID}}"

# Freeze NILM source data through the standalone Torch-side module.
export PYTHONPATH="$PWD:$PWD/nilm_experiments"
"$TORCH_ENV/bin/python" -m nilm_lab.standalone data \
  --config "$CONFIG" --run-id "$RUN_ID"

# The main process now owns discovery only; it consumes the frozen NILM data.
"$TF_ENV/bin/python" -u main.py --config "$CONFIG" --profile nilm --run-id "$RUN_ID" \
  --steps extract,segment,feature,cluster,state_merge,state_sequence

# NILM numeric stages are independent Slurm module calls; no main.py re-entry.
for COMMAND in labels train select report; do
  "$TORCH_ENV/bin/python" -m nilm_lab.standalone "$COMMAND" \
    --config "$CONFIG" --run-id "$RUN_ID"
done
# Held-out evaluation remains explicit:
# sbatch --export=ALL,COMMAND=evaluate,CONFIG="$CONFIG",RUN_ID="$RUN_ID" slurm/run_nilm_module.sh
