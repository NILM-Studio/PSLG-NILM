#!/usr/bin/env bash
#SBATCH --job-name=nilm-train
#SBATCH --partition=RTX3090
#SBATCH --nodelist=h104-slurm-a
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=nilm-train-%j.log
set -euo pipefail
model="${1:?Usage: sbatch runtime/train.slurm MODEL CONFIG APPROVAL ARM SEED WEIGHT}"
config="${2:?Absolute config path required}"
approval="${3:?Absolute approval path required}"
arm="${4:?Arm required}"
seed="${5:-0}"
weight="${6:-0.1}"
case "$model" in
  NILMFormer) env_name=nilmformer-cu118-v1 ;;
  FCN) env_name=fcn-cu118-v1 ;;
  BERT4NILM) env_name=bert4nilm-cu118-v1 ;;
  SGN) env_name=sgn-cu118-v1 ;;
  *) echo "Unknown model: $model" >&2; exit 2 ;;
esac
[[ "$config" = /* && "$approval" = /* ]] || { echo 'Use absolute config and approval paths' >&2; exit 2; }
cd /home/scnu202438025446/pslg-nilm/nilm_experiments
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
/home/scnu202438025446/miniconda3/bin/conda run --no-capture-output -n "$env_name" python -m nilm_lab.cli train --config "$config" --approval "$approval" --model "$model" --arm "$arm" --seed "$seed" --weight "$weight"
