#!/usr/bin/env bash
#SBATCH --job-name=nilm-env-check
#SBATCH --partition=RTX3090
#SBATCH --nodelist=h104-slurm-a
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=nilm-runtime-gpu-%j.log
set -euo pipefail
cd /home/scnu202438025446/pslg-nilm/nilm_experiments
nvidia-smi
for entry in NILMFormer:nilmformer FCN:fcn BERT4NILM:bert4nilm SGN:sgn; do
  model="${entry%%:*}"
  case "$model" in
    NILMFormer) model_dir=third_party/nilmformer ;;
    BERT4NILM) model_dir=third_party/bert4nilm ;;
    SGN) model_dir=third_party/nilmtk_contrib ;;
    FCN) model_dir=FCN ;;
  esac
  env_name="${entry##*:}-cu118-v1"
  /home/scnu202438025446/miniconda3/bin/conda run --no-capture-output -n "$env_name" python runtime/verify_runtime.py --model "$model" --device cuda | tee "$model_dir/runtime/verification/gpu.json"
done
