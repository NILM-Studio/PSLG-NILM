#!/bin/bash
#SBATCH -J PSLG-FEAT-MODELS
#SBATCH -p RTX3090
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -o /home/scnu2024024563/NILM/PSLG-NILM/slurm/slurm_log/job-%x-%j.out
#SBATCH -e /home/scnu2024024563/NILM/PSLG-NILM/slurm/slurm_log/job-%x-%j.err
#SBATCH --time=72:00:00

# 特征模型网格：同一切分方法下遍历所有特征提取模型。
# 特征提取有内容寻址缓存：重复提交同一 (模型, 超参, 输入) 组合会直接命中，
# 只有真正变化的部分才重训。所有选择走 CLI 参数，不再用 sed 改 config。
source "$(dirname "$0")/env.sh"

APPLIANCE="${APPLIANCE:-fridge}"
SEGMENT_METHOD="${SEGMENT_METHOD:-clasp}"
MODELS=("detsec" "bilstm_ae" "bilstm_ae_attention" "autoencoder")
if [ -n "${MODELS_ENV:-}" ]; then read -r -a MODELS <<< "$MODELS_ENV"; fi

for model in "${MODELS[@]}"; do
    echo "=================================================="
    echo "[grid] feature_model=$model  segment_method=$SEGMENT_METHOD"
    echo "=================================================="
    python main.py \
        --appliance "$APPLIANCE" \
        --steps extract,segment,feature \
        --segment-method "$SEGMENT_METHOD" \
        --feature-model "$model"
done

echo "Job finished on: $(date)"
