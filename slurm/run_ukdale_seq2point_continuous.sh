#!/bin/bash
#SBATCH --exclude=h103-slurm-a
#SBATCH -J ukdale_s2p_cont
#SBATCH -p RTX3090
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH -o /home/scnu2024024563/NILM/PSLG-NILM/slurm/slurm_log/ukdale_s2p_cont-%j.out
#SBATCH -e /home/scnu2024024563/NILM/PSLG-NILM/slurm/slurm_log/ukdale_s2p_cont-%j.err

set -euo pipefail

PROJECT_DIR="/home/scnu2024024563/NILM/PSLG-NILM"
RUN_ID="${RUN_ID:-ukdale_wm_primglr_detsec_3789}"
DATASET_DIR="${DATASET_DIR:-$PROJECT_DIR/log/$RUN_ID/nilm_continuous_dataset_strict_temporal_on_kmeans_k4_merged}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/log/$RUN_ID/nilm_seq2point_strict_continuous}"

export RUN_ID DATASET_DIR OUTPUT_ROOT
export TEST_STRIDE="${TEST_STRIDE:-1}"
export EXPERIMENTS="${EXPERIMENTS:-all}"
export SEED="${SEED:-42}"
export EPOCHS="${EPOCHS:-30}"

bash "$PROJECT_DIR/slurm/run_ukdale_seq2point.sh"
