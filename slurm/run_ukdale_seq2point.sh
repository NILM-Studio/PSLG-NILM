#!/bin/bash
#SBATCH --exclude=h103-slurm-a
#SBATCH -J ukdale_s2p
#SBATCH -p RTX3090
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH -o /home/scnu2024024563/NILM/PSLG-NILM/slurm/slurm_log/ukdale_s2p-%j.out
#SBATCH -e /home/scnu2024024563/NILM/PSLG-NILM/slurm/slurm_log/ukdale_s2p-%j.err

set -euo pipefail

source "/home/scnu2024024563/NILM/PSLG-NILM/slurm/env.sh"

RUN_ID="${RUN_ID:-ukdale_wm_primglr_detsec_3789}"
SEED="${SEED:-42}"
EXPERIMENTS="${EXPERIMENTS:-all}"
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
FORCE="${FORCE:-0}"

EXTRA_ARGS=()
if [[ -n "$OUTPUT_ROOT" ]]; then
  EXTRA_ARGS+=(--output-root "$OUTPUT_ROOT")
fi
if [[ "$FORCE" == "1" ]]; then
  EXTRA_ARGS+=(--force)
fi

python -u -m scripts.train_nilm_seq2point \
  --run-id "$RUN_ID" \
  --experiments "$EXPERIMENTS" \
  --seed "$SEED" \
  --epochs "$EPOCHS" \
  --patience "$PATIENCE" \
  "${EXTRA_ARGS[@]}"
