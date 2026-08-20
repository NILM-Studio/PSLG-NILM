#!/bin/bash
set -euo pipefail

RUN_ID="${1:-ukdale_wm_primglr_detsec_3789}"
SEED="${2:-42}"
NEIGHBORS="${3:-3 5 10}"
read -r -a NEIGHBOR_VALUES <<< "$NEIGHBORS"
CONFIG="config/config_ukdale_detsec.yaml"
CLUSTER_TAG="kmeans_k4_merged"

python main.py \
  --config "$CONFIG" \
  --steps synthesize,synthesis_eval \
  --run-id "$RUN_ID" \
  --cluster-tag "$CLUSTER_TAG" \
  --synthesis-conditioning independent \
  --synthesis-seed "$SEED"

for K in "${NEIGHBOR_VALUES[@]}"; do
  python main.py \
    --config "$CONFIG" \
    --steps synthesize,synthesis_eval \
    --run-id "$RUN_ID" \
    --cluster-tag "$CLUSTER_TAG" \
    --synthesis-conditioning cycle_neighbors \
    --conditioning-neighbors "$K" \
    --synthesis-seed "$SEED"
done

python scripts/summarize_synthesis_ablation.py \
  --run-id "$RUN_ID" \
  --cluster-tag "$CLUSTER_TAG" \
  --neighbors "${NEIGHBOR_VALUES[@]}" \
  --seeds "$SEED"
