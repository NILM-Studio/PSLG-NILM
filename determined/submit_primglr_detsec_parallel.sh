#!/usr/bin/env bash
# Submit 3 per-dataset FULL pipeline tasks to Determined in parallel:
#   extract+segment (prim-glr) + feature (detsec) + cluster (kmeans, k=2..8).
#
# Each task resamples its dataset to the UK-DALE 6s grid (fs=0.1666667) before
# activity-state extraction, segments with prim-glr, extracts features with
# detsec (batch_size=8 on 4090), and clusters with kmeans for all candidate k.
# Artifacts persist under log_det_test/<run_id>/ (manifest + features + clusters).
#
# Usage:   bash determined/submit_primglr_detsec_parallel.sh
set -uo pipefail

SRC=/labdata2/lexingruan/pslg-nilm
RUN_TAG=20260809_primglr_detsec

DATASETS=(
  "eco"
  "refit"
  "ukdale"
)

for ds in "${DATASETS[@]}"; do
    run_id="${RUN_TAG}_${ds}"
    yaml="${SRC}/determined/${run_id}.yaml"
    echo "=== submitting ${run_id} ==="
    det command run --config-file "${yaml}" --detach
done

echo "=== submitted all ${#DATASETS[@]} tasks in parallel ==="
