#!/usr/bin/env bash
# Submit all 6 per-dataset clasp-origin tasks to Determined in parallel.
#
# Each task resamples its dataset to the UK-DALE 6s grid (fs=0.1666667) before
# activity-state extraction, then runs clasp-origin segmentation, and persists
# 20 segment images under output/<run_id>/figure/segments/.
#
# Usage:   bash determined/submit_clasp_origin_parallel.sh
set -uo pipefail

SRC=/labdata2/lexingruan/pslg-nilm
RUN_TAG=20260809_clasp-origin

DATASETS=(
  "eco"
  "greend"
  "iawe"
  "redd"
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
