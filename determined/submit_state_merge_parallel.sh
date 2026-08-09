#!/usr/bin/env bash
# Submit 3 per-dataset state-merge tasks to Determined in parallel.
#
# Each task reuses its existing run_id (20260809_primglr_detsec_<ds>, whose
# feature + kmeans k2..k8 artifacts are already on disk) and adds
# <tag>_merged tagged results for every k. CPU-only, single-threaded.
#
# Usage:   bash determined/submit_state_merge_parallel.sh
set -uo pipefail

SRC=/labdata2/lexingruan/pslg-nilm
RUN_TAG=20260810_state_merge

DATASETS=(
  "eco"
  "refit"
  "ukdale"
)

for ds in "${DATASETS[@]}"; do
    yaml="${SRC}/determined/${RUN_TAG}_${ds}.yaml"
    echo "=== submitting ${RUN_TAG}_${ds} ==="
    det command run --config-file "${yaml}" --detach
done

echo "=== submitted all ${#DATASETS[@]} state-merge tasks in parallel ==="
