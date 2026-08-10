#!/usr/bin/env bash
# Submit the detsec-vs-detsec_pc controlled comparison to Determined (worker node).
# CPU-only, one light task; run_id embeds a timestamp precise to HH:MM:SS.
# Usage:   bash determined/submit_compare_parallel.sh
set -uo pipefail

SRC=/labdata2/lexingruan/pslg-nilm
RUN_TAG=compare_detsec_vs_pcdetsec
TS=$(date +%Y%m%d_%H%M%S)
run_id="${TS}_${RUN_TAG}"
yaml="${SRC}/determined/${run_id}.yaml"

echo "=== submission timestamp: $(date '+%F %T')  (TS=${TS}) ==="

cat > "${yaml}" <<YAML
# Determined AI task config — controlled comparison: detsec(kmeans) vs
# detsec_pc(dpc-kmeans) clustering quality, disentangling the feature-
# extraction factor from the clustering-algorithm factor.
# CPU-only, few threads. Output: output/${run_id}/compare_summary.json.
# Push: det command run --config-file determined/${run_id}.yaml --detach
description: ${run_id}
entrypoint:
  - bash
  - /labdata2/lexingruan/pslg-nilm/determined/run_compare_feats.sh
  - ${run_id}                                     # OUT_TAG

environment:
  image: harbor.lins.lab/determinedai/environments:cuda-11.8-pytorch-2.0-gpu-mpi-0.31.1

resources:
  slots: 1
  shm_size: 4000000000
  resource_pool: 64c128t_512_4090

bind_mounts:
  - host_path: /labdata2
    container_path: /labdata2
    read_only: false
YAML

echo "=== submitting ${run_id} ==="
det command run --config-file "${yaml}" --detach
echo "=== submitted at $(date '+%F %T') ==="
