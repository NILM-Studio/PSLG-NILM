#!/usr/bin/env bash
# Per-K clustering charts (k=2..8) for the R2 S3 runs on eco & refit.
# Renders BOTH tag sets: dpc_kmeans_k* and kmeans_k* (fair-review material).
# CPU-only, few threads; runs on the det worker node.
set -uo pipefail
SRC=/labdata2/lexingruan/pslg-nilm
TS=$(date +%Y%m%d_%H%M%S)
for ds in eco refit; do
    src_run="20260810_120812_pcdetsec_s3_${ds}"
    run_id="${TS}_viz_s3_${ds}"
    yaml="${SRC}/determined/${run_id}.yaml"
    cat > "${yaml}" <<YAML
# Per-K clustering charts for R2-S3 ${ds} (run ${src_run}), all tags.
description: ${run_id}
entrypoint:
  - bash
  - /labdata2/lexingruan/pslg-nilm/determined/run_viz_clustering.sh
  - ${src_run}                                     # RUN_ID
  - config/config_${ds}_pcdetsec_s3.yaml           # CONFIG

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
done
echo "=== submitted viz tasks for eco, refit at $(date '+%F %T') ==="
