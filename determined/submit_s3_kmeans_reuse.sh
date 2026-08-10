#!/usr/bin/env bash
# Add kmeans(n_init=30) tagged results to the R2 S3 runs (no re-training) for a
# fully fair comparison with the detsec(kmeans) baseline.
# Usage: bash determined/submit_s3_kmeans_reuse.sh eco refit ukdale ...
set -uo pipefail
SRC=/labdata2/lexingruan/pslg-nilm
TS=$(date +%Y%m%d_%H%M%S)
for ds in "$@"; do
    src_run="20260810_120812_pcdetsec_s3_${ds}"
    run_id="${TS}_s3_kmeans_${ds}"
    yaml="${SRC}/determined/${run_id}.yaml"
    cat > "${yaml}" <<YAML
# R2 fairness: kmeans(n_init=30) on the S3 detsec_pc features (run ${src_run}).
# No training. CPU-light. Adds kmeans_k2..k8 tags to the same run manifest.
description: ${run_id}
entrypoint:
  - bash
  - /labdata2/lexingruan/pslg-nilm/determined/run_cluster_reuse.sh
  - ${src_run}                                     # RUN_ID (reuse features)
  - kmeans                                        # CLUSTER_METHOD
  - config/config_${ds}_pcdetsec_s3.yaml          # CONFIG
  - "2,3,4,5,6,7,8"                               # N_CLUSTERS

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
echo "=== submitted kmeans-reuse for: $* ==="
