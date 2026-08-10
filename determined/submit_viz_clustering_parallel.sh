#!/usr/bin/env bash
# Submit 3 per-dataset clustering-CHART tasks to Determined in parallel.
#
# Each task renders per-K charts (center/stacked/tSNE) from the clustering
# results of run 20260809_160925_primglr_pcdetsec_<ds>, using the project's
# visualize.visualize_clustering (--no-item-pics) inside the container.
# CPU-only, few threads. run_id embeds a submission timestamp to HH:MM:SS.
# NOTE: chart generation runs ONLY on worker nodes via det — never on the
# login node.
#
# Usage:   bash determined/submit_viz_clustering_parallel.sh
set -uo pipefail

SRC=/labdata2/lexingruan/pslg-nilm
RUN_TAG=pcdetsec_viz
TS=$(date +%Y%m%d_%H%M%S)

DATASETS=(
  "eco"
  "refit"
  "ukdale"
)

echo "=== submission timestamp: $(date '+%F %T')  (TS=${TS}) ==="

for ds in "${DATASETS[@]}"; do
    run_id="${TS}_${RUN_TAG}_${ds}"
    src_run="20260809_160925_primglr_pcdetsec_${ds}"
    yaml="${SRC}/determined/${run_id}.yaml"
    cat > "${yaml}" <<YAML
# Determined AI task config — ${ds}: per-K clustering charts from run
# ${src_run} (dpc-kmeans k=2..8 on detsec_pc features, prim-glr segments).
# CPU-only, few threads. Figures land under output/${run_id}/.
# Push: det command run --config-file determined/${run_id}.yaml --detach
description: ${run_id}
entrypoint:
  - bash
  - /labdata2/lexingruan/pslg-nilm/determined/run_viz_clustering.sh
  - ${src_run}                                    # RUN_ID (charts source)
  - config/config_${ds}_pc_detsec.yaml            # CONFIG (visualization block)

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

echo "=== submitted all ${#DATASETS[@]} chart tasks (parallel) at $(date '+%F %T') ==="
