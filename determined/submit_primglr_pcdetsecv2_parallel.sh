#!/usr/bin/env bash
# Submit 3 per-dataset FULL pipeline tasks to Determined in parallel:
#   extract + segment (prim-glr) + feature (detsec_pc v2) + cluster (dpc-kmeans, k=2..8).
#
# detsec_pc v2 = direction 1A (per-channel 1%/99% percentile-clipped global
# MinMax normalization, preserving power levels) + direction 3 (Dense(embed,
# relu) sparse low-rank embedding projection). run_id embeds a submission
# timestamp precise to HH:MM:SS (YYYYMMDD_HHMMSS_primglr_pcdetsecv2_<ds>).
#
# Usage:   bash determined/submit_primglr_pcdetsecv2_parallel.sh
set -uo pipefail

SRC=/labdata2/lexingruan/pslg-nilm
RUN_TAG=primglr_pcdetsecv2
TS=$(date +%Y%m%d_%H%M%S)

DATASETS=(
  "eco"
  "refit"
  "ukdale"
)

declare -A RAW_SERIES=(
  [eco]="input/eco_washing_machine.csv"
  [refit]="input/refit_washing_machine.csv"
  [ukdale]="input/ukdale_washing_machine_full.csv"
)

echo "=== submission timestamp: $(date '+%F %T')  (TS=${TS}) ==="

for ds in "${DATASETS[@]}"; do
    run_id="${TS}_${RUN_TAG}_${ds}"
    yaml="${SRC}/determined/${run_id}.yaml"
    cat > "${yaml}" <<YAML
# Determined AI task config — ${ds} washing_machine, FULL pipeline v2:
#   prim-glr + detsec_pc (norm_mode=minmax, embed_proj=relu) + dpc-kmeans k=2..8.
# run_id=${run_id}  submitted $(date '+%F %T').
# Artifacts under log_det_test/${run_id}/.
# Push: det command run --config-file determined/${run_id}.yaml --detach
description: ${run_id}
entrypoint:
  - bash
  - /labdata2/lexingruan/pslg-nilm/determined/run_test.sh
  - prim-glr                                       # SEGMENT_METHOD
  - detsec_pc                                      # FEATURE_MODEL
  - dpc-kmeans                                     # CLUSTER_METHOD
  - "2,3,4,5,6,7,8"                                # N_CLUSTERS
  - ${run_id}                                      # RUN_ID
  - /labdata2/lexingruan/pslg-nilm/${RAW_SERIES[$ds]}   # RAW_SERIES
  - config/config_${ds}_pc_detsec_v2.yaml          # CONFIG

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

echo "=== submitted all ${#DATASETS[@]} tasks (parallel) at $(date '+%F %T') ==="
