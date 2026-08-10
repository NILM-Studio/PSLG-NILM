#!/usr/bin/env bash
# Submit R2: S3 scheduled-sampling detsec_pc FINAL validation on 3 datasets.
#
# Full pipeline: extract + segment (prim-glr) + feature (detsec_pc with
# tf_ratio=0.0 / tf_schedule=linear / patience=10) + cluster (dpc-kmeans
# k=2..8). After this round, 3 cheap cluster-reuse tasks add kmeans(n_init=30)
# results for a fully fair comparison with the detsec baseline.
#
# run_id embeds a submission timestamp precise to HH:MM:SS.
# Usage:   bash determined/submit_pcdetsec_s3_r2_parallel.sh
set -uo pipefail

SRC=/labdata2/lexingruan/pslg-nilm
RUN_TAG=pcdetsec_s3
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
# Determined AI task config — R2 final validation, ${ds} washing_machine:
#   prim-glr + detsec_pc (S3 scheduled sampling) + dpc-kmeans k=2..8.
# run_id=${run_id}  submitted $(date '+%F %T').
# Artifacts under log_det_test/${run_id}/.
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

echo "=== submitted all ${#DATASETS[@]} R2 tasks (parallel) at $(date '+%F %T') ==="
