#!/usr/bin/env bash
# Submit 3 per-dataset FULL pipeline tasks to Determined in parallel:
#   extract + segment (prim-glr) + feature (detsec_pc) + cluster (dpc-kmeans, k=2..8).
#
# detsec_pc  : physical-constraint DeTSEC (masked attention + gated fusion +
#              teacher-forcing Softplus nonnegative decoder + Charbonnier TV), TF port.
# dpc-kmeans : density-peak-initialized K-Means.
#
# The run_id embeds a submission timestamp precise to HH:MM:SS
# (YYYYMMDD_HHMMSS_primglr_pcdetsec_<ds>), so every task is uniquely traceable.
# Tasks run on the 4090 worker pool (never on the login node). Per-dataset yaml
# files are written under determined/ for the record, then pushed with --detach.
#
# Usage:   bash determined/submit_primglr_pcdetsec_parallel.sh
set -uo pipefail

SRC=/labdata2/lexingruan/pslg-nilm
RUN_TAG=primglr_pcdetsec
TS=$(date +%Y%m%d_%H%M%S)

DATASETS=(
  "eco"
  "refit"
  "ukdale"
)

declare -A RAW_SERIES=(
  [eco]="input/eco_washing_machine.csv        (threshold=20W)"
  [refit]="input/refit_washing_machine.csv     (threshold=5W)"
  [ukdale]="input/ukdale_washing_machine_full.csv  (threshold=10W)"
)

echo "=== submission timestamp: $(date '+%F %T')  (TS=${TS}) ==="

for ds in "${DATASETS[@]}"; do
    run_id="${TS}_${RUN_TAG}_${ds}"
    yaml="${SRC}/determined/${run_id}.yaml"
    cat > "${yaml}" <<YAML
# Determined AI task config — ${ds} washing_machine, FULL pipeline:
#   extract+segment (prim-glr) + feature (detsec_pc) + cluster (dpc-kmeans k=2..8).
# run_id=${run_id}  submitted $(date '+%F %T') — timestamp precise to HH:MM:SS.
# All datasets resampled to the UK-DALE 6s grid (fs=0.1666667).
# Artifacts under log_det_test/${run_id}/.
# Push: det command run --config-file determined/${run_id}.yaml --detach
description: ${run_id}
entrypoint:
  - bash
  - /labdata2/lexingruan/pslg-nilm/determined/run_test.sh
  - prim-glr                                       # SEGMENT_METHOD
  - detsec_pc                                      # FEATURE_MODEL
  - dpc-kmeans                                     # CLUSTER_METHOD
  - "2,3,4,5,6,7,8"                                # N_CLUSTERS (each k gets a tagged result)
  - ${run_id}                                      # RUN_ID
  - /labdata2/lexingruan/pslg-nilm/${RAW_SERIES[$ds]%% *}     # RAW_SERIES
  - config/config_${ds}_pc_detsec.yaml             # CONFIG

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
