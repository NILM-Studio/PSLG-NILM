#!/usr/bin/env bash
# Submit the R1 ablation matrix to Determined in parallel (all on ukdale).
#
#   S1    re-cluster the v2 features with kmeans(n_init=30)  [no training, cheap]
#   S2    softplus zero-point offset calibration (softplus(x)-ln2)
#   S3    scheduled sampling (teacher-forcing keep ratio 1.0 -> 0.0 linear)
#   S4a   lambda_phy 0.1 -> 0.01
#   S4b   embed_dim 32 -> 16
#
# Each training variant is a FULL pipeline (prim-glr + detsec_pc + dpc-kmeans
# k=2..8) on ukdale; run_id embeds a timestamp precise to HH:MM:SS.
#
# Usage:   bash determined/submit_ablation_parallel.sh
set -uo pipefail

SRC=/labdata2/lexingruan/pslg-nilm
RUN_TAG=abl
TS=$(date +%Y%m%d_%H%M%S)
RAW=/labdata2/lexingruan/pslg-nilm/input/ukdale_washing_machine_full.csv
V2_RUN=20260810_061444_primglr_pcdetsecv2_ukdale

echo "=== submission timestamp: $(date '+%F %T')  (TS=${TS}) ==="

# S1: clustering-only reuse task (no training)
run_id="${TS}_${RUN_TAG}_s1_kmeans_v2feat_ukdale"
yaml="${SRC}/determined/${run_id}.yaml"
cat > "${yaml}" <<YAML
# R1-S1: clustering fairness — kmeans(n_init=30) on the existing detsec_pc v2
# ukdale features (run ${V2_RUN}). No training. CPU-light.
description: ${run_id}
entrypoint:
  - bash
  - /labdata2/lexingruan/pslg-nilm/determined/run_cluster_reuse.sh
  - ${V2_RUN}                                     # RUN_ID (reuse features)
  - kmeans                                        # CLUSTER_METHOD
  - config/config_ukdale_pc_detsec_v2.yaml        # CONFIG
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

# S2..S4b: full pipeline variants
for v in s2_softplus_offset s3_tf_sampling s4a_lambda01 s4b_dim16; do
    run_id="${TS}_${RUN_TAG}_${v}_ukdale"
    yaml="${SRC}/determined/${run_id}.yaml"
    cat > "${yaml}" <<YAML
# R1 ablation ${v} on ukdale: prim-glr + detsec_pc + dpc-kmeans k=2..8.
# Full pipeline, GPU. run_id=${run_id}.
description: ${run_id}
entrypoint:
  - bash
  - /labdata2/lexingruan/pslg-nilm/determined/run_test.sh
  - prim-glr                                       # SEGMENT_METHOD
  - detsec_pc                                      # FEATURE_MODEL
  - dpc-kmeans                                     # CLUSTER_METHOD
  - "2,3,4,5,6,7,8"                                # N_CLUSTERS
  - ${run_id}                                      # RUN_ID
  - ${RAW}                                         # RAW_SERIES
  - config/config_ukdale_abl_${v}.yaml             # CONFIG

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

echo "=== submitted 5 ablation tasks (parallel) at $(date '+%F %T') ==="
