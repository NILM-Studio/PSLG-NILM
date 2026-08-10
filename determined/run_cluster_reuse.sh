#!/usr/bin/env bash
# Determined task entrypoint: re-cluster an EXISTING run's features with a
# different clustering method (no re-training). Used for clustering-algorithm
# fairness (S1): run kmeans(n_init=30) on the detsec_pc v2 features.
#
#   $1 RUN_ID           existing run whose manifest has feature_extract.features
#   $2 CLUSTER_METHOD   e.g. kmeans (default)
#   $3 CONFIG           config file under config/ (time_clustering block)
#   $4 N_CLUSTERS       candidate ks, e.g. "2,3,4,5,6,7,8"
RUN_ID="${1:-test}"
CLUSTER_METHOD="${2:-kmeans}"
CONFIG="${3:-config/config_ukdale_pc_detsec_v2.yaml}"
N_CLUSTERS="${4:-2,3,4,5,6,7,8}"
set -euo pipefail

SRC=/labdata2/lexingruan/pslg-nilm
ENV=/labdata2/lexingruan/miniconda3/envs/PSLG-NILM
PY=$ENV/bin/python
WS=/tmp/pslg_recluster

echo "=== task env ==="
echo "host: $(hostname)   date: $(date '+%F %T')   uid=$(id -u)"
echo "variant: run_id=$RUN_ID cluster=$CLUSTER_METHOD config=$CONFIG k=$N_CLUSTERS"

mkdir -p $WS
export HOME=$WS
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMBA_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=2

echo "=== stage code into writable workspace ==="
cp -r $SRC/src $SRC/models $SRC/config $SRC/scripts $SRC/main.py $SRC/requirements.txt $WS/
cd $WS

echo "=== reuse existing run artifacts ($RUN_ID) ==="
if [ -d "$SRC/log_det_test/$RUN_ID" ]; then
    mkdir -p log
    cp -rT "$SRC/log_det_test/$RUN_ID" "log/$RUN_ID"
    echo "(copied from log_det_test/$RUN_ID)"
else
    echo "FATAL: no existing run artifacts for $RUN_ID (log_det_test/)"
    exit 1
fi

echo "=== re-cluster features with $CLUSTER_METHOD (no training) ==="
$PY main.py --config "$CONFIG" \
    --appliance washing_machine --run-id "$RUN_ID" \
    --steps cluster \
    --segment-method prim-glr --feature-model detsec_pc \
    --cluster-method "$CLUSTER_METHOD" --n-clusters "$N_CLUSTERS"

echo "=== added cluster tags ==="
$PY - "$RUN_ID" <<'PYEOF'
import json, os, sys
run = sys.argv[1]
m = json.load(open(f"log/{run}/run_manifest.json"))
res = m.get("steps", {}).get("time_clustering", {}).get("results", {})
for tag in sorted(res):
    met_p = os.path.join("log", run, res[tag]["subdir"], "metrics.json")
    if os.path.exists(met_p):
        met = json.load(open(met_p))
        print(f"  {tag}: SCI={met.get('silhouette_score')} DBI={met.get('davies_bouldin_score')}")
PYEOF

echo "=== persist back to shared FS ==="
if mkdir -p "$SRC/log_det_test" 2>/dev/null && cp -rT "log/$RUN_ID" "$SRC/log_det_test/$RUN_ID" 2>/dev/null; then
    echo "(artifacts persisted to $SRC/log_det_test/$RUN_ID)"
else
    echo "(persist blocked — results captured in task logs)"
fi

echo "=== DONE ==="
