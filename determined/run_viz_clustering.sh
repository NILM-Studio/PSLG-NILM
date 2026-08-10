#!/usr/bin/env bash
# Determined task entrypoint: per-K clustering charts for one dataset's run.
#
# CPU-only, intentionally light: renders center/stacked/tSNE per candidate K
# (dpc_kmeans_k2..k8) via the project's documented visualize.visualize_clustering
# (serial, --no-item-pics), and pins OMP/MKL/OPENBLAS to 2 threads.
# Runs ONLY inside the Determined container on a worker node — never on the
# login node (chart generation stays off the login node by design).
#
#   $1 RUN_ID   existing run whose manifest has time_clustering results
#   $2 CONFIG   config file under config/ (only the visualization block is read)
RUN_ID="${1:-test}"
CONFIG="${2:-config/config_ukdale_pc_detsec.yaml}"
set -euo pipefail

SRC=/labdata2/lexingruan/pslg-nilm
ENV=/labdata2/lexingruan/miniconda3/envs/PSLG-NILM
PY=$ENV/bin/python
WS=/tmp/pslg_viz

echo "=== task env ==="
echo "host: $(hostname)   date: $(date '+%F %T')   uid=$(id -u)"
echo "variant: run_id=$RUN_ID  config=$CONFIG"

mkdir -p $WS
export HOME=$WS
# few threads on purpose — this task does not need high performance
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMBA_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=2
export MPLBACKEND=Agg

echo "=== stage code into writable workspace ==="
cp -r $SRC/src $SRC/models $SRC/config $SRC/scripts $SRC/visualize \
    $SRC/main.py $SRC/requirements.txt $WS/
cd $WS

echo "=== reuse existing run artifacts ($RUN_ID) ==="
if [ -d "$SRC/log_det_test/$RUN_ID" ]; then
    mkdir -p log
    cp -rT "$SRC/log_det_test/$RUN_ID" "log/$RUN_ID"
    echo "(copied from log_det_test/$RUN_ID)"
elif [ -d "$SRC/log/$RUN_ID" ]; then
    mkdir -p log
    cp -rT "$SRC/log/$RUN_ID" "log/$RUN_ID"
    echo "(copied from log/$RUN_ID)"
else
    echo "FATAL: no existing run artifacts for $RUN_ID (log_det_test/ or log/)"
    exit 1
fi

echo "=== render per-K clustering charts (serial, few threads) ==="
$PY -m visualize.visualize_clustering --run-id "$RUN_ID" \
    --config "$CONFIG" --no-item-pics

echo "=== generated figures ==="
find "output/$RUN_ID" -name "*.png" | sort

echo "=== persist figures back to shared FS ==="
if mkdir -p "$SRC/output" 2>/dev/null && cp -rT "output/$RUN_ID" "$SRC/output/$RUN_ID" 2>/dev/null; then
    echo "(figures persisted to $SRC/output/$RUN_ID)"
else
    echo "(persist-to-/labdata2 blocked — figure list captured in task logs)"
fi

echo "=== DONE ==="
