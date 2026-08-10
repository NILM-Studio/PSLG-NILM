#!/usr/bin/env bash
# Determined task entrypoint: controlled comparison detsec vs detsec_pc
# features x kmeans vs dpc-kmeans clustering (runs on worker node only).
#
# CPU-only, light: reads feature matrices of the two completed runs from the
# shared FS (via a symlink), runs KMeans(n_init=30) and DPC-init K-Means on
# both feature sets for k=2..8, collects PCA/loss diagnostics, and persists
# compare_summary.json under output/<OUT_TAG>/ on the shared FS.
#
#   $1 OUT_TAG  output folder name (e.g. 20260810_053000_compare_detsec_vs_pcdetsec)
OUT_TAG="${1:-compare_detsec_vs_pcdetsec}"
set -euo pipefail

SRC=/labdata2/lexingruan/pslg-nilm
ENV=/labdata2/lexingruan/miniconda3/envs/PSLG-NILM
PY=$ENV/bin/python
WS=/tmp/pslg_cmp

echo "=== task env ==="
echo "host: $(hostname)   date: $(date '+%F %T')   uid=$(id -u)"
echo "variant: out_tag=$OUT_TAG"

mkdir -p $WS
export HOME=$WS
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMBA_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=2

echo "=== stage code into writable workspace ==="
cp -r $SRC/src $SRC/models $SRC/config $SRC/scripts $SRC/main.py $SRC/requirements.txt $WS/
ln -sfn $SRC/log_det_test $WS/log_det_test   # feature matrices live on the shared FS
cd $WS

echo "=== run controlled comparison ==="
$PY scripts/compare_detsec_vs_pcdetsec.py --out-dir "output/$OUT_TAG" \
    --k 2,3,4,5,6,7,8

echo "=== persist summary to shared FS ==="
if mkdir -p "$SRC/output" 2>/dev/null && cp -rT "output/$OUT_TAG" "$SRC/output/$OUT_TAG" 2>/dev/null; then
    echo "(summary persisted to $SRC/output/$OUT_TAG/compare_summary.json)"
else
    echo "(persist blocked — summary captured in task logs)"
fi

echo "=== DONE ==="
