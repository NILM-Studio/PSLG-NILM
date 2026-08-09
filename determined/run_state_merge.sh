#!/usr/bin/env bash
# Determined task entrypoint: temporal functional restoration (state merge).
#
# Pure numpy, CPU-only, O(n) per tag — intentionally runs single-threaded.
# Reuses an EXISTING run's manifest (feature + cluster artifacts already on
# disk) and adds one "<tag>_merged" tagged result per non-merged cluster tag.
#
#   $1 SEGMENT_METHOD  $2 FEATURE_MODEL  $3 CLUSTER_METHOD
#   $4 RUN_ID          (existing run whose manifest has time_clustering results)
#   $5 CONFIG          config file under config/ (e.g. config_ukdale_detsec.yaml)
SEGMENT_METHOD="${1:-prim-glr}"
FEATURE_MODEL="${2:-detsec}"
CLUSTER_METHOD="${3:-kmeans}"
RUN_ID="${4:-test}"
CONFIG="${5:-config/config_ukdale_detsec.yaml}"
set -euo pipefail

SRC=/labdata2/lexingruan/pslg-nilm
ENV=/labdata2/lexingruan/miniconda3/envs/PSLG-NILM
PY=$ENV/bin/python
WS=/tmp/pslg_merge

echo "=== task env ==="
echo "host: $(hostname)   date: $(date -u +%FT%TZ)   uid=$(id -u)"
echo "variant: segment=$SEGMENT_METHOD feature=$FEATURE_MODEL cluster=$CLUSTER_METHOD run_id=$RUN_ID"

mkdir -p $WS
export HOME=$WS
# low thread count — the merge step is cheap and must not fight for cores
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
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
elif [ -d "$SRC/log/$RUN_ID" ]; then
    mkdir -p log
    cp -rT "$SRC/log/$RUN_ID" "log/$RUN_ID"
    echo "(copied from log/$RUN_ID)"
else
    echo "FATAL: no existing run artifacts for $RUN_ID (log_det_test/ or log/)"
    exit 1
fi

echo "=== run state_merge ==="
$PY main.py --config "$CONFIG" \
    --appliance washing_machine --run-id "$RUN_ID" \
    --steps state_merge \
    --segment-method "$SEGMENT_METHOD" --feature-model "$FEATURE_MODEL" \
    --cluster-method "$CLUSTER_METHOD"

echo "=== merge summary ==="
$PY - "$RUN_ID" <<'PYEOF'
import glob, json, os, sys
run_id = sys.argv[1]
manifest = f"log/{run_id}/run_manifest.json"
m = json.load(open(manifest))
res = m.get("steps", {}).get("time_clustering", {}).get("results", {})
merged = {t: r for t, r in res.items() if t.endswith("_merged")}
print(f"run_id={run_id}  merged tags: {len(merged)}")
for tag in sorted(merged):
    p = os.path.join("log", run_id, merged[tag]["subdir"], "metrics.json")
    if os.path.exists(p):
        met = json.load(open(p))
        print(f"  {tag}: segments={met.get('n_segments')} blocks={met.get('n_blocks')} "
              f"similar_merges={met.get('n_similar_merges')} "
              f"short_absorbed={met.get('n_short_absorbed_segments')} "
              f"merged_segment_ratio={met.get('merged_segment_ratio')}")
PYEOF

echo "=== persist artifacts back to shared FS ==="
if mkdir -p "$SRC/log_det_test" 2>/dev/null \
    && cp -rT "log/$RUN_ID" "$SRC/log_det_test/$RUN_ID" 2>/dev/null; then
    echo "(artifacts persisted to $SRC/log_det_test/$RUN_ID)"
else
    echo "(persist-to-/labdata2 blocked — manifest + metrics captured in task logs)"
fi

echo "=== DONE ==="
