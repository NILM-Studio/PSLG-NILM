#!/usr/bin/env bash
# Determined task entrypoint: extract + segment (clasp-origin / clasp / ...) for ONE dataset.
#
# Runs INSIDE the task container. Same staging strategy as run_test.sh:
#   - /labdata2 is bind-mounted RW (shared NFS) -> conda env + dataset h5/CSV.
#   - code is copied from the mount into a writable /tmp/pslg_seg at start.
#   - the shared PSLG-NILM conda env's python is exec'd directly.
#
# clasp-origin is pure-CPU (ClaSP, n_jobs=1); no GPU self-check is done.
#
#   $1 RUN_ID           e.g. 20260808_clasp-origin_eco
#   $2 RAW_SERIES       absolute CSV path of the appliance power series
#   $3 APPLIANCE        appliance display name (default washing_machine)
#   $4 CONFIG           config file under config/ (default config/config.yaml)
#   $5 SEGMENT_METHOD   clasp-origin (default) | clasp | fluss | espresso | none
RUN_ID="${1:?RUN_ID required}"
RAW_SERIES="${2:?RAW_SERIES required}"
APPLIANCE="${3:-washing_machine}"
CONFIG="${4:-config/config.yaml}"
SEGMENT_METHOD="${5:-clasp-origin}"
set -euo pipefail

SRC=/labdata2/lexingruan/pslg-nilm
ENV=/labdata2/lexingruan/miniconda3/envs/PSLG-NILM
PY=$ENV/bin/python
WS=/tmp/pslg_seg

echo "=== task env ==="
echo "host: $(hostname)   date: $(date -u +%FT%TZ)   uid=$(id -u) user=$(id -un)"
echo "run_id=$RUN_ID  raw=$RAW_SERIES  appliance=$APPLIANCE  config=$CONFIG  segment=$SEGMENT_METHOD"

mkdir -p $WS
export HOME=$WS
export PYTHONUNBUFFERED=1
export NUMBA_DISABLE_CUDA=1          # stumpy/numba CPU-only (numba.cuda segfaults on GPU nodes)
export NUMBA_THREADING_LAYER=workqueue
if [ "$SEGMENT_METHOD" = "clasp" ]; then
    # clasp runs BinaryClaSP with clasp_n_jobs=-1 (all cores); NUMBA_NUM_THREADS=1
    # would collide with n_jobs=-1 ("The number of threads must be between 1 and 1").
    export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-$(nproc)}"
    echo "clasp mode: NUMBA_NUM_THREADS=$NUMBA_NUM_THREADS"
else
    export NUMBA_NUM_THREADS=1        # clasp-origin / fluss / espresso run single-threaded
fi

echo "=== stage code into writable workspace ==="
cp -r $SRC/src $SRC/models $SRC/config $SRC/scripts $SRC/visualize $SRC/main.py $WS/
cd $WS

echo "=== check raw series ==="
$PY -c "import pandas as pd; df=pd.read_csv('$RAW_SERIES'); print(f'raw series: {len(df):,} rows  cols={list(df.columns)}')"

echo "=== run extract + segment ($SEGMENT_METHOD) ==="
$PY main.py --config "$CONFIG" \
    --appliance "$APPLIANCE" --run-id "$RUN_ID" \
    --steps extract,segment \
    --raw-series "$RAW_SERIES" \
    --segment-method "$SEGMENT_METHOD"

echo "=== manifest ==="
$PY - "$RUN_ID" <<'PYEOF'
import json, os, sys
run_id = sys.argv[1]
manifest = f"log/{run_id}/run_manifest.json"
m = json.load(open(manifest))
print("run_id:", m.get("run_id"), "| appliance:", m.get("appliance"))
print("variants:", m.get("variants", {}))
steps = m.get("steps", {})
for step_type, entry in steps.items():
    arts = list((entry.get("artifacts") or {}).keys())
    print(f"  step: {step_type} (variant={entry.get('variant','-')}) artifacts: {arts}")
for root, _, files in os.walk(f"log/{run_id}"):
    for f in files:
        print("  artifact file:", os.path.join(root, f))
PYEOF

# best-effort: persist artifacts back to shared FS (same semantics as run_test.sh)
if mkdir -p "$SRC/log_det_test" 2>/dev/null \
    && cp -rT "log/$RUN_ID" "$SRC/log_det_test/$RUN_ID" 2>/dev/null; then
    echo "(artifacts persisted to $SRC/log_det_test/$RUN_ID)"
else
    echo "(persist-to-/labdata2 blocked — manifest captured in task logs)"
fi

echo "=== generate 20 segment images ==="
if $PY -m visualize.visualize_segments --run-id "$RUN_ID" --max-plots 20; then
    NIMG=$(find "output/$RUN_ID" -name "*.png" 2>/dev/null | wc -l)
    echo "(images generated: $NIMG in output/$RUN_ID)"
    if mkdir -p "$SRC/output" 2>/dev/null \
        && cp -rT "output/$RUN_ID" "$SRC/output/$RUN_ID" 2>/dev/null; then
        echo "(images persisted to $SRC/output/$RUN_ID)"
    else
        echo "(persist-images-to-/labdata2 blocked — pngs captured in task logs)"
    fi
else
    echo "WARN: visualize_segments failed for run '$RUN_ID' — check logs above"
fi

echo "=== DONE ==="
