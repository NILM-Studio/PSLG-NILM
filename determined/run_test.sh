#!/usr/bin/env bash
# Determined task entrypoint: UK-DALE washing_machine pipeline (GPU-adapted runtime).
#
# Runs INSIDE the task container. Design:
#   - /labdata2 is bind-mounted RW (shared NFS) → conda env + 6.3 GB ukdale.h5.
#   - code is copied from the mount into a writable /tmp/pslg_test at start,
#     so outputs (input/, log/, .cache/) can be written.
#   - the shared PSLG-NILM conda env's python is exec'd directly (TF 2.16 + numpy<2).
#
# Parameterized via POSITIONAL args (each yaml passes them in its entrypoint),
# so the dtw smoke test and a bilstm_ae GPU run share one script. NB: Determined
# 0.38 generic tasks accept environment_variables in the schema but do NOT inject
# them into the container, so we cannot rely on env for the variant.
#
#   $1 SEGMENT_METHOD  $2 FEATURE_MODEL  $3 CLUSTER_METHOD  $4 N_CLUSTERS
#   $5 RUN_ID
#   $6 RAW_SERIES      absolute CSV path. "prepare" (=default) → run
#                      prepare_ukdale.py to slice a 100-row smoke window; any
#                      other value is used directly (e.g. the full-series CSV).
#   $7 CONFIG          config file under config/ (default config_ukdale_test.yaml).
#                      config_ukdale_full.yaml raises batch_size to 64 for L40.
SEGMENT_METHOD="${1:-fluss}"
FEATURE_MODEL="${2:-dtw}"            # dtw (CPU) | bilstm_ae / lstm_ae / cnn_ae / ... (GPU)
CLUSTER_METHOD="${3:-kmeans}"
N_CLUSTERS="${4:-2}"
RUN_ID="${5:-test}"
RAW_SERIES="${6:-prepare}"           # "prepare" → 100-row smoke slice; else a CSV path
CONFIG="${7:-config/config_ukdale_test.yaml}"
set -euo pipefail

SRC=/labdata2/lexingruan/pslg-nilm
ENV=/labdata2/lexingruan/miniconda3/envs/PSLG-NILM
PY=$ENV/bin/python
H5=$SRC/datasets/ukdale/ukdale.h5
WS=/tmp/pslg_test                       # writable scratch (container root FS is not writable at /)

echo "=== task env ==="
echo "host: $(hostname)   date: $(date -u +%FT%TZ)   uid=$(id -u) user=$(id -un)"
echo "variant: segment=$SEGMENT_METHOD feature=$FEATURE_MODEL cluster=$CLUSTER_METHOD k=$N_CLUSTERS run_id=$RUN_ID"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || echo "(nvidia-smi unavailable)"

mkdir -p $WS
# writable home + TF/numba caches land here, not on the read-only mount
export HOME=$WS
export TF_CPP_MIN_LOG_LEVEL=2
export TF_FORCE_GPU_ALLOW_GROWTH=true
export NUMBA_DISABLE_CUDA=1          # stumpy/numba CPU-only (numba.cuda segfaults on GPU nodes); TF still uses GPU
export NUMBA_THREADING_LAYER=workqueue
# TF 2.16 bundles CUDA-12; the determinedai image ships CUDA-11.8. Prepend TF's
# nvidia libs so they win the loader search (else TF sees no GPU).
NVIDIA_LIBS=$(ls -d $ENV/lib/python3.12/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH=$NVIDIA_LIBS$ENV/lib:$LD_LIBRARY_PATH

echo "=== stage code into writable workspace ==="
cp -r $SRC/src $SRC/models $SRC/config $SRC/scripts $SRC/main.py $SRC/requirements.txt $WS/
cd $WS

echo "=== GPU self-check (runtime must be GPU-adapted) ==="
# list_physical_devices only proves TF SEES the GPU; actually place an op on it
# to prove compute works, then report which device a TF variable lands on.
$PY - <<'PYEOF'
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print('TF', tf.__version__, '| physical GPUs:', gpus)
assert gpus, 'NO GPU VISIBLE TO TF'
# force a real matmul onto the GPU and read it back
with tf.device('/GPU:0'):
    a = tf.random.normal((512, 512)); b = tf.random.normal((512, 512))
    c = tf.matmul(a, b)
_ = c.numpy()                                   # materialize
v = tf.Variable([1.0])                          # default-placement probe
print('GPU compute OK | sample op device:', c.device, '| default Var device:', v.device)
assert 'GPU' in c.device, 'op did not land on GPU'
PYEOF

if [ "$RAW_SERIES" = "prepare" ]; then
    echo "=== prepare washing_machine slice from ukdale.h5 (nilmtk, 100-row smoke) ==="
    $PY scripts/prepare_ukdale.py --h5 "$H5" --building 1 --appliance "washing machine" --n-rows 100
    RAW="input/ukdale_washing_machine_100.csv"
else
    echo "=== using pre-built raw series: $RAW_SERIES ==="
    # the workspace only has src/models/config/scripts copied in; reference the
    # full CSV on the bind-mount by absolute path so extract_active_data reads it.
    RAW="$RAW_SERIES"
    $PY -c "import pandas as pd, sys; df=pd.read_csv('$RAW'); print(f'raw series: {len(df):,} rows  cols={list(df.columns)}')"
fi

echo "=== run pipeline: extract,segment,feature,cluster ==="
$PY main.py --config "$CONFIG" \
    --appliance washing_machine --run-id "$RUN_ID" \
    --steps extract,segment,feature,cluster \
    --raw-series "$RAW" \
    --segment-method "$SEGMENT_METHOD" --feature-model "$FEATURE_MODEL" \
    --cluster-method "$CLUSTER_METHOD" --n-clusters "$N_CLUSTERS"

echo "=== GPU utilization snapshot (post-train) ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || true

echo "=== artifacts ==="
$PY - "$RUN_ID" <<'PYEOF'
import json, os, sys
run_id = sys.argv[1]
manifest = f"log/{run_id}/run_manifest.json"
m = json.load(open(manifest))
print("run_id:", m.get("run_id"), "| appliance:", m.get("appliance"))
print("variants:", m.get("variants", {}))
# steps is a DICT keyed by step_type (run_manifest.py). time_clustering has a
# nested `results` dict; every other step is a flat entry.
steps = m.get("steps", {})
for step_type, entry in steps.items():
    if "results" in entry:                       # time_clustering: multiple tagged results
        tags = ", ".join(f"{t}({r.get('variant','-')})" for t, r in entry["results"].items())
        print(f"  step: {step_type} -> results: {tags}")
    else:
        arts = list((entry.get("artifacts") or {}).keys())
        extra = entry.get("extra") or {}
        note = f"  cache_hit={extra.get('cache_hit')}" if "cache_hit" in extra else ""
        print(f"  step: {step_type} (variant={entry.get('variant','-')}) artifacts: {arts}{note}")
for root, _, files in os.walk(f"log/{run_id}"):
    for f in files:
        print("  artifact file:", os.path.join(root, f))
print("--- full run_manifest.json (captured in task logs; survives container teardown) ---")
print(open(manifest).read())
# training history for the neural feature model (loss curve)
hist = f"log/{run_id}/FeatureExtract_{m.get('variants',{}).get('feature_model','?')}_on_{m.get('variants',{}).get('segment_method','?')}/training_history.json"
if os.path.exists(hist):
    print("--- training_history.json ---")
    print(open(hist).read())
# clustering metrics
import glob
for mj in glob.glob(f"log/{run_id}/TimeClustering_*/*/metrics.json"):
    print(f"--- {mj} ---")
    print(open(mj).read())
PYEOF

# best-effort: persist artifacts back to shared FS. Tasks run as uid 1067 here,
# so NFS root-squash does not block writes; if it ever does, the manifest +
# history + metrics are already captured in det task logs above.
if mkdir -p "$SRC/log_det_test" 2>/dev/null \
    && cp -rT "log/$RUN_ID" "$SRC/log_det_test/$RUN_ID" 2>/dev/null; then
    echo "(artifacts persisted to $SRC/log_det_test/$RUN_ID)"
else
    echo "(persist-to-/labdata2 blocked — container uid=$(id -u) vs owner=1067; "
    echo " manifest + history + metrics are captured in these task logs)"
fi

echo "=== DONE ==="
