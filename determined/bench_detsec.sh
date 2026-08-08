#!/usr/bin/env bash
# Benchmark the REFACTORED detsec feature model on the GPU cluster.
# Reuses the same GPU-adapted runtime as run_test.sh (TF2.16 sees the GPU inside
# the cuda-11.8 image). Validates: model builds, CuDNN engages (fast), features
# finite & correct shape, history keys present, per-epoch timing.
set -euo pipefail

SRC=/labdata2/lexingruan/pslg-nilm
ENV=/labdata2/lexingruan/miniconda3/envs/PSLG-NILM
PY=$ENV/bin/python
WS=/tmp/pslg_bench

echo "=== task env ==="
echo "host: $(hostname)   date: $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || echo "(nvidia-smi unavailable)"

mkdir -p $WS
export HOME=$WS
export PYTHONUNBUFFERED=1          # stream fit()/print output live (no block buffering)
export TF_CPP_MIN_LOG_LEVEL=2
export TF_FORCE_GPU_ALLOW_GROWTH=true
export NUMBA_DISABLE_CUDA=1
export NUMBA_THREADING_LAYER=workqueue
NVIDIA_LIBS=$(ls -d $ENV/lib/python3.12/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH=$NVIDIA_LIBS$ENV/lib:$LD_LIBRARY_PATH

echo "=== stage code ==="
cp -r $SRC/models $WS/models
cp -r $SRC/src $WS/src 2>/dev/null || true   # base_model import path
cp $SRC/determined/bench_detsec.py $SRC/determined/probe_bs.py $WS/ 2>/dev/null || true
cd $WS

echo "=== GPU self-check ==="
$PY - <<'PYEOF'
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print('TF', tf.__version__, '| physical GPUs:', gpus)
assert gpus, 'NO GPU VISIBLE TO TF'
with tf.device('/GPU:0'):
    a = tf.random.normal((512, 512)); b = tf.random.normal((512, 512)); c = tf.matmul(a, b)
_ = c.numpy()
assert 'GPU' in c.device, 'op did not land on GPU'
print('GPU compute OK | device:', c.device)
PYEOF

echo "=== Phase A: steady-state timing + fit per-epoch (bs=16) ==="
$PY bench_detsec.py

echo "=== Phase B: max-bs probe (each bs isolated, 120s timeout) ==="
for bs in 4 8 12 16 24; do
  out=$(timeout 120 $PY probe_bs.py $bs 2>&1 | grep -vE "cuda_fft|cuda_dnn|cuda_blas" | tail -1)
  rc=$?
  if [ $rc -eq 124 ]; then
    echo "   bs=$bs   HANG (>120s, killed)"
  elif [ -n "$out" ]; then
    echo "   $out"
  else
    echo "   bs=$bs   rc=$rc (no output)"
  fi
done
echo "=== DONE ==="
