#!/bin/bash
# Common environment for PSLG-NILM Slurm jobs.
# This file must be sourced by a submitted Slurm script.

set -e

module purge
module load miniconda3/25.5.1-0

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/bin/activate"
conda activate pslg-nilm

export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=0
export TF_FORCE_GPU_ALLOW_GROWTH=true

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMBA_NUM_THREADS=1

export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CONDA_PREFIX"

# Important: reset LD_LIBRARY_PATH instead of inheriting the login-shell value.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cufft/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/curand/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cusolver/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cusparse/lib:$LD_LIBRARY_PATH"

PROJECT_DIR="/home/scnu2024024563/NILM/PSLG-NILM"
cd "$PROJECT_DIR" || exit 1

echo "HOSTNAME=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "CONDA_PREFIX=$CONDA_PREFIX"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

which python
python --version
which nvidia-smi || true
nvidia-smi || true

python -u - <<'PY'
import sys
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")

print("TF:", tf.__version__, flush=True)
print("GPUs:", gpus, flush=True)

if not gpus:
    print(
        "ERROR: TensorFlow cannot see the allocated GPU.",
        flush=True,
    )
    sys.exit(1)

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

with tf.device("/GPU:0"):
    a = tf.random.normal((1024, 1024))
    b = tf.random.normal((1024, 1024))
    c = tf.matmul(a, b)

print("GPU compute OK:", c.device, c.shape, flush=True)
PY