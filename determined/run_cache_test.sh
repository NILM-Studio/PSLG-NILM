#!/usr/bin/env bash
# Run the M2 feature-cache unit test on the cluster env (needs TF via imports).
set -euo pipefail
SRC=/labdata2/lexingruan/pslg-nilm
ENV=/labdata2/lexingruan/miniconda3/envs/PSLG-NILM
PY=$ENV/bin/python
WS=/tmp/pslg_cache_test
mkdir -p $WS
export HOME=$WS TF_CPP_MIN_LOG_LEVEL=2 TF_FORCE_GPU_ALLOW_GROWTH=true NUMBA_DISABLE_CUDA=1
NVIDIA_LIBS=$(ls -d $ENV/lib/python3.12/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH=$NVIDIA_LIBS$ENV/lib:$LD_LIBRARY_PATH
cp -r $SRC/src $SRC/models $SRC/config $SRC/tests $SRC/main.py $WS/ 2>/dev/null || true
cd $WS
$PY -m pytest tests/test_m2_cache.py -v 2>&1 | tail -40 || $PY -m unittest tests.test_m2_cache -v 2>&1 | tail -40
echo "=== DONE ==="
