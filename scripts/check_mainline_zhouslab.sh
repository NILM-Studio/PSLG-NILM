#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# Tests create isolated temporary project roots; do not route their fixtures into this execution.
unset PSLG_EXECUTION_DIR
export PYTHONPATH="$ROOT/runtime/nilm/zhouslab/mainline_test_site"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export NILM_TORCH_PYTHON="$ROOT/.venv-generation-torch/bin/python"
"$ROOT/.venv-generation-tf/bin/python" -m pytest -q --ignore=tests/test_layout_refactor.py
"$ROOT/.venv-generation-torch/bin/python" -m pytest -q tests/test_layout_refactor.py
