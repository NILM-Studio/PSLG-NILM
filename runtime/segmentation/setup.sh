#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="$repo_root/.venv-generation-torch/bin/python"
overlay="$repo_root/runtime/segmentation/overlay_site"

if [[ ! -x "$python_bin" ]]; then
  echo "Missing Torch runtime: $python_bin" >&2
  exit 1
fi
mkdir -p "$overlay"
"$python_bin" -m pip install --no-deps --target "$overlay" \
  -r "$repo_root/runtime/segmentation/requirements-overlay.txt"
NUMBA_DISABLE_CUDA=1 PYTHONPATH="$overlay${PYTHONPATH:+:$PYTHONPATH}" "$python_bin" -c \
  'import claspy, numba, ruptures, statsmodels, stumpy, torch; from claspy.state_detection import AgglomerativeCLaPDetection; print("segmentation imports ready")'

echo "Run with PYTHONPATH=$overlay and $python_bin"
