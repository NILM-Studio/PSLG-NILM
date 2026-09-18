#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
CONDA_EXE="${CONDA_EXE:-/home/scnu202438025446/miniconda3/bin/conda}"
base_env=nilmformer-cu118-v1
"$CONDA_EXE" run --no-capture-output -n "$base_env" python -m pip install --timeout 120 --retries 8 torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
"$CONDA_EXE" run --no-capture-output -n "$base_env" python -m pip install --timeout 120 --retries 8 -r runtime/requirements-runtime.txt
"$CONDA_EXE" run -n "$base_env" python -m pip check
for entry in NILMFormer:nilmformer FCN:fcn BERT4NILM:bert4nilm SGN:sgn; do
  model="${entry%%:*}"
  case "$model" in
    NILMFormer) model_dir=third_party/nilmformer ;;
    BERT4NILM) model_dir=third_party/bert4nilm ;;
    SGN) model_dir=third_party/nilmtk_contrib ;;
    FCN) model_dir=FCN ;;
  esac
  env_name="${entry##*:}-cu118-v1"
  if [[ "$env_name" != "$base_env" ]]; then
    if "$CONDA_EXE" run -n "$env_name" python --version >/dev/null 2>&1; then
      echo "Existing environment found; validating $env_name"
    else
      "$CONDA_EXE" create -y -n "$env_name" --clone "$base_env"
    fi
  fi
  mkdir -p "$model_dir/runtime/verification"
  "$CONDA_EXE" run --no-capture-output -n "$env_name" python -m pip check
  "$CONDA_EXE" run --no-capture-output -n "$env_name" python runtime/verify_runtime.py --model "$model" | tee "$model_dir/runtime/verification/cpu.json"
  "$CONDA_EXE" run -n "$env_name" python -m pip freeze > "$model_dir/runtime/pip-freeze.txt"
  "$CONDA_EXE" list -n "$env_name" --explicit > "$model_dir/runtime/conda-explicit.txt"
done
