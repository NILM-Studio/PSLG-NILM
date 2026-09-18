#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
runtime_python=/home/scnu202438025446/miniconda3/envs/nilmformer-cu118-v1/bin/python
test -f NILMFormer/runtime/verification/cpu.json || { echo 'Verify the base Conda environment first'; exit 1; }
mkdir -p runtime/wheelhouse
"$runtime_python" -m pip list --format=freeze > runtime/requirements-lock.txt
"$runtime_python" runtime/fill_wheelhouse.py
sha256sum runtime/wheelhouse/*.whl > runtime/wheelhouse-sha256.txt
