#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="$ROOT/../../runtime/envs/timegan"
CONDA_BIN="${CONDA_BIN:-/opt/miniconda3/bin/conda}"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  "$CONDA_BIN" create -y -p "$ENV_DIR" --override-channels -c conda-forge python=3.7 pip
fi

"$ENV_DIR/bin/python" -m pip install --timeout 180 --retries 5 \
  tensorflow==1.15.0 numpy==1.18.5 protobuf==3.20.3 \
  tqdm==4.66.5 argparse==1.4.0 pandas==1.3.5 \
  scikit-learn==1.0.2 matplotlib==3.5.3

"$ENV_DIR/bin/python" - <<'PY'
import numpy
import tensorflow
print("TimeGAN legacy runtime ready:", tensorflow.__version__, numpy.__version__)
PY
