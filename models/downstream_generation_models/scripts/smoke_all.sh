#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT="$(cd "$ROOT/../.." && pwd)"
: "${PSLG_EXECUTION_DIR:?Register an execution and use scripts/run_execution.py}"
TORCH_PY="$PROJECT/.venv-generation-torch/bin/python"
TF_PY="$PROJECT/.venv-generation-tf/bin/python"
DATA="$PSLG_EXECUTION_DIR/analysis/self_collected_smoke.npz"
OUT="$PSLG_EXECUTION_DIR/analysis"
cd "$ROOT"
"$TORCH_PY" scripts/prepare_self_collected.py --project-root "$PROJECT" --output "$DATA"
cp "${DATA%.npz}.manifest.json" "$PSLG_EXECUTION_DIR/inputs.json"
PYTHONPATH="$PROJECT/runtime/generation/torch_site${PYTHONPATH:+:$PYTHONPATH}" "$TORCH_PY" scripts/smoke_torch_models.py --data "$DATA" --output "$OUT/torch_smoke.json" --strict
"$TF_PY" scripts/smoke_timevae.py --data "$DATA" --output "$OUT/timevae_smoke.json"
CUDA_VISIBLE_DEVICES="" "$PROJECT/runtime/envs/timegan/bin/python" scripts/smoke_timegan.py --data "$DATA" --output "$OUT/timegan_smoke.json"
