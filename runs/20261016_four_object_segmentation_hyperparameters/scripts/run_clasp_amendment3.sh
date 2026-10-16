#!/usr/bin/env bash
set -euo pipefail
cd /home/avis/pslg-nilm
exp=20261016_four_object_segmentation_hyperparameters
base="runs/$exp"
config="$base/configs/search_space_amendment3.json"
for obj in refit_h7_dishwasher refit_h20_tumble_dryer ukdale_h2_washing_machine ukdale_h1_washing_machine_hotpoint; do
  path=$(python scripts/create_execution.py "$exp" "screen_amendment3_${obj}_clasp" --config "$config")
  echo "START $obj $path" $(date -u +%FT%TZ)
  PSLG_SEARCH_SPACE="$config" CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
    PYTHONPATH=runtime/segmentation/overlay_site \
    .venv-generation-torch/bin/python scripts/run_execution.py "$path" -- \
    .venv-generation-torch/bin/python "$base/scripts/screen.py" --object "$obj" --method clasp
  python - "$path" <<'PY'
import json,sys
from pathlib import Path
run=json.loads((Path(sys.argv[1])/'run.json').read_text())
print('DONE',sys.argv[1],run['exit_code'],flush=True)
raise SystemExit(run['exit_code'])
PY
done
