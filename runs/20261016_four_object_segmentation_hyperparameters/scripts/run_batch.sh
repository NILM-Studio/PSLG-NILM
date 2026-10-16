#!/usr/bin/env bash
set -euo pipefail
cd /home/avis/pslg-nilm
exp=20261016_four_object_segmentation_hyperparameters
base="runs/$exp"
case "${1:?queue}" in
  light)
    jobs=(
      refit_h20_tumble_dryer:prim-glr
      ukdale_h2_washing_machine:prim-glr
      ukdale_h1_washing_machine_hotpoint:prim-glr
      refit_h7_dishwasher:pelt-rbf
      refit_h20_tumble_dryer:pelt-rbf
      ukdale_h2_washing_machine:pelt-rbf
      ukdale_h1_washing_machine_hotpoint:pelt-rbf
      refit_h7_dishwasher:clasp
      refit_h20_tumble_dryer:clasp
      ukdale_h2_washing_machine:clasp
      ukdale_h1_washing_machine_hotpoint:clasp
    ) ;;
  fluss)
    jobs=(
      refit_h7_dishwasher:fluss-threshold
      refit_h20_tumble_dryer:fluss-threshold
      ukdale_h2_washing_machine:fluss-threshold
      ukdale_h1_washing_machine_hotpoint:fluss-threshold
    ) ;;
  gpu)
    jobs=(
      refit_h7_dishwasher:e2usd
      refit_h20_tumble_dryer:e2usd
      ukdale_h2_washing_machine:e2usd
      ukdale_h1_washing_machine_hotpoint:e2usd
    ) ;;
  *) echo "Unknown queue $1" >&2; exit 2 ;;
esac
for job in "${jobs[@]}"; do
  obj="${job%%:*}"
  method="${job##*:}"
  condition="screen_${obj}_${method//-/_}"
  execution=$(python scripts/create_execution.py "$exp" "$condition" --config "$base/configs/search_space.json")
  echo "START $job $execution" $(date -u +%FT%TZ)
  CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
    PYTHONPATH=runtime/segmentation/overlay_site \
    .venv-generation-torch/bin/python scripts/run_execution.py "$execution" -- \
    .venv-generation-torch/bin/python "$base/scripts/screen.py" --object "$obj" --method "$method"
  python - "$execution" <<'PY'
import json,sys
from pathlib import Path
run=json.loads((Path(sys.argv[1])/'run.json').read_text())
print('DONE',sys.argv[1],run['exit_code'],flush=True)
if run['exit_code']:
    raise SystemExit(run['exit_code'])
PY
done
