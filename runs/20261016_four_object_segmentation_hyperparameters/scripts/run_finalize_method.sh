#!/usr/bin/env bash
set -euo pipefail
cd /home/avis/pslg-nilm
method="${1:?method}"
exp=20261016_four_object_segmentation_hyperparameters
base="runs/$exp"
safe="${method//-/_}"
config="$base/configs/search_space.json"
if [[ "$method" == "fluss-threshold" && -f "$base/configs/search_space_amendment1.json" ]]; then
  config="$base/configs/search_space_amendment1.json"
fi
if [[ "$method" == "pelt-rbf" && -f "$base/configs/search_space_amendment2.json" ]]; then
  config="$base/configs/search_space_amendment2.json"
fi
if [[ "$method" == "clasp" && -f "$base/configs/search_space_amendment3.json" ]]; then
  config="$base/configs/search_space_amendment3.json"
fi
if [[ ! -f "$base/data/finalists_${safe}.json" ]]; then
  path=$(python scripts/create_execution.py "$exp" "select_${safe}" --config "$config")
  echo "SELECT $method $path"
  PYTHONPATH=runtime/segmentation/overlay_site .venv-generation-torch/bin/python \
    scripts/run_execution.py "$path" -- .venv-generation-torch/bin/python \
    "$base/scripts/choose_finalists.py" --method "$method"
  python - "$path" <<'PY'
import json,sys
from pathlib import Path
raise SystemExit(json.loads((Path(sys.argv[1])/'run.json').read_text())['exit_code'])
PY
fi
objects=(refit_h7_dishwasher refit_h20_tumble_dryer ukdale_h2_washing_machine ukdale_h1_washing_machine_hotpoint)
for obj in "${objects[@]}"; do
  condition="validate_${obj}_${safe}"
  path=$(python scripts/create_execution.py "$exp" "$condition" --config "$config")
  echo "VALIDATE $obj $method $path" $(date -u +%FT%TZ)
  CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
    PYTHONPATH=runtime/segmentation/overlay_site \
    .venv-generation-torch/bin/python scripts/run_execution.py "$path" -- \
    .venv-generation-torch/bin/python "$base/scripts/validate.py" --object "$obj" --method "$method"
  python - "$path" <<'PY'
import json,sys
from pathlib import Path
run=json.loads((Path(sys.argv[1])/'run.json').read_text())
print('DONE',sys.argv[1],run['exit_code'],flush=True)
raise SystemExit(run['exit_code'])
PY
done
echo "VALIDATION_COMPLETE $method — pending ten-panel human review; no configuration frozen"
