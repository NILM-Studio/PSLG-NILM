#!/usr/bin/env bash
# Evaluate the existing study; no workflow fitting, synthesis or GPU training.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ID="${RUN_ID:-ukdale_wm_composition_prechange_20260917}"
CONTROL_TAG="${CONTROL_TAG:-real_cycle_controls_20260917}"
"$PYTHON_BIN" -c 'import sys; from scripts.prepare_downstream_run import validate_run_id; [validate_run_id(value) for value in sys.argv[1:]]' "$RUN_ID" "$CONTROL_TAG"
stage="$PROJECT_DIR/log/$RUN_ID/$CONTROL_TAG"
archive="$PROJECT_DIR/log/${RUN_ID}_${CONTROL_TAG}_diagnostics.tar.gz"
if [[ -e "$archive" || -L "$archive" ]]; then
  printf 'Existing archive left unchanged: %s\n' "$archive"
  exit 1
fi
# Exclusive creation: a rerun cannot replace even a failed diagnostic attempt.
mkdir "$stage"
run_controls() (
  set -euo pipefail
  git rev-parse HEAD
  "$PYTHON_BIN" --version
  "$PYTHON_BIN" -u -m scripts.run_real_cycle_controls \
    --run-id "$RUN_ID" --output-dir "$stage/results" \
    --repeats "${CONTROL_REPEATS:-100}" --sampling-seed "${CONTROL_SEED:-20260917}"
)
set +e
run_controls 2>&1 | tee "$stage/controls_execution.log"
statuses=("${PIPESTATUS[@]}")
set -e
result=${statuses[0]}
if [[ "${statuses[1]}" -ne 0 ]]; then result=1; fi
if ! printf 'REAL_CONTROLS_EXIT_STATUS=%s\n' "$result" | tee -a "$stage/controls_execution.log"; then
  result=1
fi
if tar -czf "$archive" -C "$stage" .; then
  printf 'Return diagnostic archive: %s\n' "$archive"
else
  printf 'Packaging failed; return the controls execution log.\n'
  result=1
fi
exit "$result"
