#!/usr/bin/env bash
# Run from an activated project Python environment. CPU only; do not source the
# GPU-only slurm/env.sh. Existing experiments and the inherited inputs stay intact.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
SOURCE_RUN_ID="${SOURCE_RUN_ID:-ukdale_wm_primglr_detsec_3789}"
RUN_ID="${RUN_ID:-ukdale_wm_composition_20260917}"
CLUSTER_TAG="${CLUSTER_TAG:-kmeans_k4_merged}"

# Exclusively creates a fresh run, so a repeated invocation cannot overwrite
# the old holdout, learned structure, study or its execution log.
"$PYTHON_BIN" -m scripts.prepare_downstream_run \
  --source-run-id "$SOURCE_RUN_ID" --run-id "$RUN_ID"

exec > >(tee "$PROJECT_DIR/log/$RUN_ID/composition_execution.log") 2>&1
printf 'COMPOSITION_RUN_ID=%s\n' "$RUN_ID"
git rev-parse HEAD
"$PYTHON_BIN" --version

"$PYTHON_BIN" -u -m scripts.validate_generation_run \
  --run-id "$RUN_ID" --cluster-tag "$CLUSTER_TAG" --stage upstream \
  --device-change-date '2015-09-08T00:00:00+01:00' \
  --output "log/$RUN_ID/upstream_validation.json"

"$PYTHON_BIN" -u main.py \
  --config config/config_ukdale_detsec.yaml \
  --run-id "$RUN_ID" --cluster-tag "$CLUSTER_TAG" \
  --segment-method prim-glr --feature-model detsec \
  --steps temporal_holdout,cycle_classify,cycle_validate,cycle_split

"$PYTHON_BIN" -u -m scripts.run_primitive_composition \
  --run-id "$RUN_ID" --cluster-tag "$CLUSTER_TAG" \
  --output-dir "log/$RUN_ID/composition" \
  --ratios "${RATIOS:-0.05,0.1,0.2,1.0}" \
  --seeds "${SEEDS:-42}" --max-anchors "${MAX_ANCHORS:-30}" \
  --candidates "${CANDIDATES:-8}" --min-fit-cycles 3 --max-warp 2 \
  --device-change-date '2015-09-08T00:00:00+01:00'

"$PYTHON_BIN" -m scripts.audit_primitive_composition --study-dir "log/$RUN_ID/composition"
printf 'Return: %s\n' "$PROJECT_DIR/log/$RUN_ID/composition/composition_summary.json"
printf 'Return: %s\n' "$PROJECT_DIR/log/$RUN_ID/upstream_validation.json"
printf 'Return: %s\n' "$PROJECT_DIR/log/$RUN_ID/composition_execution.log"
