#!/usr/bin/env bash
# Run from an activated project Python environment. CPU only; do not source the
# GPU-only slurm/env.sh. Existing experiments and the inherited inputs stay intact.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
SOURCE_RUN_ID="${SOURCE_RUN_ID:-ukdale_wm_primglr_detsec_3789}"
RUN_ID="${RUN_ID:-ukdale_wm_composition_units_20260917}"
CLUSTER_TAG="${CLUSTER_TAG:-kmeans_k4_merged}"

# Exclusively creates a fresh run, so a repeated invocation cannot overwrite
# the old holdout, learned structure, study or its execution log.
"$PYTHON_BIN" -m scripts.prepare_downstream_run \
  --source-run-id "$SOURCE_RUN_ID" --run-id "$RUN_ID"

# A subshell gives the pipeline strict early-exit behavior. The outer shell
# waits for tee before packaging, without /dev/fd process substitution.
run_pipeline() (
set -euo pipefail
printf 'COMPOSITION_RUN_ID=%s\n' "$RUN_ID"
git rev-parse HEAD
"$PYTHON_BIN" --version
# Run against a recorded copy of the actual server config, including thresholds.
cp config/config_ukdale_detsec.yaml "log/$RUN_ID/downstream_config.yaml"

cohort_args=()
if [[ -n "${COHORT_START:-}" ]]; then cohort_args+=(--cohort-start "$COHORT_START"); fi
if [[ -n "${COHORT_END:-}" ]]; then cohort_args+=(--cohort-end "$COHORT_END"); fi
printf 'COHORT_START=%s COHORT_END=%s\n' "${COHORT_START:-unbounded}" "${COHORT_END:-unbounded}"

"$PYTHON_BIN" -u -m scripts.validate_generation_run \
  --run-id "$RUN_ID" --cluster-tag "$CLUSTER_TAG" --stage upstream \
  --device-change-date '2015-09-08T00:00:00+01:00' \
  --output "log/$RUN_ID/upstream_validation.json"

"$PYTHON_BIN" -u main.py \
  --config "log/$RUN_ID/downstream_config.yaml" \
  --run-id "$RUN_ID" --cluster-tag "$CLUSTER_TAG" \
  --segment-method prim-glr --feature-model detsec \
  --steps temporal_holdout,cycle_classify,cycle_validate,cycle_split \
  "${cohort_args[@]}"

# Does not relax class/member thresholds. Empty or disjoint holdouts stop here,
# with the exclusion funnel recorded for review before any new synthesis.
"$PYTHON_BIN" -m scripts.audit_cycle_cohort --run-id "$RUN_ID" \
  --output "log/$RUN_ID/cycle_cohort_audit.json" --require-evaluation-ready

composition_status=0
"$PYTHON_BIN" -u -m scripts.run_primitive_composition \
  --run-id "$RUN_ID" --cluster-tag "$CLUSTER_TAG" \
  --output-dir "log/$RUN_ID/composition" \
  --ratios "${RATIOS:-0.05,0.1,0.2,1.0}" \
  --seeds "${SEEDS:-42}" --max-anchors "${MAX_ANCHORS:-30}" \
  --candidates "${CANDIDATES:-8}" --min-fit-cycles 3 --max-warp 2 \
  --target-weight "${TARGET_WEIGHT:-1}" \
  --device-change-date '2015-09-08T00:00:00+01:00' || composition_status=$?

# Exit 2 still leaves diagnostic outputs when paired evaluation is unavailable.
if [[ "$composition_status" -ne 0 && "$composition_status" -ne 2 ]]; then
  exit "$composition_status"
fi

"$PYTHON_BIN" -m scripts.audit_primitive_composition --study-dir "log/$RUN_ID/composition"
"$PYTHON_BIN" -m scripts.report_primitive_composition \
  --study-dir "log/$RUN_ID/composition" --output-dir "log/$RUN_ID/composition_review" \
  --max-cases "${REPORT_CASES:-6}"
printf 'Return: %s\n' "$PROJECT_DIR/log/$RUN_ID/composition_review/report.md"
printf 'Return: %s\n' "$PROJECT_DIR/log/$RUN_ID/composition/composition_summary.json"
printf 'Return: %s\n' "$PROJECT_DIR/log/$RUN_ID/upstream_validation.json"
printf 'Return: %s\n' "$PROJECT_DIR/log/$RUN_ID/composition_execution.log"
exit "$composition_status"
)

set +e
run_pipeline 2>&1 | tee "$PROJECT_DIR/log/$RUN_ID/composition_execution.log"
pipeline_status=("${PIPESTATUS[@]}")
set -e
run_status=${pipeline_status[0]}
if [[ "${pipeline_status[1]}" -ne 0 ]]; then run_status=1; fi
if ! printf 'COMPOSITION_EXIT_STATUS=%s\n' "$run_status" | tee -a "$PROJECT_DIR/log/$RUN_ID/composition_execution.log"; then
  run_status=1
fi

# Always leave diagnostics on preflight/study failure, never raw/source arrays.
# The new run contains downstream outputs and upstream references, not source data.
archive="$PROJECT_DIR/log/${RUN_ID}_diagnostics.tar.gz"
if [[ -e "$archive" ]]; then
  printf 'Existing diagnostic archive left unchanged: %s\n' "$archive"
  if [[ "$run_status" -eq 0 ]]; then run_status=1; fi
elif tar --exclude='*.npz' --exclude='*.npy' \
    -czf "$archive" -C "$PROJECT_DIR/log/$RUN_ID" .; then
  printf 'Return diagnostic archive: %s\n' "$archive"
else
  printf 'Diagnostic packaging failed; return the run directory reports.\n'
  if [[ "$run_status" -eq 0 ]]; then run_status=1; fi
fi
exit "$run_status"
