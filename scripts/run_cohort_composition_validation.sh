#!/usr/bin/env bash
# Prespecified early-period prototype. A calendar filter is not proof of device
# identity, and inherited segmentation/representation fitting remains unverified.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export RUN_ID="${RUN_ID:-ukdale_wm_composition_prechange_20260917}"
export COHORT_END="${COHORT_END:-2015-09-08T00:00:00+01:00}"
exec bash "$PROJECT_DIR/scripts/run_composition_validation.sh"
