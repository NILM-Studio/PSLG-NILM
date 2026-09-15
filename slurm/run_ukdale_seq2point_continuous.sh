#!/bin/bash
#SBATCH --exclude=h103-slurm-a
#SBATCH -J ukdale_s2p_cont
#SBATCH -p RTX3090
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH -o /home/scnu2024024563/NILM/PSLG-NILM/slurm/slurm_log/ukdale_s2p_cont-%j.out
#SBATCH -e /home/scnu2024024563/NILM/PSLG-NILM/slurm/slurm_log/ukdale_s2p_cont-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/scnu2024024563/NILM/PSLG-NILM}"
RUN_ID="${RUN_ID:-ukdale_wm_primglr_detsec_3789}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASET_DIR="$("$PYTHON_BIN" - "$PROJECT_DIR" "$RUN_ID" "${DATASET_DIR:-}" <<'PY'
import json
import sys
from pathlib import Path

project, run_id, explicit = sys.argv[1:]
project = Path(project).resolve()
run_root = project / "log" / run_id
try:
    if explicit:
        directory = Path(explicit)
        if not directory.is_absolute():
            directory = project / directory
        dataset_manifest = directory / "nilm_dataset_manifest.json"
    else:
        with (run_root / "run_manifest.json").open(encoding="utf-8") as source:
            run = json.load(source)
        reference = run["steps"]["nilm_continuous_dataset"]["artifacts"]["dataset_manifest"]
        dataset_manifest = Path(reference)
        if not dataset_manifest.is_absolute():
            dataset_manifest = run_root / dataset_manifest
    if not dataset_manifest.is_file():
        raise FileNotFoundError(dataset_manifest)
except (OSError, ValueError, KeyError, TypeError) as exc:
    raise SystemExit(
        f"Cannot resolve continuous dataset: {exc}. "
        "Build nilm_continuous first or set DATASET_DIR explicitly.") from exc
print(dataset_manifest.resolve().parent)
PY
)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/log/$RUN_ID/nilm_seq2point/$(basename "$DATASET_DIR")}"

export RUN_ID DATASET_DIR OUTPUT_ROOT
export TEST_STRIDE="${TEST_STRIDE:-1}"
export EXPERIMENTS="${EXPERIMENTS:-all}"
export SEED="${SEED:-42}"
export EPOCHS="${EPOCHS:-30}"

bash "$PROJECT_DIR/slurm/run_ukdale_seq2point.sh"
