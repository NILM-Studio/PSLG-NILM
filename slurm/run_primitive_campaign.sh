#!/usr/bin/env bash
#SBATCH --job-name=primitive-formal
#SBATCH --partition=RTX3090
#SBATCH --nodelist=h104-slurm-a
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --output=runs/slurm/primitive-formal-%j.log
set -euo pipefail
cd /home/scnu202438025446/pslg-nilm
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TF_FORCE_GPU_ALLOW_GROWTH=true
TF_ENV=/home/scnu202438025446/miniconda3/envs/nilm-discovery-tf-v1
TORCH_ENV=/home/scnu202438025446/miniconda3/envs/nilmformer-cu118-v1
export NILM_TORCH_PYTHON="$TORCH_ENV/bin/python"
RUN_ID=primitive_source_20260915_v2
"$TF_ENV/bin/python" -m unittest discover -s tests -p 'test_nilm_sequence_labels.py' -v
"$TF_ENV/bin/python" -m unittest discover -s tests -p 'test_primitive_protocol.py' -v
"$TF_ENV/bin/python" -m unittest discover -s tests -p 'test_nilm_workflow_integration.py' -v
(cd nilm_experiments && "$TORCH_ENV/bin/python" -m unittest nilm_lab.test_sequence_models nilm_lab.test_formal_models -v)
for lib in "$TF_ENV"/lib/python3.12/site-packages/nvidia/*/lib; do
  export LD_LIBRARY_PATH="$lib:${LD_LIBRARY_PATH:-}"
done
RUN_DIR=$("$TF_ENV/bin/python" -m src.framework.run_paths --run-id "$RUN_ID")
if [ ! -f "$RUN_DIR/occlusion_audit.json" ]; then
  "$TF_ENV/bin/python" -u scripts/audit_primitive_labels.py --run-id "$RUN_ID"
fi
unset LD_LIBRARY_PATH
"$TF_ENV/bin/python" -u main.py --config config/config_nilm_primitive_formal.yaml --profile nilm \
  --run-id "$RUN_ID" --steps nilm_train,nilm_select,nilm_report
# Explicit final test executes only after every source gate and formal freeze succeeds.
"$TF_ENV/bin/python" -u main.py --config config/config_nilm_primitive_formal.yaml --profile nilm \
  --run-id "$RUN_ID" --steps nilm_evaluate,nilm_report
