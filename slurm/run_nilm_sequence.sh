#!/usr/bin/env bash
#SBATCH --job-name=nilm-sequence
#SBATCH --partition=RTX3090
#SBATCH --nodelist=h104-slurm-a
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=runs/slurm/nilm-sequence-%j.log
set -euo pipefail
cd /home/scnu202438025446/pslg-nilm
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TF_FORCE_GPU_ALLOW_GROWTH=true
TF_ENV=/home/scnu202438025446/miniconda3/envs/nilm-discovery-tf-v1
TORCH_ENV=/home/scnu202438025446/miniconda3/envs/nilmformer-cu118-v1
CONFIG="${CONFIG:-config/config_nilm_sequence_pilot.yaml}"
RUN_ID="${RUN_ID:-nilm_sequence_${SLURM_JOB_ID}}"
# Each main.py invocation and Torch worker is isolated; never mix vendor src with project src.
"$TF_ENV/bin/python" -m unittest discover -s tests -p 'test_nilm_sequence_labels.py' -v
(cd nilm_experiments && "$TORCH_ENV/bin/python" -m unittest nilm_lab.test_sequence_models -v)
for lib in "$TF_ENV"/lib/python3.12/site-packages/nvidia/*/lib; do
  export LD_LIBRARY_PATH="$lib:${LD_LIBRARY_PATH:-}"
done
"$TF_ENV/bin/python" -u main.py --config "$CONFIG" --profile nilm --run-id "$RUN_ID" \
  --steps nilm_data,extract,segment,feature,cluster,state_merge,state_sequence,nilm_labels
# End the TF process before invoking any NILM model. Training worker uses its own interpreter.
unset LD_LIBRARY_PATH
"$TF_ENV/bin/python" -u main.py --config "$CONFIG" --profile nilm --run-id "$RUN_ID" \
  --steps nilm_train,nilm_select,nilm_report
# Held-out evaluation is intentionally absent. Invoke nilm_evaluate explicitly after source protocol freeze.
