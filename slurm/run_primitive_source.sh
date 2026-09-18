#!/usr/bin/env bash
#SBATCH --job-name=primitive-source
#SBATCH --partition=RTX3090
#SBATCH --nodelist=h104-slurm-a
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=runs/slurm/primitive-source-%j.log
set -euo pipefail
cd /home/scnu202438025446/pslg-nilm
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TF_FORCE_GPU_ALLOW_GROWTH=true
TF_ENV=/home/scnu202438025446/miniconda3/envs/nilm-discovery-tf-v1
TORCH_ENV=/home/scnu202438025446/miniconda3/envs/nilmformer-cu118-v1
"$TF_ENV/bin/python" -m unittest discover -s tests -p 'test_nilm_sequence_labels.py' -v
"$TF_ENV/bin/python" -m unittest discover -s tests -p 'test_primitive_protocol.py' -v
(cd nilm_experiments && "$TORCH_ENV/bin/python" -m unittest nilm_lab.test_sequence_models nilm_lab.test_formal_models -v)
for lib in "$TF_ENV"/lib/python3.12/site-packages/nvidia/*/lib; do
  export LD_LIBRARY_PATH="$lib:${LD_LIBRARY_PATH:-}"
done
"$TF_ENV/bin/python" -m unittest discover -s tests -p 'test_discovery_refit.py' -v
"$TF_ENV/bin/python" -u main.py --config config/config_nilm_primitive_formal.yaml --profile nilm \
  --run-id primitive_source_20260915_v2 --steps nilm_data,extract,segment,feature,cluster,state_merge,state_sequence,nilm_labels
