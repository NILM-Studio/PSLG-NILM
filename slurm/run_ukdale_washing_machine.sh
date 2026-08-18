#!/bin/bash
#SBATCH --exclude=h103-slurm-a
#SBATCH -J ukdale_wm
#SBATCH -p RTX3090
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH -o /home/scnu2024024563/NILM/PSLG-NILM/slurm/slurm_log/ukdale_wm-%j.out
#SBATCH -e /home/scnu2024024563/NILM/PSLG-NILM/slurm/slurm_log/ukdale_wm-%j.err

set -euo pipefail

source "/home/scnu2024024563/NILM/PSLG-NILM/slurm/env.sh"

CONFIG="${CONFIG:-config/config_ukdale_detsec.yaml}"
RAW_SERIES="${RAW_SERIES:-input/ukdale_washing_machine_full.csv}"
RUN_ID="${RUN_ID:-ukdale_wm_primglr_detsec_${SLURM_JOB_ID}}"

echo "CONFIG=$CONFIG"
echo "RAW_SERIES=$RAW_SERIES"
echo "RUN_ID=$RUN_ID"

if [ ! -f "$RAW_SERIES" ]; then
    echo "ERROR: washing-machine CSV not found: $RAW_SERIES"
    exit 1
fi

echo "=== Data check ==="
ls -lh "$RAW_SERIES"

python - "$RAW_SERIES" <<'PY'
import sys
import pandas as pd

path = sys.argv[1]
df = pd.read_csv(path)

print("rows:", len(df))
print("columns:", list(df.columns))
print(df.head())
PY

echo "=== TensorFlow GPU check ==="
python - <<'PY'
import tensorflow as tf

print("TensorFlow:", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("Physical GPUs:", gpus)

if not gpus:
    raise RuntimeError("TensorFlow cannot detect the Slurm GPU")

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

with tf.device("/GPU:0"):
    a = tf.random.normal((512, 512))
    b = tf.random.normal((512, 512))
    c = tf.matmul(a, b)

_ = c.numpy()
print("GPU compute OK:", c.device, c.shape)
PY

echo "=== Start PSLG-NILM pipeline ==="

python main.py \
    --config "$CONFIG" \
    --appliance washing_machine \
    --run-id "$RUN_ID" \
    --raw-series "$RAW_SERIES" \
    --steps extract,segment,feature,cluster \
    --segment-method prim-glr \
    --feature-model detsec \
    --cluster-method kmeans \
    --n-clusters 2,3,4,5,6

echo "=== Pipeline completed ==="
echo "Artifacts: /home/scnu2024024563/NILM/PSLG-NILM/log/$RUN_ID"
echo "Finished on: $(date)"