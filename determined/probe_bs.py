"""Single-batch-size probe for DETSEC: build, one train_on_batch, report.

Each invocation is a fresh process (called by bench_detsec.sh under `timeout`),
so a GPU-memory hang at one bs cannot block the others. Prints OK + mem, or the
exception. Exit 0 on success, non-zero on failure/OOM.
"""
import sys
import time
import numpy as np
import tensorflow as tf
from models.feature_extract.detsec_model import DETSECModel

bs = int(sys.argv[1])
rng = np.random.RandomState(0)
m = DETSECModel(config=dict(latent_dim=16, nunits=128, attention_size=32,
                            batch_size=bs, learning_rate=1e-3))
m._build_model(1500, 4)
xb = rng.rand(bs, 1500, 4).astype(np.float32)
t0 = time.time()
try:
    m.autoencoder_model.train_on_batch(xb, xb)
    mem = tf.config.experimental.get_memory_usage('GPU:0')/1e6
    print(f"bs={bs:>3} OK {time.time()-t0:.2f}s mem={mem:.0f}MB", flush=True)
    sys.exit(0)
except Exception as e:
    print(f"bs={bs:>3} FAILED {type(e).__name__}: {str(e)[:160]}", flush=True)
    sys.exit(1)
