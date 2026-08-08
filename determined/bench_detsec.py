"""DETSEC refactored-model benchmark (Phase A): correctness + steady-state timing.

Run on the GPU cluster. One model at bs=16: warms the XLA compile, then times
steady-state per-batch and a full fit() per-epoch on 256 samples.
"""
import time
import numpy as np
import tensorflow as tf
from models.feature_extract.detsec_model import detsec_ae, DETSECModel

rng = np.random.RandomState(0)
BS = 8        # bs<=8 fits; the backward of 2x return_sequences BiGRUs (128u, 1500
              # steps) hits the 4090 24GB memory cliff between bs=8 and bs=16.

# ---- build + param count ----
m = DETSECModel(config=dict(latent_dim=16, nunits=128, attention_size=32,
                            batch_size=BS, learning_rate=1e-3))
m._build_model(1500, 4)
nparams = int(sum(np.prod(v.shape) for v in m.autoencoder_model.trainable_variables))
print(f"[A0] params: {nparams:,}", flush=True)

xb = rng.rand(BS, 1500, 4).astype(np.float32)

# ---- warm compile (first 2 calls eat the XLA trace) ----
for w in range(2):
    t0 = time.time(); m.autoencoder_model.train_on_batch(xb, xb)
    print(f"[Awarm{w}] {time.time()-t0:.2f}s", flush=True)

# ---- steady-state per-batch ----
ts = []
for _ in range(5):
    t0 = time.time(); m.autoencoder_model.train_on_batch(xb, xb); ts.append(time.time()-t0)
mean, mn = float(np.mean(ts)), float(np.min(ts))
print(f"[Asteady] bs={BS} per-batch mean={mean:.3f}s min={mn:.3f}s "
      f"-> {BS/mean:.1f} samples/s", flush=True)
mem = tf.config.experimental.get_memory_usage('GPU:0')/1e6
print(f"[Amem] {mem:.0f} MB at bs={BS}", flush=True)

# ---- full fit() per-epoch on 256 samples ----
tf.keras.backend.clear_session()
X = rng.rand(256, 1500, 4).astype(np.float32)
lengths = np.full((256,), 1500, dtype=np.int32)
cfg = dict(latent_dim=16, nunits=128, attention_size=32,
           epochs=3, batch_size=BS, learning_rate=1e-3, patience=99, lengths=lengths)
t0 = time.time(); feats, hist = detsec_ae(X, cfg); dt = time.time()-t0
losses = [round(float(x), 4) for x in hist['loss']]
print(f"[Afit] 256 samples bs={BS} epochs=3 wall={dt:.1f}s losses={losses} "
      f"-> ~{dt/3:.1f}s/epoch (epoch1 incl compile)", flush=True)
print(f"       feats {feats.shape} finite={np.isfinite(feats).all()}", flush=True)
assert feats.shape == (256, 16) and np.isfinite(feats).all()
print("PHASE_A_DONE", flush=True)
