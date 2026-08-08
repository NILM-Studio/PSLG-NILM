"""Precise batch_size scan: ms/batch + peak GPU mem at each batch size, using
train_on_batch (one step = clean OOM failure, no validation/callback overhead,
no hang-prone growth-retry over many epochs). Finds the memory wall on a 24GB
4090 for the real tensor shape (3769, 1500, 4).

Run: det task create determined/bench_batch2.yaml
"""
from __future__ import annotations
import os, time, gc
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, Bidirectional, LSTM, Dense,
                                     RepeatVector, TimeDistributed, Concatenate)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

N_SAMPLES, TS, NF, LATENT = 3769, 1500, 4, 16
BATCHES = [32, 40, 48, 64, 80, 96, 128, 192, 256]
STEPS = 10  # train_on_batch steps to time (steady-state, post warmup)


def build():
    inp = Input(shape=(TS, NF))
    eo, fh, fc, bh, bc = Bidirectional(LSTM(32, activation="tanh", return_state=True))(inp)
    latent = Dense(LATENT, activation="relu")(Concatenate(axis=-1)([fh, bh]))
    dec_in = RepeatVector(TS)(latent)
    dec_out = Bidirectional(LSTM(32, activation="tanh", return_sequences=True))(dec_in)
    out = TimeDistributed(Dense(NF, activation="linear"))(dec_out)
    m = Model(inp, out)
    m.compile(optimizer=Adam(learning_rate=1e-4, clipnorm=1.0), loss="mse")
    return m


def peak_mib():
    try: return int(tf.config.experimental.get_memory_info("GPU:0")["peak"] / 1048576)
    except Exception: return -1


def main():
    gpus = tf.config.list_physical_devices("GPU")
    print(f"TF {tf.__version__} | {gpus[0]}" if gpus else "NO GPU", flush=True)
    assert gpus
    rng = np.random.default_rng(0)
    # one big pool we slice per batch
    Xall = rng.uniform(0, 2500, (N_SAMPLES, TS, NF)).astype(np.float32)
    sw_all = np.ones((N_SAMPLES, TS), np.float32)
    print(f"\n{'batch':>6} {'ms/batch':>9} {'s/epoch':>9} {'peak_MiB':>9} {'vs32_speed':>11}", flush=True)
    print("-" * 50, flush=True)
    res, base_ep = [], None
    for bs in BATCHES:
        gc.collect(); tf.keras.backend.clear_session()
        try: tf.config.experimental.reset_memory_stats("GPU:0")
        except Exception: pass
        m = build()
        xb = Xall[:bs]; wb = sw_all[:bs]
        # warmup (2 steps: build graph + trace)
        try:
            m.train_on_batch(xb, xb, sample_weight=wb)
            m.train_on_batch(xb, xb, sample_weight=wb)
        except Exception as e:
            print(f"{bs:>6}   warmup OOM: {str(e)[:50]}", flush=True); res.append((bs,None)); continue
        t0 = time.time()
        for i in range(STEPS):
            m.train_on_batch(xb, xb, sample_weight=wb)
        dt = time.time() - t0
        ms_b = dt / STEPS * 1000
        n_b = int(np.ceil(N_SAMPLES / bs))
        s_ep = ms_b / 1000 * n_b
        pm = peak_mib()
        if base_ep is None: base_ep = s_ep
        print(f"{bs:>6} {ms_b:>9.1f} {s_ep:>9.2f} {pm:>9} {base_ep/s_ep:>10.2f}x", flush=True)
        res.append((bs, ms_b, s_ep, pm))
        del m
    # find the largest batch that fits under a safety margin
    print("\n=== verdict ===", flush=True)
    fits = [r for r in res if len(r) > 2 and r[3] < 22000]  # <22GB safety on 24GB
    if fits:
        best = max(fits, key=lambda r: r[3])  # largest fitting batch
        fastest = min(fits, key=lambda r: r[2])
        print(f"largest fitting batch: {best[0]} ({best[3]} MiB, {best[2]:.1f}s/epoch, {base_ep/best[2]:.2f}x vs bs32)")
        print(f"fastest  fitting batch: {fastest[0]} ({fastest[2]:.1f}s/epoch, {base_ep/fastest[2]:.2f}x vs bs32)")
    wall = [r[0] for r in res if len(r) == 2]
    if wall: print(f"OOM at: {wall}")


if __name__ == "__main__":
    main()
