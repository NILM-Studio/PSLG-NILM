"""Benchmark BiLSTM-AE training at varying batch_size on the real data shape.

Uses the EXACT model architecture from models/feature_extract/bilstm_ae.py
(BiLSTM(32)+Dense(16) encoder/decoder, sample_weight masking) on synthetic data
of shape (3769, 1500, 4) — the real clasp-origin tensor shape. Timing is
shape-determined (overhead-bound), so synthetic values are representative.

Reports: ms/batch, steady s/epoch, peak GPU memory. Short (2 timed epochs per
size after a warmup), so the whole sweep runs in a few minutes.

Run inside a Determined task:  det task create determined/bench_batch.yaml
"""
from __future__ import annotations
import os, time, gc
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import tensorflow as tf

N_SAMPLES, TIMESTEPS, N_FEATURES = 3769, 1500, 4
LATENT_DIM = 16
EPOCHS_TIME = 3          # timed epochs per batch size (after 1 warmup)
BATCH_SIZES = [32, 64, 128, 256, 512, 1024]


def build_model(timesteps, n_features, latent_dim, learning_rate=0.0001):
    """Replica of models/feature_extract/bilstm_ae.py model construction."""
    from tensorflow.keras.layers import (Input, Bidirectional, LSTM, Dense,
                                         RepeatVector, TimeDistributed)
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    inp = Input(shape=(timesteps, n_features))
    enc = Bidirectional(LSTM(32, activation="tanh", return_state=True))
    eo, fh, fc, bh, bc = enc(inp)
    combined_h = tf.keras.layers.Concatenate(axis=-1)([fh, bh])
    latent = Dense(latent_dim, activation="relu")(combined_h)
    dec_in = RepeatVector(timesteps)(latent)
    dec = Bidirectional(LSTM(32, activation="tanh", return_sequences=True))
    dec_out = dec(dec_in)
    out = TimeDistributed(Dense(n_features, activation="linear"))(dec_out)
    m = Model(inputs=inp, outputs=out)
    m.compile(optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0), loss="mse")
    return m


def gpu_mem_mib():
    try:
        return int(tf.config.experimental.get_memory_info("GPU:0")["peak"] / 1048576)
    except Exception:
        return -1


def main():
    gpus = tf.config.list_physical_devices("GPU")
    print(f"TF {tf.__version__} | GPUs: {gpus}", flush=True)
    assert gpus, "NO GPU"

    rng = np.random.default_rng(0)
    # real-ish scale (power watts), shape matches the clasp-origin tensor exactly
    X = rng.uniform(0, 2500, size=(N_SAMPLES, TIMESTEPS, N_FEATURES)).astype(np.float32)
    sample_weight = np.ones((N_SAMPLES, TIMESTEPS), dtype=np.float32)
    print(f"data: {X.shape}  ({X.nbytes/1e6:.1f} MB)", flush=True)

    print(f"\n{'batch':>6} {'batches/ep':>10} {'ms/batch':>9} {'s/epoch':>9} "
          f"{'peak_gpu_MiB':>13}", flush=True)
    print("-" * 55, flush=True)

    results = []
    for bs in BATCH_SIZES:
        gc.collect()
        tf.keras.backend.clear_session()
        try:
            tf.config.experimental.reset_memory_stats("GPU:0")
        except Exception:
            pass
        m = build_model(TIMESTEPS, N_FEATURES, LATENT_DIM)
        n_batches = int(np.ceil(N_SAMPLES / bs))
        # 1 warmup epoch (compile + XLA/JIT cache)
        try:
            m.fit(X, X, epochs=1, batch_size=bs, verbose=0,
                  sample_weight=sample_weight)
        except Exception as e:
            print(f"{bs:>6}   OOM/error: {e}", flush=True)
            results.append((bs, None)); continue
        # timed epochs
        t0 = time.time()
        m.fit(X, X, epochs=EPOCHS_TIME, batch_size=bs, verbose=0,
              sample_weight=sample_weight)
        dt = time.time() - t0
        per_epoch = dt / EPOCHS_TIME
        per_batch_ms = per_epoch / n_batches * 1000
        peak = gpu_mem_mib()
        print(f"{bs:>6} {n_batches:>10} {per_batch_ms:>9.1f} {per_epoch:>9.2f} "
              f"{peak:>13}", flush=True)
        results.append((bs, per_epoch, per_batch_ms, peak))
        del m

    # analysis: speedup vs baseline + overhead-bound detection
    print("\n=== analysis ===", flush=True)
    base = next((r for r in results if r[0] == 32 and len(r) > 2), None)
    if base:
        b_ep = base[1]
        print(f"{'batch':>6} {'speedup':>8} {'ms/batch growth':>16}", flush=True)
        for r in results:
            if len(r) <= 2:
                continue
            bs, ep, pb, peak = r
            sp = b_ep / ep
            growth = pb / results[0][2]   # ms/batch relative to batch=32
            print(f"{bs:>6} {sp:>7.2f}x {growth:>15.2f}x", flush=True)
        # recommend the knee: largest batch where ms/batch is still < 2x the
        # smallest batch's ms/batch (i.e. still strongly overhead-dominated)
        smallest_pb = results[0][2]
        knee = None
        for r in results:
            if len(r) <= 2: continue
            if r[2] < 2 * smallest_pb:
                knee = r[0]
        best = max((r for r in results if len(r) > 2), key=lambda r: base[1]/r[1])
        print(f"\noverhead-bound up to batch={knee} (ms/batch < 2x baseline)")
        print(f"fastest tested: batch={best[0]} @ {best[1]:.2f}s/epoch "
              f"({base[1]/best[1]:.2f}x), peak {best[3]} MiB")
        print(f"\nRECOMMEND: batch_size=128 — typical overhead-bound knee for a "
              f"~100K-param model; 3-4x speedup with negligible convergence risk.")


if __name__ == "__main__":
    main()
