"""Physical-constraint DeTSEC feature extractor (TF/Keras port).

Based on the standalone script ``特征提取_物理约束DeTSEC.py``, ported to the
TensorFlow/Keras runtime used by every other feature model in
``models/feature_extract/`` so it runs in the existing PSLG-NILM conda env on
the 4090 GPU pool (no torch dependency). The architecture is preserved
layer-for-layer:

  1. Masked temporal attention
     ``λ_j = exp(v_j^T u_a)·m_j / Σ_k exp(v_k^T u_a)·m_k`` — padding positions
     never contribute to the attention weights, so the embedding does not leak
     primitive-length information.

  2. Teacher-forcing nonnegative structural decoder
     Decoder input is ``[x_{t-1}; z]``; nonnegative physical channels (active
     power etc.) use a Softplus output activation, so ``x̂ ≥ 0`` holds by
     construction.

  3. Masked, length-normalized bidirectional reconstruction loss
     ``L_ae`` is normalized by each sequence's true length ``T_pri,i`` so
     variable-length primitives contribute evenly.

  4. Edge-preserving smoothing constraint (Charbonnier TV, scheme A)
     ``L_phy`` is an L1-type penalty over nonnegative channels: steady-state
     plateau stays flat while sparse large steps (resistance wire on/off) are
     only linearly penalized — "flat steady states, free changepoints".

Total objective: ``L = L_ae + λ_phy · L_phy``.

Adapter contract (used by ``FeatureExtractStep``)::

    features, history = detsec_pc(np_data, model_config)
        np_data      : (n, timesteps, F) float32 padded tensor
        model_config : latent_dim / embed_dim / lambda_phy / nonneg_channels /
                       norm_mode ("znorm" | "minmax") / embed_proj ("relu" |
                       "none") / epochs / batch_size / learning_rate / patience
                       / lengths
        features     : (n, embed_dim)   <- encoder embeddings
        history      : dict with loss / val_loss / l_ae / l_phy / epochs_trained

``norm_mode="minmax"`` is the power-level-preserving per-channel percentile-
clipped (1%/99%) global MinMax; ``embed_proj="relu"`` adds the sparse
low-rank embedding projection (directions 1A + 3 of the v2 analysis).
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import ops
from tensorflow.keras.layers import (Bidirectional, Dense, GRU, Layer)

# ============================================================
# 1. 掩码时序注意力 / 门控融合 / 非负解码器
# ============================================================


class MaskedTemporalAttention(Layer):
    """sv maskable temporal attention: h_att = Σ λ_j h_j, λ over real steps only."""

    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = int(dim)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="W", shape=(self.dim, self.dim),
            initializer=tf.keras.initializers.GlorotNormal(), trainable=True)
        self.b_a = self.add_weight(
            name="b_a", shape=(self.dim,),
            initializer=tf.keras.initializers.Zeros(), trainable=True)
        self.u = self.add_weight(
            name="u", shape=(self.dim, 1),
            initializer=tf.keras.initializers.GlorotNormal(), trainable=True)
        super().build(input_shape)

    def call(self, H, mask):
        # H: (B, T, l)  mask: (B, T)
        score = ops.squeeze(ops.matmul(
            ops.tanh(ops.matmul(H, self.W) + self.b_a), self.u), axis=-1)   # (B, T)
        score = ops.where(mask < 0.5, tf.constant(-1e9, dtype=score.dtype), score)
        lam = ops.softmax(score, axis=-1)                                   # (B, T)
        h_att = ops.sum(ops.expand_dims(lam, -1) * H, axis=1)               # (B, l)
        return h_att, lam


class GatedFusion(Layer):
    """z = sigmoid(g_f)·h_fw + sigmoid(g_b)·h_bw (adaptive gated fusion)."""

    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.gate_forw = Dense(dim, activation="sigmoid", name="gate_forw")
        self.gate_back = Dense(dim, activation="sigmoid", name="gate_back")

    def call(self, h_forw, h_back):
        return self.gate_forw(h_forw) * h_forw + self.gate_back(h_back) * h_back


class NonNegTeacherForcingDecoder(Layer):
    """Teacher-forcing GRU decoder + per-channel output activation.

    Training input at each step is the previous true sample concatenated with
    the embedding (``[x_{t-1}; z]``), so the decoder focuses on waveform
    dynamics (steps / surges / ripple). The GRU is seeded with ``h_0 = z``.

    ``keep_mask`` (B, T, 1) implements scheduled-sampling-style decoder input
    mixing (direction 3 ablation): keep=1 feeds the true x_{t-1}, keep=0 feeds
    a zero vector, so with a low keep ratio the decoder is forced to rely on
    the embedding z alone (z-only conditioning, like the detsec baseline).

    Nonnegative physical channels f∈C+ use the configured activation:
      - softplus       : ln(1+e^x)  (vanilla, x=0 maps to ln2≈0.693)
      - softplus_offset: softplus(x)-ln2  (zero-point calibrated, 0 maps to 0)
      - relu           : max(x, 0)  (hard, no offset bias)
    """

    def __init__(self, embed_dim, n_features, nonneg_channels,
                 nonneg_activation="softplus", **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = int(embed_dim)
        self.n_features = int(n_features)
        self.nonneg_channels = sorted(int(c) for c in (nonneg_channels or []))
        self.nonneg_activation = str(nonneg_activation).lower()
        if self.nonneg_activation == "softplus":
            self._nn_act = ops.softplus
        elif self.nonneg_activation == "softplus_offset":
            self._nn_act = lambda v: ops.softplus(v) - 0.6931471805599453
        elif self.nonneg_activation == "relu":
            self._nn_act = ops.relu
        else:
            raise ValueError(f"[detsec_pc] unknown nonneg_activation "
                             f"'{nonneg_activation}' (softplus | "
                             "softplus_offset | relu)")
        self.rnn = GRU(
            self.embed_dim, activation="tanh", recurrent_activation="sigmoid",
            recurrent_dropout=0.0, use_bias=True, return_sequences=True)
        self.head = Dense(self.n_features, name="dec_head")

    def call(self, z, x_shift, keep_mask=None):
        # z: (B, l) ; x_shift: (B, T, F) ; keep_mask: (B, T, 1) or None
        if keep_mask is not None:
            x_shift = x_shift * keep_mask            # scheduled-sampling mixing
        T = tf.shape(x_shift)[1]
        z_t = tf.tile(z[:, None, :], [1, T, 1])                             # (B, T, l)
        inp = ops.concatenate([x_shift, z_t], axis=-1)                      # (B, T, F+l)
        out = self.rnn(inp, initial_state=[z])                              # (B, T, l)
        x_tilde = self.head(out)                                            # (B, T, F)
        if not self.nonneg_channels:
            return x_tilde
        chan = tf.reduce_sum(tf.one_hot(self.nonneg_channels, self.n_features,
                                        dtype=x_tilde.dtype), axis=0)        # (F,)
        chan_mask = chan[None, None, :]                                      # (1,1,F)
        x_hat = x_tilde * (1.0 - chan_mask) + self._nn_act(x_tilde) * chan_mask
        return x_hat


# ============================================================
# 2. 完整模型：双向 GRU 编码器 + 双向非负解码器
# ============================================================


class PhyConstrainedDeTSEC(Layer):
    """Physical-constraint DeTSEC stage-1 feature extractor (no clustering).

    embed_proj: "relu" adds a Dense(embed_dim, relu) projection on top of the
    gated fusion (sparse, low-rank embedding geometry — direction 3), "none"
    uses the gated fusion output directly.
    """

    def __init__(self, n_features, embed_dim, nonneg_channels,
                 embed_proj="none", nonneg_activation="softplus", **kwargs):
        super().__init__(**kwargs)
        self.encoder = Bidirectional(
            GRU(embed_dim, activation="tanh", recurrent_activation="sigmoid",
                recurrent_dropout=0.0, use_bias=True, return_sequences=True),
            merge_mode=None, name="encoder_biGRU")
        self.att_forw = MaskedTemporalAttention(embed_dim, name="attn_fw")
        self.att_back = MaskedTemporalAttention(embed_dim, name="attn_bw")
        self.fusion = GatedFusion(embed_dim, name="gated_fuse")
        self.dec_forw = NonNegTeacherForcingDecoder(
            embed_dim, n_features, nonneg_channels,
            nonneg_activation=nonneg_activation, name="dec_forw")
        self.dec_back = NonNegTeacherForcingDecoder(
            embed_dim, n_features, nonneg_channels,
            nonneg_activation=nonneg_activation, name="dec_back")
        if str(embed_proj).lower() == "relu":
            self.embed_proj = Dense(embed_dim, activation="relu", name="embedding")
        elif str(embed_proj).lower() in ("none", ""):
            self.embed_proj = None
        else:
            raise ValueError(f"[detsec_pc] unknown embed_proj '{embed_proj}' "
                             "(relu | none)")

    def encode(self, x, mask):
        """(B, T, F) x (B, T) -> embedding (B, l)."""
        H_fw, H_bw = self.encoder(x)            # 各 (B, T, l)
        h_fw, _ = self.att_forw(H_fw, mask)
        h_bw, _ = self.att_back(H_bw, mask)
        z = self.fusion(h_fw, h_bw)             # (B, l)
        if self.embed_proj is not None:
            z = self.embed_proj(z)              # 稀疏非负投影 -> 低秩几何
        return z

    def call(self, x, mask, lengths, keep_mask=None):
        z = self.encode(x, mask)
        x_rev = reverse_sequences(x, lengths)
        x_shift_f = ops.concatenate([ops.zeros_like(x[:, :1]), x[:, :-1]], axis=1)
        x_shift_b = ops.concatenate([ops.zeros_like(x_rev[:, :1]), x_rev[:, :-1]], axis=1)
        x_hat_f = self.dec_forw(z, x_shift_f, keep_mask)   # 非负通道 ≥ 0
        x_hat_b = self.dec_back(z, x_shift_b, keep_mask)   # 对应反转序列
        return z, x_hat_f, x_hat_b


# ============================================================
# 3. 掩码双向重构损失 + Charbonnier TV 保边平滑
# ============================================================


def reverse_sequences(x, lengths):
    """Reverse only each sequence's valid region (padding untouched)."""
    B = tf.shape(x)[0]
    T = tf.shape(x)[1]
    F = x.shape[-1]
    idx = tf.broadcast_to(tf.range(T)[None, :], (B, T))
    rev = tf.clip_by_value(lengths[:, None] - 1 - idx, 0, T - 1)            # (B, T)
    bb = tf.broadcast_to(tf.range(B)[:, None, None], (B, T, F))
    rr = tf.broadcast_to(ops.expand_dims(rev, -1), (B, T, F))
    col = tf.broadcast_to(tf.range(F)[None, None, :], (B, T, F))
    return tf.gather_nd(x, tf.stack([bb, rr, col], axis=-1))                # (B, T, F)


def masked_reconstruction_loss(x, x_hat, mask):
    """Length-normalized masked MSE: only positions with m_j = 1 count."""
    se = ops.sum((x - x_hat) ** 2, axis=-1) * mask                          # (B, T)
    per_sample = ops.sum(se, axis=1) / ops.maximum(ops.sum(mask, axis=1), 1.0)
    return ops.mean(per_sample)


def charbonnier_tv_loss(x_hat, mask, channels, eps=1e-6):
    """Edge-preserving smoothing (scheme A): L1-type on adjacent differences."""
    if not channels:
        return x_hat * 0.0
    diff = (tf.gather(x_hat[:, 1:], channels, axis=-1)
            - tf.gather(x_hat[:, :-1], channels, axis=-1))                 # (B, T-1, |C+|)
    w = mask[:, 1:] * mask[:, :-1]                                         # (B, T-1)
    tv = ops.sum(ops.sqrt(diff * diff + eps), axis=-1) * w                 # (B, T-1)
    per_sample = ops.sum(tv, axis=1) / ops.maximum(ops.sum(w, axis=1), 1.0)
    return ops.mean(per_sample)


def total_loss(x, x_hat_f, x_hat_b, mask, lengths, nonneg_channels, lambda_phy):
    """L = L_ae + λ_phy · L_phy (bidirectional). Returns (loss, l_ae, l_phy)."""
    x_rev = reverse_sequences(x, lengths)
    l_ae = (masked_reconstruction_loss(x, x_hat_f, mask)
            + masked_reconstruction_loss(x_rev, x_hat_b, mask))
    l_phy = (charbonnier_tv_loss(x_hat_f, mask, nonneg_channels)
             + charbonnier_tv_loss(x_hat_b, mask, nonneg_channels))
    return l_ae + lambda_phy * l_phy, l_ae, l_phy


# ============================================================
# 4. 训练 + 嵌入提取
# ============================================================


def _per_sequence_znorm(X, lengths):
    """Per-sequence z-score normalization (shape/transition patterns, not power)."""
    X = np.asarray(X, dtype=np.float32)
    lengths = np.asarray(lengths).reshape(-1).astype(np.int32)
    n, T, F = X.shape
    Xn = X.copy()
    for i in range(n):
        L = min(int(lengths[i]), T)
        seg = X[i, :L]
        mu = seg.mean(axis=0, keepdims=True)
        sd = seg.std(axis=0, keepdims=True)
        Xn[i, :L] = (seg - mu) / (sd + 1e-8)
        Xn[i, L:] = 0.0
    return Xn


def _per_channel_minmax(X, lengths, q_low=1.0, q_high=99.0):
    """Per-channel global MinMax with percentile clipping (direction 1A).

    Quantiles are computed over VALID positions only (padding excluded), so
    rare surge spikes cannot stretch the scale: values are clipped to the
    [q_low, q_high] percentiles then mapped to [0, 1] per channel. Absolute
    power levels are preserved, which is the strongest state-separation
    signal for real NILM primitives.
    """
    X = np.asarray(X, dtype=np.float32)
    lengths = np.asarray(lengths).reshape(-1).astype(np.int32)
    n, T, F = X.shape
    mask = np.zeros((n, T), dtype=bool)
    for i in range(n):
        mask[i, :min(int(lengths[i]), T)] = True
    Xn = X.copy()
    for f in range(F):
        vals = Xn[mask, f]
        lo, hi = np.percentile(vals, [q_low, q_high])
        scale = float(hi - lo)
        Xn[mask, f] = (np.clip(Xn[mask, f], lo, hi) - lo) / (scale + 1e-7)
    Xn[~mask] = 0.0
    return Xn


def _normalize(X, lengths, mode="znorm"):
    """Config-driven input normalization (znorm | minmax)."""
    mode = str(mode).lower()
    if mode == "znorm":
        return _per_sequence_znorm(X, lengths)
    if mode == "minmax":
        return _per_channel_minmax(X, lengths)
    raise ValueError(f"[detsec_pc] unknown norm_mode '{mode}' (znorm | minmax)")


def _make_batches(Xn, lengths, batch_size, shuffle=True, random_state=0):
    # NOTE: always emit the global padded width so the tf.function train/eval
    # steps trace exactly ONE graph (per-batch T would retrace per length).
    n = Xn.shape[0]
    T = Xn.shape[1]
    idx_grid = np.arange(T)[None, :]
    order = np.random.RandomState(random_state).permutation(n) if shuffle \
        else np.arange(n)
    for s in range(0, n, batch_size):
        ids = order[s:s + batch_size]
        x = Xn[ids]
        mask = (idx_grid < lengths[ids, None]).astype(np.float32)
        yield (tf.convert_to_tensor(x),
               tf.convert_to_tensor(mask),
               tf.convert_to_tensor(lengths[ids], dtype=tf.int32))


@tf.function
def _train_step(model, optimizer, x, mask, lens, keep_mask,
                nonneg_channels, lambda_phy):
    with tf.GradientTape() as tape:
        _, x_hat_f, x_hat_b = model(x, mask, lens, keep_mask)
        loss, l_ae, l_phy = total_loss(x, x_hat_f, x_hat_b, mask, lens,
                                       nonneg_channels, lambda_phy)
    grads = tape.gradient(loss, model.trainable_variables)
    grads = [g if g is not None else tf.zeros_like(v)
             for g, v in zip(grads, model.trainable_variables)]
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, l_ae, l_phy


@tf.function
def _eval_step(model, x, mask, lens, keep_mask, nonneg_channels, lambda_phy):
    _, x_hat_f, x_hat_b = model(x, mask, lens, keep_mask)
    loss, _, _ = total_loss(x, x_hat_f, x_hat_b, mask, lens,
                            nonneg_channels, lambda_phy)
    return loss


def _ones_mask(x, mask):
    """Full teacher-forcing keep mask (all ones) for eval/extract."""
    return tf.ones_like(x[:, :, :1]) * mask[:, :, None]


def _sampling_mask(x, ratio, seed):
    """Bernoulli keep mask (B, T, 1) for scheduled-sampling input mixing."""
    if ratio >= 1.0:
        return None  # full teacher forcing: no mixing ops at all
    B, T = tf.shape(x)[0], tf.shape(x)[1]
    rnd = tf.random.uniform((B, T, 1), seed=seed)
    return tf.cast(rnd < ratio, dtype=tf.float32)


def _train_epoch(model, optimizer, Xn, lengths, nonneg_channels, lambda_phy,
                 batch_size, seed, tf_ratio=1.0):
    tot = tot_ae = tot_phy = 0.0
    nb = 0
    for x, mask, lens in _make_batches(Xn, lengths, batch_size, True, seed):
        keep = _sampling_mask(x, tf_ratio, seed + nb)
        loss, l_ae, l_phy = _train_step(model, optimizer, x, mask, lens,
                                        keep, nonneg_channels, lambda_phy)
        tot += float(loss); tot_ae += float(l_ae); tot_phy += float(l_phy)
        nb += 1
    return tot / max(nb, 1), tot_ae / max(nb, 1), tot_phy / max(nb, 1)


def _eval_loss(model, Xn, lengths, nonneg_channels, lambda_phy, batch_size):
    tot = 0.0
    nb = 0
    for x, mask, lens in _make_batches(Xn, lengths, batch_size, False):
        keep = _ones_mask(x, mask)  # full teacher forcing for validation
        loss = _eval_step(model, x, mask, lens, keep,
                          nonneg_channels, lambda_phy)
        tot += float(loss); nb += 1
    return tot / max(nb, 1)


def _extract(model, Xn, lengths, batch_size):
    zs = []
    for x, mask, lens in _make_batches(Xn, lengths, batch_size, False):
        keep = _ones_mask(x, mask)
        z, _, _ = model(x, mask, lens, keep)
        zs.append(z.numpy())
    return np.concatenate(zs, axis=0)


def train_feature_extractor(X, lengths, n_features, nonneg_channels,
                            embed_dim=32, lambda_phy=0.1, lr=1e-4,
                            batch_size=16, epochs=50, patience=5,
                            random_state=0, verbose=True,
                            norm_mode="znorm", embed_proj="none",
                            nonneg_activation="softplus", tf_ratio=1.0,
                            tf_schedule="constant"):
    """Train the physical-constraint DeTSEC stage-1 feature extractor.

    X       : (n, timesteps, F) padded tensor
    lengths : (n,) / (n,1) true lengths
    norm_mode : "znorm" (per-sequence z-score) | "minmax" (per-channel
                percentile-clipped global MinMax — preserves power levels)
    embed_proj : "relu" (sparse low-rank embedding projection) | "none"
    nonneg_activation : "softplus" | "softplus_offset" (zero-point
                calibrated) | "relu"
    tf_ratio : teacher-forcing keep ratio in [0,1]; 1.0 = full teacher
                forcing, 0.0 = decoder input is z-only (no x_{t-1})
    tf_schedule : "constant" (use tf_ratio every epoch) | "linear"
                (ratio decays 1.0 -> tf_ratio across epochs)
    Returns (model, Xn, training_history).
    """
    Xn = _normalize(X, lengths, norm_mode)
    lengths = np.asarray(lengths).reshape(-1).astype(np.int32)
    n = Xn.shape[0]
    model = PhyConstrainedDeTSEC(n_features, embed_dim, nonneg_channels,
                                 embed_proj=embed_proj,
                                 nonneg_activation=nonneg_activation)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)

    # validation split + early stopping (mirror detsec_model)
    val_ids = None
    if n >= 5:
        rng = np.random.RandomState(random_state)
        val_ids = rng.choice(n, size=max(1, n // 5), replace=False)
    train_ids = np.setdiff1d(np.arange(n), val_ids) if val_ids is not None \
        else np.arange(n)

    X_tr, len_tr = Xn[train_ids], lengths[train_ids]
    X_va, len_va = (Xn[val_ids], lengths[val_ids]) if val_ids is not None \
        else (None, None)

    tf_ratio = float(tf_ratio)
    tf_schedule = str(tf_schedule).lower()

    def _ratio_for_epoch(epoch):
        if tf_schedule == "linear":
            frac = (epoch - 1) / max(epochs - 1, 1)   # 0 -> 1 across epochs
            return max(0.0, tf_ratio + (1.0 - tf_ratio) * (1.0 - frac))
        return tf_ratio

    history = {"loss": [], "val_loss": [], "l_ae": [], "l_phy": [],
               "tf_ratio_used": [], "epochs_trained": 0, "model_name": "detsec_pc"}
    best_val, best_weights, best_epoch = float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        ratio = _ratio_for_epoch(epoch)
        l, l_ae, l_phy = _train_epoch(model, optimizer, X_tr, len_tr,
                                      nonneg_channels, lambda_phy, batch_size,
                                      random_state + epoch, tf_ratio=ratio)
        val = (_eval_loss(model, X_va, len_va, nonneg_channels, lambda_phy,
                          batch_size) if X_va is not None else l)
        history["loss"].append(l)
        history["val_loss"].append(val)
        history["l_ae"].append(l_ae)
        history["l_phy"].append(l_phy)
        history["tf_ratio_used"].append(ratio)
        history["epochs_trained"] = epoch
        if val < best_val:
            best_val, best_weights, best_epoch = val, model.get_weights(), epoch
        if verbose and (epoch == 1 or epoch % max(1, epochs // 5) == 0):
            print(f"[detsec_pc] Epoch {epoch:3d}/{epochs}  L={l:.4f}  "
                  f"L_ae={l_ae:.4f}  L_phy={l_phy:.4f}  val={val:.4f}  "
                  f"tf={ratio:.2f}")
        if patience > 0 and epoch - best_epoch >= patience:
            if verbose:
                print(f"[detsec_pc] early stop @ epoch {epoch} (best val "
                      f"{best_val:.4f} @ {best_epoch})")
            break
    if best_weights is not None:
        model.set_weights(best_weights)
    return model, Xn, history


def detsec_pc(data, model_config):
    """FeatureExtractStep adapter: (n, T, F) + config -> (features, history)."""
    X = data["X"] if isinstance(data, dict) else data
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError(f"[detsec_pc] expected (n, timesteps, F), got {X.shape}")
    n_samples, timesteps, n_features = X.shape
    lengths = model_config.get("lengths")
    lengths = np.asarray(lengths).reshape(-1).astype(np.int32) \
        if lengths is not None else np.full(n_samples, timesteps, dtype=np.int32)
    if len(lengths) != n_samples:
        raise ValueError(f"[detsec_pc] lengths {len(lengths)} != samples {n_samples}")

    embed_dim = int(model_config.get("embed_dim", 32))
    lambda_phy = float(model_config.get("lambda_phy", 0.1))
    nonneg_channels = model_config.get("nonneg_channels", [0, 1, 2, 3])
    norm_mode = str(model_config.get("norm_mode", "znorm"))
    embed_proj = str(model_config.get("embed_proj", "none"))
    nonneg_activation = str(model_config.get("nonneg_activation", "softplus"))
    tf_ratio = float(model_config.get("tf_ratio", 1.0))
    tf_schedule = str(model_config.get("tf_schedule", "constant"))
    lr = float(model_config.get("learning_rate", 1e-4))
    batch_size = int(model_config.get("batch_size", 16))
    epochs = int(model_config.get("epochs", 50))
    patience = int(model_config.get("patience", 5))

    print(f"[detsec_pc] training embed_dim={embed_dim} lambda_phy={lambda_phy} "
          f"nonneg_channels={nonneg_channels} norm_mode={norm_mode} "
          f"embed_proj={embed_proj} nonneg_activation={nonneg_activation} "
          f"tf_ratio={tf_ratio} tf_schedule={tf_schedule} input={X.shape}")
    model, Xn, history = train_feature_extractor(
        X, lengths, n_features, nonneg_channels,
        embed_dim=embed_dim, lambda_phy=lambda_phy, lr=lr,
        batch_size=batch_size, epochs=epochs, patience=patience,
        norm_mode=norm_mode, embed_proj=embed_proj,
        nonneg_activation=nonneg_activation, tf_ratio=tf_ratio,
        tf_schedule=tf_schedule)

    features = _extract(model, Xn, lengths, batch_size)
    print(f"[detsec_pc] features: {features.shape}")
    return features, history
