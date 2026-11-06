"""Explicit split, update-budget trainers for the approved representation protocol.

Inputs are already scaled float arrays [N,128,1]. No fit/transform of input
scales, random validation split, PCA, or clustering is performed here.
"""
from __future__ import annotations

import json
import os
import pickle
import random
import time
from pathlib import Path

# Set before importing TF; no CUDA visibility is changed by this module.
for _thread_variable in ("TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "OMP_NUM_THREADS"):
    os.environ[_thread_variable] = str(min(4, max(1, int(os.environ.get(_thread_variable, "2")))))
import numpy as np
import tensorflow as tf

for _gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError as exc:
        raise RuntimeError("Import tf_formal before initializing GPU tensors") from exc


@tf.keras.utils.register_keras_serializable(package="pslg_formal")
class AttentionLayer(tf.keras.layers.Layer):
    """Local copy of the existing DeTSEC stage-one temporal attention."""
    def __init__(self, nunits=128, attention_size=32, **kwargs):
        super().__init__(**kwargs)
        self.nunits = nunits
        self.attention_size = attention_size

    def build(self, input_shape):
        self.W_omega = self.add_weight(name="W_omega", shape=(self.nunits, self.attention_size), initializer=tf.keras.initializers.GlorotNormal())
        self.b_omega = self.add_weight(name="b_omega", shape=(self.attention_size,), initializer=tf.keras.initializers.RandomNormal(stddev=0.1))
        self.u_omega = self.add_weight(name="u_omega", shape=(self.attention_size,), initializer=tf.keras.initializers.RandomNormal(stddev=0.1))
        super().build(input_shape)

    def call(self, inputs):
        scores = tf.einsum("bta,a->bt", tf.tanh(tf.einsum("btu,ua->bta", inputs, self.W_omega) + self.b_omega), self.u_omega)
        return tf.reduce_sum(inputs * tf.nn.softmax(scores, axis=1)[:, :, None], axis=1)

    def get_config(self):
        return {**super().get_config(), "nunits": self.nunits, "attention_size": self.attention_size}


def _array(X, name, allow_empty=False):
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3 or X.shape[1:] != (128, 1):
        raise ValueError(f"{name} must have shape [N,128,1], got {X.shape}")
    if (not allow_empty and len(X) == 0) or not np.isfinite(X).all():
        raise ValueError(f"{name} must be finite and nonempty")
    return X


def _build(config):
    L = tf.keras.layers
    name = config["model"]
    dim = int(config.get("native_dim", 64))
    if dim != 64:
        raise ValueError("Approved CNN-AE/DeTSEC architecture has native_dim=64")
    inp = L.Input((128, 1), name="inputs")
    latent = L.Input((dim,), name="latent_input")
    if name == "cnn_ae":
        if list(config.get("conv_channels", [32, 64])) != [32, 64]:
            raise ValueError("Approved convolution widths are [32,64]")
        x = L.Conv1D(32, 5, padding="same", activation="relu", name="enc_conv1")(inp)
        x = L.BatchNormalization(name="enc_bn1")(x)
        x = L.MaxPooling1D(2, padding="same", name="enc_pool1")(x)
        x = L.Conv1D(64, 3, padding="same", activation="relu", name="enc_conv2")(x)
        x = L.BatchNormalization(name="enc_bn2")(x)
        x = L.MaxPooling1D(2, padding="same", name="enc_pool2")(x)
        x = L.GlobalAveragePooling1D(name="enc_gap")(x)
        z = L.Dense(dim, activation="linear", name="embedding")(x)
        y = L.Dense(32 * 64, activation="relu", name="dec_dense")(latent)
        y = L.Reshape((32, 64), name="dec_reshape")(y)
        y = L.UpSampling1D(2, name="dec_up1")(y)
        y = L.Conv1D(64, 3, padding="same", activation="relu", name="dec_conv1")(y)
        y = L.BatchNormalization(name="dec_bn1")(y)
        y = L.UpSampling1D(2, name="dec_up2")(y)
        y = L.Conv1D(32, 3, padding="same", activation="relu", name="dec_conv2")(y)
        y = L.BatchNormalization(name="dec_bn2")(y)
        y = L.Conv1D(1, 3, padding="same", activation="linear", name="reconstruct_raw")(y)
    elif name == "detsec_stage1":
        units = int(config.get("gru_units_per_direction", 128))
        attn = int(config.get("attention_size", 32))
        if (units, attn) != (128, 32):
            raise ValueError("Approved DeTSEC widths are GRU128/attention32")
        def gru():
            return L.GRU(units, activation="tanh", recurrent_activation="sigmoid", recurrent_dropout=0., use_bias=True, return_sequences=True)
        fw, bw = L.Bidirectional(gru(), merge_mode=None, name="encoder_biGRU")(inp)
        fw = AttentionLayer(units, attn, name="attn_fw")(fw)
        bw = AttentionLayer(units, attn, name="attn_bw")(bw)
        fw = L.Multiply(name="gate_mul_fw")([L.Dense(units, activation="sigmoid", name="gate_fw")(fw), fw])
        bw = L.Multiply(name="gate_mul_bw")([L.Dense(units, activation="sigmoid", name="gate_bw")(bw), bw])
        z = L.Dense(dim, activation="relu", name="embedding")(L.Add(name="gated_fuse")([fw, bw]))
        y = L.RepeatVector(128, name="repeat")(latent)
        y = L.Bidirectional(gru(), merge_mode="concat", name="decoder_biGRU")(y)
        y = L.TimeDistributed(L.Dense(1, activation="linear"), name="recon")(y)
    else:
        raise ValueError(f"Unsupported TensorFlow representation: {name}")
    encoder = tf.keras.Model(inp, z, name=name + "_encoder")
    decoder = tf.keras.Model(latent, y, name=name + "_decoder")
    autoencoder = tf.keras.Model(inp, decoder(encoder(inp)), name=name + "_autoencoder")
    return encoder, decoder, autoencoder


def _json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


class FrozenModel:
    """Frozen inference interface. Scaling is owned by the caller."""
    def __init__(self, encoder, decoder, config, optimizer=None, checkpoint=None):
        self.encoder, self.decoder, self.config = encoder, decoder, config
        self.optimizer, self.checkpoint = optimizer, checkpoint

    def encode(self, X, batch_size=32):
        X = _array(X, "X", allow_empty=True)
        if not len(X):
            return np.empty((0, 64), dtype=np.float32)
        values = [self.encoder(X[i:i + batch_size], training=False).numpy() for i in range(0, len(X), batch_size)]
        out = np.concatenate(values)
        if not np.isfinite(out).all():
            raise FloatingPointError("Nonfinite encoder output")
        return out

    def reconstruct(self, X, batch_size=32):
        X = _array(X, "X", allow_empty=True)
        if not len(X):
            return X.copy()
        return np.concatenate([self.decoder(self.encoder(X[i:i + batch_size], training=False), training=False).numpy() for i in range(0, len(X), batch_size)])


def encode(model, X, batch_size=32):
    return model.encode(X, batch_size=batch_size)


def load(output_dir, checkpoint="selected"):
    root = Path(output_dir)
    config = json.loads((root / "config.actual.json").read_text())
    if checkpoint == "selected":
        checkpoint = json.loads((root / "result.json").read_text())["selected_checkpoint"]
    directory = root / checkpoint
    encoder, decoder, model = _build(config)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"], clipnorm=1.0)
    optimizer.build(model.trainable_variables)
    counter = tf.Variable(0, dtype=tf.int64, trainable=False)
    generator = tf.random.Generator.from_seed(config["seed"])
    state = tf.train.Checkpoint(encoder=encoder, decoder=decoder, optimizer=optimizer, step=counter, rng=generator)
    # write(), unlike save(), has no save_counter entry; model and slots must match.
    state.read(str(directory / "state")).assert_consumed()
    return FrozenModel(encoder, decoder, config, optimizer, state)


def train(config, X_train, X_val, output_dir, seed=0, steps_cap=5000,
          min_steps=500, validate_every=100, patience=10, fixed_steps=None):
    """Train and save a selected checkpoint; return a JSON-compatible manifest.

    config requires model; learning_rate or lr_multiplier are optional.
    fixed_steps disables early stopping and selects final weights, for full refit.
    Short pilots use steps_cap<min_steps and explicitly select the final fallback.
    """
    X_train = _array(X_train, "X_train")
    if X_val is None:
        X_val = np.empty((0, 128, 1), dtype=np.float32)
    X_val = _array(X_val, "X_val", allow_empty=True)
    if fixed_steps is None and not len(X_val):
        raise ValueError("Heldout training requires explicit activity validation")
    if min(steps_cap, validate_every, patience) <= 0 or min_steps < 0:
        raise ValueError("Update/patience limits must be positive and min_steps nonnegative")
    if fixed_steps is not None and int(fixed_steps) <= 0:
        raise ValueError("fixed_steps must be positive")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "config.actual.json").exists():
        raise FileExistsError("Use a new output directory; never overwrite a prior training")
    config = dict(config)
    name = config["model"]
    if name == "cnn_ae":
        config.setdefault("conv_channels", [32, 64])
        config["embedding_activation"] = "linear"
    elif name == "detsec_stage1":
        config.setdefault("gru_units_per_direction", 128)
        config.setdefault("attention_size", 32)
        config["embedding_activation"] = "relu"
    base_lr = {"cnn_ae": 1e-3, "detsec_stage1": 1e-4}[name]
    config.update(seed=int(seed), native_dim=int(config.get("native_dim", 64)),
                  learning_rate=float(config.get("learning_rate", base_lr * config.get("lr_multiplier", 1.))),
                  batch_size=min(int(config.get("batch_size", 32)), len(X_train)),
                  steps_cap=int(steps_cap), min_steps=int(min_steps), validate_every=int(validate_every),
                  patience=int(patience), fixed_steps=None if fixed_steps is None else int(fixed_steps),
                  relative_improvement=float(config.get("relative_improvement", .001)),
                  n_train=len(X_train), n_validation=len(X_val), input_shape=[128, 1],
                  preprocessing="caller_train_fitted_scaler_no_internal_scaling", validation_split=None,
                  learning_rate_schedule="constant", optimizer="Adam", clipnorm=1.0,
                  weight_decay=0.0, objective="mean_squared_reconstruction_error")
    if config["batch_size"] < 1:
        raise ValueError("batch_size must be positive")
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    generator = tf.random.Generator.from_seed(seed)
    tf.random.set_global_generator(generator)
    sampler = np.random.default_rng(seed)
    encoder, decoder, model = _build(config)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"], clipnorm=1.)
    optimizer.build(model.trainable_variables)
    counter = tf.Variable(0, dtype=tf.int64, trainable=False)
    state = tf.train.Checkpoint(encoder=encoder, decoder=decoder, optimizer=optimizer, step=counter, rng=generator)
    config["parameter_count"] = model.count_params()
    config["visible_gpu_names"] = [x.name for x in tf.config.list_logical_devices("GPU")]
    config["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    config["tensorflow_version"] = tf.__version__
    config["thread_limits"] = {key: int(os.environ[key]) for key in ("TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "OMP_NUM_THREADS")}
    _json(root / "config.actual.json", config)
    spec = tf.TensorSpec([None, 128, 1], tf.float32)

    @tf.function(input_signature=[spec], reduce_retracing=True)
    def update(batch):
        with tf.GradientTape() as tape:
            reconstructed = model(batch, training=True)
            loss = tf.reduce_mean(tf.square(batch - reconstructed))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        counter.assign_add(1)
        return loss

    @tf.function(input_signature=[spec], reduce_retracing=True)
    def validation_batch(batch):
        return tf.reduce_sum(tf.square(batch - model(batch, training=False)))

    permutation = sampler.permutation(len(X_train))
    cursor = 0
    best_loss, best_step, checks = None, None, 0
    stopping_reference = None
    history = []
    total_steps = int(fixed_steps if fixed_steps is not None else steps_cap)
    started = time.monotonic()
    prev_time, prev_step = started, 0
    status = "running"

    def save_checkpoint(label, step):
        directory = root / label
        directory.mkdir(exist_ok=True)
        state.write(str(directory / "state"))
        with (directory / "rng.pkl").open("wb") as handle:
            pickle.dump({"python": random.getstate(), "numpy_global": np.random.get_state(),
                         "sampler": sampler.bit_generator.state, "permutation": permutation,
                         "cursor": cursor, "step": step, "best_loss": best_loss,
                         "best_step": best_step, "patience_checks": checks,
                         "stopping_reference": stopping_reference,
                         "tensorflow_generator": generator.state.numpy()}, handle)
        _json(directory / "metadata.json", {"step": step, "best_step": best_step,
              "best_validation_loss": best_loss, "optimizer_iterations": int(optimizer.iterations.numpy()),
              "encoder_and_decoder_saved_in": "state", "rng_file_trusted_local_only": "rng.pkl"})

    with (root / "progress.jsonl").open("a", buffering=1) as progress:
        for step in range(1, total_steps + 1):
            if cursor >= len(permutation):
                permutation = sampler.permutation(len(X_train))
                cursor = 0
            rows = permutation[cursor:cursor + config["batch_size"]]
            cursor += len(rows)
            loss = float(update(X_train[rows]).numpy())
            if not np.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training objective at step {step}")
            validate = fixed_steps is None and bool(len(X_val)) and (step % validate_every == 0 or step == total_steps)
            val_loss = None
            improved = False
            if validate:
                val_sum = sum(float(validation_batch(X_val[i:i+config["batch_size"]]).numpy()) for i in range(0, len(X_val), config["batch_size"]))
                val_loss = val_sum / X_val.size
                if not np.isfinite(val_loss):
                    raise FloatingPointError(f"Nonfinite validation objective at step {step}")
                if fixed_steps is None and step >= min_steps:
                    improved = best_loss is None or val_loss < best_loss
                    significant = stopping_reference is None or val_loss < stopping_reference - config["relative_improvement"] * max(abs(stopping_reference), 1e-12)
                    if significant:
                        stopping_reference, checks = val_loss, 0
                    else:
                        checks += 1
                    if improved:
                        best_loss, best_step = val_loss, step
                        save_checkpoint("best", step)
                history.append({"step": step, "train_loss": loss, "validation_loss": val_loss,
                                "best_eligible": step >= min_steps, "improved": improved,
                                "patience_checks": checks})
            if step % 10 == 0 or validate or step == total_steps:
                now = time.monotonic()
                event = {"status": "running", "step": step, "total_steps": total_steps,
                         "train_loss": loss, "validation_loss": val_loss,
                         "best_step": best_step, "best_validation_loss": best_loss,
                         "current_validation_loss": val_loss,
                         "elapsed_seconds": now-started,
                         "updates_per_second": (step-prev_step)/max(now-prev_time, 1e-9)}
                progress.write(json.dumps(event, allow_nan=False) + "\n")
                _json(root / "progress.json", event)
                prev_time, prev_step = now, step
            if validate:
                save_checkpoint("latest", step)
            if fixed_steps is None and step >= min_steps and checks >= patience:
                status = "early_stopped"
                break
    save_checkpoint("latest", step)
    selected = "best" if best_step is not None and fixed_steps is None else "latest"
    if status == "running":
        status = "completed"
    result = {"status": status, "model": name, "steps_trained": step,
              "selected_checkpoint": selected, "best_step": best_step,
              "selected_step": best_step if selected == "best" else step,
              "best_validation_loss": best_loss, "best_checkpoint_eligible": best_step is not None,
              "selection": "fixed_steps_final" if fixed_steps is not None else ("minimum_validation_objective" if selected == "best" else "short_pilot_final_no_eligible_checkpoint"),
              "native_dim": 64, "elapsed_seconds": time.monotonic()-started, "history": history}
    _json(root / "result.json", result)
    _json(root / "progress.json", {**result, "history": None, "step": step,
          "current_validation_loss": val_loss,
          "updates_per_second": step/max(time.monotonic()-started, 1e-9)})
    return result
