"""Seq2Point CNN utilities for cycle-based NILM experiments."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np


class CycleWindowCorpus:
    """Load variable-length NILM cycles and expose midpoint windows."""

    def __init__(self, root: Path, files: list[str], window_length: int = 599,
                 stride: int = 1, mains_scale: float = 10_000.0,
                 appliance_scale: float = 4_000.0):
        if window_length < 3 or window_length % 2 == 0:
            raise ValueError("window_length must be an odd integer >= 3")
        if stride < 1:
            raise ValueError("stride must be positive")
        self.root = Path(root)
        self.files = list(files)
        self.window_length = int(window_length)
        self.stride = int(stride)
        self.mains_scale = float(mains_scale)
        self.appliance_scale = float(appliance_scale)
        self.half_window = self.window_length // 2
        self.cycles = []
        self.indices = []
        for cycle_id, relative in enumerate(self.files):
            path = self.root / relative
            with np.load(path) as payload:
                mains = np.asarray(payload["mains"], dtype=np.float32)
                appliance = np.asarray(payload["appliance"], dtype=np.float32)
            if mains.ndim != 1 or appliance.ndim != 1 or len(mains) != len(appliance):
                raise ValueError(f"invalid NILM cycle pair: {path}")
            if len(mains) == 0 or not np.isfinite(mains).all() or not np.isfinite(appliance).all():
                raise ValueError(f"empty or non-finite NILM cycle: {path}")
            normalized_mains = np.clip(mains, 0.0, self.mains_scale) / self.mains_scale
            normalized_appliance = (
                np.clip(appliance, 0.0, self.appliance_scale) / self.appliance_scale)
            padded = np.pad(
                normalized_mains, (self.half_window, self.half_window),
                mode="constant")
            self.cycles.append({
                "file": relative,
                "mains": mains,
                "appliance": appliance,
                "padded_mains": padded.astype(np.float32, copy=False),
                "normalized_appliance": normalized_appliance.astype(
                    np.float32, copy=False),
            })
            self.indices.extend(
                (cycle_id, center) for center in range(0, len(mains), self.stride))

    def __len__(self) -> int:
        return len(self.indices)

    def batch(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.empty((len(indices), self.window_length, 1), dtype=np.float32)
        y = np.empty((len(indices), 1), dtype=np.float32)
        for row, sample_index in enumerate(indices):
            cycle_id, center = self.indices[int(sample_index)]
            cycle = self.cycles[cycle_id]
            x[row, :, 0] = cycle["padded_mains"][
                center:center + self.window_length]
            y[row, 0] = cycle["normalized_appliance"][center]
        return x, y

    def all_targets(self) -> np.ndarray:
        return np.concatenate([
            cycle["appliance"][::self.stride] for cycle in self.cycles
        ]).astype(np.float32)


def make_keras_sequence(corpus: CycleWindowCorpus, batch_size: int,
                        shuffle: bool, seed: int):
    import tensorflow as tf

    class WindowSequence(tf.keras.utils.Sequence):
        def __init__(self):
            super().__init__()
            self.order = np.arange(len(corpus), dtype=np.int64)
            self.rng = np.random.default_rng(seed)
            if shuffle:
                self.rng.shuffle(self.order)

        def __len__(self):
            return int(math.ceil(len(self.order) / batch_size))

        def __getitem__(self, index):
            rows = self.order[index * batch_size:(index + 1) * batch_size]
            return corpus.batch(rows)

        def on_epoch_end(self):
            if shuffle:
                self.rng.shuffle(self.order)

    return WindowSequence()


def build_seq2point(window_length: int = 599, learning_rate: float = 1e-3,
                    dropout: float = 0.0):
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(window_length, 1), name="mains_window")
    x = inputs
    for filters, kernel in ((30, 10), (30, 8), (40, 6), (50, 5), (50, 5)):
        x = tf.keras.layers.Conv1D(
            filters, kernel, activation="relu", padding="same")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(
        1, activation="linear", name="appliance_power")(x)
    model = tf.keras.Model(inputs, outputs, name="seq2point_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse")
    return model


def regression_metrics(target: np.ndarray, prediction: np.ndarray,
                       on_threshold: float = 20.0) -> dict:
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    prediction = np.maximum(
        np.asarray(prediction, dtype=np.float64).reshape(-1), 0.0)
    if len(target) != len(prediction) or len(target) == 0:
        raise ValueError("target and prediction must be non-empty and aligned")
    error = prediction - target
    true_on = target >= on_threshold
    predicted_on = prediction >= on_threshold
    tp = int(np.sum(true_on & predicted_on))
    fp = int(np.sum(~true_on & predicted_on))
    fn = int(np.sum(true_on & ~predicted_on))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "mae_watts": float(np.mean(np.abs(error))),
        "sae": float(abs(np.sum(prediction) - np.sum(target))
                     / max(np.sum(target), 1e-12)),
        "nde": float(np.sum(error ** 2) / max(np.sum(target ** 2), 1e-12)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * precision * recall / (precision + recall)
                    if precision + recall else 0.0),
        "target_mean_watts": float(np.mean(target)),
        "target_max_watts": float(np.max(target)),
        "target_on_fraction": float(np.mean(true_on)),
        "prediction_mean_watts": float(np.mean(prediction)),
        "prediction_max_watts": float(np.max(prediction)),
        "prediction_on_fraction": float(np.mean(predicted_on)),
        "on_threshold_watts": float(on_threshold),
        "n_points": int(len(target)),
    }
