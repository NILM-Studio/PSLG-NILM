"""PrimGLR DTW prototype representation, extracted from the 2026-10-16 public pipeline.

The legacy `dtw.py` uses different prototype selection and normalization. This
module preserves the nine-object algorithm and permits train-only fitting.
"""
import math

import numpy as np

try:
    from numba import njit
except ImportError:
    # Useful for small checks; large matrices should install numba.
    def njit(*args, **kwargs):
        return lambda fn: fn


@njit(cache=True)
def dtw_distance(x, y, band):
    """Square-root of min squared path cost divided by path cell count."""
    n, m = len(x), len(y)
    cost = np.full((n + 1, m + 1), np.inf)
    length = np.zeros((n + 1, m + 1), np.int32)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - band), min(m, i + band) + 1):
            best, count = cost[i - 1, j - 1], length[i - 1, j - 1]
            if cost[i - 1, j] < best:
                best, count = cost[i - 1, j], length[i - 1, j]
            if cost[i, j - 1] < best:
                best, count = cost[i, j - 1], length[i, j - 1]
            cost[i, j] = best + (x[i - 1] - y[j - 1]) ** 2
            length[i, j] = count + 1
    return math.sqrt(cost[n, m] / max(length[n, m], 1))


@njit(cache=True)
def distance_features(waves, prototypes, band):
    out = np.empty((len(waves), len(prototypes)))
    for i in range(len(waves)):
        for j in range(len(prototypes)):
            out[i, j] = dtw_distance(waves[i], prototypes[j], band)
    return out


def phase_wave(x, n=128):
    x = np.asarray(x, float)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("DTW needs a finite, unpadded valid waveform")
    return np.interp(np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, len(x)), x)


def prototype_rows(segment_rows, eligible, seed, maximum=16):
    rng = np.random.default_rng(seed)
    aids = sorted({segment_rows[i]["activity_id"] for i in eligible})
    rng.shuffle(aids)
    selected = []
    for aid in aids[:maximum]:
        options = [i for i in eligible if segment_rows[i]["activity_id"] == aid]
        selected.append(int(rng.choice(options)))
    rest = np.array([i for i in eligible if i not in selected], int)
    rng.shuffle(rest)
    return np.array(selected + rest[:max(0, maximum - len(selected))].tolist(), int)


class DTWReferenceEncoder:
    """Fit train-only power scale and prototypes, then export N x M distances."""

    def __init__(self, points=128, band=13, maximum=16, seed=0):
        self.points, self.band, self.maximum, self.seed = points, band, maximum, seed
        self.power_min = self.power_max = self.prototypes = self.prototype_indices = None

    def fit(self, waves, segment_rows, train_indices):
        train = np.asarray(train_indices, dtype=int)
        if not len(train) or train.min() < 0 or train.max() >= len(waves):
            raise ValueError("Invalid training rows")
        self.power_min = min(float(np.min(waves[i])) for i in train)
        self.power_max = max(float(np.max(waves[i])) for i in train)
        scale = max(self.power_max - self.power_min, 1e-7)
        self.prototype_indices = prototype_rows(segment_rows, train, self.seed, self.maximum)
        self.prototypes = np.stack([phase_wave((waves[i] - self.power_min) / scale,
                                               self.points) for i in self.prototype_indices])
        return self

    def transform(self, waves):
        if self.prototypes is None:
            raise ValueError("Fit the DTW encoder first")
        scale = max(self.power_max - self.power_min, 1e-7)
        shaped = np.stack([phase_wave((w - self.power_min) / scale, self.points)
                           for w in waves])
        return distance_features(shaped, self.prototypes, self.band)
