"""Content-addressed cache for the feature-extract step (M2).

Feature extraction (autoencoder training) is by far the slowest step, and its
output depends only on:

- the segmentation tensor ``X`` (content, dtype, shape) and optional ``lengths``,
- the model name,
- every hyperparameter in ``model_config``.

So the cache key is a SHA256 over exactly those inputs, PLUS a fingerprint of
the model's source code (so editing the architecture invalidates old entries).

Upstream choices (dataset / appliance, segmentation method, segmentation
hyperparameters) are covered *transitively*: they can only influence the
features through ``X``/``lengths``, and those are hashed byte-for-byte. They
are deliberately NOT hashed as labels — if two different upstream paths produce
a byte-identical tensor, the features are identical too and sharing one cache
entry is correct. Provenance (appliance, segment_method) is recorded in
``meta.json`` for humans, not in the key.

This replaces the legacy per-trajectory RunKey caching (which cached every step
and needed cartesian upstream expansion) with a single cache at the single
expensive point.

Layout::

    <cache_dir>/features/<key>/
        features.npy           # the latent feature matrix
        training_history.json  # loss curves etc. (kept for the visualize scripts)
        meta.json              # human-readable provenance; NOT used for lookup
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Bump when the cache entry format changes, so old entries are simply ignored.
CACHE_SCHEMA = "v1"


def _hash_array(h: "hashlib._Hash", arr: np.ndarray) -> None:
    a = np.ascontiguousarray(arr)
    h.update(str(a.shape).encode("utf-8"))
    h.update(str(a.dtype).encode("utf-8"))
    h.update(a.tobytes())


def _sanitize_config(model_config: Dict[str, Any]) -> Dict[str, Any]:
    """JSON-safe copy of the hyperparameters (array values replaced by a hash)."""
    out: Dict[str, Any] = {}
    for k, v in sorted(model_config.items()):
        if isinstance(v, np.ndarray):
            out[k] = f"ndarray{tuple(v.shape)}:{v.dtype}"
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def file_fingerprint(path: str) -> str:
    """SHA256 of a file's bytes; a stable placeholder if the file is absent."""
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return f"missing:{path}"


def compute_key(X: np.ndarray, lengths: Optional[np.ndarray],
                model_name: str, model_config: Dict[str, Any],
                code_id: str = "") -> str:
    """Content-addressed cache key for one (tensor, model, hyperparams, code) tuple."""
    h = hashlib.sha256()
    h.update(CACHE_SCHEMA.encode("utf-8"))
    h.update(model_name.encode("utf-8"))
    h.update(code_id.encode("utf-8"))
    _hash_array(h, X)
    if lengths is not None:
        _hash_array(h, np.asarray(lengths))
    h.update(json.dumps(_sanitize_config(model_config), sort_keys=True,
                        ensure_ascii=False).encode("utf-8"))
    return h.hexdigest()


def _entry_dir(cache_dir: str, key: str) -> str:
    return os.path.join(cache_dir, "features", key)


def load(cache_dir: str, key: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
    """Return (features, training_history) on a cache hit, else ``None``."""
    d = _entry_dir(cache_dir, key)
    f_path = os.path.join(d, "features.npy")
    h_path = os.path.join(d, "training_history.json")
    if not (os.path.exists(f_path) and os.path.exists(h_path)):
        return None
    try:
        features = np.load(f_path)
        with open(h_path, "r", encoding="utf-8") as f:
            history = json.load(f)
    except (OSError, ValueError, json.JSONDecodeError):
        return None  # corrupt entry: treat as a miss, the store will overwrite
    return features, history


def store(cache_dir: str, key: str, features: np.ndarray,
          training_history: Optional[Dict[str, Any]],
          meta: Optional[Dict[str, Any]] = None) -> str:
    """Write a cache entry (tmp-then-rename so a crash never leaves half files)."""
    d = _entry_dir(cache_dir, key)
    os.makedirs(d, exist_ok=True)

    tmp_f = os.path.join(d, ".features.tmp.npy")
    with open(tmp_f, "wb") as f:
        np.save(f, features)
    os.replace(tmp_f, os.path.join(d, "features.npy"))

    tmp_h = os.path.join(d, ".history.tmp")
    with open(tmp_h, "w", encoding="utf-8") as f:
        json.dump(training_history or {}, f, ensure_ascii=False)
    os.replace(tmp_h, os.path.join(d, "training_history.json"))

    if meta is not None:
        tmp_m = os.path.join(d, ".meta.tmp")
        with open(tmp_m, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
        os.replace(tmp_m, os.path.join(d, "meta.json"))
    return d
