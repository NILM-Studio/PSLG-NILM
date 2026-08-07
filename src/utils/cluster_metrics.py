"""Small, self-contained clustering metric helpers.

Copied/condensed from the legacy ``src/utils/clustering_utils.py`` — only the
computation parts. Everything visualization-related from that module belongs to
the decoupled visualize scripts (M4), not to compute steps.
"""
from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Optional

import numpy as np
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)


def compute_cluster_metrics(feature_matrix: np.ndarray, labels: np.ndarray,
                            dist_matrix: Optional[np.ndarray] = None) -> dict:
    """Silhouette / DBI / CHI on the non-noise subset, plus distribution stats.

    Returns a JSON-safe dict; metric values are ``None`` when they cannot be
    computed (fewer than 2 non-noise clusters, degenerate input, ...).
    """
    labels = np.asarray(labels)
    unique, counts = np.unique(labels, return_counts=True)
    metrics: dict = {
        "cluster_distribution": {
            ("noise" if int(l) == -1 else f"cluster_{int(l)}"): int(c)
            for l, c in zip(unique, counts)
        },
        "n_clusters": int(len(set(labels.tolist())) - (1 if -1 in labels else 0)),
        "n_noise": int(np.sum(labels == -1)),
        "silhouette_score": None,
        "davies_bouldin_score": None,
        "calinski_harabasz_score": None,
    }

    valid = labels != -1
    valid_labels = labels[valid]
    if valid.sum() < 2 or len(np.unique(valid_labels)) < 2:
        return metrics

    valid_feat = np.asarray(feature_matrix)[valid]
    try:
        if dist_matrix is not None:
            valid_dist = np.asarray(dist_matrix)[valid][:, valid]
            sil = silhouette_score(valid_dist, valid_labels, metric="precomputed")
        else:
            sil = silhouette_score(valid_feat, valid_labels, metric="euclidean")
        metrics["silhouette_score"] = float(sil)
        metrics["davies_bouldin_score"] = float(davies_bouldin_score(valid_feat, valid_labels))
        metrics["calinski_harabasz_score"] = float(calinski_harabasz_score(valid_feat, valid_labels))
    except Exception as e:  # degenerate geometry — keep distribution stats
        print(f"[cluster_metrics] metric computation skipped: {e}")
    return metrics


@lru_cache(maxsize=1)
def _get_hdbscan_validity_index():
    """Lazily import hdbscan validity_index (optional dependency)."""
    try:
        module = importlib.import_module("hdbscan.validity")
        return getattr(module, "validity_index", None)
    except Exception:
        return None


def dbcv_score(dist_matrix: np.ndarray, labels: np.ndarray, d: Optional[int] = None):
    """DBCV via hdbscan.validity.validity_index on precomputed distances.

    Returns ``None`` when hdbscan is unavailable or the score is degenerate.
    """
    labels = np.asarray(labels)
    if labels.size == 0 or len(np.unique(labels[labels != -1])) < 2:
        return None
    validity_index = _get_hdbscan_validity_index()
    if validity_index is None:
        print("[cluster_metrics] hdbscan not installed; DBCV skipped.")
        return None
    if d is None or int(d) <= 0:
        d = 2
    try:
        score = float(validity_index(dist_matrix, labels, metric="precomputed", d=int(d)))
        return score if np.isfinite(score) else None
    except Exception as e:
        print(f"[cluster_metrics] DBCV failed: {e}")
        return None


def detect_few_shot_clusters(cluster_labels: np.ndarray, method: str = "avg_percent",
                             n_percent: float = 50.0, threshold: int = 5) -> dict:
    """Detect few-shot clusters from cluster size statistics (excluding noise -1).

    ``avg_percent``: clusters smaller than n_percent% of the average size.
    ``threshold``:   clusters smaller than ``threshold`` samples.
    """
    labels = np.asarray(cluster_labels)
    result = {
        "method": str(method).lower(),
        "n_percent": float(n_percent),
        "threshold": int(threshold),
        "average_cluster_size": None,
        "few_shot_clusters": [],
    }
    if labels.size == 0:
        return result

    unique_labels, counts = np.unique(labels, return_counts=True)
    valid_pairs = [(int(l), int(c)) for l, c in zip(unique_labels, counts) if int(l) != -1]
    if not valid_pairs:
        return result

    cluster_sizes = np.array([c for _, c in valid_pairs], dtype=np.float64)
    avg_cluster_size = float(np.mean(cluster_sizes))
    method_norm = result["method"]

    if method_norm in ("avg_percent", "percent_avg", "avg_ratio", "ratio"):
        if n_percent < 0:
            raise ValueError(f"n_percent must be >= 0, got {n_percent}")
        cutoff = avg_cluster_size * (float(n_percent) / 100.0)
        few_shot_pairs = [(cid, cnt) for cid, cnt in valid_pairs if float(cnt) < cutoff]
    elif method_norm == "threshold":
        if threshold < 0:
            raise ValueError(f"threshold must be >= 0, got {threshold}")
        few_shot_pairs = [(cid, cnt) for cid, cnt in valid_pairs if int(cnt) < int(threshold)]
    else:
        raise ValueError(
            f"Unsupported few_shot_detection method: {method}. "
            "Supported methods: avg_percent, threshold")

    result["average_cluster_size"] = avg_cluster_size
    result["few_shot_clusters"] = [
        {"cluster_id": int(cid), "sample_count": int(cnt)}
        for cid, cnt in sorted(few_shot_pairs, key=lambda x: (x[1], x[0]))
    ]
    return result
