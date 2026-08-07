"""TimeClustering step: cluster the latent features — for EVERY candidate k.

Redesign vs. the legacy step (which scanned k, kept only the best-k labels and
discarded the rest):

- ``kmeans``: every k in ``n_clusters`` produces its own tagged result
  (``kmeans_k3``, ``kmeans_k4``, ...), each registered in the run manifest via
  ``add_cluster_result``. No best-k selection, nothing discarded.
- ``kmeans-scan``: demoted to an *optional diagnostic*. It writes one
  ``kmeans_scan.json`` (metrics per k + a max-SCI recommendation) and registers
  NO cluster results.
- ``dbscan`` / ``hdbscan``: density methods have no k — a single result tagged
  ``dbscan`` / ``hdbscan``.
- No figure generation anywhere (charts are the visualize scripts' job, M4).
- No redundant artifact copies: the cleaned feature matrix / seq_len / kept-row
  map are saved ONCE at the step level; each tagged result only adds its
  labels, 3-column indices and metrics.json. The raw tensor X is never copied —
  consumers resolve it from the manifest.

Inputs resolve from ``context['data']`` first, then from the manifest
(``feature_extract.features``, ``time_segmentation.{X,lengths,indices}``), so
the step also works standalone via ``--steps cluster --run-id <existing>``.

Heavy backends (hdbscan, tslearn for DTW) are imported lazily.
"""
from __future__ import annotations

import gc
import json
import os

import numpy as np

from src.framework.step import Step
from src.utils.cluster_metrics import compute_cluster_metrics, dbcv_score


class TimeClusteringStep(Step):
    step_type = "time_clustering"

    def __init__(self, cluster_method: str = "kmeans",
                 feature_model: str = "detsec", segment_method: str = "clasp",
                 n_clusters=(3, 4, 5),
                 metric: str = "euclidean", normalization_method: str = "zscore",
                 col_index: int = 2,
                 kmeans_n_init: int = 30, kmeans_max_iter: int = 300,
                 kmeans_random_state: int = 42,
                 dbscan_eps: float = 1.25, dbscan_min_pts: int = 20,
                 hdbscan_min_cluster_size: int = 20, hdbscan_min_samples=None,
                 hdbscan_cluster_selection_method: str = "eom",
                 hdbscan_cluster_selection_epsilon: float = 0.0):
        self.cluster_method = str(cluster_method).lower()
        self.feature_model = feature_model
        self.segment_method = segment_method
        super().__init__(variant=f"{self.cluster_method}_on_{feature_model}_on_{segment_method}")
        self.n_clusters = sorted({int(k) for k in n_clusters})
        self.metric = str(metric).lower()
        self.normalization_method = str(normalization_method).lower()
        self.col_index = int(col_index)
        self.kmeans_n_init = int(kmeans_n_init)
        self.kmeans_max_iter = int(kmeans_max_iter)
        self.kmeans_random_state = int(kmeans_random_state)
        self.dbscan_eps = float(dbscan_eps)
        self.dbscan_min_pts = int(dbscan_min_pts)
        self.hdbscan_min_cluster_size = int(hdbscan_min_cluster_size)
        self.hdbscan_min_samples = None if hdbscan_min_samples is None else int(hdbscan_min_samples)
        self.hdbscan_cluster_selection_method = str(hdbscan_cluster_selection_method).lower()
        self.hdbscan_cluster_selection_epsilon = float(hdbscan_cluster_selection_epsilon)

    def log_subdir(self) -> str:
        return f"TimeClustering_{self.variant}"

    # ── input ────────────────────────────────────────────────────

    def _load_inputs(self, context: dict):
        data = context.get("data", {}) or {}

        features = data.get("features")
        if features is None:
            p = self.resolve(context, "feature_extract", "features")
            if p and os.path.exists(p):
                features = np.load(p)
        if features is None:
            raise ValueError(
                "[time_clustering] no feature matrix. Run feature_extract first, or reuse a "
                "--run-id whose manifest has feature_extract.features.")
        features = np.asarray(features)
        if features.ndim != 2:
            raise ValueError(f"[time_clustering] expected 2-D features, got {features.shape}")

        def _load(key):
            arr = data.get(key)
            if arr is None:
                p = self.resolve(context, "time_segmentation", key)
                if p and os.path.exists(p):
                    arr = np.load(p)
            return arr

        X = _load("X")               # only used on the DTW path
        lengths = _load("lengths")
        indices = _load("indices")   # 2-col (csv_idx, start) from segmentation
        return features, X, lengths, indices

    def _filter_invalid(self, features, lengths, indices):
        """Drop samples with NaN/Inf features; keep everything else row-aligned."""
        invalid = ~np.isfinite(features).all(axis=1)
        kept_rows = np.nonzero(~invalid)[0]
        if invalid.any():
            print(f"[time_clustering] dropping {int(invalid.sum())} samples with NaN/Inf features")
            features = features[kept_rows]
            if lengths is not None:
                lengths = np.asarray(lengths)[kept_rows]
            if indices is not None:
                indices = np.asarray(indices)[kept_rows]
        return features, lengths, indices, kept_rows

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        if self.normalization_method == "minmax":
            scaler = MinMaxScaler()
        elif self.normalization_method == "zscore":
            scaler = StandardScaler()
        else:
            raise ValueError(f"[time_clustering] unknown normalization: {self.normalization_method}")
        return scaler.fit_transform(np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0))

    # ── clustering backends ──────────────────────────────────────

    def _kmeans(self, feats: np.ndarray, k: int) -> np.ndarray:
        from sklearn.cluster import KMeans
        if k >= len(feats):
            raise ValueError(f"[time_clustering] k={k} needs k < n_samples ({len(feats)})")
        labels = KMeans(n_clusters=k, random_state=self.kmeans_random_state,
                        n_init=self.kmeans_n_init, max_iter=self.kmeans_max_iter
                        ).fit_predict(feats)
        self._print_distribution(f"kmeans k={k}", labels)
        return labels

    def _dist_matrix(self, feats: np.ndarray, X, lengths) -> np.ndarray:
        if self.metric in ("dtw", "fastdtw"):
            if X is None:
                raise ValueError("[time_clustering] DTW metric needs the raw tensor X "
                                 "(time_segmentation.X in the manifest).")
            from tslearn.metrics import cdist_dtw
            from tslearn.utils import to_time_series_dataset
            series = []
            for i in range(len(X)):
                n = int(np.asarray(lengths[i]).reshape(-1)[0]) if lengths is not None else X.shape[1]
                row = np.asarray(X[i][:max(1, min(n, X.shape[1]))])
                series.append(row[:, self.col_index] if row.ndim == 2 else row.reshape(-1))
            print(f"[time_clustering] computing DTW distance matrix for {len(series)} series...")
            return cdist_dtw(to_time_series_dataset(series), n_jobs=-1)
        from scipy.spatial.distance import cdist
        return cdist(feats, feats, metric=self.metric)

    def _dbscan(self, dist: np.ndarray) -> np.ndarray:
        from sklearn.cluster import DBSCAN
        labels = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_pts,
                        metric="precomputed").fit_predict(dist)
        self._print_distribution(f"dbscan eps={self.dbscan_eps}", labels)
        return labels

    def _hdbscan(self, feats: np.ndarray) -> np.ndarray:
        import hdbscan
        labels = hdbscan.HDBSCAN(
            min_cluster_size=self.hdbscan_min_cluster_size,
            min_samples=self.hdbscan_min_samples,
            metric=self.metric,
            cluster_selection_method=self.hdbscan_cluster_selection_method,
            cluster_selection_epsilon=self.hdbscan_cluster_selection_epsilon,
        ).fit_predict(feats)
        self._print_distribution("hdbscan", labels)
        return labels

    @staticmethod
    def _print_distribution(title: str, labels: np.ndarray) -> None:
        n_clusters = len(set(labels.tolist())) - (1 if -1 in labels else 0)
        print(f"[time_clustering] {title}: {n_clusters} clusters, "
              f"{int(np.sum(labels == -1))} noise")
        for l, c in zip(*np.unique(labels, return_counts=True)):
            print(f"  {'noise' if l == -1 else f'cluster_{l}'}: {c} samples")

    # ── result persistence ───────────────────────────────────────

    def _save_result(self, context: dict, tag: str, labels: np.ndarray,
                     indices, metrics: dict, shared: dict) -> None:
        out_dir = os.path.join(self.log_dir(context), tag)
        os.makedirs(out_dir, exist_ok=True)

        labels_path = os.path.join(out_dir, "cluster_labels.npy")
        np.save(labels_path, labels)

        artifacts = {
            "labels": self.rel(context, labels_path),
            "metrics": self.rel(context, os.path.join(out_dir, "metrics.json")),
            # shared, saved once at the step level — recorded per tag so a
            # consumer only ever needs the tag to resolve everything
            "feature_matrix": shared["feature_matrix"],
            "seq_len": shared["seq_len"],
            "kept_rows": shared["kept_rows"],
        }
        if indices is not None and len(indices) == len(labels):
            merged = np.column_stack((np.asarray(indices), labels))
            idx_path = os.path.join(out_dir, "indices.npy")
            np.save(idx_path, merged)
            artifacts["indices"] = self.rel(context, idx_path)

        with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        context["manifest"].add_cluster_result(
            tag, os.path.join(self.log_subdir(), tag).replace(os.sep, "/"),
            artifacts, extra={"n_clusters": metrics.get("n_clusters"),
                              "n_noise": metrics.get("n_noise")})
        print(f"[time_clustering] result '{tag}' -> {out_dir}")

    # ── main ─────────────────────────────────────────────────────

    def run(self, context: dict) -> dict:
        log_dir = self.log_dir(context)
        features, X, lengths, indices = self._load_inputs(context)
        features, lengths, indices, kept_rows = self._filter_invalid(features, lengths, indices)
        print(f"[time_clustering] method={self.cluster_method}  features={features.shape}  "
              f"candidates={self.n_clusters if self.cluster_method.startswith('kmeans') else '-'}")

        # shared artifacts, saved ONCE (NaN-cleaned, row-aligned with labels)
        feat_path = os.path.join(log_dir, "feature_matrix.npy")
        np.save(feat_path, features)
        seq_path = os.path.join(log_dir, "seq_len.npy")
        np.save(seq_path, np.asarray(lengths).reshape(-1) if lengths is not None
                else np.full(len(features), -1, dtype=np.int64))
        kept_path = os.path.join(log_dir, "kept_rows.npy")
        np.save(kept_path, kept_rows)
        shared = {"feature_matrix": self.rel(context, feat_path),
                  "seq_len": self.rel(context, seq_path),
                  "kept_rows": self.rel(context, kept_path)}

        use_dtw = self.metric in ("dtw", "fastdtw") and not self.cluster_method.startswith("kmeans")
        norm = None if use_dtw else self._normalize(features)
        dist = self._dist_matrix(norm, X, lengths) if self.cluster_method == "dbscan" else None

        tags = []
        if self.cluster_method == "kmeans":
            for k in self.n_clusters:
                labels = self._kmeans(norm, k)
                metrics = compute_cluster_metrics(norm, labels)
                metrics.update({"cluster_method": "kmeans", "n_clusters_requested": k,
                                "feature_model": self.feature_model,
                                "segment_method": self.segment_method})
                tag = f"kmeans_k{k}"
                self._save_result(context, tag, labels, indices, metrics, shared)
                tags.append(tag)

        elif self.cluster_method == "kmeans-scan":
            # optional diagnostic: metrics per k + max-SCI recommendation, no results
            records = []
            for k in self.n_clusters:
                if k >= len(norm):
                    print(f"[time_clustering] skip k={k}: needs k < n_samples ({len(norm)})")
                    continue
                labels = self._kmeans(norm, k)
                m = compute_cluster_metrics(norm, labels)
                records.append({"n_clusters": k,
                                "sci": m["silhouette_score"], "dbi": m["davies_bouldin_score"],
                                "chi": m["calinski_harabasz_score"]})
            valid = [r for r in records if r["sci"] is not None]
            best_k = max(valid, key=lambda r: r["sci"])["n_clusters"] if valid else None
            scan = {"scan_method": "kmeans-scan", "selection_rule": "max_sci (diagnostic only)",
                    "recommended_n_clusters": best_k, "feature_model": self.feature_model,
                    "segment_method": self.segment_method, "records": records}
            scan_path = os.path.join(log_dir, "kmeans_scan.json")
            with open(scan_path, "w", encoding="utf-8") as f:
                json.dump(scan, f, indent=2, ensure_ascii=False)
            # merge WITHOUT clobbering any existing cluster results in the manifest
            context["manifest"].add_step_artifact(
                "time_clustering", "kmeans_scan", self.rel(context, scan_path))
            print(f"[time_clustering] kmeans-scan diagnostic -> {scan_path} "
                  f"(recommended k={best_k}); no cluster results registered")

        elif self.cluster_method == "dbscan":
            labels = self._dbscan(dist)
            metrics = compute_cluster_metrics(features if norm is None else norm, labels,
                                              dist_matrix=dist)
            metrics.update({"cluster_method": "dbscan", "eps": self.dbscan_eps,
                            "min_pts": self.dbscan_min_pts, "metric": self.metric,
                            "feature_model": self.feature_model,
                            "segment_method": self.segment_method})
            self._save_result(context, "dbscan", labels, indices, metrics, shared)
            tags.append("dbscan")

        elif self.cluster_method == "hdbscan":
            labels = self._hdbscan(norm)
            metrics = compute_cluster_metrics(norm, labels)
            d = dbcv_score(self._dist_matrix(norm, X, lengths), labels, d=features.shape[1])
            if d is not None:
                metrics["dbcv_score"] = d
            metrics.update({"cluster_method": "hdbscan",
                            "min_cluster_size": self.hdbscan_min_cluster_size,
                            "feature_model": self.feature_model,
                            "segment_method": self.segment_method})
            self._save_result(context, "hdbscan", labels, indices, metrics, shared)
            tags.append("hdbscan")
        else:
            raise ValueError(
                f"[time_clustering] unknown cluster_method '{self.cluster_method}'. "
                "Supported: kmeans, kmeans-scan, dbscan, hdbscan")

        context.setdefault("data", {})["cluster_tags"] = tags
        # big objects no longer needed downstream
        context["data"].pop("features", None)
        del features, X
        gc.collect()
        return context
