"""FewShotClusterExtract step: find genuine few-shot clusters and export them.

Consumes ONE tagged clustering result (``--cluster-tag``, e.g. ``kmeans_k4``).
If the run has exactly one tagged result the tag may be omitted; with several
results an explicit tag is required — no silent "best k" choice anymore.

Algorithm preserved from the legacy step: candidate detection by cluster size
(``detect_few_shot_clusters``), then artifact-like filtering via temporal
adjacency (``adj_threshold``) and centroid-support (``center_margin`` /
``center_support_threshold``); surviving clusters' segments are exported per
cluster with a manifest.json.

All inputs resolve through the run manifest (tag artifacts for the clustering
result, ``extract_active_data.segments_dir`` for the source CSVs) — no more
directory-name guessing.
"""
from __future__ import annotations

import gc
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from src.framework.step import Step
from src.utils.cluster_metrics import detect_few_shot_clusters


class FewShotClusterExtractStep(Step):
    step_type = "few_shot_cluster_extract"

    def __init__(self, cluster_tag: str = None, n_percent: float = 50.0,
                 adj_threshold: float = 0.6, center_margin: float = 0.1,
                 center_support_threshold: float = 0.6,
                 export_format: str = "csv", export_channel: str = "original",
                 normalization_method: str = "zscore"):
        self.cluster_tag = cluster_tag  # may be None -> resolved at run time
        self.n_percent = float(n_percent)
        self.adj_threshold = float(adj_threshold)
        self.center_margin = float(center_margin)
        self.center_support_threshold = float(center_support_threshold)
        self.export_format = str(export_format).lower()
        self.export_channel = str(export_channel).lower()
        self.normalization_method = str(normalization_method).lower()
        super().__init__(variant=f"{self.n_percent:g}pct_on_{cluster_tag or 'auto'}")

    # ── input resolution (manifest only, no path guessing) ───────

    def _resolve_tag(self, context: dict) -> str:
        if self.cluster_tag:
            return self.cluster_tag
        tags = context["manifest"].cluster_tags()
        if len(tags) == 1:
            print(f"[few_shot] only one cluster result available, using tag '{tags[0]}'")
            return tags[0]
        raise ValueError(
            f"[few_shot] --cluster-tag is required: {len(tags)} cluster results exist "
            f"{tags}. Pick one, e.g. --cluster-tag {tags[0] if tags else 'kmeans_k3'}.")

    def _load_cluster_artifacts(self, context: dict, tag: str):
        m = context["manifest"]

        def _get(key, required=True):
            p = m.cluster_artifact_path(tag, key)
            if p and os.path.exists(p):
                return np.load(p, allow_pickle=False)
            if required:
                raise FileNotFoundError(
                    f"[few_shot] cluster result '{tag}' is missing artifact '{key}'. "
                    f"Run --steps cluster first (same --run-id).")
            return None

        labels = np.asarray(_get("labels")).reshape(-1).astype(np.int64)
        feature_matrix = np.asarray(_get("feature_matrix"), dtype=np.float64)
        seq_len = np.asarray(_get("seq_len")).reshape(-1).astype(np.int64)

        indices_3col = _get("indices", required=False)
        if indices_3col is None:
            seg_idx = self.resolve(context, "time_segmentation", "indices")
            kept = _get("kept_rows", required=False)
            if seg_idx and os.path.exists(seg_idx):
                seg_idx = np.load(seg_idx)
                if kept is not None:
                    seg_idx = seg_idx[kept]
                if len(seg_idx) == len(labels):
                    indices_3col = np.column_stack((seg_idx, labels))
        if indices_3col is None or len(indices_3col) != len(labels):
            raise ValueError(
                f"[few_shot] cannot align segmentation indices with labels of '{tag}'.")
        return labels, np.asarray(indices_3col).astype(np.int64), feature_matrix, seq_len

    def _resolve_input_root(self, context: dict) -> str:
        if context.get("input_root") and os.path.isdir(context["input_root"]):
            return context["input_root"]
        p = self.resolve(context, "extract_active_data", "segments_dir")
        if p and os.path.isdir(p):
            return p
        raise FileNotFoundError(
            "[few_shot] no source CSV directory. Run extract_active_data first "
            "(or reuse a --run-id whose manifest has it).")

    # ── analysis (preserved from legacy) ─────────────────────────

    def _normalize_features(self, feature_matrix: np.ndarray) -> np.ndarray:
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        data = np.nan_to_num(np.asarray(feature_matrix, dtype=np.float64),
                             nan=0.0, posinf=0.0, neginf=0.0)
        scaler = MinMaxScaler() if self.normalization_method == "minmax" else StandardScaler()
        return scaler.fit_transform(data)

    @staticmethod
    def _compute_centroids(features: np.ndarray, labels: np.ndarray) -> dict:
        centroids = {}
        for cid in np.unique(labels):
            if int(cid) == -1:
                continue
            mask = labels == cid
            if np.any(mask):
                centroids[int(cid)] = np.mean(features[mask], axis=0)
        return centroids

    def _analyze_artifacts(self, indices_3col, labels, features_norm,
                           few_shot_set, big_set, centroids) -> dict:
        csv_idx = indices_3col[:, 0]
        start_idx = indices_3col[:, 1]

        per_file = defaultdict(list)
        for i in range(len(labels)):
            per_file[int(csv_idx[i])].append(i)
        pos_in_file = {}
        for fid, idxs in per_file.items():
            idxs.sort(key=lambda x: int(start_idx[x]))
            for pos, i in enumerate(idxs):
                pos_in_file[i] = (fid, pos)

        neighbor_counts_by_small = defaultdict(lambda: defaultdict(int))
        neighbor_valid_by_small = defaultdict(int)

        for i in range(len(labels)):
            c = int(labels[i])
            if c not in few_shot_set or i not in pos_in_file:
                continue
            fid, pos = pos_in_file[i]
            idxs = per_file.get(fid, [])
            neighbors = ([idxs[pos - 1]] if pos > 0 else []) + \
                        ([idxs[pos + 1]] if pos + 1 < len(idxs) else [])
            valid = [int(labels[j]) for j in neighbors
                     if int(labels[j]) != -1 and int(labels[j]) in big_set]
            if valid:
                neighbor_valid_by_small[c] += 1
                for lj in valid:
                    neighbor_counts_by_small[c][lj] += 1

        dominant_big_by_small = {}
        for c in few_shot_set:
            counts = neighbor_counts_by_small.get(c, {})
            dominant_big_by_small[c] = (int(max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0])
                                        if counts else None)

        big_ids = sorted(big_set)
        big_centroids = (np.asarray([centroids[cid] for cid in big_ids], dtype=np.float64)
                         if big_ids else None)

        center_support_by_small = defaultdict(int)
        total_by_small = defaultdict(int)
        eps = 1e-12
        for i in range(len(labels)):
            c = int(labels[i])
            if c not in few_shot_set:
                continue
            total_by_small[c] += 1
            dom_big = dominant_big_by_small.get(c)
            if dom_big is None or big_centroids is None:
                continue
            x = features_norm[i]
            self_centroid = centroids.get(c)
            big_d2 = np.einsum("ij,ij->i", big_centroids - x, big_centroids - x)
            best_big = int(big_ids[int(np.argmin(big_d2))])
            if self_centroid is None:
                continue
            d_self = float(np.sqrt(np.sum((x - self_centroid) ** 2)))
            d_best = float(np.sqrt(big_d2[int(np.argmin(big_d2))]))
            if best_big == int(dom_big) and d_best + eps < d_self * (1.0 - self.center_margin) + eps:
                center_support_by_small[c] += 1

        cluster_sizes = {int(l): int(cn) for l, cn in zip(*np.unique(labels, return_counts=True))}
        cluster_reports = {}
        for c in sorted(few_shot_set):
            denom = int(neighbor_valid_by_small.get(c, 0))
            dom = dominant_big_by_small.get(c)
            hits = int(neighbor_counts_by_small[c].get(int(dom), 0)) if (denom > 0 and dom is not None) else 0
            adj_ratio = float(hits) / float(denom) if denom > 0 else 0.0
            total = int(total_by_small.get(c, 0))
            center_ratio = float(center_support_by_small.get(c, 0)) / float(total) if total > 0 else 0.0
            cluster_reports[int(c)] = {
                "cluster_id": int(c),
                "cluster_size": int(cluster_sizes.get(int(c), 0)),
                "dominant_big_cluster": None if dom is None else int(dom),
                "adj_ratio": adj_ratio,
                "center_support_ratio": center_ratio,
                "artifact_like": bool(adj_ratio >= self.adj_threshold
                                      and center_ratio >= self.center_support_threshold),
                "params": {"n_percent": self.n_percent, "adj_threshold": self.adj_threshold,
                           "center_margin": self.center_margin,
                           "center_support_threshold": self.center_support_threshold},
            }
        return {"cluster_reports": cluster_reports}

    # ── export (preserved from legacy) ───────────────────────────

    def _export_segments(self, context, export_dir, indices_3col, seq_len, labels,
                         true_few_shot_set, cluster_reports):
        os.makedirs(export_dir, exist_ok=True)
        input_root = self._resolve_input_root(context)
        csv_paths = [os.path.join(input_root, f)
                     for f in sorted(os.listdir(input_root)) if f.lower().endswith(".csv")]
        if not csv_paths:
            raise FileNotFoundError(f"[few_shot] input_root has no csv files: {input_root}")

        csv_idx, start_idx = indices_3col[:, 0], indices_3col[:, 1]
        per_file_segments = defaultdict(list)
        for i in range(len(labels)):
            if int(labels[i]) in true_few_shot_set:
                per_file_segments[int(csv_idx[i])].append(i)

        manifest = []
        for fid, seg_indices in sorted(per_file_segments.items()):
            if fid < 0 or fid >= len(csv_paths):
                continue
            try:
                df = pd.read_csv(csv_paths[fid])
            except Exception as e:
                print(f"[few_shot] failed to read {csv_paths[fid]}: {e}")
                continue
            n_rows = len(df)
            base_name = os.path.splitext(os.path.basename(csv_paths[fid]))[0]
            seg_indices.sort(key=lambda x: int(start_idx[x]))
            for i in seg_indices:
                cid = int(labels[i])
                start = int(start_idx[i])
                length = int(seq_len[i]) if i < len(seq_len) else 0
                if length <= 0:
                    continue
                end = min(n_rows, start + length)
                if start < 0 or start >= end:
                    continue
                seg_df = df.iloc[start:end].copy()
                if self.export_channel == "cleaned" and "power" in seg_df.columns:
                    from scipy.signal import medfilt
                    seg_df["power"] = medfilt(seg_df["power"].to_numpy(), kernel_size=5)

                cluster_folder = os.path.join(export_dir, f"cluster_{cid}")
                os.makedirs(cluster_folder, exist_ok=True)
                out_name = f"{base_name}_start{start}_end{end}_i{i}.csv"
                out_path = os.path.join(cluster_folder, out_name)
                if self.export_format == "csv":
                    seg_df.to_csv(out_path, index=False)
                else:
                    out_path = out_path.replace(".csv", ".npy")
                    np.save(out_path, seg_df.to_numpy())
                manifest.append({
                    "cluster_id": cid, "csv_idx": int(fid),
                    "source_csv": os.path.basename(csv_paths[fid]),
                    "start": start, "end": end, "length": int(end - start),
                    "export_path": out_path,
                    "cluster_report": cluster_reports.get(cid, {}),
                })
                del seg_df
            del df
            gc.collect()

        manifest_path = os.path.join(export_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        return manifest_path, manifest

    # ── main ─────────────────────────────────────────────────────

    def run(self, context: dict) -> dict:
        tag = self._resolve_tag(context)
        self.variant = f"{self.n_percent:g}pct_on_{tag}"
        log_dir = self.log_dir(context)
        labels, indices_3col, feature_matrix, seq_len = self._load_cluster_artifacts(context, tag)
        print(f"[few_shot] tag={tag}  samples={len(labels)}  n_percent={self.n_percent:g}")

        few_shot_info = detect_few_shot_clusters(labels, method="avg_percent",
                                                 n_percent=self.n_percent, threshold=5)
        few_shot_set = {int(item["cluster_id"]) for item in few_shot_info["few_shot_clusters"]}
        big_set = {int(x) for x in np.unique(labels) if int(x) != -1} - few_shot_set

        features_norm = self._normalize_features(feature_matrix)
        centroids = self._compute_centroids(features_norm, labels)
        cluster_reports = self._analyze_artifacts(
            indices_3col, labels, features_norm, few_shot_set, big_set, centroids)["cluster_reports"]

        artifact_like_set = {int(cid) for cid, rep in cluster_reports.items()
                             if rep.get("artifact_like", False)}
        true_few_shot_set = few_shot_set - artifact_like_set

        summary = {
            "cluster_tag": tag,
            "few_shot_detection": few_shot_info,
            "few_shot_candidates": sorted(few_shot_set),
            "artifact_like_clusters": sorted(artifact_like_set),
            "true_few_shot_clusters": sorted(true_few_shot_set),
            "cluster_reports": cluster_reports,
        }
        summary_path = os.path.join(log_dir, "few_shot_cluster_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        export_dir = os.path.join(context.get("output_root", "output"), "few_shot_active_data", tag)
        manifest_path, manifest = self._export_segments(
            context, export_dir, indices_3col, seq_len, labels, true_few_shot_set, cluster_reports)

        self.record(context, artifacts={
            "summary": self.rel(context, summary_path),
            "export_manifest": self.rel(context, manifest_path),
            "export_dir": self.rel(context, export_dir),
        }, extra={
            "cluster_tag": tag,
            "true_few_shot_clusters": sorted(true_few_shot_set),
            "artifact_like_clusters": sorted(artifact_like_set),
            "n_exported_segments": len(manifest),
        })

        del feature_matrix, features_norm, indices_3col, labels
        gc.collect()
        print(f"[few_shot] done: {len(manifest)} segments from "
              f"{len(true_few_shot_set)} true few-shot clusters -> {export_dir}")
        return context
