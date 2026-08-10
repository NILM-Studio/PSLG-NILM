"""Chart renderers for clustering results.

Adapted from the legacy ``clustering_utils`` (visualization parts only) with:
- no workflow coupling — pure functions taking arrays + an output dir;
- heavy deps (tslearn DTW barycenter, sklearn tSNE) imported lazily;
- unified file naming: ``<tag>_<kind>.png`` inside the given figure dir.
"""
from __future__ import annotations

import os
import shutil

import numpy as np

from visualize.viz_common import setup_fonts, texts


def per_cluster_item_pics(org_data, seq_length, labels, save_dir,
                          threshold=200, col_index=1, language="en"):
    """One small png per series, grouped into cluster_<id>/ folders."""
    import matplotlib.pyplot as plt
    t = texts(language)

    groups = {}
    for i in range(len(org_data)):
        groups.setdefault(int(labels[i]), []).append(i)

    written = 0
    for cluster_id, indices in groups.items():
        indices = indices[:threshold]
        cluster_dir = os.path.join(save_dir, f"cluster_{cluster_id}")
        if os.path.exists(cluster_dir):
            shutil.rmtree(cluster_dir)
        os.makedirs(cluster_dir, exist_ok=True)
        for idx, data_idx in enumerate(indices):
            length = (int(np.asarray(seq_length[data_idx]).reshape(-1)[0])
                      if seq_length is not None and len(seq_length) > data_idx
                      else len(org_data[data_idx]))
            row = np.asarray(org_data[data_idx][:length])
            series = row[:, col_index] if row.ndim == 2 else row.reshape(-1)
            plt.figure(figsize=(10, 6))
            plt.plot(series)
            plt.title(f"{t['cluster_prefix']} {cluster_id} - {t['series']} {idx + 1}")
            plt.xlabel(t["time"])
            plt.ylabel(t["value"])
            plt.savefig(os.path.join(cluster_dir, f"item_{idx + 1}.png"))
            plt.close()
            written += 1
    return written


def _series_list(org_data, seq_length, col_index):
    out = []
    for i in range(len(org_data)):
        raw_len = (seq_length[i] if seq_length is not None and i < len(seq_length)
                   else np.asarray(org_data[i]).shape[0])
        try:
            eff_len = int(np.asarray(raw_len).reshape(-1)[0])
        except Exception:
            eff_len = np.asarray(org_data[i]).shape[0]
        arr = np.asarray(org_data[i])
        eff_len = max(1, min(eff_len, arr.shape[0]))
        series = arr[:eff_len, col_index] if arr.ndim >= 2 else arr[:eff_len].reshape(-1)
        out.append(np.asarray(series, dtype=np.float64).reshape(-1))
    return out


def center_stacked_tsne(labels, org_data, feature_matrix, seq_length, save_dir,
                        tag, dist_method="euclidean", col_index=1,
                        sampling_threshold=200, cluster_stack_count=50,
                        visualize_noise=True, language="en"):
    """The three standard charts: cluster centers, stacked series, tSNE."""
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    t = texts(language)
    setup_fonts(language)

    labels = np.asarray(labels)
    valid_mask = labels != -1
    valid_labels = labels[valid_mask]
    n_clusters = len(np.unique(valid_labels))
    if n_clusters == 0:
        print("[cluster_charts] no non-noise clusters — skipped")
        return

    series = _series_list(org_data, seq_length, col_index)
    colors = plt.cm.tab10(np.arange(10))

    # ── 1. cluster centers ───────────────────────────────────────
    center_title = t["center_dtw"] if dist_method == "dtw" else t["center_mean"]
    fig, axes = plt.subplots(n_clusters, 1, figsize=(12, max(8, n_clusters * 2)))
    fig.suptitle(center_title, fontsize=14, fontweight="bold")
    axes = [axes] if n_clusters == 1 else list(np.atleast_1d(axes).flatten())

    for i, cluster_id in enumerate(np.unique(valid_labels)):
        idx_all = np.where(labels == cluster_id)[0]
        cluster_seq = [series[j] for j in idx_all]
        if len(cluster_seq) > sampling_threshold:
            rng = np.random.RandomState(42)
            pick = rng.choice(len(cluster_seq), size=sampling_threshold, replace=False)
            cluster_seq = [cluster_seq[j] for j in pick]
        if not cluster_seq:
            continue
        min_len = min(len(s) for s in cluster_seq)
        if min_len <= 0:
            continue
        aligned = np.asarray([s[:min_len] for s in cluster_seq], dtype=np.float64)
        if dist_method == "dtw":
            from tslearn.barycenters import dtw_barycenter_averaging
            center = dtw_barycenter_averaging(aligned)
        else:
            center = np.mean(aligned, axis=0)
        axes[i].plot(center, color=colors[int(cluster_id) % 10], linewidth=2.5,
                     label=f"{t['cluster_prefix']} {cluster_id} "
                           f"({t['sample_count']}: {len(idx_all)})")
        axes[i].set_title(f"{t['cluster_prefix']} {cluster_id} {t['cluster_center_suffix']}",
                          fontsize=12)
        axes[i].set_xlabel(t["time_step"], fontsize=10)
        axes[i].set_ylabel(t["series_value"], fontsize=10)
        axes[i].legend(fontsize=9)
        axes[i].grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{tag}_center.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # ── 2. stacked series ────────────────────────────────────────
    has_noise = bool(np.any(labels == -1))
    show_noise = bool(visualize_noise and has_noise)
    stack_count = max(1, int(cluster_stack_count))
    total_plots = n_clusters + (1 if show_noise else 0)
    n_cols = min(3, total_plots)
    n_rows = (total_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    fig.suptitle(t["stacked_with_noise"] if has_noise else t["stacked_without_noise"],
                 fontsize=16, fontweight="bold")
    axes = [axes] if total_plots == 1 else list(np.atleast_1d(axes).flatten())

    for i, cluster_id in enumerate(np.unique(valid_labels)):
        subset = [series[j] for j in np.where(labels == cluster_id)[0]][:stack_count]
        for j, s in enumerate(subset):
            axes[i].plot(s, alpha=0.6, label=f"{t['series']} {j}" if j < 3 else "")
        axes[i].set_title(f"{t['cluster_prefix']} {cluster_id} "
                          f"({t['first_n_data'].format(n=len(subset))})", fontsize=12)
        axes[i].set_xlabel(t["time_step"], fontsize=10)
        axes[i].set_ylabel(t["series_value"], fontsize=10)
        axes[i].grid(alpha=0.3, linestyle="--")
    if show_noise:
        ax = axes[n_clusters]
        subset = [series[j] for j in np.where(labels == -1)[0]][:stack_count]
        for j, s in enumerate(subset):
            ax.plot(s, alpha=0.6, color="gray", label=f"{t['noise']} {j}" if j < 3 else "")
        ax.set_title(t["noise_points_with_n"].format(
            first_n=t["first_n_data"].format(n=len(subset))), fontsize=12)
        ax.set_xlabel(t["time_step"], fontsize=10)
        ax.set_ylabel(t["series_value"], fontsize=10)
        ax.grid(alpha=0.3, linestyle="--")
    for j in range(total_plots, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{tag}_stacked.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # ── 3. tSNE ──────────────────────────────────────────────────
    feature_matrix = np.asarray(feature_matrix)
    fstd = float(feature_matrix.std()) if feature_matrix.size else 0.0
    if not np.isfinite(fstd) or fstd < 1e-9:
        print("[cluster_charts] tSNE skipped: degenerate/near-constant "
              "feature matrix (std={:.2e})".format(fstd))
        return
    perplexity = 2 if len(feature_matrix) < 5 else max(2, min(30, len(feature_matrix) // 10))
    # exact for small n (fast, no Barnes-Hut degeneracy with duplicate rows);
    # barnes_hut for the large embedding matrices (e.g. ukdale 9k+ samples).
    method = "exact" if len(feature_matrix) <= 5000 else "barnes_hut"
    import time as _time
    _t0 = _time.time()
    tsne_2d = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                   init="pca", method=method).fit_transform(feature_matrix)
    print(f"[cluster_charts] tSNE({method}) n={len(feature_matrix)} -> "
          f"{_time.time() - _t0:.1f}s")
    plt.figure(figsize=(10, 8))
    for cluster_id in np.unique(valid_labels):
        idx = labels == cluster_id
        plt.scatter(tsne_2d[idx, 0], tsne_2d[idx, 1], c=[colors[int(cluster_id) % 10]],
                    label=f"{t['cluster_prefix']} {cluster_id}", s=70, alpha=0.8,
                    edgecolors="white", linewidth=0.5)
    if show_noise:
        noise_idx = labels == -1
        plt.scatter(tsne_2d[noise_idx, 0], tsne_2d[noise_idx, 1], c="black",
                    marker="x", label=t["noise"], s=90, alpha=0.8)
    plt.title(t["tsne_title"], fontsize=14, fontweight="bold")
    plt.xlabel(t["tsne_dim1"], fontsize=11)
    plt.ylabel(t["tsne_dim2"], fontsize=11)
    plt.legend(fontsize=10, loc="best")
    plt.grid(alpha=0.2, linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{tag}_tsne.png"), dpi=300, bbox_inches="tight")
    plt.close()


def kmeans_scan_chart(scan_payload: dict, save_dir: str) -> str:
    """SCI/DBI/CHI per candidate k from a kmeans_scan.json diagnostic payload."""
    import matplotlib.pyplot as plt

    records = scan_payload["records"]
    ks = [r["n_clusters"] for r in records]
    series = [("sci", "SCI (higher is better)", "tab:blue"),
              ("dbi", "DBI (lower is better)", "tab:orange"),
              ("chi", "CHI (higher is better)", "tab:green")]
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), dpi=150)
    fig.suptitle(f"KMeans scan ({scan_payload.get('feature_model', '?')}, "
                 f"{scan_payload.get('segment_method', '?')}) — diagnostic, "
                 f"recommended k={scan_payload.get('recommended_n_clusters')}",
                 fontsize=13, fontweight="bold")
    for ax, (key, ylabel, color) in zip(axes, series):
        ax.plot(ks, [r[key] for r in records], marker="o", color=color)
        ax.set_xlabel("n_clusters")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(save_dir, "kmeans_scan.png")
    plt.savefig(out)
    plt.close()
    return out
