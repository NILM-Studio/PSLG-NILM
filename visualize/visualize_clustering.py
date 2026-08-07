"""Clustering charts for one or ALL tagged results of a run.

Usage:
    python -m visualize.visualize_clustering --run-id <id>                    # all tags
    python -m visualize.visualize_clustering --run-id <id> --cluster-tag kmeans_k4

Per tag: prints metrics.json, renders per-cluster item pics + center/stacked/
tSNE charts into output/<run_id>/figure/clustering/<tag>/. If a kmeans-scan
diagnostic exists, its SCI/DBI/CHI chart is rendered too.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from visualize import cluster_charts
from visualize.viz_common import (fig_dir, load_manifest, load_viz_config,
                                  require, require_cluster, resolve_cluster_tags)


def main():
    ap = argparse.ArgumentParser(description="Clustering charts (manifest-driven).")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--cluster-tag", default=None, help="default: ALL tags in the run")
    ap.add_argument("--no-item-pics", action="store_true",
                    help="skip the one-png-per-series cluster_<id>/ folders")
    args = ap.parse_args()

    viz = load_viz_config(args.config)
    language = viz.get("language", "en")
    col_index = viz.get("col_index", 2)
    manifest = load_manifest(args.run_id)
    tags = resolve_cluster_tags(manifest, args.cluster_tag)

    # shared inputs, resolved once
    X = np.load(require(manifest, "time_segmentation", "X"))

    for tag in tags:
        out_dir = fig_dir(args.run_id, os.path.join("clustering", tag))
        labels = np.load(require_cluster(manifest, tag, "labels"))
        kept = np.load(require_cluster(manifest, tag, "kept_rows"))
        org_data = X[kept]
        seq_len = np.load(require_cluster(manifest, tag, "seq_len"))
        features = np.load(require_cluster(manifest, tag, "feature_matrix"))

        with open(require_cluster(manifest, tag, "metrics"), encoding="utf-8") as f:
            metrics = json.load(f)
        print(f"[{tag}] silhouette={metrics.get('silhouette_score')}, "
              f"dbi={metrics.get('davies_bouldin_score')}, "
              f"chi={metrics.get('calinski_harabasz_score')}, "
              f"clusters={metrics.get('n_clusters')}, noise={metrics.get('n_noise')}")

        metric = metrics.get("metric", "euclidean")
        dist_method = "dtw" if metric in ("dtw", "fastdtw") else "euclidean"

        if not args.no_item_pics:
            n = cluster_charts.per_cluster_item_pics(
                org_data, seq_len, labels, out_dir, threshold=200,
                col_index=col_index, language=language)
            print(f"[{tag}] {n} item pics")
        cluster_charts.center_stacked_tsne(
            labels, org_data, features, seq_len, out_dir, tag,
            dist_method=dist_method, col_index=col_index,
            cluster_stack_count=viz.get("cluster_stack_count", 50),
            visualize_noise=viz.get("visualize_noise", True), language=language)
        print(f"[{tag}] center/stacked/tsne -> {out_dir}")

    # optional kmeans-scan diagnostic chart
    scan_path = manifest.artifact_path("time_clustering", "kmeans_scan")
    if scan_path and os.path.exists(scan_path):
        with open(scan_path, encoding="utf-8") as f:
            payload = json.load(f)
        out = cluster_charts.kmeans_scan_chart(payload, fig_dir(args.run_id, "clustering"))
        print(f"kmeans-scan chart -> {out}")


if __name__ == "__main__":
    main()
