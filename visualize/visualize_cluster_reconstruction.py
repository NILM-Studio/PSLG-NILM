"""Cluster-label spans over the original activity signals.

Usage:
    python -m visualize.visualize_cluster_reconstruction --run-id <id> [--cluster-tag T]

Uses the tag's 3-column indices (csv_idx, start, label) + aligned seq_len —
both saved row-aligned by the clustering step, so the legacy "are indices and
lengths still matched?" guesswork is gone.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from visualize.viz_common import (fig_dir, load_manifest, load_viz_config,
                                  require_cluster, resolve_cluster_tags,
                                  segments_dir, setup_fonts)


def main():
    ap = argparse.ArgumentParser(description="Cluster reconstruction charts (manifest-driven).")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--cluster-tag", default=None, help="default: ALL tags in the run")
    ap.add_argument("--max-files", type=int, default=20)
    args = ap.parse_args()

    viz = load_viz_config(args.config)
    setup_fonts(viz.get("language", "en"))
    import matplotlib.pyplot as plt

    manifest = load_manifest(args.run_id)
    seg_dir = segments_dir(manifest)
    csv_files = sorted(f for f in os.listdir(seg_dir) if f.lower().endswith(".csv"))

    for tag in resolve_cluster_tags(manifest, args.cluster_tag):
        indices = np.load(require_cluster(manifest, tag, "indices"))  # (n, 3)
        seq_len = np.load(require_cluster(manifest, tag, "seq_len")).reshape(-1)
        out_dir = fig_dir(args.run_id, os.path.join("cluster_reconstruction", tag))

        cmap = plt.get_cmap("tab10")
        noise_color = (0.7, 0.7, 0.7, 0.3)

        for i, csv_name in enumerate(csv_files[: args.max_files]):
            mask = indices[:, 0] == i
            rows = np.where(mask)[0]
            if len(rows) == 0:
                continue
            df = pd.read_csv(os.path.join(seg_dir, csv_name))
            signal = (df["power"] if "power" in df.columns else df.iloc[:, -1]).to_numpy()

            plt.figure(figsize=(15, 6))
            plt.plot(signal, color="black", linewidth=1, alpha=0.8, label="Original Signal")
            seen_labels = set()
            for r in rows:
                start, cluster_id = int(indices[r, 1]), int(indices[r, 2])
                end = start + int(seq_len[r])
                if cluster_id == -1:
                    color, label = noise_color, "Noise"
                else:
                    c = cmap(cluster_id % 10)
                    color, label = (c[0], c[1], c[2], 0.4), f"Cluster {cluster_id}"
                plt.axvspan(start, end, color=color,
                            label=label if label not in seen_labels else None)
                seen_labels.add(label)
                plt.axvline(x=start, color="red", linestyle="--", alpha=0.3)
                plt.text(start + (end - start) / 2, np.max(signal) * 0.9, str(cluster_id),
                         ha="center", fontsize=8, color="darkred")
            plt.title(f"Cluster Reconstruction [{tag}] - {csv_name}")
            plt.xlabel("Time Index")
            plt.ylabel("Power")
            plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
            plt.grid(True, linestyle=":", alpha=0.6)
            plt.tight_layout()
            out = os.path.join(out_dir,
                               f"cluster_reconstruction_{os.path.splitext(csv_name)[0]}.png")
            plt.savefig(out, bbox_inches="tight")
            plt.close()
            print(f"saved {out}")
        print(f"[{tag}] done -> {out_dir}")


if __name__ == "__main__":
    main()
