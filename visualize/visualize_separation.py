"""Reconstruction charts: segment boundaries + tensor channels per source CSV.

Usage:  python -m visualize.visualize_separation --run-id <id> [--max-files 20]

Fixes vs legacy: the file limit is a real CLI arg (the legacy script read a
module-level ``nums`` that only existed under __main__), and inputs come from
the manifest instead of reconstructed folder names.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from visualize.viz_common import (fig_dir, load_manifest, load_viz_config,
                                  require, segments_dir, setup_fonts)

_CHANNEL_STYLE = [(90 / 255, 164 / 255, 174 / 255), (250 / 255, 192 / 255, 61 / 255),
                  (74 / 255, 75 / 255, 157 / 255), (200 / 255, 22 / 255, 29 / 255)]


def main():
    ap = argparse.ArgumentParser(description="Plot segmentation reconstruction (manifest-driven).")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--max-files", type=int, default=20)
    args = ap.parse_args()

    viz = load_viz_config(args.config)
    setup_fonts(viz.get("language", "en"))
    import matplotlib.pyplot as plt

    manifest = load_manifest(args.run_id)
    seg_dir = segments_dir(manifest)
    X = np.load(require(manifest, "time_segmentation", "X"))
    lengths = np.load(require(manifest, "time_segmentation", "lengths")).reshape(-1)
    indices = np.load(require(manifest, "time_segmentation", "indices"))
    out_dir = fig_dir(args.run_id, "separation")

    csv_files = sorted(f for f in os.listdir(seg_dir) if f.lower().endswith(".csv"))
    red = (200 / 255, 22 / 255, 29 / 255)
    n_channels = X.shape[2]

    for i, csv_name in enumerate(csv_files[: args.max_files]):
        mask = indices[:, 0] == i
        rows = np.where(mask)[0]
        if len(rows) == 0:
            continue
        starts = indices[mask, 1]
        internal_cps = starts[starts > 0]

        df = pd.read_csv(os.path.join(seg_dir, csv_name))
        signal = (df["power"] if "power" in df.columns else df.iloc[:, -1]).to_numpy()

        n_sub = 1 + max(0, n_channels - 1)
        fig, axes = plt.subplots(n_sub, 1, figsize=(15, 3 * n_sub))
        axes = [axes] if n_sub == 1 else list(axes)

        axes[0].plot(signal, color="gray", alpha=0.7, label="Original Signal")
        for j, cp in enumerate(internal_cps):
            axes[0].axvline(x=cp, color=red, linestyle="--", alpha=0.8,
                            label="Segment Boundary" if j == 0 else None)
        axes[0].set_title(f"Reconstructed Analysis - {csv_name}")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, linestyle=":", alpha=0.6)

        for ch in range(1, n_channels):
            ax = axes[ch]
            offset = 0
            for r in rows:
                n = int(lengths[r])
                ax.plot(np.arange(offset, offset + n), X[r, :n, ch],
                        color=_CHANNEL_STYLE[(ch - 1) % len(_CHANNEL_STYLE)])
                offset += n
            ax.set_title(f"Channel {ch}")
            ax.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        out = os.path.join(out_dir, f"reconstruction_{os.path.splitext(csv_name)[0]}.png")
        plt.savefig(out)
        plt.close()
        print(f"saved {out}")
    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
