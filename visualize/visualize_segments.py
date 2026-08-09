"""Plot the first N active-segment CSVs of a run.

Usage:  python -m visualize.visualize_segments --run-id <id> [--max-plots 30]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from visualize.viz_common import (fig_dir, load_manifest, load_viz_config,
                                  require, segments_dir, setup_fonts)


def main():
    ap = argparse.ArgumentParser(description="Plot active segments (manifest-driven).")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--max-plots", type=int, default=30)
    args = ap.parse_args()

    viz = load_viz_config(args.config)
    setup_fonts(viz.get("language", "en"))
    import matplotlib.pyplot as plt

    manifest = load_manifest(args.run_id)
    seg_dir = segments_dir(manifest)
    out_dir = fig_dir(args.run_id, "segments")

    indices = None
    try:
        idx_path = require(manifest, "time_segmentation", "indices")
        indices = np.load(idx_path)
    except SystemExit:
        print(f"WARN: no time_segmentation.indices for run '{args.run_id}' "
              f"-> segment boundaries not annotated")

    csv_files = sorted(f for f in os.listdir(seg_dir) if f.lower().endswith(".csv"))
    if not csv_files:
        raise SystemExit(f"no CSV segments in {seg_dir}")

    signal_color = (74 / 255, 75 / 255, 157 / 255)
    boundary_color = (200 / 255, 22 / 255, 29 / 255)
    for i, name in enumerate(csv_files[: args.max_plots]):
        df = pd.read_csv(os.path.join(seg_dir, name))
        col = "power" if "power" in df.columns else df.columns[-1]
        plt.figure(figsize=(12, 6))
        plt.plot(df[col].to_numpy(), color=signal_color, alpha=0.8, label="Power Signal")
        if indices is not None:
            mask = indices[:, 0] == i
            starts = indices[mask, 1]
            internal_cps = np.sort(starts[starts > 0])
            for j, cp in enumerate(internal_cps):
                plt.axvline(x=cp, color=boundary_color, linestyle="--", alpha=0.8,
                            label="Segment Boundary" if j == 0 else None)
        plt.title(f"Segment - {name}")
        plt.xlabel("Time Steps")
        plt.ylabel("Power (W)")
        plt.legend(loc="upper right")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        out = os.path.join(out_dir, f"segment_{os.path.splitext(name)[0]}.png")
        plt.savefig(out)
        plt.close()
        print(f"saved {out}")
    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
