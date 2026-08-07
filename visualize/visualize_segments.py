"""Plot the first N active-segment CSVs of a run.

Usage:  python -m visualize.visualize_segments --run-id <id> [--max-plots 30]
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from visualize.viz_common import fig_dir, load_manifest, load_viz_config, segments_dir, setup_fonts


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

    csv_files = sorted(f for f in os.listdir(seg_dir) if f.lower().endswith(".csv"))
    if not csv_files:
        raise SystemExit(f"no CSV segments in {seg_dir}")

    signal_color = (74 / 255, 75 / 255, 157 / 255)
    for name in csv_files[: args.max_plots]:
        df = pd.read_csv(os.path.join(seg_dir, name))
        col = "power" if "power" in df.columns else df.columns[-1]
        plt.figure(figsize=(12, 6))
        plt.plot(df[col].to_numpy(), color=signal_color, alpha=0.8, label="Power Signal")
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
