"""Render medoid/near/far real cycles for every discovered physical mode.

Usage:
    python -m visualize.visualize_cycle_validation --run-id <id>
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from visualize.viz_common import (fig_dir, load_manifest, load_viz_config,
                                  require, segments_dir, setup_fonts)


def main():
    parser = argparse.ArgumentParser(
        description="Cycle-validation mode representative charts.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--class-id", type=int, default=None)
    parser.add_argument("--max-modes", type=int, default=0,
                        help="Maximum class/mode figures; 0 renders all.")
    args = parser.parse_args()

    viz = load_viz_config(args.config)
    setup_fonts(viz.get("language", "en"))
    import matplotlib.pyplot as plt

    manifest = load_manifest(args.run_id)
    representatives_path = require(
        manifest, "cycle_validation", "mode_representatives")
    source_dir = segments_dir(manifest)
    representatives = pd.read_csv(representatives_path)
    if args.class_id is not None:
        representatives = representatives[
            representatives["class_id"] == args.class_id]
    groups = list(representatives.groupby(["class_id", "mode_id"], sort=True))
    if args.max_modes > 0:
        groups = groups[:args.max_modes]
    if not groups:
        raise SystemExit("no validation mode representatives matched the request")

    output_dir = fig_dir(args.run_id, "cycle_validation_modes")
    for (class_id, mode_id), rows in groups:
        rows = rows.sort_values("role", key=lambda col: col.map(
            {"medoid": 0, "near": 1, "far": 2}).fillna(3))
        fig, axes = plt.subplots(len(rows), 1, figsize=(15, 3.5 * len(rows)),
                                 squeeze=False)
        for ax, (_, row) in zip(axes[:, 0], rows.iterrows()):
            path = os.path.join(source_dir, str(row["file"]))
            frame = pd.read_csv(path)
            power_col = "power" if "power" in frame.columns else frame.columns[-1]
            power = pd.to_numeric(frame[power_col], errors="coerce").fillna(0.0).to_numpy()
            if "timestamp" in frame.columns:
                timestamp = pd.to_numeric(
                    frame["timestamp"], errors="coerce").to_numpy(dtype=np.float64)
                time_minutes = (timestamp - timestamp[0]) / 60.0
            else:
                time_minutes = np.arange(len(power), dtype=np.float64) / 0.1666667 / 60.0
            ax.plot(time_minutes, power, color="black", linewidth=0.9)
            ax.set_title(
                f"{row['role']} | activity {row['activity_id']} | {row['file']}",
                fontsize=10)
            ax.set_xlabel("Time (minutes)")
            ax.set_ylabel("Power (W)")
            ax.grid(True, linestyle=":", alpha=0.5)
        fig.suptitle(f"Cycle class {class_id} - physical mode {mode_id}",
                     fontsize=14)
        fig.tight_layout()
        output_path = os.path.join(
            output_dir, f"class_{int(class_id)}_mode_{int(mode_id)}.png")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {output_path}")
    print(f"[cycle_validation] figures -> {output_dir}")


if __name__ == "__main__":
    main()
