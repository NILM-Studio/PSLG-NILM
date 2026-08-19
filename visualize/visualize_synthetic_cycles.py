"""Render synthesized appliance cycles with state-colored spans.

Usage:
    python -m visualize.visualize_synthetic_cycles --run-id <id> --max-files 20
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from visualize.viz_common import (fig_dir, load_manifest, load_viz_config,
                                  require, setup_fonts)


def main():
    ap = argparse.ArgumentParser(
        description="Synthetic appliance-cycle charts (manifest-driven).")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--max-files", type=int, default=20)
    args = ap.parse_args()

    viz = load_viz_config(args.config)
    setup_fonts(viz.get("language", "en"))
    import matplotlib.pyplot as plt

    manifest = load_manifest(args.run_id)
    cycles_dir = require(manifest, "primitive_synthesis", "cycles_dir")
    files = sorted(f for f in os.listdir(cycles_dir)
                   if f.startswith("synthetic_cycle_") and f.endswith(".csv"))
    if not files:
        raise SystemExit(f"no synthetic cycle CSV files in {cycles_dir}")

    out_dir = fig_dir(args.run_id, "primitive_synthesis")
    cmap = plt.get_cmap("tab10")
    for filename in files[:max(0, args.max_files)]:
        frame = pd.read_csv(os.path.join(cycles_dir, filename))
        if frame.empty:
            continue

        cycle_class = (int(frame["cycle_class"].iloc[0])
                       if "cycle_class" in frame.columns else None)
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(frame["time_seconds"], frame["power"], color="black",
                linewidth=1, alpha=0.85, label="Synthetic power")
        seen = set()
        for block_id, block in frame.groupby("block_id", sort=False):
            state = int(block["state_label"].iloc[0])
            start = float(block["time_seconds"].iloc[0])
            end = float(block["time_seconds"].iloc[-1])
            color = cmap(state % 10)
            label = f"State {state}"
            ax.axvspan(start, end, color=color, alpha=0.3,
                       label=label if state not in seen else None)
            ax.axvline(start, color="red", linestyle="--", alpha=0.3, linewidth=0.8)
            seen.add(state)
            y = float(frame["power"].max()) * 0.92
            ax.text((start + end) / 2, y, str(state), ha="center",
                    va="top", fontsize=8, color="darkred")

        class_text = f" - Class {cycle_class}" if cycle_class is not None else ""
        ax.set_title(f"Synthetic appliance cycle{class_text} - {filename}")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Power")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right")
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"{os.path.splitext(filename)[0]}.png")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out_path}")

    print(f"[primitive_synthesis] figures -> {out_dir}")


if __name__ == "__main__":
    main()
