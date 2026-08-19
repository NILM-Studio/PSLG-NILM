"""Visualize discovered full-cycle classes and representative state patterns."""
from __future__ import annotations

import argparse
import json
import os

from visualize.viz_common import (fig_dir, load_manifest, load_viz_config,
                                  require, setup_fonts)


def main():
    ap = argparse.ArgumentParser(
        description="Cycle-class support and representative state patterns.")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    viz = load_viz_config(args.config)
    setup_fonts(viz.get("language", "en"))
    import matplotlib.pyplot as plt

    manifest = load_manifest(args.run_id)
    path = require(manifest, "cycle_classification", "cycle_classes")
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    classes = payload.get("classes", [])
    if not classes:
        raise SystemExit("cycle classification contains no valid classes")

    fig, (ax_support, ax_pattern) = plt.subplots(
        1, 2, figsize=(15, max(5, len(classes) * 0.7)),
        gridspec_kw={"width_ratios": [1, 2.2]})
    labels = [f"Class {entry['class_id']}" for entry in classes]
    supports = [entry["support"] for entry in classes]
    y_positions = list(range(len(classes)))

    ax_support.barh(y_positions, supports, color="#4778a8")
    ax_support.set_yticks(y_positions, labels)
    ax_support.invert_yaxis()
    ax_support.set_xlabel("Real activity count")
    ax_support.set_title("Cycle class support")
    for y, support in zip(y_positions, supports):
        ax_support.text(support, y, f" {support}", va="center")
    ax_support.grid(True, axis="x", linestyle=":", alpha=0.5)

    cmap = plt.get_cmap("tab10")
    for y, entry in zip(y_positions, classes):
        signature = entry["representative_signature"]
        for index, state in enumerate(signature):
            ax_pattern.barh(y, 1, left=index, color=cmap(int(state) % 10),
                            edgecolor="white", alpha=0.75)
            ax_pattern.text(index + 0.5, y, str(state), ha="center", va="center")
    ax_pattern.set_yticks(y_positions, labels)
    ax_pattern.invert_yaxis()
    ax_pattern.set_xlabel("Ordered functional-state blocks")
    ax_pattern.set_title("Representative primitive-state combination")
    ax_pattern.grid(True, axis="x", linestyle=":", alpha=0.5)

    fig.suptitle(
        f"Discovered cycle classes: {payload['n_classes']} classes, "
        f"{payload['n_outliers']} outliers")
    fig.tight_layout()
    out_dir = fig_dir(args.run_id, "cycle_classification")
    out_path = os.path.join(out_dir, "cycle_classes.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
