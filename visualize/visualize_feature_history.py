"""Training-history (loss curve) chart for the feature-extract step.

Usage:  python -m visualize.visualize_feature_history --run-id <id>

The step itself only stores ``training_history.json`` (decoupled charts);
this script turns it into a png on demand.
"""
from __future__ import annotations

import argparse
import json
import os

from visualize.viz_common import fig_dir, load_manifest, load_viz_config, require, setup_fonts


def main():
    ap = argparse.ArgumentParser(description="Feature-extract training history chart.")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    viz = load_viz_config(args.config)
    setup_fonts(viz.get("language", "en"))
    import matplotlib.pyplot as plt

    manifest = load_manifest(args.run_id)
    with open(require(manifest, "feature_extract", "training_history"), encoding="utf-8") as f:
        history = json.load(f)

    curves = {k: v for k, v in history.items()
              if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v)}
    if not curves:
        raise SystemExit("training_history.json has no plottable numeric series")

    extra = manifest.data.get("steps", {}).get("feature_extract", {})
    tag = extra.get("variant", "feature_extract")
    plt.figure(figsize=(10, 6))
    for name, values in curves.items():
        plt.plot(values, label=name)
    plt.title(f"Training History - {tag}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    out = os.path.join(fig_dir(args.run_id, "feature_history"), f"{tag}_history.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"saved {out}")


if __name__ == "__main__":
    main()
