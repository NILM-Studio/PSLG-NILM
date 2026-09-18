"""Split-point density charts for a primitive segmentation run.

For each working-state (activity) segment produced by ``extract_active_data``,
a primitive segmentation method (prim-glr, clasp, clasp-origin, fluss, ...)
divides it into sub-events (primitives). The internal boundaries (starts > 0)
are the "split points".

This script renders two panels + a colorbar showing how those split points are
distributed over the segment:

  (A) stacked raster   — one row per working segment, split points plotted at
                         their absolute in-segment step position, colored by
                         their ordinal (1st -> 2nd -> ... from the front).
  (B) stacked histogram— binned over absolute step position; per-ordinal counts
                         stacked so both the total density and the per-ordinal
                         contribution are visible.

``render_split_density`` is the reusable core (used by the manifest CLI below
and by ``run_split_density_all`` for cross-method comparison on one dataset).

Usage (manifest-driven):
    python -m visualize.visualize_split_density --run-id <date-prefixed-id>
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from visualize.viz_common import (load_manifest, load_viz_config,
                                  require, setup_fonts, fig_dir)

DEFAULT_N_ORDINALS = 8       # cap distinct ordinal colors; ordinals > N group as "N+"


def collect_cut_points(indices: np.ndarray) -> dict:
    """Group internal (start>0) primitive boundaries per working segment.

    Returns a dict keyed by working-segment index; each value is a list of
    ``(ordinal, absolute_step)`` cut points (ordinal 1-based, from the front).
    """
    per_file = {}
    order = {}
    for f, s in indices:
        order.setdefault(int(f), []).append(int(s))
    for f, starts in order.items():
        starts = sorted(starts)
        internal = [s for s in starts if s > 0]
        per_file[f] = [(ordinal + 1, pos) for ordinal, pos in enumerate(internal)]
    return per_file


def ordinal_color(ordinal: int, cmap) -> tuple:
    """Map an ordinal to a color; ordinals beyond the cap map to the last color."""
    cap = cmap.N
    t = (min(ordinal, cap) - 1) / max(cap - 1, 1)
    return tuple(cmap(t))


def render_split_density(indices: np.ndarray, out_dir: str, method: str,
                         n_ordinals: int = DEFAULT_N_ORDINALS, n_bins: int = 40,
                         active_segments: int = None,
                         title_suffix: str = "") -> dict:
    """Render the two density panels for a given set of indices.

    ``indices`` shape (N, 2) = [working-segment index, primitive start position].
    Writes ``split_points_raster.png`` and ``split_points_density.png`` into
    ``out_dir``. Returns a summary dict.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    per_file = collect_cut_points(indices)
    n_files = max(active_segments or 0, max(per_file.keys(), default=-1) + 1)
    n_ord = max(1, n_ordinals)
    n_bins = max(5, n_bins)

    all_pos = [p for pts in per_file.values() for _, p in pts]
    if not all_pos:
        raise SystemExit(f"no split points for method '{method}'.")
    x_max = int(np.ceil(max(all_pos) * 1.05))
    bins = np.linspace(0, x_max, n_bins + 1)

    total = sum(len(pts) for pts in per_file.values())
    max_ord = max((o for pts in per_file.values() for o, _ in pts), default=0)

    cmap = plt.get_cmap("turbo", n_ord)
    sm = ScalarMappable(norm=Normalize(vmin=1, vmax=max(2, n_ord)), cmap=cmap)
    os.makedirs(out_dir, exist_ok=True)

    # ── Panel A: stacked raster ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 9))
    for row, pts in per_file.items():
        base_color = (0.85, 0.85, 0.85)
        seg_len = max([p for _, p in pts], default=0)
        ax.plot([0, max(seg_len, 1)], [row, row], color=base_color, lw=0.6, zorder=1)
        for ordinal, pos in pts:
            ax.scatter(pos, row, s=6, color=ordinal_color(ordinal, cmap),
                       edgecolors="none", zorder=2)
    ax.set_xlim(0, x_max)
    ax.set_ylim(-0.5, max(0, n_files - 0.5))
    ax.set_xlabel("Absolute in-segment step")
    ax.set_ylabel("Working-state segment")
    ax.set_title(f"[{method}] Split points per working segment (colored by ordinal)"
                 f"{title_suffix}\n{total} split points across {n_files} segments, "
                 f"max ordinal={max_ord}")
    fig.colorbar(sm, ax=ax, label="Ordinal (from front)")
    fig.tight_layout()
    raster_out = os.path.join(out_dir, "split_points_raster.png")
    fig.savefig(raster_out, dpi=150)
    plt.close(fig)

    # ── Panel B: stacked histogram ──────────────────────────────
    counts = np.zeros((n_ord, n_bins), dtype=int)
    for pts in per_file.values():
        for ordinal, pos in pts:
            o = min(ordinal, n_ord) - 1
            b = int(np.clip(np.searchsorted(bins, pos, side="right") - 1, 0, n_bins - 1))
            counts[o, b] += 1

    labels = [str(i + 1) for i in range(n_ord - 1)] + [f"{n_ord}+"]

    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(n_bins)
    for i in range(n_ord):
        color = ordinal_color(i + 1, cmap)
        ax.bar(bins[:-1], counts[i], width=bins[1] - bins[0],
               bottom=bottom, color=color, label=labels[i], edgecolor="none")
        bottom += counts[i]

    try:
        from scipy.stats import gaussian_kde
        if len(all_pos) > 1:
            kde = gaussian_kde(all_pos, bw_method=0.15)
            xs = np.linspace(0, x_max, 200)
            ymax = bottom.max() if bottom.max() > 0 else 1
            scale = ymax * ((xs[-1] - xs[0]) / (n_bins * (bins[1] - bins[0])))
            ax.plot(xs, kde(xs) * scale, color="black", lw=1.2, alpha=0.7,
                    label="total KDE")
    except Exception:
        pass

    ax.set_xlim(0, x_max)
    ax.set_xlabel("Absolute in-segment step")
    ax.set_ylabel("Split-point count (stacked by ordinal)")
    ax.set_title(f"[{method}] Split-point density by ordinal{title_suffix}\n"
                 f"(total {total} split points)")
    ax.legend(title="Ordinal", loc="upper right", ncol=2, fontsize=8)
    fig.tight_layout()
    hist_out = os.path.join(out_dir, "split_points_density.png")
    fig.savefig(hist_out, dpi=150)
    plt.close(fig)

    print(f"[{method}] raster={raster_out}")
    print(f"[{method}] density={hist_out}")
    summary = {"method": method, "segments": n_files, "split_points": total,
               "max_ordinal": max_ord, "x_max": x_max}
    print(f"[{method}] summary={summary}")
    return summary


def main():
    ap = argparse.ArgumentParser(description="Split-point density charts (manifest-driven).")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-ordinals", type=int, default=DEFAULT_N_ORDINALS)
    ap.add_argument("--n-bins", type=int, default=40)
    args = ap.parse_args()

    viz = load_viz_config(args.config)
    language = viz.get("language", "en")
    setup_fonts(language)

    manifest = load_manifest(args.run_id)
    indices = np.load(require(manifest, "time_segmentation", "indices"))
    method = (manifest.data.get("variants", {}) or {}).get("segment_method",
                                                           "unknown")
    nseg = int((manifest.data.get("steps", {})
                .get("extract_active_data", {})
                .get("artifacts", {})
                .get("count", "0")) or 0)

    out_dir = fig_dir(args.run_id, "split_density")
    render_split_density(indices, out_dir, method,
                         n_ordinals=args.n_ordinals, n_bins=args.n_bins,
                         active_segments=nseg)


if __name__ == "__main__":
    main()
