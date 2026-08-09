"""State-merge reconstruction charts: one working state, colored function spans.

Usage:
    python -m visualize.visualize_state_merge --run-id <id>                  # all *_merged tags
    python -m visualize.visualize_state_merge --run-id <id> --cluster-tag kmeans_k4_merged
    python -m visualize.visualize_state_merge --run-id <id> --max-figs 5

Per tag (a merged tag like ``kmeans_k4_merged``) renders up to ``--max-figs``
(5 by default) figures, each showing ONE activity (a working state / CSV):

    top panel    segment-level cluster labels (over-segmented: A,B,B,C ...)
    bottom panel merged functional states (A,B,C) — the temporal restoration

Colors are keyed by the *state label* so the same function keeps its color in
both panels. For a non-merged tag (``--cluster-tag kmeans_k4``) only the top
panel is drawn. Figures land in ``output/<run_id>/figure/state_merge/<tag>/``.

Activities are ranked by merge effect (changed labels first, then RLE-merged
segments, then segment count) so the figures showcase the restoration.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from visualize.viz_common import (fig_dir, load_manifest, load_viz_config,
                                  require_cluster, resolve_cluster_tags,
                                  segments_dir, setup_fonts, texts)


def _merged_tag(base_tag: str) -> str:
    return f"{base_tag}_merged"


def _resolve_tags(manifest, tag_arg: str) -> list:
    """--cluster-tag value -> tag list. Default: all *_merged tags (fall back
    to all base tags when no merge step has run yet)."""
    if tag_arg:
        return resolve_cluster_tags(manifest, tag_arg)
    tags = manifest.cluster_tags()
    merged = [t for t in tags if t.endswith("_merged")]
    if merged:
        return merged
    return tags


def _load_tag(manifest, tag: str):
    """Segment-aligned cluster data for one tag."""
    indices = np.load(require_cluster(manifest, tag, "indices")).astype(np.int64)
    seq_len = np.load(require_cluster(manifest, tag, "seq_len")).reshape(-1).astype(np.int64)
    labels = np.load(require_cluster(manifest, tag, "labels")).reshape(-1).astype(np.int64)
    return labels, indices, seq_len


def _activity_rows(indices: np.ndarray) -> dict:
    per = defaultdict(list)
    for r in range(len(indices)):
        per[int(indices[r, 0])].append(r)
    return per


def _rank_activities(manifest, tag: str, base_tag: str | None, per: dict,
                     indices: np.ndarray, seq_len: np.ndarray,
                     n_csv: int, max_figs: int) -> list:
    """Pick up to ``max_figs`` activities, preferring ones where merging
    actually changed the label sequence (best demonstrations)."""
    state_seq = None
    if base_tag is not None:
        with open(require_cluster(manifest, tag, "state_sequences"), encoding="utf-8") as f:
            state_seq = json.load(f)
        base_idx = np.load(require_cluster(manifest, base_tag, "indices")).astype(np.int64)

    scored = []
    for fid, rows in per.items():
        if int(fid) >= n_csv:
            continue
        n_seg = len(rows)
        if base_tag is not None:
            fin = [int(indices[r, 2]) for r in rows]
            base = [int(base_idx[r, 2]) for r in rows]
            n_changed = sum(1 for a, b in zip(fin, base) if a != b)
            n_blocks = len(state_seq.get(str(fid), []))
            n_rle = n_seg - n_blocks
            scored.append((n_changed, n_rle, n_seg, int(fid)))
        else:
            scored.append((0, 0, n_seg, int(fid)))
    scored.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
    return [t[3] for t in scored[: max_figs]]


def _render_one(plt, out_path: str, tag: str, csv_path: str, csv_idx: int,
                per_rows: list, indices: np.ndarray, seq_len: np.ndarray,
                final_indices: np.ndarray | None, state_seq: dict | None,
                language: str) -> None:
    """One activity figure: signal + colored spans (top: cluster labels,
    bottom: merged functional states when available)."""
    t = texts(language)
    df = pd.read_csv(csv_path)
    col = "power" if "power" in df.columns else df.columns[-1]
    signal = df[col].to_numpy()

    cmap = plt.get_cmap("tab10")
    noise_color = (0.7, 0.7, 0.7, 0.35)

    def color_for(label: int):
        if label == -1:
            return noise_color
        c = cmap(label % 10)
        return (c[0], c[1], c[2], 0.35)

    rows = sorted(per_rows, key=lambda r: int(indices[r, 1]))
    spans_top = []
    for r in rows:
        s = int(indices[r, 1])
        e = s + int(seq_len[r])
        spans_top.append((s, e, int(indices[r, 2])))

    has_bottom = final_indices is not None and state_seq is not None
    spans_bottom = []
    if has_bottom:
        for b in state_seq.get(str(csv_idx), []):
            spans_bottom.append((int(b["start"]), int(b["end"]), int(b["state_label"])))

    # legend from all unique labels present
    seen = []
    for s, e, lab in spans_top + spans_bottom:
        if lab not in seen:
            seen.append(lab)
    seen.sort()

    fig, axs = plt.subplots(2 if has_bottom else 1, 1,
                            figsize=(15, 9 if has_bottom else 5), sharex=True)
    if has_bottom:
        ax_top, ax_bottom = axs
    else:
        ax_top = axs

    for ax in (ax_top,) if not has_bottom else (ax_top, ax_bottom):
        ax.plot(signal, color="black", linewidth=1, alpha=0.85, label=t["power_signal"])

    for s, e, lab in spans_top:
        ax_top.axvspan(s, e, color=color_for(lab))
        ax_top.axvline(s, color="red", linestyle="--", alpha=0.3, linewidth=0.8)
        ax_top.text(s + (e - s) / 2, np.max(signal) * 0.92, str(lab),
                    ha="center", va="top", fontsize=8, color="darkred")
    ax_top.set_title(t["state_merge_raw"], fontsize=11)

    if has_bottom:
        for s, e, lab in spans_bottom:
            ax_bottom.axvspan(s, e, color=(color_for(lab)[0], color_for(lab)[1],
                                           color_for(lab)[2], 0.5))
            ax_bottom.axvline(s, color="red", linestyle="-", alpha=0.4, linewidth=0.8)
            ax_bottom.text(s + (e - s) / 2, np.max(signal) * 0.92, str(lab),
                           ha="center", va="top", fontsize=9, color="darkred")
        ax_bottom.set_title(t["state_merge_states"], fontsize=11)

    legend_handles = [plt.Line2D([0], [0], marker="s", color="w", alpha=0.0,
                                 markerfacecolor=color_for(lab)[:3], markersize=10,
                                 label=f"{t['cluster_prefix']} {lab}") for lab in seen]
    for ax in ((ax_top, ax_bottom) if has_bottom else (ax_top,)):
        ax.legend(handles=legend_handles, loc="upper right",
                  bbox_to_anchor=(1.0, 1.0), fontsize=8, ncol=max(1, len(seen) // 6 + 1))
        ax.set_xlabel(t["time"])
        ax.set_ylabel(t["power"])
        ax.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle(f"{tag} - {t['state_merge_title']} - {os.path.basename(csv_path)}",
                 fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="State-merge reconstruction charts (manifest-driven).")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--cluster-tag", default=None,
                    help="default: ALL *_merged tags (fallback: all tags)")
    ap.add_argument("--max-figs", type=int, default=5,
                    help="figures per tag, one activity each (default 5)")
    args = ap.parse_args()

    viz = load_viz_config(args.config)
    language = viz.get("language", "en")
    setup_fonts(language)
    import matplotlib.pyplot as plt

    manifest = load_manifest(args.run_id)
    seg_dir = segments_dir(manifest)
    csv_files = sorted(f for f in os.listdir(seg_dir) if f.lower().endswith(".csv"))
    if not csv_files:
        raise SystemExit(f"no activity CSV segments in {seg_dir}")

    for tag in _resolve_tags(manifest, args.cluster_tag):
        base_tag = tag[:-len("_merged")] if tag.endswith("_merged") else None
        labels, indices, seq_len = _load_tag(manifest, tag)
        per = _activity_rows(indices)

        final_indices, state_seq = None, None
        if base_tag is not None:
            final_indices = indices  # segment-aligned final labels in col 3
            with open(require_cluster(manifest, tag, "state_sequences"), encoding="utf-8") as f:
                state_seq = json.load(f)

        out_dir = fig_dir(args.run_id, os.path.join("state_merge", tag))
        chosen = _rank_activities(manifest, tag, base_tag, per, indices, seq_len,
                                  len(csv_files), args.max_figs)
        n = 0
        for csv_idx in chosen:
            out_path = os.path.join(
                out_dir, f"state_{os.path.splitext(csv_files[csv_idx])[0]}.png")
            _render_one(plt, out_path, tag, os.path.join(seg_dir, csv_files[csv_idx]),
                        csv_idx, per[csv_idx], indices, seq_len,
                        final_indices, state_seq, language)
            n += 1
            print(f"  saved {out_path}")
        print(f"[{tag}] {n} figures -> {out_dir}")


if __name__ == "__main__":
    main()
