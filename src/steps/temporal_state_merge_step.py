"""TemporalStateMerge step: temporal functional restoration of cluster labels.

Over-segmentation (prim-glr / clasp) produces unstable change points, so a
single appliance *working state* gets chopped into several segments that the
cluster step then labels — often yielding a repeated adjacent pattern like
``A,B,B,C`` where the true state sequence is ``A,B,C``. A cluster represents the
power-waveform primitive state behind a function, and that function is stable
inside one working state; only the *boundaries* are unstable.

This step therefore rebuilds the state sequence along the time axis of each
activity (per CSV), merging temporally adjacent fragments into **continuous
functional-state blocks**, and emits a new tagged result ``<tag>_merged`` per
existing cluster tag (e.g. ``kmeans_k4_merged``).

Merge algorithm (per activity, segments sorted by start):
  1. RLE — join consecutive segments with the SAME cluster label (``A,B,B,C``
     -> ``A,B,C``).
  2. Short-run absorption (方案1: minimum duration constraint) — a block
     shorter than ``min_block_seconds`` is a spurious over-segmentation
     fragment; absorb it into the longer of its two temporal neighbours
     (ties go left), labelling it with the neighbour's state. Real fast
     switching (e.g. PWM dimming) survives because its durations exceed the
     threshold; the threshold is configurable.
  3. Similar-feature merge (same-label + similar mode) — two *adjacent* blocks
     with different labels but very close aggregated features (distance below
     ``similar_feature_tol`` in the same z-score space used by clustering) are
     the label-drift case (a stable state split and clustered into B and B');
     merge them, keeping the longer block's label.

Outputs per tag (registered via ``add_cluster_result`` so downstream can use
``--cluster-tag <tag>_merged`` unchanged):

  Segment-aligned (kept for the existing fewshot/pam contract):
    cluster_labels.npy   final state label per ORIGINAL segment row
    indices.npy          (n,3) = (csv_idx, start, final_label)
    seq_len.npy          original segment lengths
    feature_matrix.npy   original (raw) feature matrix
    kept_rows.npy        original kept_rows

  Block-aligned (the reconstructed functional states — the new merged clusters):
    block_labels.npy     (n_blocks,) final state label per block
    block_indices.npy    (n_blocks,3) = (csv_idx, block_start, label)
    block_seq_len.npy    (n_blocks,) block duration in samples
    block_features.npy   (n_blocks,d) length-weighted aggregated raw features
    segment_to_block.npy (n,2) = (segment_row, block_id)
    blocks.json          full block records (spans, members, merge provenance)
    state_sequences.json per-activity reconstructed state sequence
    metrics.json         merge statistics + cluster metrics

The step is pure numpy O(n) (no GPU), deterministic, and intentionally runs
single-threaded; the determined entrypoint pins OMP/MKL/OPENBLAS/NUMBA threads
to 1.
"""
from __future__ import annotations

import gc
import json
import os
from collections import defaultdict

import numpy as np

from src.framework.step import Step
from src.utils.cluster_metrics import compute_cluster_metrics


# ============================================================================
# Pure merge primitives (module-level so tests exercise them directly)
# ============================================================================

def rle_blocks(rows, labels, starts, lengths, feat_norm, feat_raw):
    """Group consecutive same-label rows into initial blocks (RLE).

    Parameters are GLOBAL arrays indexed by segment row; ``rows`` is one
    activity's rows in chronological order. Returns a list of block dicts.
    """
    blocks = []
    cur = None
    for r in rows:
        lab = int(labels[r])
        ln = int(lengths[r])
        if cur is not None and lab == cur["label"]:
            cur["rows"].append(r)
            cur["length"] += ln
            cur["n_segments"] += 1
            cur["end"] = int(starts[r]) + ln
            cur["feat_norm_sum"] += feat_norm[r] * ln
            cur["feat_raw_sum"] += feat_raw[r] * ln
        else:
            if cur is not None:
                blocks.append(cur)
            cur = {
                "label": lab,
                "rows": [r],
                "start": int(starts[r]),
                "end": int(starts[r]) + ln,
                "length": ln,
                "n_segments": 1,
                "feat_norm_sum": np.asarray(feat_norm[r]) * ln,
                "feat_raw_sum": np.asarray(feat_raw[r]) * ln,
                "absorbed_labels": [],
                "n_absorbed_segments": 0,
                "similar_merged": False,
            }
    if cur is not None:
        blocks.append(cur)
    return blocks


def block_len(block) -> int:
    return int(block["length"])


def block_norm_feat(block) -> np.ndarray:
    return np.asarray(block["feat_norm_sum"]) / max(int(block["length"]), 1)


def block_raw_feat(block) -> np.ndarray:
    return np.asarray(block["feat_raw_sum"]) / max(int(block["length"]), 1)


def merge_blocks(left, right, label, reason):
    """Merge two temporally adjacent blocks (``left`` before ``right``).

    ``label`` is the surviving state label. ``reason`` is "short" (one side was
    a short spurious fragment) or "similar" (similar-feature merge). The result
    is a single block with concatenated members and provenance bookkeeping.
    """
    merged = {
        "label": label,
        "rows": left["rows"] + right["rows"],
        "start": left["start"],
        "end": right["end"],
        "length": left["length"] + right["length"],
        "n_segments": left["n_segments"] + right["n_segments"],
        "feat_norm_sum": np.asarray(left["feat_norm_sum"]) + np.asarray(right["feat_norm_sum"]),
        "feat_raw_sum": np.asarray(left["feat_raw_sum"]) + np.asarray(right["feat_raw_sum"]),
        "absorbed_labels": list(left["absorbed_labels"]) + list(right["absorbed_labels"]),
        "n_absorbed_segments": int(left["n_absorbed_segments"]) + int(right["n_absorbed_segments"]),
        "similar_merged": bool(left["similar_merged"]) or bool(right["similar_merged"]),
    }
    if reason == "short":
        absorbed = right if label == left["label"] else left
        merged["absorbed_labels"].append(absorbed["label"])
        merged["n_absorbed_segments"] += int(absorbed["n_segments"])
    elif reason == "similar":
        merged["similar_merged"] = True
    return merged


def absorb_short_blocks(blocks, min_len):
    """方案1: absorb blocks shorter than ``min_len`` samples into the longer
    temporal neighbour (ties go left). Iterates until stable."""
    blocks = list(blocks)
    changed = True
    while changed and len(blocks) > 1:
        changed = False
        for i, blk in enumerate(blocks):
            if block_len(blk) >= min_len:
                continue
            left_len = block_len(blocks[i - 1]) if i > 0 else -1.0
            right_len = block_len(blocks[i + 1]) if i + 1 < len(blocks) else -1.0
            if left_len >= right_len and i > 0:
                blocks[i - 1] = merge_blocks(blocks[i - 1], blk, blocks[i - 1]["label"],
                                             reason="short")
                blocks.pop(i)
            else:
                merged = merge_blocks(blk, blocks[i + 1], blocks[i + 1]["label"],
                                      reason="short")
                blocks[i] = merged
                blocks.pop(i + 1)
            changed = True
            break
    return blocks


def similar_merge_blocks(blocks, tol):
    """Merge ADJACENT blocks whose aggregated-feature distance < ``tol`` (in the
    clustering's z-score space) even when their labels differ. The surviving
    label is the longer block's. Skips noise (-1). Iterates until stable."""
    blocks = list(blocks)
    changed = True
    while changed and len(blocks) > 1:
        changed = False
        for i in range(len(blocks) - 1):
            a, b = blocks[i], blocks[i + 1]
            if a["label"] == b["label"] or a["label"] == -1 or b["label"] == -1:
                continue
            d = float(np.linalg.norm(block_norm_feat(a) - block_norm_feat(b)))
            if d < tol:
                label = a["label"] if block_len(a) >= block_len(b) else b["label"]
                blocks[i] = merge_blocks(a, b, label, reason="similar")
                blocks.pop(i + 1)
                changed = True
                break
    return blocks


def merge_activity(rows, labels, starts, lengths, feat_norm, feat_raw,
                   min_len, enable_similar, similar_tol):
    """Full per-activity merge: RLE -> short-run absorption -> similar merge."""
    blocks = rle_blocks(rows, labels, starts, lengths, feat_norm, feat_raw)
    if min_len > 0:
        blocks = absorb_short_blocks(blocks, min_len)
    if enable_similar:
        blocks = similar_merge_blocks(blocks, similar_tol)
    return blocks


# ============================================================================
# Step
# ============================================================================

class TemporalStateMergeStep(Step):
    step_type = "state_merge"

    def __init__(self, cluster_method: str = "kmeans",
                 feature_model: str = "detsec", segment_method: str = "prim-glr",
                 merge_mode: str = "same_label_plus_similar",
                 enable_similar_merge: bool = True,
                 similar_feature_tol: float = 2.0,
                 min_block_seconds: float = 90.0,
                 fs: float = 0.1666667):
        self.cluster_method = str(cluster_method).lower()
        self.feature_model = feature_model
        self.segment_method = segment_method
        super().__init__(variant=f"{self.cluster_method}_on_{self.feature_model}_on_{self.segment_method}")
        self.merge_mode = str(merge_mode).lower()
        self.enable_similar_merge = bool(enable_similar_merge)
        self.similar_feature_tol = float(similar_feature_tol)
        self.min_block_seconds = float(min_block_seconds)
        self.fs = float(fs)

    def log_subdir(self) -> str:
        return f"TemporalStateMerge_{self.variant}"

    # ── helpers ──────────────────────────────────────────────────

    def _min_block_len_samples(self) -> int:
        """Convert the min-block-seconds threshold into samples at ``fs`` Hz."""
        return int(round(max(0.0, self.min_block_seconds) * self.fs))

    def _zscore(self, features: np.ndarray) -> np.ndarray:
        from sklearn.preprocessing import StandardScaler
        return StandardScaler().fit_transform(
            np.nan_to_num(np.asarray(features, dtype=np.float64),
                          nan=0.0, posinf=0.0, neginf=0.0))

    def _resolve_tags(self, context: dict) -> list:
        tags = context["manifest"].cluster_tags()
        originals = [t for t in tags if not str(t).endswith("_merged")]
        if not originals:
            print(f"[state_merge] no non-merged cluster tags found in manifest "
                  f"(got {tags}); nothing to do.")
        return originals

    def _load_tag_artifacts(self, context: dict, tag: str):
        m = context["manifest"]

        def _get(key, required=True):
            p = m.cluster_artifact_path(tag, key)
            if p and os.path.exists(p):
                return np.load(p, allow_pickle=False)
            if required:
                raise FileNotFoundError(
                    f"[state_merge] cluster result '{tag}' missing artifact '{key}'. "
                    f"Run --steps cluster first (same --run-id).")
            return None

        labels = np.asarray(_get("labels")).reshape(-1).astype(np.int64)
        indices = _get("indices")  # (n,3): csv_idx, start, label
        if indices is None or indices.shape[1] < 2:
            # fall back: segmentation indices + labels aligned via kept_rows
            seg_idx = self.resolve(context, "time_segmentation", "indices")
            kept = _get("kept_rows", required=False)
            if seg_idx and os.path.exists(seg_idx):
                seg_idx = np.load(seg_idx)
                if kept is not None:
                    seg_idx = np.asarray(seg_idx)[np.asarray(kept).astype(np.int64)]
                indices = np.column_stack((seg_idx, labels))
        indices = np.asarray(indices).astype(np.int64)
        seq_len = np.asarray(_get("seq_len")).reshape(-1).astype(np.int64)
        feats = np.asarray(_get("feature_matrix"), dtype=np.float64)
        kept = _get("kept_rows", required=False)
        kept_rows = (np.asarray(kept).reshape(-1).astype(np.int64)
                     if kept is not None else np.arange(len(labels), dtype=np.int64))

        n = len(labels)
        if len(indices) != n or len(seq_len) != n or len(feats) != n:
            raise ValueError(
                f"[state_merge] misaligned tag '{tag}' artifacts: "
                f"labels={n} indices={len(indices)} seq_len={len(seq_len)} "
                f"feature_matrix={len(feats)}")
        return labels, indices, seq_len, feats, kept_rows

    # ── persistence ──────────────────────────────────────────────

    def _save_result(self, context, tag, out, extra):
        out_dir = os.path.join(self.log_dir(context), tag)
        os.makedirs(out_dir, exist_ok=True)

        def _save(name, obj):
            p = os.path.join(out_dir, name)
            if name.endswith(".json"):
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(obj, f, indent=2, ensure_ascii=False)
            else:
                np.save(p, obj)
            return p

        artifacts = {
            # segment-aligned (existing downstream contract)
            "labels": self.rel(context, _save("cluster_labels.npy", out["seg_labels"])),
            "indices": self.rel(context, _save("indices.npy", out["seg_indices"])),
            "seq_len": self.rel(context, _save("seq_len.npy", out["seg_seq_len"])),
            "feature_matrix": self.rel(context, _save("feature_matrix.npy", out["seg_feats"])),
            "kept_rows": self.rel(context, _save("kept_rows.npy", out["kept_rows"])),
            # block-aligned (the reconstructed functional states)
            "block_labels": self.rel(context, _save("block_labels.npy", out["block_labels"])),
            "block_indices": self.rel(context, _save("block_indices.npy", out["block_indices"])),
            "block_seq_len": self.rel(context, _save("block_seq_len.npy", out["block_seq_len"])),
            "block_features": self.rel(context, _save("block_features.npy", out["block_features"])),
            "segment_to_block": self.rel(context, _save("segment_to_block.npy", out["seg_to_block"])),
            "blocks": self.rel(context, _save("blocks.json", out["blocks_json"])),
            "state_sequences": self.rel(context, _save("state_sequences.json", out["state_sequences"])),
            "metrics": self.rel(context, _save("metrics.json", out["metrics"])),
        }
        context["manifest"].add_cluster_result(
            tag, os.path.join(self.log_subdir(), tag).replace(os.sep, "/"),
            artifacts, extra=extra)
        print(f"[state_merge] result '{tag}' -> {out_dir} "
              f"(segments={out['metrics'].get('n_segments')} "
              f"blocks={out['metrics'].get('n_blocks')})")

    # ── main ─────────────────────────────────────────────────────

    def _merge_one(self, context: dict, tag: str):
        labels, indices, seq_len, feats, kept_rows = self._load_tag_artifacts(context, tag)
        n = len(labels)
        csv_idx, start = indices[:, 0], indices[:, 1]
        feat_norm = self._zscore(feats)
        min_len = self._min_block_len_samples()
        print(f"[state_merge] tag={tag}  samples={n}  min_block_len={min_len} samples "
              f"({self.min_block_seconds:g}s @ {self.fs:g}Hz)  similar_tol={self.similar_feature_tol}")

        per_file = defaultdict(list)
        for r in range(n):
            per_file[int(csv_idx[r])].append(r)

        blocks_all = []
        seg_to_block = np.full(n, -1, dtype=np.int64)
        seg_labels = np.array(labels, dtype=np.int64)
        blocks_by_fid = {}

        for fid in sorted(per_file):
            rows = sorted(per_file[fid], key=lambda r: int(start[r]))
            blocks = merge_activity(rows, labels, start, seq_len, feat_norm, feats,
                                    min_len, self.enable_similar_merge,
                                    self.similar_feature_tol)
            blocks_by_fid[fid] = blocks
            for b in blocks:
                bid = len(blocks_all)
                blocks_all.append(b)
                for r in b["rows"]:
                    seg_to_block[r] = bid
                    seg_labels[r] = b["label"]

        n_blocks = len(blocks_all)
        block_labels = np.array([b["label"] for b in blocks_all], dtype=np.int64)
        block_indices = np.array(
            [[csv_idx[b["rows"][0]], b["start"], b["label"]] for b in blocks_all],
            dtype=np.int64)
        block_seq_len = np.array([b["length"] for b in blocks_all], dtype=np.int64)
        block_features = np.array([block_raw_feat(b) for b in blocks_all], dtype=np.float64)
        seg_to_block2 = np.column_stack((np.arange(n, dtype=np.int64), seg_to_block))

        blocks_json = []
        for bid, b in enumerate(blocks_all):
            blocks_json.append({
                "block_id": bid,
                "csv_idx": int(csv_idx[b["rows"][0]]),
                "start": int(b["start"]),
                "end": int(b["end"]),
                "length_samples": int(b["length"]),
                "length_seconds": round(int(b["length"]) / self.fs, 3),
                "state_label": int(b["label"]),
                "n_segments": int(b["n_segments"]),
                "member_rows": [int(r) for r in b["rows"]],
                "absorbed_labels": [int(l) for l in b["absorbed_labels"]],
                "n_absorbed_segments": int(b["n_absorbed_segments"]),
                "similar_merged": bool(b["similar_merged"]),
                "feature": [round(float(x), 6) for x in block_raw_feat(b)],
            })

        # global block ids are assigned in fid order: derive each activity's ids
        fid_to_block_id = {}
        offset = 0
        for fid in sorted(blocks_by_fid):
            fid_to_block_id[fid] = list(range(offset, offset + len(blocks_by_fid[fid])))
            offset += len(blocks_by_fid[fid])
        state_sequences = {}
        for fid in sorted(blocks_by_fid):
            state_sequences[str(fid)] = [{
                "block_id": fid_to_block_id[fid][k],
                "start": int(blocks_by_fid[fid][k]["start"]),
                "end": int(blocks_by_fid[fid][k]["end"]),
                "length_samples": int(blocks_by_fid[fid][k]["length"]),
                "length_seconds": round(int(blocks_by_fid[fid][k]["length"]) / self.fs, 3),
                "state_label": int(blocks_by_fid[fid][k]["label"]),
                "n_segments": int(blocks_by_fid[fid][k]["n_segments"]),
            } for k in range(len(blocks_by_fid[fid]))]

        # per-state summary (the "new merged clusters")
        states = {}
        for b in blocks_all:
            s = states.setdefault(int(b["label"]), {
                "n_blocks": 0, "n_segments": 0, "length_samples": 0,
                "absorbed_labels": set()})
            s["n_blocks"] += 1
            s["n_segments"] += int(b["n_segments"])
            s["length_samples"] += int(b["length"])
            s["absorbed_labels"] |= {int(l) for l in b["absorbed_labels"]}
        states = {
            str(k): {"n_blocks": v["n_blocks"], "n_segments": v["n_segments"],
                     "length_samples": v["length_samples"],
                     "length_seconds": round(v["length_samples"] / self.fs, 3),
                     "absorbed_labels": sorted(v["absorbed_labels"])}
            for k, v in sorted(states.items())}

        n_changed = int(np.sum(seg_labels != labels))
        metrics = {
            "cluster_tag": tag,
            "merge_mode": self.merge_mode,
            "min_block_seconds": self.min_block_seconds,
            "min_block_len_samples": min_len,
            "enable_similar_merge": bool(self.enable_similar_merge),
            "similar_feature_tol": self.similar_feature_tol,
            "n_segments": n,
            "n_blocks": n_blocks,
            "n_merged_segments": n_changed,
            "merged_segment_ratio": round(n_changed / n, 4) if n else 0.0,
            "n_short_absorbed_segments": int(sum(b["n_absorbed_segments"] for b in blocks_all)),
            "n_similar_merges": int(sum(1 for b in blocks_all if b["similar_merged"])),
            "n_activities": len(per_file),
            "states": states,
            "cluster_metrics": compute_cluster_metrics(feats, seg_labels),
        }

        out = {
            "seg_labels": seg_labels,
            "seg_indices": np.column_stack((csv_idx, start, seg_labels)).astype(np.int64),
            "seg_seq_len": seq_len,
            "seg_feats": feats,
            "kept_rows": kept_rows,
            "block_labels": block_labels,
            "block_indices": block_indices,
            "block_seq_len": block_seq_len,
            "block_features": block_features,
            "seg_to_block": seg_to_block2,
            "blocks_json": blocks_json,
            "state_sequences": state_sequences,
            "metrics": metrics,
        }
        merged_tag = f"{tag}_merged"
        self._save_result(context, merged_tag, out, extra={
            "n_clusters": int(len(states)),
            "n_blocks": int(n_blocks),
            "n_merged_segments": int(n_changed),
        })

    def run(self, context: dict) -> dict:
        tags = self._resolve_tags(context)
        if not tags:
            return context
        for tag in tags:
            self._merge_one(context, tag)
            gc.collect()
        context.setdefault("data", {})["state_merge_tags"] = [f"{t}_merged" for t in tags]
        return context
