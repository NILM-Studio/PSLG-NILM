"""PrimitiveActivityMapping step: map segments (primitives) onto activities.

Definitions (cleaned up from the legacy step):
- **activity**  = one CSV produced by extract_active_data (a contiguous run of
  the appliance), resolved via the manifest (``extract_active_data.segments_dir``).
- **primitive** = one segment produced by time_segmentation. Its parent activity
  is known EXACTLY (``indices[:, 0]`` = csv index), so matching is index-based —
  the legacy fuzzy/tolerant timestamp matching is gone.
- cluster labels come from ONE tagged clustering result (``--cluster-tag``),
  aligned back to segmentation rows via the result's ``kept_rows``.
- an activity is **few-shot** if it contains at least one primitive whose
  cluster is a *true few-shot cluster* per few_shot_cluster_extract's summary.

Outputs (same names as legacy, for downstream compatibility):
``activity_sequence_ranges.json``, ``primitive_sequence_ranges.json``,
``primitive_activity_mapping.json``, ``few_shot_activity_sequences.json``,
``non_few_shot_activity_sequences.json``, ``few_shot_activity_tensor.npy``,
``non_few_shot_activity_tensor.npy`` (+ ``*_seq_lens.npy``).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.framework.step import Step


class PrimitiveActivityMappingStep(Step):
    step_type = "primitive_activity_mapping"

    def __init__(self, cluster_tag: str = None):
        self.cluster_tag = cluster_tag
        super().__init__(variant=cluster_tag or "auto")

    # ── input resolution (manifest only) ─────────────────────────

    def _resolve_tag(self, context: dict) -> str:
        if self.cluster_tag:
            return self.cluster_tag
        tags = context["manifest"].cluster_tags()
        if len(tags) == 1:
            return tags[0]
        raise ValueError(
            f"[pam] --cluster-tag is required: {len(tags)} cluster results exist {tags}.")

    def _load_inputs(self, context: dict, tag: str):
        m = context["manifest"]

        seg_dir = self.resolve(context, "extract_active_data", "segments_dir")
        if not (seg_dir and os.path.isdir(seg_dir)):
            raise FileNotFoundError("[pam] extract_active_data.segments_dir not resolvable.")

        indices = self.resolve(context, "time_segmentation", "indices")
        lengths = self.resolve(context, "time_segmentation", "lengths")
        if not (indices and lengths and os.path.exists(indices) and os.path.exists(lengths)):
            raise FileNotFoundError("[pam] time_segmentation indices/lengths not resolvable.")
        indices = np.load(indices).astype(np.int64)
        lengths = np.load(lengths).reshape(-1).astype(np.int64)

        labels_path = m.cluster_artifact_path(tag, "labels")
        kept_path = m.cluster_artifact_path(tag, "kept_rows")
        if not (labels_path and os.path.exists(labels_path)):
            raise FileNotFoundError(f"[pam] cluster result '{tag}' labels not resolvable.")
        labels = np.load(labels_path).reshape(-1).astype(np.int64)
        kept_rows = (np.load(kept_path).reshape(-1).astype(np.int64)
                     if kept_path and os.path.exists(kept_path)
                     else np.arange(len(indices), dtype=np.int64))

        summary_path = self.resolve(context, "few_shot_cluster_extract", "summary")
        if not (summary_path and os.path.exists(summary_path)):
            raise FileNotFoundError(
                "[pam] few_shot_cluster_extract summary not resolvable — run --steps fewshot first.")
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
        if summary.get("cluster_tag") not in (None, tag):
            print(f"[pam] WARNING: few-shot summary is for tag '{summary.get('cluster_tag')}', "
                  f"but pam uses '{tag}'.")
        true_few_shot = {int(c) for c in summary.get("true_few_shot_clusters", [])}

        return seg_dir, indices, lengths, labels, kept_rows, true_few_shot

    # ── helpers ──────────────────────────────────────────────────

    @staticmethod
    def _timestamps_from_csv(file_path: str, start_idx: int, length: int):
        """Start/end timestamps of rows [start_idx, start_idx+length) of a CSV's column 0."""
        try:
            df = pd.read_csv(file_path, usecols=[0],
                             skiprows=range(1, start_idx + 1), nrows=length)
            if df.empty:
                return None, None
            s, e = df.iloc[0, 0], df.iloc[-1, 0]
            s = s.item() if hasattr(s, "item") else s
            e = e.item() if hasattr(e, "item") else e
            return float(s), float(e)
        except Exception as exc:
            print(f"[pam] timestamp read failed for {file_path}: {exc}")
            return None, None

    @staticmethod
    def _build_activity_tensor(records):
        """Stack activity CSVs into a padded (n, max_len, dim) float32 tensor."""
        feature_columns, samples, seq_lens, valid = None, [], [], []
        for rec in records:
            fp = rec.get("file_path")
            if not fp or not Path(fp).exists():
                continue
            try:
                df = pd.read_csv(fp)
            except Exception as exc:
                print(f"[pam] skip unreadable csv {fp}: {exc}")
                continue
            if df.empty:
                continue
            if feature_columns is None:
                feature_columns = [str(c) for c in df.columns]
            else:
                df = df.reindex(columns=feature_columns)
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            arr = df.to_numpy(dtype=np.float32, copy=False)
            if arr.ndim != 2 or arr.shape[0] == 0:
                continue
            samples.append(arr)
            seq_lens.append(int(arr.shape[0]))
            valid.append(rec)

        if not samples:
            return (np.zeros((0, 0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int32),
                    feature_columns or [], [])
        max_len = max(seq_lens)
        tensor = np.zeros((len(samples), max_len, samples[0].shape[1]), dtype=np.float32)
        for i, arr in enumerate(samples):
            tensor[i, : arr.shape[0]] = arr
        return tensor, np.asarray(seq_lens, dtype=np.int32), feature_columns, valid

    # ── main ─────────────────────────────────────────────────────

    def run(self, context: dict) -> dict:
        tag = self._resolve_tag(context)
        self.variant = tag
        log_dir = self.log_dir(context)
        seg_dir, indices, lengths, labels, kept_rows, true_few_shot = \
            self._load_inputs(context, tag)

        csv_files = sorted(f for f in os.listdir(seg_dir) if f.lower().endswith(".csv"))
        csv_paths = [os.path.join(seg_dir, f) for f in csv_files]

        # activities
        activity_records = []
        for name, path in zip(csv_files, csv_paths):
            try:
                df = pd.read_csv(path, usecols=[0])
            except Exception as exc:
                print(f"[pam] skip unreadable activity {name}: {exc}")
                continue
            if df.empty:
                continue
            s, e = df.iloc[0, 0], df.iloc[-1, 0]
            activity_records.append({
                "file_name": name, "file_path": path,
                "start_timestamp": s.item() if hasattr(s, "item") else s,
                "end_timestamp": e.item() if hasattr(e, "item") else e,
                "row_count": int(len(df)),
            })

        # primitives: segmentation rows; cluster label aligned via kept_rows
        label_by_seg_row = {int(r): int(l) for r, l in zip(kept_rows, labels)}
        primitive_records, match_records = [], []
        for i in range(len(indices)):
            csv_idx, start = int(indices[i, 0]), int(indices[i, 1])
            length = int(lengths[i]) if i < len(lengths) else 0
            cluster = label_by_seg_row.get(i)  # None if dropped by NaN filtering
            rec = {"primitive_index": i, "activity_csv_idx": csv_idx,
                   "start_index_in_csv": start, "sample_length": length,
                   "cluster_label": cluster}
            primitive_records.append(rec)

            if 0 <= csv_idx < len(csv_paths) and length > 0:
                a = activity_records[csv_idx]
                p_start, p_end = self._timestamps_from_csv(a["file_path"], start, length)
                match_records.append({
                    **rec,
                    "primitive_start_timestamp": p_start,
                    "primitive_end_timestamp": p_end,
                    "activity_file_name": a["file_name"],
                    "activity_file_path": a["file_path"],
                    "activity_start_timestamp": a["start_timestamp"],
                    "activity_end_timestamp": a["end_timestamp"],
                    "match_type": "index_match" if p_start is not None else "unmatched",
                })

        # few-shot split: activity contains >=1 primitive in a true few-shot cluster
        few_shot_files = {r["activity_file_name"] for r in match_records
                          if r.get("cluster_label") in true_few_shot}
        few_records = [a for a in activity_records if a["file_name"] in few_shot_files]
        non_few_records = [a for a in activity_records if a["file_name"] not in few_shot_files]

        few_tensor, few_lens, _, few_records = self._build_activity_tensor(few_records)
        non_tensor, non_lens, _, non_few_records = self._build_activity_tensor(non_few_records)

        def _dump(name, obj):
            p = os.path.join(log_dir, name)
            if name.endswith(".npy"):
                np.save(p, obj)
            else:
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(obj, f, ensure_ascii=False, indent=2)
            return p

        paths = {
            "activity_ranges": _dump("activity_sequence_ranges.json", activity_records),
            "primitive_ranges": _dump("primitive_sequence_ranges.json", primitive_records),
            "mapping": _dump("primitive_activity_mapping.json", match_records),
            "few_shot_activities": _dump("few_shot_activity_sequences.json", few_records),
            "non_few_shot_activities": _dump("non_few_shot_activity_sequences.json", non_few_records),
            "few_shot_tensor": _dump("few_shot_activity_tensor.npy", few_tensor),
            "non_few_shot_tensor": _dump("non_few_shot_activity_tensor.npy", non_tensor),
            "few_shot_seq_lens": _dump("few_shot_activity_seq_lens.npy", few_lens),
            "non_few_shot_seq_lens": _dump("non_few_shot_activity_seq_lens.npy", non_lens),
        }
        self.record(context,
                    artifacts={k: self.rel(context, p) for k, p in paths.items()},
                    extra={"cluster_tag": tag,
                           "n_activities": len(activity_records),
                           "n_primitives": len(primitive_records),
                           "few_shot_activities": len(few_records),
                           "non_few_shot_activities": len(non_few_records)})

        print(f"[pam] tag={tag}: activities={len(activity_records)} "
              f"(few-shot={len(few_records)}, non-few-shot={len(non_few_records)}), "
              f"primitives={len(primitive_records)} -> {log_dir}")
        return context
