"""TimeSegmentation step: segment each active-data CSV into sub-event regimes.

Each input activity is cut once on raw power and encoded as a padded
4-channel tensor (raw, raw, raw, zero), matching the downstream shape.
Changes:
- adapts to the new ``Step`` base + run manifest (registers X/lengths/indices);
- resolves its input directory from ``context['input_root']`` or, when run in
  isolation, from the manifest's ``extract_active_data.segments_dir`` — no more
  guessing ``log_root/DataLoader``;
- drops the dead matplotlib import and the unused ``dataset_split_step`` import;
- drops the per-N checkpoint .npy files (redundant intermediate artifacts).

Comparison backends are imported lazily; enhanced ClaSP preprocessing is removed.
"""
from __future__ import annotations

import gc
import os

import numpy as np
import pandas as pd

from src.framework.step import Step


class TimeSegmentationStep(Step):
    step_type = "time_segmentation"

    def __init__(self, segment_method: str = "clasp", appliance_name: str = "",
                 max_seg_len: int = 0):
        super().__init__(variant=segment_method)
        self.segment_method = segment_method
        self.appliance_name = appliance_name
        self.max_seg_len = int(max_seg_len or 0)

    # ── input resolution ─────────────────────────────────────────

    def _resolve_input_dir(self, context: dict):
        candidates = []
        if context.get("input_root"):
            candidates.append(context["input_root"])
        seg_dir = self.resolve(context, "extract_active_data", "segments_dir")
        if seg_dir:
            candidates.append(seg_dir)
        candidates.append(os.path.join(context["log_root"], "DataLoader"))  # legacy fallback
        for c in candidates:
            if c and os.path.isdir(c):
                return c
        return None

    # ── segmentation ─────────────────────────────────────────────

    def get_segmentation_points(self, time_series, distance=None):
        from models.time_segmentation.comparison import run

        cuts, _ = run(np.asarray(time_series).flatten(), self.segment_method)
        return cuts

    # ── main ─────────────────────────────────────────────────────

    def run(self, context: dict) -> dict:
        # Sliding context release: free ExtractActiveData's payload once we have read it.
        if "data" in context and "extract_active_data" in context["data"]:
            del context["data"]["extract_active_data"]
            gc.collect()

        input_dir = self._resolve_input_dir(context)
        log_dir = self.log_dir(context)
        if not input_dir:
            print(f"[time_segmentation] no input dir found for segments; skipping.")
            return context

        target_files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(".csv"))
        print(f"[time_segmentation] {len(target_files)} CSV files, method={self.segment_method}")

        all_samples, all_lengths, all_indices = [], [], []

        for i, file_name in enumerate(target_files):
            print(f"  [{i + 1}/{len(target_files)}] {file_name}", flush=True)
            df = pd.read_csv(os.path.join(input_dir, file_name))
            if "power" not in df.columns:
                print(f"  skip {file_name}: no 'power' column")
                continue
            signal = df["power"].values

            signal_cleaned = signal
            low_freq, high_freq = signal, np.zeros_like(signal)
            synth_cp = self.get_segmentation_points(signal)

            cps = sorted({int(round(cp)) for cp in synth_cp})
            boundaries = [0] + cps + [len(signal_cleaned)]
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                if end <= start:
                    continue
                sample = np.stack([
                    signal[start:end], signal_cleaned[start:end],
                    low_freq[start:end], high_freq[start:end]], axis=1)
                all_samples.append(sample)
                all_lengths.append(len(sample))
                all_indices.append([i, start])

            del df, signal, signal_cleaned
            gc.collect()

        if not all_samples:
            print("[time_segmentation] no segments produced.")
            return context

        # Optional hard cap on primitive length (max_seg_len>0). Guard against
        # pathological longest-primitive tensors: very long BiGRU sequences can
        # hang TF/CuDNN training (observed at 1765 steps on 4090 while <=1536
        # trains). Truncating the tail keeps indices/coverage intact; only the
        # longest few primitives are affected.
        if self.max_seg_len > 0:
            for idx, sample in enumerate(all_samples):
                if len(sample) > self.max_seg_len:
                    all_samples[idx] = sample[: self.max_seg_len]
                    all_lengths[idx] = self.max_seg_len

        max_len = max(all_lengths)
        n_samples = len(all_samples)
        X = np.zeros((n_samples, max_len, 4), dtype=np.float32)
        for idx, sample in enumerate(all_samples):
            X[idx, : all_lengths[idx], :] = sample
        lengths = np.array(all_lengths, dtype=np.int32).reshape(-1, 1)
        indices = np.array(all_indices, dtype=np.int32)

        x_path = os.path.join(log_dir, "X.npy")
        l_path = os.path.join(log_dir, "lengths.npy")
        i_path = os.path.join(log_dir, "indices.npy")
        np.save(x_path, X)
        np.save(l_path, lengths)
        np.save(i_path, indices)

        context.setdefault("data", {}).update({"X": X, "lengths": lengths, "indices": indices})
        context["segment_method"] = self.segment_method

        self.record(context, artifacts={
            "X": self.rel(context, x_path),
            "lengths": self.rel(context, l_path),
            "indices": self.rel(context, i_path),
        }, extra={"n_samples": n_samples, "max_len": int(max_len)})

        print(f"[time_segmentation] tensor: {n_samples} samples, max_len={max_len} -> {log_dir}")

        del all_samples, all_lengths, all_indices
        gc.collect()
        return context
