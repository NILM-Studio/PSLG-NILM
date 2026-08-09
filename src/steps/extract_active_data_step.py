"""ExtractActiveData step: cut working intervals out of a raw power series.

Behavior is preserved from the legacy project (simple threshold vs. adaptive
clustering detector). Changes:
- adapts to the new ``Step`` base + run manifest (registers ``segments_dir``);
- drops the per-N checkpoint JSON files (redundant intermediate artifacts); the
  segments directory itself is the durable, resumable output.
"""
from __future__ import annotations

import gc
import os
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd

from src.framework.step import Step


class ExtractActiveDataStep(Step):
    step_type = "extract_active_data"

    def __init__(self, method: str = "simple", appliance_name: str = "",
                 input_file: str = "", resample_fs: float = 0, **method_kwargs):
        super().__init__(variant=method.lower())
        self.method = method.lower()
        self.appliance_name = appliance_name
        self.input_file = input_file
        self.resample_fs = float(resample_fs or 0)
        self.method_kwargs = method_kwargs

    # ── helpers ──────────────────────────────────────────────────

    def _get_detector(self, context: dict):
        from models.extract_active_data.simple_threshold import SimpleThresholdDetector
        from models.extract_active_data.adaptive_clustering import AdaptiveClusteringDetector

        app_name = self.appliance_name or context.get("appliance_name", "appliance")
        config = {"appliance_name": app_name, **self.method_kwargs}
        if self.method == "adaptive":
            return AdaptiveClusteringDetector("AdaptiveDetector", config)
        return SimpleThresholdDetector("SimpleDetector", config)

    def _resample_to_fs(self, timestamps: np.ndarray, powers: np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray]:
        """Resample (timestamp, power) onto a uniform grid at ``self.resample_fs``.

        Datasets have heterogeneous native sampling rates (ECO/GREEND/IAWE=1s,
        REDD=4s, REFIT=7s, UK-DALE=6s). To make activity-state extraction
        comparable, the series is first resampled to the UK-DALE 6s grid
        (fs=0.1666667). ``origin="epoch"`` anchors the grid to epoch multiples of
        the bin width so every dataset lands on the same aligned timeline.

        Two kinds of "empty" bins are treated differently:
          - short gaps (<= t_drop, e.g. native 7s -> 6s bins): linearly
            interpolated so the resampled series stays gapless within a work
            cycle;
          - long data-absence gaps (> t_drop, real holes in the recording): left
            as NaN. The detector compares ``power >= threshold``, which is False
            for NaN, so long holes keep separating independent work cycles just
            as they did on the native timeline.
        """
        import pandas as pd

        if len(timestamps) < 2:
            return timestamps, powers
        dt = 1.0 / self.resample_fs
        rule = f"{int(round(dt))}s"
        t_drop = float(self.method_kwargs.get("t_drop", 0) or 0)
        df = pd.DataFrame({"timestamp": timestamps, "power": powers})
        df.index = pd.to_datetime(df["timestamp"], unit="s")
        out = df["power"].resample(rule, origin="epoch").mean()
        out = out.astype(np.float64)
        if out.isna().any():
            keep_nan = (self._null_long_gaps(out, t_drop)
                        if t_drop > 0 else None)
            # interpolate only the short holes; long holes stay NaN so work
            # cycles are not bridged across recording gaps. Edge NaN (series
            # starting/ending mid-recording) is left as-is: NaN is inactive.
            out = out.interpolate(method="time", limit_area="inside")
            if keep_nan is not None:
                out[np.asarray(keep_nan)] = np.nan
        new_t = np.asarray(out.index.view(np.int64) // 10**9)
        new_p = np.asarray(out.to_numpy(dtype=np.float64))
        n_nan = int(np.isnan(new_p).sum())
        print(f"[{self.step_type}] resampled to {self.resample_fs} Hz ({rule}) "
              f"-> {len(new_t):,} samples ({n_nan:,} NaN holes)")
        return new_t, new_p

    @staticmethod
    def _null_long_gaps(out: pd.Series, t_drop: float) -> np.ndarray:
        """Mark resampled bins inside a recording hole > t_drop as keep-NaN.

        On the native timeline, ``t_drop`` separates independent work cycles, so
        any data-absence hole wider than ``t_drop`` must NOT be interpolated —
        otherwise independent cycles get bridged. Returns a bool array (same
        length as ``out``) where True means the bin is inside such a long hole
        and must remain NaN. Edge holes (recording starts/ends mid-series) are
        also kept NaN.
        """
        secs = np.asarray(out.index.view(np.int64) // 10**9)
        vals = out.to_numpy(dtype=np.float64)
        n = len(vals)
        nan = np.isnan(vals)
        keep = np.zeros(n, dtype=bool)
        i = 0
        while i < n:
            if not nan[i]:
                i += 1
                continue
            j = i
            while j < n and nan[j]:
                j += 1
            # NaN run is [i:j)
            if i == 0 or j == n:
                keep[i:j] = True      # edge hole: no neighbours to bridge
            elif secs[j] - secs[i - 1] > t_drop:
                keep[i:j] = True      # long recording hole: keep as NaN
            i = j
        return keep

    def _read_data(self, input_file: str) -> Tuple[np.ndarray, np.ndarray]:
        ext = os.path.splitext(input_file)[1].lower()
        if ext == ".csv":
            df = pd.read_csv(input_file)
            if "datetime" in df.columns and "power" in df.columns:
                return df["datetime"].values, df["power"].values
            if "timestamp" in df.columns and "power" in df.columns:
                return df["timestamp"].values, df["power"].values
            return df.iloc[:, 0].values, df.iloc[:, 1].values  # fallback: first two cols
        if ext == ".npy":
            data = np.load(input_file)
            return data[:, 0], data[:, 1]
        data = np.loadtxt(input_file)
        return data[:, 0], data[:, 1]

    # ── main ─────────────────────────────────────────────────────

    def run(self, context: dict) -> dict:
        if not self.input_file:
            print(f"[{self.step_type}] no input_file set; skipping.")
            return context

        log_dir = self.log_dir(context)
        segments_dir = os.path.join(log_dir, "segments")
        os.makedirs(segments_dir, exist_ok=True)

        # 1. load
        print(f"[{self.step_type}] loading {self.input_file}")
        timestamps, powers = self._read_data(self.input_file)

        # 1b. align sampling frequency to the target grid (UK-DALE 6s) BEFORE
        #     activity-state detection — datasets natively sample at 1/4/6/7s.
        if self.resample_fs > 0:
            timestamps, powers = self._resample_to_fs(timestamps, powers)
            self.method_kwargs["fs"] = self.resample_fs

        # 2. detect
        detector = self._get_detector(context)
        detector.train(powers, timestamps)
        print(f"[{self.step_type}] detecting active intervals (method={self.method})")
        work_intervals = detector.detect(powers, timestamps)

        # 3. export one CSV per interval
        app_name = self.appliance_name or context.get("appliance_name", "appliance")
        output_files = []
        for interval in work_intervals:
            start_dt = datetime.fromtimestamp(interval["start_time"])
            end_dt = datetime.fromtimestamp(interval["end_time"])
            fname = (f"{app_name}_{start_dt.strftime('%Y%m%d_%H%M%S')}_"
                     f"{end_dt.strftime('%Y%m%d_%H%M%S')}_{int(interval['duration_sec'])}s.csv")
            df = interval["data"]
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
            df.to_csv(os.path.join(segments_dir, fname), index=False)
            output_files.append(fname)
            if len(output_files) % 50 == 0:
                print(f"  saved {len(output_files)}/{len(work_intervals)} intervals")

        print(f"[{self.step_type}] extracted {len(output_files)} active intervals -> {segments_dir}")

        # 4. context handoff (in-memory, for steps running in the same invocation)
        context.setdefault("data", {})["extract_active_data"] = {
            "segments_dir": segments_dir,
            "segment_files": output_files,
            "method": self.method,
        }
        if output_files:
            context["input_root"] = segments_dir

        # 5. manifest (path relative to log_root) — source of truth for later runs
        self.record(context, artifacts={"segments_dir": self.rel(context, segments_dir),
                                        "count": str(len(output_files))})

        del timestamps, powers, work_intervals, detector
        gc.collect()
        return context
