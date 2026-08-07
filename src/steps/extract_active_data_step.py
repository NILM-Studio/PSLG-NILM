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
                 input_file: str = "", **method_kwargs):
        super().__init__(variant=method.lower())
        self.method = method.lower()
        self.appliance_name = appliance_name
        self.input_file = input_file
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
