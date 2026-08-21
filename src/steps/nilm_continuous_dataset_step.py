"""Build a strict temporal NILM benchmark with real OFF-state background."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.framework.step import Step


class NilmContinuousDatasetStep(Step):
    step_type = "nilm_continuous_dataset"

    def __init__(self, cluster_tag: str, aligned_series_path: str,
                 sample_period_seconds: int = 6, max_gap_seconds: int = 150,
                 active_threshold_watts: float = 10.0,
                 min_off_samples: int = 599, max_chunk_samples: int = 100_000,
                 off_to_real_sample_ratio: float = 1.0,
                 random_seed: int = 42,
                 require_train_only_structure: bool = True):
        if not cluster_tag:
            raise ValueError("nilm_continuous_dataset requires --cluster-tag")
        if not aligned_series_path:
            raise ValueError("nilm_continuous_dataset.aligned_series is required")
        if sample_period_seconds <= 0 or max_gap_seconds <= 0:
            raise ValueError("continuous NILM periods must be positive")
        if min_off_samples < 1 or max_chunk_samples < min_off_samples:
            raise ValueError("invalid continuous NILM chunk limits")
        if off_to_real_sample_ratio < 0:
            raise ValueError("off_to_real_sample_ratio must be non-negative")
        super().__init__(variant=f"strict_temporal_on_{cluster_tag}")
        self.cluster_tag = cluster_tag
        self.aligned_series_path = str(aligned_series_path)
        self.sample_period_seconds = int(sample_period_seconds)
        self.max_gap_seconds = int(max_gap_seconds)
        self.active_threshold_watts = float(active_threshold_watts)
        self.min_off_samples = int(min_off_samples)
        self.max_chunk_samples = int(max_chunk_samples)
        self.off_to_real_sample_ratio = float(off_to_real_sample_ratio)
        self.random_seed = int(random_seed)
        self.require_train_only_structure = bool(require_train_only_structure)

    @staticmethod
    def _relative(path: str | Path, root: str | Path) -> str:
        return os.path.relpath(path, root).replace(os.sep, "/")

    def _uniform_chunks(self, timestamp: np.ndarray, mains: np.ndarray,
                        appliance: np.ndarray, start: int | None,
                        end: int | None) -> list[dict]:
        left = 0 if start is None else int(np.searchsorted(timestamp, start, side="left"))
        right = (len(timestamp) if end is None else
                 int(np.searchsorted(timestamp, end, side="right")))
        t = timestamp[left:right]
        m = mains[left:right]
        a = appliance[left:right]
        if len(t) < 2:
            return []
        boundaries = np.flatnonzero(np.diff(t) > self.max_gap_seconds) + 1
        runs = np.split(np.arange(len(t)), boundaries)
        chunks = []
        for indices in runs:
            if len(indices) < 2:
                continue
            run_t, run_m, run_a = t[indices], m[indices], a[indices]
            grid_start = int(np.ceil(run_t[0] / self.sample_period_seconds)
                             * self.sample_period_seconds)
            grid_end = int(np.floor(run_t[-1] / self.sample_period_seconds)
                           * self.sample_period_seconds)
            if grid_end <= grid_start:
                continue
            grid = np.arange(
                grid_start, grid_end + 1, self.sample_period_seconds,
                dtype=np.int64)
            uniform_mains = np.interp(grid, run_t, run_m).astype(np.float32)
            uniform_appliance = np.interp(grid, run_t, run_a).astype(np.float32)
            for offset in range(0, len(grid), self.max_chunk_samples):
                stop = min(len(grid), offset + self.max_chunk_samples)
                if stop - offset < 2:
                    continue
                chunks.append({
                    "timestamp": grid[offset:stop],
                    "mains": uniform_mains[offset:stop],
                    "appliance": uniform_appliance[offset:stop],
                })
        return chunks

    def _off_chunks(self, chunks: list[dict]) -> list[dict]:
        result = []
        for chunk in chunks:
            inactive = chunk["appliance"] <= self.active_threshold_watts
            changes = np.flatnonzero(np.diff(inactive.astype(np.int8)) != 0) + 1
            for indices in np.split(np.arange(len(inactive)), changes):
                if len(indices) < self.min_off_samples or not inactive[indices[0]]:
                    continue
                for offset in range(0, len(indices), self.max_chunk_samples):
                    selected = indices[offset:offset + self.max_chunk_samples]
                    if len(selected) < self.min_off_samples:
                        continue
                    result.append({key: value[selected] for key, value in chunk.items()})
        return result

    @staticmethod
    def _write_chunks(directory: str, prefix: str,
                      chunks: list[dict]) -> list[dict]:
        os.makedirs(directory, exist_ok=True)
        records = []
        for index, payload in enumerate(chunks):
            path = os.path.join(directory, f"{prefix}_{index:05d}.npz")
            np.savez_compressed(path, **payload)
            records.append({
                "path": path,
                "length_samples": int(len(payload["timestamp"])),
                "start_timestamp": int(payload["timestamp"][0]),
                "end_timestamp": int(payload["timestamp"][-1]),
            })
        return records

    @staticmethod
    def _sample_count(root: Path, files: list[str]) -> int:
        count = 0
        for relative in files:
            with np.load(root / relative) as payload:
                count += int(len(payload["appliance"]))
        return count

    def _select_off(self, records: list[dict], target_samples: int,
                    seed_suffix: int) -> list[dict]:
        if target_samples <= 0 or not records:
            return []
        order = np.arange(len(records))
        rng = np.random.default_rng(np.random.SeedSequence([
            self.random_seed, int(seed_suffix), 49979687,
        ]))
        rng.shuffle(order)
        selected, total = [], 0
        for index in order:
            selected.append(records[int(index)])
            total += int(records[int(index)]["length_samples"])
            if total >= target_samples:
                break
        return selected

    @staticmethod
    def _repeat_off(records: list[dict], target_samples: int) -> list[dict]:
        """Repeat one fixed OFF pool without introducing new background data."""
        if target_samples <= 0 or not records:
            return []
        selected, total, index = [], 0, 0
        while total < target_samples:
            record = records[index % len(records)]
            selected.append(record)
            total += int(record["length_samples"])
            index += 1
        return selected

    def run(self, context: dict) -> dict:
        aligned_path = Path(self.aligned_series_path)
        if not aligned_path.exists():
            raise FileNotFoundError(
                f"[nilm_continuous_dataset] aligned series not found: {aligned_path}")
        holdout_path = self.resolve(context, "temporal_holdout", "assignments")
        holdout_summary_path = self.resolve(context, "temporal_holdout", "summary")
        cycle_manifest_path = self.resolve(context, "nilm_dataset", "dataset_manifest")
        if not all(path and os.path.exists(path) for path in (
                holdout_path, holdout_summary_path, cycle_manifest_path)):
            raise FileNotFoundError(
                "[nilm_continuous_dataset] strict holdout and cycle dataset are required")
        split_entry = context["manifest"].get_step("cycle_split") or {}
        if (self.require_train_only_structure
                and (split_entry.get("extra") or {}).get(
                    "structure_fit_scope") != "train_only"):
            raise ValueError(
                "[nilm_continuous_dataset] structure must be fit on train only")

        with open(holdout_path, newline="", encoding="utf-8") as f:
            assignments = list(csv.DictReader(f))
        validation_starts = [int(row["start_timestamp"]) for row in assignments
                             if row["split"] == "validation"]
        test_starts = [int(row["start_timestamp"]) for row in assignments
                       if row["split"] == "test"]
        if not validation_starts or not test_starts:
            raise ValueError(
                "[nilm_continuous_dataset] validation and test periods must be non-empty")
        validation_start = min(validation_starts)
        test_start = min(test_starts)
        if validation_start >= test_start:
            raise ValueError("[nilm_continuous_dataset] temporal boundaries overlap")

        pair = pd.read_csv(
            aligned_path, usecols=["timestamp", "mains", "appliance"],
            dtype={"timestamp": np.int64, "mains": np.float32,
                   "appliance": np.float32})
        timestamp = pair["timestamp"].to_numpy(copy=False)
        mains = pair["mains"].to_numpy(copy=False)
        appliance = pair["appliance"].to_numpy(copy=False)
        if np.any(np.diff(timestamp) <= 0):
            raise ValueError(
                "[nilm_continuous_dataset] aligned timestamps must increase")

        train_chunks = self._uniform_chunks(
            timestamp, mains, appliance, None, validation_start - 1)
        validation_chunks = self._uniform_chunks(
            timestamp, mains, appliance, validation_start, test_start - 1)
        test_chunks = self._uniform_chunks(
            timestamp, mains, appliance, test_start, None)
        off_chunks = self._off_chunks(train_chunks)
        if not validation_chunks or not test_chunks or not off_chunks:
            raise ValueError(
                "[nilm_continuous_dataset] a temporal split produced no usable chunks")

        log_dir = self.log_dir(context)
        validation_records = self._write_chunks(
            os.path.join(log_dir, "continuous", "validation"),
            "validation", validation_chunks)
        test_records = self._write_chunks(
            os.path.join(log_dir, "continuous", "test"), "test", test_chunks)
        off_records = self._write_chunks(
            os.path.join(log_dir, "continuous", "train_off"), "off", off_chunks)

        cycle_root = Path(cycle_manifest_path).parent
        with open(cycle_manifest_path, encoding="utf-8") as f:
            cycle_manifest = json.load(f)

        def imported(files):
            return [self._relative(cycle_root / path, log_dir) for path in files]

        experiments = {}
        ratio_entries = [
            (tag, entry) for tag, entry in cycle_manifest["experiments"].items()
            if tag != "full"
        ]
        ratio_entries.sort(key=lambda item: float(item[1]["real_ratio"]))
        for tag, entry in ratio_entries:
            real_files = entry["A_real_only"]
            real_active_samples = self._sample_count(cycle_root, real_files)
            selected_off = self._select_off(
                off_records,
                int(round(real_active_samples * self.off_to_real_sample_ratio)),
                int(round(float(entry["real_ratio"]) * 10_000)))
            augmented_active_samples = max(
                self._sample_count(
                    cycle_root, entry["B_real_plus_traditional"]),
                self._sample_count(
                    cycle_root, entry["C_real_plus_generated"]),
            )
            repeated_off = self._repeat_off(
                selected_off,
                int(round(augmented_active_samples
                          * self.off_to_real_sample_ratio)))
            off_files = [self._relative(row["path"], log_dir)
                         for row in selected_off]
            augmented_off_files = [self._relative(row["path"], log_dir)
                                   for row in repeated_off]
            experiments[tag] = {
                "real_ratio": float(entry["real_ratio"]),
                "A_real_only": imported(entry["A_real_only"]) + off_files,
                "B_real_plus_traditional": imported(
                    entry["B_real_plus_traditional"]) + augmented_off_files,
                "C_real_plus_generated": imported(
                    entry["C_real_plus_generated"]) + augmented_off_files,
                "selected_real_count": int(entry["selected_real_count"]),
                "selected_traditional_count": int(
                    entry["selected_traditional_count"]),
                "selected_generated_count": int(entry["selected_generated_count"]),
                "selected_off_chunks": len(off_files),
                "selected_off_samples": int(sum(
                    row["length_samples"] for row in selected_off)),
                "augmented_off_file_references": len(augmented_off_files),
                "augmented_off_samples": int(sum(
                    row["length_samples"] for row in repeated_off)),
                "off_background_policy": "same_unique_pool_repeated_for_B_and_C",
            }

        full_real = cycle_manifest["experiments"]["full"]["D_full_real"]
        full_active_samples = self._sample_count(cycle_root, full_real)
        full_off = self._select_off(
            off_records,
            int(round(full_active_samples * self.off_to_real_sample_ratio)), 100_000)
        experiments["full"] = {
            "D_full_real": imported(full_real) + [
                self._relative(row["path"], log_dir) for row in full_off],
            "validation": [self._relative(row["path"], log_dir)
                           for row in validation_records],
            "test": [self._relative(row["path"], log_dir) for row in test_records],
        }

        manifest = {
            "benchmark_type": "strict_temporal_continuous_test",
            "aligned_series": str(aligned_path),
            "cluster_tag": self.cluster_tag,
            "structure_fit_scope": "train_only",
            "temporal_boundaries": {
                "validation_start": validation_start,
                "test_start": test_start,
            },
            "sample_period_seconds": self.sample_period_seconds,
            "max_gap_seconds": self.max_gap_seconds,
            "active_threshold_watts": self.active_threshold_watts,
            "off_to_real_sample_ratio": self.off_to_real_sample_ratio,
            "continuous_chunks": {
                "validation": len(validation_records),
                "test": len(test_records),
                "train_off_pool": len(off_records),
            },
            "continuous_samples": {
                "validation": int(sum(row["length_samples"]
                                      for row in validation_records)),
                "test": int(sum(row["length_samples"] for row in test_records)),
                "train_off_pool": int(sum(row["length_samples"]
                                          for row in off_records)),
            },
            "experiments": experiments,
        }
        manifest_path = os.path.join(log_dir, "nilm_dataset_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        self.record(context, artifacts={
            "dataset_manifest": self.rel(context, manifest_path),
            "continuous_dir": self.rel(context, os.path.join(log_dir, "continuous")),
        }, extra={
            "cluster_tag": self.cluster_tag,
            "benchmark_type": manifest["benchmark_type"],
            "structure_fit_scope": "train_only",
            "continuous_samples": manifest["continuous_samples"],
        })
        print(f"[nilm_continuous_dataset] validation/test samples="
              f"{manifest['continuous_samples']['validation']}/"
              f"{manifest['continuous_samples']['test']} -> {log_dir}")
        return context
