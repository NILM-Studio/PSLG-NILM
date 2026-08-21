"""Build leakage-controlled real and synthetic NILM cycle datasets."""
from __future__ import annotations

import csv
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.framework.step import Step


class NilmDatasetStep(Step):
    step_type = "nilm_dataset"

    def __init__(self, cluster_tag: str, aligned_series_path: str,
                 real_ratios=(0.05, 0.10, 0.20), sample_period_seconds: int = 6,
                 max_gap_seconds: int = 150, random_seed: int = 42,
                 expected_conditioning_neighbors: int = 10,
                 traditional_scale_range=(0.9, 1.1),
                 traditional_noise_ratio: float = 0.01,
                 active_threshold_watts: float = 10.0,
                 require_train_only_structure: bool = False):
        if not cluster_tag:
            raise ValueError("nilm_dataset requires --cluster-tag")
        if not aligned_series_path:
            raise ValueError("nilm_dataset.aligned_series is required")
        ratios = [float(value) for value in real_ratios]
        if not ratios or any(value <= 0 or value > 1 for value in ratios):
            raise ValueError("nilm_dataset.real_ratios must be in (0, 1]")
        scope = "strict_" if require_train_only_structure else ""
        super().__init__(variant=f"{scope}cycle_augmentation_on_{cluster_tag}")
        self.cluster_tag = cluster_tag
        self.aligned_series_path = aligned_series_path
        self.real_ratios = ratios
        self.sample_period_seconds = int(sample_period_seconds)
        self.max_gap_seconds = int(max_gap_seconds)
        self.random_seed = int(random_seed)
        self.expected_conditioning_neighbors = int(expected_conditioning_neighbors)
        self.traditional_scale_range = tuple(float(value)
                                             for value in traditional_scale_range)
        if (len(self.traditional_scale_range) != 2
                or self.traditional_scale_range[0] <= 0
                or self.traditional_scale_range[1] < self.traditional_scale_range[0]):
            raise ValueError("traditional_scale_range must contain positive [low, high]")
        self.traditional_noise_ratio = max(0.0, float(traditional_noise_ratio))
        self.active_threshold_watts = max(0.0, float(active_threshold_watts))
        self.require_train_only_structure = bool(require_train_only_structure)

    @staticmethod
    def _load_assignments(path: str) -> list[dict]:
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    @staticmethod
    def _stratified_select(records: list[dict], count: int,
                           rng: np.random.Generator) -> list[dict]:
        if count >= len(records):
            return list(records)
        groups = {}
        for record in records:
            key = (int(record["class_id"]), int(record["mode_id"]))
            groups.setdefault(key, []).append(record)
        for values in groups.values():
            rng.shuffle(values)
        selected = []
        keys = sorted(groups)
        while len(selected) < count:
            progressed = False
            for key in keys:
                if groups[key] and len(selected) < count:
                    selected.append(groups[key].pop())
                    progressed = True
            if not progressed:
                break
        return selected

    @staticmethod
    def _resample_interval(timestamp: np.ndarray, mains: np.ndarray,
                           appliance: np.ndarray, start: int, end: int,
                           period: int, max_gap: int):
        left = int(np.searchsorted(timestamp, start, side="left"))
        right = int(np.searchsorted(timestamp, end, side="right"))
        t = timestamp[left:right]
        if len(t) < 2:
            return None, "insufficient_aligned_points"
        largest_gap = int(np.max(np.diff(t)))
        if largest_gap > max_gap:
            return None, f"gap_exceeds_{max_gap}s"
        grid_start = int(np.ceil(start / period) * period)
        grid_end = int(np.floor(end / period) * period)
        if grid_end <= grid_start:
            return None, "interval_too_short"
        grid = np.arange(grid_start, grid_end + 1, period, dtype=np.int64)
        return {
            "timestamp": grid,
            "mains": np.interp(grid, t, mains[left:right]).astype(np.float32),
            "appliance": np.interp(
                grid, t, appliance[left:right]).astype(np.float32),
            "largest_source_gap_seconds": largest_gap,
        }, None

    @staticmethod
    def _write_npz(directory: str, name: str, payload: dict) -> str:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{name}.npz")
        np.savez_compressed(path, **payload)
        return path

    @staticmethod
    def _traditional_augment(mains: np.ndarray, appliance: np.ndarray,
                             rng: np.random.Generator, scale_range,
                             noise_ratio: float, active_threshold: float):
        """Apply magnitude scaling and active-state jitter to an NILM pair."""
        low, high = (float(scale_range[0]), float(scale_range[1]))
        scale = float(rng.uniform(low, high))
        active = appliance > active_threshold
        augmented = appliance.astype(np.float32, copy=True) * scale
        active_values = appliance[active]
        sigma = (float(np.std(active_values)) * float(noise_ratio)
                 if len(active_values) else 0.0)
        if sigma > 0:
            augmented[active] += rng.normal(0.0, sigma, int(np.sum(active))).astype(
                np.float32)
        augmented = np.maximum(augmented, 0.0).astype(np.float32)
        background = np.maximum(mains - appliance, 0.0).astype(np.float32)
        return (background + augmented).astype(np.float32), augmented, {
            "scale": scale, "active_noise_sigma": sigma,
        }

    @staticmethod
    def _relative(path: str, root: str) -> str:
        return os.path.relpath(path, root).replace(os.sep, "/")

    def _validate_synthesis(self, context: dict) -> tuple[str, str]:
        entry = context["manifest"].get_step("primitive_synthesis") or {}
        extra = entry.get("extra") or {}
        if extra.get("conditioning_method") != "cycle_neighbors":
            raise ValueError("[nilm_dataset] synthesis must use cycle_neighbors")
        if int(extra.get("conditioning_neighbors", -1)) != self.expected_conditioning_neighbors:
            raise ValueError(
                "[nilm_dataset] synthesis neighbor count does not match selected method")
        if extra.get("source_split") != "train":
            raise ValueError("[nilm_dataset] synthesis must use train split only")
        cycles = self.resolve(context, "primitive_synthesis", "cycles_dir")
        manifest = self.resolve(context, "primitive_synthesis", "synthesis_manifest")
        if not (cycles and os.path.isdir(cycles) and manifest and os.path.exists(manifest)):
            raise FileNotFoundError("[nilm_dataset] selected synthesis artifacts not found")
        return cycles, manifest

    def run(self, context: dict) -> dict:
        aligned_path = Path(self.aligned_series_path)
        if not aligned_path.exists():
            raise FileNotFoundError(f"[nilm_dataset] aligned series not found: {aligned_path}")
        assignments_path = self.resolve(context, "cycle_split", "assignments")
        segments_dir = self.resolve(context, "extract_active_data", "segments_dir")
        if not (assignments_path and os.path.exists(assignments_path)):
            raise FileNotFoundError("[nilm_dataset] cycle split assignments not found")
        split_entry = context["manifest"].get_step("cycle_split") or {}
        if (self.require_train_only_structure
                and (split_entry.get("extra") or {}).get(
                    "structure_fit_scope") != "train_only"):
            raise ValueError("[nilm_dataset] train-only structure fit is required")
        if not (segments_dir and os.path.isdir(segments_dir)):
            raise FileNotFoundError("[nilm_dataset] extracted activity directory not found")
        synthetic_dir, synthetic_manifest_path = self._validate_synthesis(context)

        pair = pd.read_csv(
            aligned_path, usecols=["timestamp", "mains", "appliance"],
            dtype={"timestamp": np.int64, "mains": np.float32,
                   "appliance": np.float32})
        timestamp = pair["timestamp"].to_numpy(copy=False)
        mains = pair["mains"].to_numpy(copy=False)
        appliance = pair["appliance"].to_numpy(copy=False)
        if np.any(np.diff(timestamp) <= 0):
            raise ValueError("[nilm_dataset] aligned timestamps must be strictly increasing")

        log_dir = self.log_dir(context)
        cycle_root = os.path.join(log_dir, "cycles")
        real_records, rejected = [], []
        real_by_activity = {}
        assignments = self._load_assignments(assignments_path)
        for row in assignments:
            segment_path = os.path.join(segments_dir, row["file"])
            if not os.path.exists(segment_path):
                rejected.append({**row, "reason": "segment_file_missing"})
                continue
            segment = pd.read_csv(segment_path, usecols=["timestamp"])
            start, end = int(segment["timestamp"].min()), int(segment["timestamp"].max())
            values, reason = self._resample_interval(
                timestamp, mains, appliance, start, end,
                self.sample_period_seconds, self.max_gap_seconds)
            if reason:
                rejected.append({**row, "reason": reason})
                continue
            activity_id = str(row["activity_id"])
            path = self._write_npz(
                os.path.join(cycle_root, "real", row["split"]),
                f"activity_{int(activity_id):05d}", values)
            record = {
                "kind": "real", "activity_id": activity_id,
                "class_id": int(row["class_id"]), "mode_id": int(row["mode_id"]),
                "split": row["split"], "length_samples": int(len(values["mains"])),
                "file": self._relative(path, log_dir),
            }
            real_records.append(record)
            real_by_activity[activity_id] = {**record, "payload": values}

        with open(synthetic_manifest_path, encoding="utf-8") as f:
            synthetic_manifest = json.load(f)
        synthetic_records, rejected_synthetic = [], []
        for row in synthetic_manifest:
            source_id = str(row.get("source_activity_id", ""))
            source = real_by_activity.get(source_id)
            assignment = next(
                (value for value in assignments
                 if str(value["activity_id"]) == source_id), None)
            if assignment and assignment["split"] != "train":
                raise ValueError(
                    f"[nilm_dataset] synthetic cycle references non-training activity {source_id}")
            if not source:
                rejected_synthetic.append({
                    "cycle_id": int(row["cycle_id"]),
                    "source_activity_id": source_id,
                    "reason": "source_training_cycle_unavailable",
                })
                continue
            frame = pd.read_csv(
                os.path.join(synthetic_dir, row["file"]), usecols=["power"])
            target = frame["power"].to_numpy(dtype=np.float32, copy=False)
            source_payload = source["payload"]
            background = np.maximum(
                source_payload["mains"] - source_payload["appliance"], 0.0)
            source_axis = np.linspace(0.0, 1.0, len(background))
            target_axis = np.linspace(0.0, 1.0, len(target))
            synthetic_background = np.interp(
                target_axis, source_axis, background).astype(np.float32)
            synthetic_mains = synthetic_background + target
            payload = {
                "timestamp": np.arange(len(target), dtype=np.int64)
                * self.sample_period_seconds,
                "mains": synthetic_mains.astype(np.float32),
                "appliance": target.astype(np.float32),
            }
            path = self._write_npz(
                os.path.join(cycle_root, "synthetic"),
                f"synthetic_{int(row['cycle_id']):05d}", payload)
            synthetic_records.append({
                "kind": "synthetic", "cycle_id": int(row["cycle_id"]),
                "source_activity_id": source_id,
                "class_id": int(row["cycle_class"]),
                "mode_id": int(row["cycle_mode"]),
                "split": "train", "length_samples": int(len(target)),
                "file": self._relative(path, log_dir),
            })

        train = [row for row in real_records if row["split"] == "train"]
        validation = [row for row in real_records if row["split"] == "validation"]
        test = [row for row in real_records if row["split"] == "test"]
        traditional_records, traditional_by_activity = [], {}
        for record in train:
            activity_id = str(record["activity_id"])
            source_payload = real_by_activity[activity_id]["payload"]
            augment_rng = np.random.default_rng(np.random.SeedSequence([
                self.random_seed, int(activity_id), 32452843,
            ]))
            augmented_mains, augmented_appliance, parameters = (
                self._traditional_augment(
                    source_payload["mains"], source_payload["appliance"],
                    augment_rng, self.traditional_scale_range,
                    self.traditional_noise_ratio, self.active_threshold_watts))
            payload = {
                "timestamp": source_payload["timestamp"],
                "mains": augmented_mains,
                "appliance": augmented_appliance,
            }
            path = self._write_npz(
                os.path.join(cycle_root, "traditional"),
                f"traditional_{int(activity_id):05d}", payload)
            augmented_record = {
                "kind": "traditional", "source_activity_id": activity_id,
                "class_id": record["class_id"], "mode_id": record["mode_id"],
                "split": "train", "length_samples": record["length_samples"],
                "file": self._relative(path, log_dir), **parameters,
            }
            traditional_records.append(augmented_record)
            traditional_by_activity[activity_id] = augmented_record
        experiments = {}
        for ratio in self.real_ratios:
            count = max(1, int(round(len(train) * ratio)))
            ratio_seed = np.random.SeedSequence(
                [self.random_seed, int(round(ratio * 10_000))])
            rng = np.random.default_rng(ratio_seed)
            real_subset = self._stratified_select(train, count, rng)
            generated_subset = self._stratified_select(
                synthetic_records, min(len(synthetic_records), len(real_subset)), rng)
            traditional_subset = [
                traditional_by_activity[str(row["activity_id"])]
                for row in real_subset
            ]
            tag = f"{int(round(ratio * 100)):02d}pct"
            experiments[tag] = {
                "real_ratio": ratio,
                "A_real_only": [row["file"] for row in real_subset],
                "B_real_plus_traditional": (
                    [row["file"] for row in real_subset]
                    + [row["file"] for row in traditional_subset]),
                "C_real_plus_generated": (
                    [row["file"] for row in real_subset]
                    + [row["file"] for row in generated_subset]),
                "selected_real_count": len(real_subset),
                "selected_traditional_count": len(traditional_subset),
                "selected_generated_count": len(generated_subset),
            }
        experiments["full"] = {
            "D_full_real": [row["file"] for row in train],
            "validation": [row["file"] for row in validation],
            "test": [row["file"] for row in test],
        }

        expected_groups = Counter(
            (row["split"], int(row["class_id"]), int(row["mode_id"]))
            for row in assignments)
        accepted_groups = Counter(
            (row["split"], int(row["class_id"]), int(row["mode_id"]))
            for row in real_records)
        group_retention = []
        for (split, class_id, mode_id), total in sorted(expected_groups.items()):
            accepted = accepted_groups[(split, class_id, mode_id)]
            group_retention.append({
                "split": split, "class_id": class_id, "mode_id": mode_id,
                "total": total, "accepted": accepted,
                "rejected": total - accepted,
                "acceptance_ratio": accepted / total if total else 0.0,
            })

        manifest_path = os.path.join(log_dir, "nilm_dataset_manifest.json")
        total_real = len(real_records) + len(rejected)
        audit = {
            "aligned_series": str(aligned_path),
            "sample_period_seconds": self.sample_period_seconds,
            "max_gap_seconds": self.max_gap_seconds,
            "real_counts": {"train": len(train), "validation": len(validation),
                            "test": len(test), "rejected": len(rejected)},
            "real_acceptance_ratio": (
                len(real_records) / total_real if total_real else 0.0),
            "real_rejection_reasons": dict(Counter(
                row["reason"] for row in rejected)),
            "class_mode_retention": group_retention,
            "synthetic_count": len(synthetic_records),
            "synthetic_rejected": len(rejected_synthetic),
            "synthetic_acceptance_ratio": (
                len(synthetic_records) / len(synthetic_manifest)
                if synthetic_manifest else 0.0),
            "synthetic_rejection_reasons": dict(Counter(
                row["reason"] for row in rejected_synthetic)),
            "synthetic_background": "max(source_train_mains-source_train_appliance,0)",
            "traditional_augmentation": {
                "method": "magnitude_scaling_plus_active_jitter",
                "count": len(traditional_records),
                "scale_range": list(self.traditional_scale_range),
                "noise_ratio": self.traditional_noise_ratio,
                "active_threshold_watts": self.active_threshold_watts,
                "paired_with_selected_real_cycles": True,
            },
            "test_waveform_used_by_synthesis": False,
            "experiments": experiments,
            "rejected_cycles": rejected,
            "rejected_synthetic_cycles": rejected_synthetic,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(audit, f, indent=2, ensure_ascii=False)
        traditional_manifest_path = os.path.join(
            log_dir, "traditional_augmentation_manifest.json")
        with open(traditional_manifest_path, "w", encoding="utf-8") as f:
            json.dump(traditional_records, f, indent=2, ensure_ascii=False)
        self.record(context, artifacts={
            "dataset_manifest": self.rel(context, manifest_path),
            "cycles_dir": self.rel(context, cycle_root),
            "traditional_manifest": self.rel(context, traditional_manifest_path),
        }, extra={
            "cluster_tag": self.cluster_tag,
            "real_counts": audit["real_counts"],
            "synthetic_count": len(synthetic_records),
            "waveform_holdout": True,
        })
        print(f"[nilm_dataset] real={audit['real_counts']} synthetic="
              f"{len(synthetic_records)} -> {log_dir}")
        return context
