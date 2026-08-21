"""Create a global chronological cycle holdout before structure discovery."""
from __future__ import annotations

import csv
import json
import os

import pandas as pd

from src.framework.step import Step


class TemporalHoldoutStep(Step):
    """Assign appliance cycles to contiguous train/validation/test periods."""

    step_type = "temporal_holdout"

    def __init__(self, cluster_tag: str, train_ratio: float = 0.7,
                 validation_ratio: float = 0.1, test_ratio: float = 0.2):
        if not cluster_tag:
            raise ValueError("temporal holdout requires --cluster-tag")
        ratios = [float(train_ratio), float(validation_ratio), float(test_ratio)]
        if any(value < 0 for value in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
            raise ValueError("temporal_holdout ratios must be non-negative and sum to 1")
        if ratios[0] <= 0:
            raise ValueError("temporal_holdout.train_ratio must be positive")
        super().__init__(variant=f"global_chronological_on_{cluster_tag}")
        self.cluster_tag = cluster_tag
        self.train_ratio, self.validation_ratio, self.test_ratio = ratios

    @staticmethod
    def _counts(n: int, train_ratio: float, validation_ratio: float,
                test_ratio: float) -> tuple[int, int, int]:
        if n <= 0:
            return 0, 0, 0
        train = max(1, int(n * train_ratio))
        validation = int(n * validation_ratio)
        if validation_ratio > 0 and n - train >= 2:
            validation = max(1, validation)
        test = n - train - validation
        if test_ratio > 0 and test == 0 and train > 1:
            train -= 1
            test = 1
        return train, validation, test

    def run(self, context: dict) -> dict:
        sequence_path = context["manifest"].cluster_artifact_path(
            self.cluster_tag, "state_sequences")
        if not (sequence_path and os.path.exists(sequence_path)):
            raise FileNotFoundError(
                f"[temporal_holdout] missing {self.cluster_tag}.state_sequences")
        with open(sequence_path, encoding="utf-8") as f:
            sequences = json.load(f)
        segments_dir = self.resolve(context, "extract_active_data", "segments_dir")
        if not (segments_dir and os.path.isdir(segments_dir)):
            raise FileNotFoundError("[temporal_holdout] activity directory not found")
        files = sorted(name for name in os.listdir(segments_dir)
                       if name.lower().endswith(".csv"))

        records = []
        for activity_id in sequences:
            index = int(activity_id)
            if not 0 <= index < len(files):
                raise ValueError(
                    f"[temporal_holdout] activity {activity_id} has no segment file")
            filename = files[index]
            frame = pd.read_csv(
                os.path.join(segments_dir, filename), usecols=["timestamp"])
            if frame.empty:
                raise ValueError(
                    f"[temporal_holdout] empty timestamp series: {filename}")
            records.append({
                "activity_id": str(activity_id),
                "file": filename,
                "start_timestamp": int(frame["timestamp"].min()),
                "end_timestamp": int(frame["timestamp"].max()),
            })
        records.sort(key=lambda row: (
            row["start_timestamp"], row["end_timestamp"], int(row["activity_id"])))
        n_train, n_validation, n_test = self._counts(
            len(records), self.train_ratio, self.validation_ratio, self.test_ratio)
        boundaries = n_train, n_train + n_validation
        for position, row in enumerate(records):
            if position < boundaries[0]:
                row["split"] = "train"
            elif position < boundaries[1]:
                row["split"] = "validation"
            else:
                row["split"] = "test"

        log_dir = self.log_dir(context)
        assignments_path = os.path.join(log_dir, "temporal_holdout_assignments.csv")
        with open(assignments_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "activity_id", "file", "start_timestamp", "end_timestamp", "split"])
            writer.writeheader()
            writer.writerows(records)
        split_rows = {
            split: [row for row in records if row["split"] == split]
            for split in ("train", "validation", "test")
        }
        summary = {
            "method": "global_chronological_cycles_before_structure_fit",
            "cluster_tag": self.cluster_tag,
            "ratios": {"train": self.train_ratio,
                       "validation": self.validation_ratio,
                       "test": self.test_ratio},
            "counts": {key: len(value) for key, value in split_rows.items()},
            "timestamp_ranges": {
                key: ({"start": int(value[0]["start_timestamp"]),
                       "end": int(value[-1]["end_timestamp"])} if value else None)
                for key, value in split_rows.items()
            },
            "structure_fit_scope": "train_only",
            "globally_chronological": True,
        }
        summary_path = os.path.join(log_dir, "temporal_holdout_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        self.record(context, artifacts={
            "assignments": self.rel(context, assignments_path),
            "summary": self.rel(context, summary_path),
        }, extra={
            "cluster_tag": self.cluster_tag,
            "counts": summary["counts"],
            "structure_fit_scope": "train_only",
        })
        print(f"[temporal_holdout] train/validation/test={summary['counts']} "
              f"-> {log_dir}")
        return context
