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

    @staticmethod
    def _purge_boundary_overlaps(records: list[dict]) -> tuple[list[dict], dict]:
        """Keep later periods fixed; exclude earlier cycles reaching their start.

        Intervals include the extracted context and both endpoints. Purging a
        whole activity, rather than trimming its waveform, preserves the row
        indices used by the existing segmentation and clustering artifacts.
        """
        splits = ("train", "validation", "test")
        starts = {
            split: min(row["start_timestamp"] for row in records
                       if row["split"] == split)
            for split in splits if any(row["split"] == split for row in records)
        }
        assignments = []
        for record in records:
            row = {**record, "original_split": record["split"],
                   "purge_reason": "", "boundary_split": "",
                   "boundary_timestamp": ""}
            later = splits[splits.index(row["split"]) + 1:]
            boundary_split = next((split for split in later if split in starts), None)
            if (boundary_split is not None
                    and row["end_timestamp"] >= starts[boundary_split]):
                row.update(
                    split="purged", purge_reason="interval_reaches_later_split",
                    boundary_split=boundary_split,
                    boundary_timestamp=int(starts[boundary_split]))
            assignments.append(row)
        return assignments, starts

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

        nominal_counts = {"train": n_train, "validation": n_validation,
                          "test": n_test}
        records, boundary_starts = self._purge_boundary_overlaps(records)
        excluded = [row for row in records if row["split"] == "purged"]

        log_dir = self.log_dir(context)
        assignments_path = os.path.join(log_dir, "temporal_holdout_assignments.csv")
        fields = ["activity_id", "file", "start_timestamp", "end_timestamp",
                  "split", "original_split", "purge_reason", "boundary_split",
                  "boundary_timestamp"]
        with open(assignments_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
        exclusions_path = os.path.join(log_dir, "temporal_holdout_excluded.csv")
        with open(exclusions_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(excluded)
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
            "nominal_counts": nominal_counts,
            "boundary_policy": "purge_earlier_activity_reaching_later_period_start",
            "nominal_period_starts": boundary_starts,
            "excluded_count": len(excluded),
            "excluded_activity_ids": [row["activity_id"] for row in excluded],
            "timestamp_ranges": {
                key: ({"start": int(min(row["start_timestamp"] for row in value)),
                       "end": int(max(row["end_timestamp"] for row in value))}
                      if value else None)
                for key, value in split_rows.items()
            },
            "structure_fit_scope": "train_only",
            "globally_chronological": True,
            "cross_split_interval_overlap": False,
        }
        summary_path = os.path.join(log_dir, "temporal_holdout_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        emptied = [split for split, count in nominal_counts.items()
                   if count and not split_rows[split]]
        if emptied:
            raise ValueError(
                f"[temporal_holdout] boundary purge emptied {emptied}; choose "
                f"different split ratios or more data. Exclusions: {exclusions_path}")
        self.record(context, artifacts={
            "assignments": self.rel(context, assignments_path),
            "summary": self.rel(context, summary_path),
            "excluded_assignments": self.rel(context, exclusions_path),
        }, extra={
            "cluster_tag": self.cluster_tag,
            "counts": summary["counts"],
            "structure_fit_scope": "train_only",
            "excluded_count": len(excluded),
            "cross_split_interval_overlap": False,
        })
        print(f"[temporal_holdout] train/validation/test={summary['counts']} "
              f"purged={len(excluded)} -> {log_dir}")
        return context
