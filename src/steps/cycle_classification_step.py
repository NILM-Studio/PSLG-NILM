"""Classify real appliance activities by merged primitive-state patterns."""
from __future__ import annotations

import csv
import json
import os

from src.framework.step import Step
from src.generation.cycle_patterns import CyclePatternClassifier


class CycleClassificationStep(Step):
    step_type = "cycle_classification"

    def __init__(self, cluster_tag: str, min_support: int = 3,
                 max_classes: int = 12, rare_max_distance: float = 0.34,
                 min_pattern_blocks: int = 3, min_unique_states: int = 2,
                 require_temporal_holdout: bool = False):
        if not cluster_tag:
            raise ValueError("cycle classification requires --cluster-tag")
        scope = "train_only_" if require_temporal_holdout else ""
        super().__init__(variant=f"{scope}on_{cluster_tag}")
        self.cluster_tag = cluster_tag
        self.require_temporal_holdout = bool(require_temporal_holdout)
        self.classifier = CyclePatternClassifier(
            min_support=min_support,
            max_classes=max_classes,
            rare_max_distance=rare_max_distance,
            min_pattern_blocks=min_pattern_blocks,
            min_unique_states=min_unique_states,
        )

    def run(self, context: dict) -> dict:
        source = context["manifest"].cluster_artifact_path(
            self.cluster_tag, "state_sequences")
        if not (source and os.path.exists(source)):
            raise FileNotFoundError(
                f"[cycle_classification] missing {self.cluster_tag}.state_sequences; "
                "run state_merge first and use a merged cluster tag")
        with open(source, encoding="utf-8") as f:
            sequences = json.load(f)

        split_by_activity = {}
        if self.require_temporal_holdout:
            assignment_path = self.resolve(context, "temporal_holdout", "assignments")
            if not (assignment_path and os.path.exists(assignment_path)):
                raise FileNotFoundError(
                    "[cycle_classification] temporal holdout assignments not found")
            with open(assignment_path, newline="", encoding="utf-8") as f:
                split_by_activity = {
                    str(row["activity_id"]): row["split"] for row in csv.DictReader(f)
                }
            missing = set(map(str, sequences)) - set(split_by_activity)
            if missing:
                raise ValueError(
                    f"[cycle_classification] {len(missing)} activities lack holdout split")
        fit_ids = ([key for key, split in split_by_activity.items()
                    if split == "train"] if split_by_activity else None)
        result = self.classifier.fit(sequences, fit_ids=fit_ids)
        if split_by_activity:
            for activity_id, record in result["activities"].items():
                record["source_split"] = split_by_activity[activity_id]
        log_dir = self.log_dir(context)
        classes_path = os.path.join(log_dir, "cycle_classes.json")
        with open(classes_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        assignments_path = os.path.join(log_dir, "cycle_class_assignments.csv")
        with open(assignments_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "activity_id", "class_id", "signature", "length_samples",
                "distance_to_representative", "outlier_reason"])
            writer.writeheader()
            for activity_id, record in sorted(result["activities"].items()):
                writer.writerow({
                    "activity_id": activity_id,
                    "class_id": record["class_id"],
                    "signature": "->".join(map(str, record["signature"])),
                    "length_samples": record["length_samples"],
                    "distance_to_representative": record["distance_to_representative"],
                    "outlier_reason": record.get("outlier_reason", ""),
                })

        self.record(context, artifacts={
            "cycle_classes": self.rel(context, classes_path),
            "assignments": self.rel(context, assignments_path),
        }, extra={
            "cluster_tag": self.cluster_tag,
            "n_classes": result["n_classes"],
            "n_outliers": result["n_outliers"],
            "structure_fit_scope": result["fit_scope"],
        })
        print(f"[cycle_classification] {result['n_activities']} activities -> "
              f"{result['n_classes']} classes, {result['n_outliers']} outliers")
        for entry in result["classes"]:
            print(f"  class_{entry['class_id']}: support={entry['support']} "
                  f"pattern={'->'.join(map(str, entry['representative_signature']))}")
        print(f"[cycle_classification] result -> {log_dir}")
        return context
