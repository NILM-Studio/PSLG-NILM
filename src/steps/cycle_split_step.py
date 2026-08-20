"""Chronological, class/mode-stratified source split for cycle synthesis."""
from __future__ import annotations

import copy
import csv
import json
import os
from typing import Dict, List

from src.framework.step import Step


class CycleSplitStep(Step):
    """Hold out validated source cycles before building primitive libraries.

    Splitting is chronological inside each validated (class, mode) group. This
    preserves every supported program in train/validation/test while ensuring
    held-out waveforms cannot be sampled by the generator.
    """

    step_type = "cycle_split"

    def __init__(self, cluster_tag: str, train_ratio: float = 0.7,
                 validation_ratio: float = 0.1, test_ratio: float = 0.2):
        if not cluster_tag:
            raise ValueError("cycle split requires --cluster-tag")
        ratios = [float(train_ratio), float(validation_ratio), float(test_ratio)]
        if any(value < 0 for value in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
            raise ValueError("cycle_split ratios must be non-negative and sum to 1")
        if ratios[0] <= 0:
            raise ValueError("cycle_split.train_ratio must be positive")
        super().__init__(variant=f"chronological_stratified_on_{cluster_tag}")
        self.cluster_tag = cluster_tag
        self.train_ratio, self.validation_ratio, self.test_ratio = ratios

    @staticmethod
    def _counts(n: int, train_ratio: float, validation_ratio: float,
                test_ratio: float) -> tuple[int, int, int]:
        """Allocate counts while keeping at least one training member."""
        if n <= 0:
            return 0, 0, 0
        validation = int(round(n * validation_ratio))
        test = int(round(n * test_ratio))
        if validation_ratio > 0 and n >= 3:
            validation = max(1, validation)
        if test_ratio > 0 and n >= 2:
            test = max(1, test)
        while validation + test > n - 1:
            if validation > (1 if validation_ratio > 0 and n >= 3 else 0):
                validation -= 1
            elif test > (1 if test_ratio > 0 and n >= 2 else 0):
                test -= 1
            elif validation > 0:
                validation -= 1
            else:
                test -= 1
        return n - validation - test, validation, test

    @staticmethod
    def _catalog_for(payload: dict, member_ids: set[str], split: str) -> dict:
        result = copy.deepcopy(payload)
        result["activities"] = {
            key: value for key, value in result.get("activities", {}).items()
            if str(key) in member_ids
        }
        classes = []
        for entry in result.get("classes", []):
            entry["member_ids"] = [str(value) for value in entry.get("member_ids", [])
                                   if str(value) in member_ids]
            entry["support"] = len(entry["member_ids"])
            if entry["member_ids"]:
                classes.append(entry)
        result["classes"] = classes
        result["n_classes"] = len(classes)
        result["n_activities"] = len(result["activities"])
        result["source_split"] = {
            "name": split,
            "waveform_holdout": True,
            "structure_fit_scope": "all_validated_cycles",
        }
        return result

    def run(self, context: dict) -> dict:
        validation = context["manifest"].get_step("cycle_validation") or {}
        validated_tag = (validation.get("extra") or {}).get("cluster_tag")
        if validated_tag and validated_tag != self.cluster_tag:
            raise ValueError(
                f"[cycle_split] validation uses {validated_tag}, requested {self.cluster_tag}")
        catalog_path = self.resolve(
            context, "cycle_validation", "validated_cycle_classes")
        if not (catalog_path and os.path.exists(catalog_path)):
            raise FileNotFoundError(
                "[cycle_split] validated catalog not found; run cycle_validate first")
        with open(catalog_path, encoding="utf-8") as f:
            payload = json.load(f)

        segments_dir = self.resolve(context, "extract_active_data", "segments_dir")
        files = (sorted(name for name in os.listdir(segments_dir)
                        if name.lower().endswith(".csv"))
                 if segments_dir and os.path.isdir(segments_dir) else [])

        groups: Dict[tuple[int, int], List[str]] = {}
        for class_entry in payload.get("classes", []):
            class_id = int(class_entry["class_id"])
            for value in class_entry.get("member_ids", []):
                activity_id = str(value)
                activity = payload["activities"][activity_id]
                mode_id = int(activity.get("validation_mode_id", -1))
                if mode_id < 0:
                    raise ValueError(
                        f"[cycle_split] activity {activity_id} has no validated mode")
                groups.setdefault((class_id, mode_id), []).append(activity_id)

        split_ids = {"train": set(), "validation": set(), "test": set()}
        assignments, group_summary = [], []
        for (class_id, mode_id), members in sorted(groups.items()):
            members = sorted(members, key=int)
            n_train, n_validation, n_test = self._counts(
                len(members), self.train_ratio, self.validation_ratio, self.test_ratio)
            boundaries = (n_train, n_train + n_validation)
            slices = {
                "train": members[:boundaries[0]],
                "validation": members[boundaries[0]:boundaries[1]],
                "test": members[boundaries[1]:],
            }
            for split, ids in slices.items():
                split_ids[split].update(ids)
                for activity_id in ids:
                    index = int(activity_id)
                    assignments.append({
                        "activity_id": activity_id,
                        "file": files[index] if 0 <= index < len(files) else "",
                        "class_id": class_id,
                        "mode_id": mode_id,
                        "split": split,
                    })
            group_summary.append({
                "class_id": class_id, "mode_id": mode_id,
                "total": len(members), "train": n_train,
                "validation": n_validation, "test": n_test,
            })

        all_ids = set().union(*split_ids.values())
        if len(all_ids) != len(payload.get("activities", {})):
            raise ValueError("[cycle_split] split did not cover every validated activity")
        if any(split_ids[left] & split_ids[right]
               for left, right in (("train", "validation"),
                                   ("train", "test"),
                                   ("validation", "test"))):
            raise ValueError("[cycle_split] source splits overlap")

        log_dir = self.log_dir(context)
        artifacts = {}
        for split, ids in split_ids.items():
            path = os.path.join(log_dir, f"{split}_cycle_catalog.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._catalog_for(payload, ids, split), f,
                          indent=2, ensure_ascii=False)
            artifacts[f"{split}_catalog"] = self.rel(context, path)

        assignment_path = os.path.join(log_dir, "cycle_split_assignments.csv")
        with open(assignment_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["activity_id", "file", "class_id", "mode_id", "split"])
            writer.writeheader()
            writer.writerows(sorted(assignments, key=lambda row: int(row["activity_id"])))
        artifacts["assignments"] = self.rel(context, assignment_path)

        summary = {
            "method": "chronological_within_class_mode",
            "ratios": {"train": self.train_ratio,
                       "validation": self.validation_ratio,
                       "test": self.test_ratio},
            "counts": {key: len(value) for key, value in split_ids.items()},
            "groups": group_summary,
            "waveform_holdout": True,
            "structure_fit_scope": "all_validated_cycles",
        }
        summary_path = os.path.join(log_dir, "cycle_split_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        artifacts["summary"] = self.rel(context, summary_path)

        self.record(context, artifacts=artifacts, extra={
            "cluster_tag": self.cluster_tag,
            "counts": summary["counts"],
            "waveform_holdout": True,
            "structure_fit_scope": "all_validated_cycles",
        })
        print(f"[cycle_split] train/validation/test={summary['counts']} -> {log_dir}")
        return context
