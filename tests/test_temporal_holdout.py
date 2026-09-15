import csv
import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.framework.step import Step
from src.framework.workflow import Workflow
from src.generation.cycle_patterns import CyclePatternClassifier
from src.steps.cycle_classification_step import CycleClassificationStep
from src.steps.cycle_split_step import CycleSplitStep
from src.steps.cycle_validation_step import CycleValidationStep
from src.steps.temporal_holdout_step import TemporalHoldoutStep


class TemporalUpstreamStub(Step):
    step_type = "temporal_upstream_stub"

    def __init__(self, overlap=False, empty_validation=False):
        super().__init__()
        self.overlap = overlap
        self.empty_validation = empty_validation

    def run(self, context):
        segments = os.path.join(context["log_root"], "activities")
        os.makedirs(segments)
        sequences = {}
        for activity_id in range(10):
            start = 1_000 + activity_id * 100
            power = np.asarray([0, 0, 20, 20, 20, 0, 0, 0], dtype=float)
            period = 18 if self.overlap and activity_id in (5, 7) else 6
            if self.empty_validation and activity_id == 7:
                period = 18
            pd.DataFrame({
                "timestamp": start + np.arange(len(power)) * period,
                "power": power,
            }).to_csv(os.path.join(
                segments, f"activity_{activity_id:02d}.csv"), index=False)
            labels = [0, 1, 0] if activity_id % 2 == 0 else [1, 0, 1]
            sequences[str(activity_id)] = [
                {"state_label": label, "length_samples": 8}
                for label in labels
            ]
        context["manifest"].add_step(
            "extract_active_data", "stub", "activities",
            {"segments_dir": self.rel(context, segments)})
        result_dir = os.path.join(context["log_root"], "merged")
        os.makedirs(result_dir)
        sequence_path = os.path.join(result_dir, "state_sequences.json")
        with open(sequence_path, "w", encoding="utf-8") as f:
            json.dump(sequences, f)
        context["manifest"].add_cluster_result(
            "kmeans_k2_merged", "merged",
            {"state_sequences": self.rel(context, sequence_path)})
        return context


class TemporalHoldoutTests(unittest.TestCase):
    @staticmethod
    def _blocks(labels):
        return [{"state_label": label, "length_samples": 10}
                for label in labels]

    def test_classifier_anchors_are_learned_from_fit_ids_only(self):
        sequences = {
            "0": self._blocks([0, 1, 0]),
            "1": self._blocks([0, 1, 0]),
            "2": self._blocks([0, 1, 0]),
            **{str(i): self._blocks([2, 3, 2]) for i in range(3, 9)},
        }
        result = CyclePatternClassifier(min_support=2).fit(
            sequences, fit_ids={"0", "1", "2"})
        self.assertEqual(result["fit_scope"], "train_only")
        self.assertEqual(result["n_classes"], 1)
        self.assertEqual(result["classes"][0]["representative_signature"], [0, 1, 0])
        self.assertTrue(all(result["activities"][str(i)]["class_id"] == -1
                            for i in range(3, 9)))

    def _assert_strict_workflow(self, overlap=False):
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                wf = Workflow("strict", "washing_machine", {})
                wf.add(TemporalUpstreamStub(overlap=overlap))
                ratios = (0.6, 0.2, 0.2) if overlap else (0.7, 0.1, 0.2)
                wf.add(TemporalHoldoutStep(
                    "kmeans_k2_merged", *ratios))
                wf.add(CycleClassificationStep(
                    "kmeans_k2_merged", min_support=1,
                    require_temporal_holdout=True))
                wf.add(CycleValidationStep(
                    "kmeans_k2_merged", fs=1.0,
                    min_class_support=1, min_signature_purity=0.0,
                    min_valid_member_ratio=0.0,
                    core_state_min_prevalence=0.0,
                    terminal_state_min_prevalence=0.0,
                    min_duration_seconds=0.0,
                    boundary_absolute_watts=100.0,
                    robust_z_threshold=float("inf"),
                    min_mode_support=2,
                    require_train_only_structure=True))
                wf.add(CycleSplitStep(
                    "kmeans_k2_merged", *ratios,
                    require_temporal_holdout=True))
                wf.run()

                with open(wf.manifest.artifact_path(
                        "temporal_holdout", "summary"), encoding="utf-8") as f:
                    holdout = json.load(f)
                self.assertEqual(
                    holdout["counts"],
                    {"train": 5 if overlap else 7, "validation": 1, "test": 2})
                self.assertFalse(holdout["cross_split_interval_overlap"])
                ranges = holdout["timestamp_ranges"]
                self.assertLess(ranges["train"]["end"], ranges["validation"]["start"])
                self.assertLess(ranges["validation"]["end"], ranges["test"]["start"])
                excluded = {"5", "7"} if overlap else set()
                self.assertEqual(set(holdout["excluded_activity_ids"]), excluded)
                with open(wf.manifest.artifact_path(
                        "temporal_holdout", "excluded_assignments"),
                        newline="", encoding="utf-8") as f:
                    excluded_rows = list(csv.DictReader(f))
                self.assertEqual({row["activity_id"] for row in excluded_rows}, excluded)
                self.assertTrue(all(row["purge_reason"] for row in excluded_rows))

                with open(wf.manifest.artifact_path(
                        "cycle_classification", "cycle_classes"), encoding="utf-8") as f:
                    classes = json.load(f)
                self.assertEqual(classes["fit_scope"], "train_only")
                self.assertFalse(excluded & set(classes["activities"]))
                self.assertEqual(set(classes["temporal_excluded_activity_ids"]), excluded)

                with open(wf.manifest.artifact_path(
                        "cycle_split", "summary"), encoding="utf-8") as f:
                    split = json.load(f)
                self.assertEqual(
                    split["method"],
                    "inherited_global_chronological_before_structure_fit")
                self.assertEqual(split["structure_fit_scope"], "train_only")

                with open(wf.manifest.artifact_path(
                        "cycle_split", "train_catalog"), encoding="utf-8") as f:
                    train = json.load(f)
                self.assertTrue(train["activities"])
                self.assertTrue(all(
                    row["source_split"] == "train"
                    for row in train["activities"].values()))
                self.assertFalse(excluded & set(train["activities"]))
            finally:
                os.chdir(cwd)

    def test_strict_workflow_inherits_global_temporal_split(self):
        self._assert_strict_workflow()

    def test_purged_activities_are_excluded_from_classification_and_split(self):
        self._assert_strict_workflow(overlap=True)

    def test_purge_removes_shared_endpoint_and_long_earlier_intervals(self):
        records = [
            {"activity_id": "0", "split": "train", "start_timestamp": 0,
             "end_timestamp": 250},
            {"activity_id": "1", "split": "train", "start_timestamp": 10,
             "end_timestamp": 99},
            {"activity_id": "2", "split": "train", "start_timestamp": 20,
             "end_timestamp": 100},
            {"activity_id": "3", "split": "validation", "start_timestamp": 100,
             "end_timestamp": 150},
            {"activity_id": "4", "split": "test", "start_timestamp": 200,
             "end_timestamp": 220},
        ]
        result, starts = TemporalHoldoutStep._purge_boundary_overlaps(records)
        self.assertEqual(starts, {"train": 0, "validation": 100, "test": 200})
        self.assertEqual([row["split"] for row in result],
                         ["purged", "train", "purged", "validation", "test"])
        self.assertEqual(result[0]["original_split"], "train")
        self.assertEqual(result[0]["boundary_timestamp"], 100)
        self.assertEqual(records[0]["split"], "train")

    def test_empty_split_after_purge_fails_and_keeps_exclusion_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                wf = Workflow("empty_after_purge", "washing_machine", {})
                wf.add(TemporalUpstreamStub(empty_validation=True))
                step = TemporalHoldoutStep("kmeans_k2_merged", 0.7, 0.1, 0.2)
                wf.add(step)
                with self.assertRaisesRegex(ValueError, "purge emptied.*validation"):
                    wf.run()
                audit_path = os.path.join(
                    "log", "empty_after_purge", step.log_subdir(),
                    "temporal_holdout_excluded.csv")
                with open(audit_path, newline="", encoding="utf-8") as f:
                    excluded = list(csv.DictReader(f))
                self.assertEqual([row["activity_id"] for row in excluded], ["7"])
                self.assertEqual(excluded[0]["boundary_split"], "test")
            finally:
                os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
