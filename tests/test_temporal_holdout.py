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

    def run(self, context):
        segments = os.path.join(context["log_root"], "activities")
        os.makedirs(segments)
        sequences = {}
        for activity_id in range(10):
            start = 1_000 + activity_id * 100
            power = np.asarray([0, 0, 20, 20, 20, 0, 0, 0], dtype=float)
            pd.DataFrame({
                "timestamp": start + np.arange(len(power)) * 6,
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

    def test_strict_workflow_inherits_global_temporal_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                wf = Workflow("strict", "washing_machine", {})
                wf.add(TemporalUpstreamStub())
                wf.add(TemporalHoldoutStep(
                    "kmeans_k2_merged", 0.7, 0.1, 0.2))
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
                    "kmeans_k2_merged", 0.7, 0.1, 0.2,
                    require_temporal_holdout=True))
                wf.run()

                with open(wf.manifest.artifact_path(
                        "temporal_holdout", "summary"), encoding="utf-8") as f:
                    holdout = json.load(f)
                self.assertEqual(
                    holdout["counts"], {"train": 7, "validation": 1, "test": 2})

                with open(wf.manifest.artifact_path(
                        "cycle_classification", "cycle_classes"), encoding="utf-8") as f:
                    classes = json.load(f)
                self.assertEqual(classes["fit_scope"], "train_only")

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
            finally:
                os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
