"""Budget donor isolation, waveform behavior, provenance and measurement gates."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.framework.run_manifest import RunManifest
from src.generation.primitive_library import Primitive
from src.steps.nilm_dataset_step import NilmDatasetStep


class BudgetConditioningTests(unittest.TestCase):
    @staticmethod
    def fixture():
        records, activities, paired, primitives = [], {}, {}, []
        for activity_id, level in enumerate([10.0, 11.0, 100.0, 1000.0]):
            mode = 1 if activity_id == 3 else 0
            power = np.asarray([level] * 3 + [level * 2] * 3, dtype=np.float32)
            records.append({"activity_id": str(activity_id), "class_id": 0,
                            "mode_id": mode})
            activities[str(activity_id)] = {
                "class_id": 0, "validation_mode_id": mode,
                "blocks": [{"state_label": 0, "length_samples": 3},
                           {"state_label": 1, "length_samples": 3}],
            }
            paired[str(activity_id)] = {"payload": {
                "appliance": power, "mains": power + 50,
            }}
            for state in (0, 1):
                primitives.append(Primitive(
                    2 * activity_id + state, state, activity_id, state * 3,
                    power[state * 3:state * 3 + 3]))
        # An invalid, out-of-budget source must never enter a profile or library.
        primitives.append(Primitive(999, 0, 99, 0, np.array([np.nan], dtype=np.float32)))
        activities["99"] = {"blocks": []}
        return records, {"activities": activities}, paired, primitives

    @staticmethod
    def step(method="independent", neighbors=1, seed=42, **kwargs):
        return NilmDatasetStep(
            "kmeans_k2_merged", "pair.csv", synthesis_scope="budget_local",
            budget_conditioning_method=method,
            budget_conditioning_neighbors=neighbors, random_seed=seed,
            within_state_smooth_samples=0, boundary_smooth_samples=0, **kwargs)

    def generate(self, directory, method="independent", neighbors=1, selected=None):
        records, catalog, paired, primitives = self.fixture()
        selected = records if selected is None else [records[index] for index in selected]
        return self.step(method, neighbors)._generate_budget_cycles(
            directory, "10pct", selected, paired, catalog, primitives)

    def test_neighbors_fit_only_selected_class_mode_and_exclude_anchor(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = self.generate(directory, "cycle_neighbors")
            for row in rows[:3]:
                anchor = int(row["anchor_activity_id"])
                self.assertEqual(row["conditioning_fit_activity_ids"], [0, 1, 2])
                self.assertEqual(row["conditioning_neighbor_count"], 1)
                self.assertEqual(row["donor_activity_count"], 1)
                self.assertNotIn(anchor, row["donor_activity_ids"])
                self.assertTrue(set(row["primitive_source_activity_ids"]) <= {0, 1, 2})
                self.assertEqual(row["self_source_sample_ratio"], 0.0)
                self.assertTrue(row["cross_cycle_generation"])
                self.assertEqual(row["actual_conditioning_method"], "cycle_neighbors")
            self.assertEqual(rows[0]["donor_activity_ids"], [1])
            self.assertEqual(rows[1]["donor_activity_ids"], [0])
            self.assertEqual(rows[3]["conditioning_fit_activity_ids"], [3])

    def test_budget_change_removes_even_physically_nearest_outside_donor(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = self.generate(directory, "cycle_neighbors", selected=[0, 2])
            self.assertEqual(rows[0]["donor_activity_ids"], [2])
            self.assertEqual(rows[1]["donor_activity_ids"], [0])
            for row in rows:
                self.assertEqual(row["conditioning_fit_activity_ids"], [0, 2])
                self.assertEqual(row["budget_activity_ids"], [0, 2])
                self.assertEqual(row["self_source_sample_ratio"], 0.0)

    def test_independent_and_neighbors_change_waveforms_not_templates(self):
        with tempfile.TemporaryDirectory() as independent_dir, \
                tempfile.TemporaryDirectory() as neighbors_dir:
            independent = self.generate(independent_dir, "independent", selected=[0, 1, 2])
            neighbors = self.generate(neighbors_dir, "cycle_neighbors", selected=[0, 1, 2])
            changed = []
            for left, right in zip(independent, neighbors):
                self.assertEqual(left["anchor_activity_id"], right["anchor_activity_id"])
                self.assertEqual(left["length_samples"], right["length_samples"])
                self.assertEqual([(b["state_label"], b["length_samples"]) for b in left["blocks"]],
                                 [(b["state_label"], b["length_samples"]) for b in right["blocks"]])
                self.assertEqual(left["donor_activity_count"], 2)
                self.assertEqual(left["conditioning_neighbor_count"], 0)
                self.assertEqual(left["actual_conditioning_method"], "independent")
                self.assertEqual(right["donor_activity_count"], 1)
                self.assertEqual(left["self_source_sample_ratio"], 0.0)
                with np.load(Path(independent_dir) / left["file"]) as a, \
                        np.load(Path(neighbors_dir) / right["file"]) as b:
                    changed.append(not np.array_equal(a["appliance"], b["appliance"]))
                    np.testing.assert_allclose(a["mains"] - a["appliance"], 50)
                    np.testing.assert_allclose(b["mains"] - b["appliance"], 50)
            self.assertTrue(any(changed), "the method selection must affect sampled power")

    def test_singleton_is_explicitly_self_resampling_for_both_methods(self):
        for method in ("independent", "cycle_neighbors"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                row = self.generate(directory, method, selected=[0])[0]
                self.assertEqual(row["conditioning_method"], method)
                self.assertEqual(row["actual_conditioning_method"], "singleton_self_resample")
                self.assertTrue(row["conditioning_fallback"])
                self.assertEqual(row["conditioning_fallback_reason"], "singleton_class_mode_budget")
                self.assertFalse(row["conditioning_anchor_excluded"])
                self.assertEqual(row["conditioning_neighbors"], [])
                self.assertEqual(row["conditioning_neighbor_count"], 0)
                self.assertEqual(row["primitive_source_activity_ids"], [0])
                self.assertEqual(row["primitive_source_activity_count"], 1)
                self.assertEqual(row["primitive_source_count"], 2)
                self.assertEqual(row["primitive_source_samples_by_activity"], {"0": 6})
                self.assertEqual(row["self_source_sample_ratio"], 1.0)
                self.assertEqual(row["cross_cycle_source_sample_ratio"], 0.0)
                self.assertEqual(row["cross_cycle_source_activity_count"], 0)
                self.assertFalse(row["cross_cycle_generation"])

    def test_neighbor_count_caps_at_available_non_anchor_cycles(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = self.generate(directory, "cycle_neighbors", neighbors=10, selected=[0, 1, 2])
            for row in rows:
                self.assertEqual(row["conditioning_neighbors_requested"], 10)
                self.assertEqual(row["conditioning_neighbor_count"], 2)
                self.assertTrue(row["conditioning_anchor_excluded"])

    def test_provenance_lengths_and_power_metrics_match_written_waveform(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = self.generate(directory, "cycle_neighbors")
            for row in rows:
                self.assertEqual(sum(row["primitive_source_samples_by_activity"].values()),
                                 row["length_samples"])
                self.assertEqual(row["blocks"][0]["start"], 0)
                self.assertEqual(row["blocks"][-1]["end"], row["length_samples"])
                with np.load(Path(directory) / row["file"]) as output:
                    self.assertAlmostEqual(row["energy_wh"],
                                           float(output["appliance"].sum()) * 6 / 3600)
                    self.assertAlmostEqual(row["mean_power"], float(output["appliance"].mean()))

    def test_same_seed_reproduces_conditioned_waveforms_and_provenance(self):
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = self.generate(first_dir, "cycle_neighbors", neighbors=2)
            second = self.generate(second_dir, "cycle_neighbors", neighbors=2)
            self.assertEqual(first, second)
            for a, b in zip(first, second):
                with np.load(Path(first_dir) / a["file"]) as left, \
                        np.load(Path(second_dir) / b["file"]) as right:
                    np.testing.assert_array_equal(left["appliance"], right["appliance"])

    def test_truncated_catalog_source_length_is_rejected(self):
        records, catalog, paired, primitives = self.fixture()
        catalog["activities"]["0"]["blocks"][0]["length_samples"] = 2
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(
                ValueError, "source length 6 differs from catalog duration 5"):
            self.step()._generate_budget_cycles(
                directory, "10pct", records[:1], paired, catalog, primitives)

    def test_primitive_gap_and_overlap_are_rejected(self):
        for wrong_start in (2, 4):
            records, catalog, paired, primitives = self.fixture()
            original = primitives[1]
            primitives[1] = Primitive(original.primitive_id, original.state_label,
                                      original.activity_index, wrong_start, original.power)
            with self.subTest(start=wrong_start), tempfile.TemporaryDirectory() as directory, \
                    self.assertRaisesRegex(ValueError, "coverage gap/overlap"):
                self.step()._generate_budget_cycles(
                    directory, "10pct", records[:1], paired, catalog, primitives)

    def test_catalog_class_mode_mismatch_is_rejected(self):
        records, catalog, paired, primitives = self.fixture()
        records[0]["mode_id"] = 999
        with tempfile.TemporaryDirectory() as directory, \
                self.assertRaisesRegex(ValueError, "class/mode differs"):
            self.step()._generate_budget_cycles(
                directory, "10pct", records[:1], paired, catalog, primitives)

    def test_optional_template_coordinates_must_follow_accumulated_duration(self):
        for coordinate, value in (("start", 3), ("end", 6), ("start", 0.5)):
            records, catalog, paired, primitives = self.fixture()
            catalog["activities"]["0"]["blocks"][0][coordinate] = value
            with self.subTest(coordinate=coordinate, value=value), \
                    tempfile.TemporaryDirectory() as directory, \
                    self.assertRaisesRegex(ValueError, "catalog block 0"):
                self.step()._generate_budget_cycles(
                    directory, "10pct", records[:1], paired, catalog, primitives)

    def test_valid_optional_template_coordinates_preserve_generation(self):
        records, catalog, paired, primitives = self.fixture()
        for index, block in enumerate(catalog["activities"]["0"]["blocks"]):
            block.update(start=index * 3, end=(index + 1) * 3)
        with tempfile.TemporaryDirectory() as directory:
            rows = self.step()._generate_budget_cycles(
                directory, "10pct", records[:1], paired, catalog, primitives)
            self.assertEqual(rows[0]["length_samples"], 6)

    def test_fractional_catalog_duration_is_rejected_without_int_truncation(self):
        records, catalog, paired, primitives = self.fixture()
        catalog["activities"]["0"]["blocks"][0]["length_samples"] = 3.5
        with tempfile.TemporaryDirectory() as directory, \
                self.assertRaisesRegex(ValueError, "catalog block 0 length must be a finite integer"):
            self.step()._generate_budget_cycles(
                directory, "10pct", records[:1], paired, catalog, primitives)

    def test_budget_variant_separates_method_neighbor_count_and_seed(self):
        variants = {self.step(method, neighbors, seed).variant
                    for method, neighbors, seed in [
                        ("independent", 1, 42), ("cycle_neighbors", 1, 42),
                        ("cycle_neighbors", 10, 42), ("cycle_neighbors", 10, 43)]}
        self.assertEqual(len(variants), 4)
        self.assertEqual(NilmDatasetStep("tag", "pair.csv").variant,
                         "cycle_augmentation_on_tag")

    def test_run_persists_budget_profiles_and_measurement_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            segments = root / "segments"
            segments.mkdir()
            manifest = RunManifest(str(root / "run_manifest.json"), "test", "appliance")
            assignments, aligned, activities, indices, labels = [], [], {}, [], []
            for activity_id in range(5):
                timestamps = np.arange(6, dtype=np.int64) * 6 + activity_id * 60
                power = np.asarray([10.0 + activity_id] * 3
                                   + [20.0 + activity_id] * 3, dtype=np.float32)
                filename = f"activity_{activity_id:05d}.csv"
                pd.DataFrame({"timestamp": timestamps, "power": power}).to_csv(
                    segments / filename, index=False)
                aligned.append(pd.DataFrame({"timestamp": timestamps, "appliance": power,
                                             "mains": power + 50}))
                split = "train" if activity_id < 3 else ("validation" if activity_id == 3 else "test")
                assignments.append({"activity_id": str(activity_id), "file": filename,
                                    "class_id": 0, "mode_id": 0, "split": split})
                if split == "train":
                    activities[str(activity_id)] = {
                        "class_id": 0, "validation_mode_id": 0,
                        "blocks": [{"state_label": 0, "length_samples": 3},
                                   {"state_label": 1, "length_samples": 3}],
                    }
                for state in (0, 1):
                    indices.append([activity_id, state * 3, state])
                    labels.append(state)
            aligned_path = root / "pair.csv"
            pd.concat(aligned).to_csv(aligned_path, index=False)
            Path(str(aligned_path) + ".audit.json").write_text(json.dumps({
                "output_path": str(aligned_path), "mains_power_type": "active",
                "appliance_power_type": "active",
                "measurement_compatible_for_additive_synthesis": True,
            }))
            pd.DataFrame(assignments).to_csv(root / "assignments.csv", index=False)
            (root / "train_catalog.json").write_text(json.dumps({"activities": activities}))
            for name, values in (("labels", labels), ("indices", indices),
                                 ("seq_len", [3] * len(labels))):
                np.save(root / f"{name}.npy", values)
            manifest.add_step("extract_active_data", "simple", "segments",
                              {"segments_dir": "segments"})
            manifest.add_step("cycle_split", "strict", "split", {
                "assignments": "assignments.csv", "train_catalog": "train_catalog.json",
            }, extra={"structure_fit_scope": "train_only"})
            manifest.add_cluster_result("kmeans_k2_merged", "cluster", {
                name: f"{name}.npy" for name in ("labels", "indices", "seq_len")})
            step = NilmDatasetStep(
                "kmeans_k2_merged", str(aligned_path), real_ratios=(0.34, 1.0),
                synthesis_scope="budget_local", budget_conditioning_method="cycle_neighbors",
                budget_conditioning_neighbors=1, require_train_only_structure=True,
                require_additive_measurement=True)
            step.run({"log_root": str(root), "manifest": manifest})
            audit_path = Path(manifest.artifact_path("nilm_dataset", "dataset_manifest"))
            audit = json.loads(audit_path.read_text())
            generated = json.loads(Path(manifest.artifact_path(
                "nilm_dataset", "budget_synthesis_manifest")).read_text())
            self.assertTrue(audit["measurement_compatible_for_additive_synthesis"])
            self.assertEqual(audit["real_counts"], {
                "train": 3, "validation": 1, "test": 1, "rejected": 0})
            self.assertTrue(audit["budget_leakage_check"]["passed"])
            self.assertEqual(audit["budget_conditioning"]["budgets"]["34pct"][
                "singleton_fallback_count"], 1)
            self.assertEqual(audit["budget_conditioning"]["budgets"]["100pct"][
                "cross_cycle_generated_count"], 3)
            for tag, rows in generated.items():
                selected = {int(value) for value in audit["experiments"][tag][
                    "selected_real_activity_ids"]}
                for row in rows:
                    self.assertTrue(set(row["conditioning_fit_activity_ids"]) <= selected)
                    self.assertTrue(set(row["primitive_source_activity_ids"]) <= selected)
                    self.assertNotIn(3, row["donor_activity_ids"])
                    self.assertNotIn(4, row["donor_activity_ids"])
                    self.assertTrue((audit_path.parent / row["file"]).is_file())


class BudgetResourceValidationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.segments = self.root / "segments"
        self.segments.mkdir()
        self.power = np.asarray([10, 10, 10, 20, 20, 20], dtype=np.float32)
        self.write_power(self.power)
        (self.root / "catalog.json").write_text(json.dumps({"activities": {"0": {
            "blocks": [{"state_label": 0, "length_samples": 3},
                       {"state_label": 1, "length_samples": 3}],
        }}}))
        self.manifest = RunManifest(str(self.root / "run_manifest.json"))
        self.manifest.add_step("extract_active_data", "simple", "segments",
                               {"segments_dir": "segments"})
        self.manifest.add_step("cycle_split", "train", "split",
                               {"train_catalog": "catalog.json"})
        self.manifest.add_cluster_result("tag", "cluster", {
            name: f"{name}.npy" for name in ("labels", "indices", "seq_len")})
        self.step = NilmDatasetStep("tag", "pair.csv", synthesis_scope="budget_local")
        self.context = {"manifest": self.manifest}
        self.write_arrays()

    def write_power(self, power):
        pd.DataFrame({"timestamp": np.arange(len(power)) * 6, "power": power}).to_csv(
            self.segments / "activity_00000.csv", index=False)

    def write_arrays(self, indices=None, lengths=None, labels=None):
        # The out-of-scope row has no corresponding file and corrupt geometry.
        values = {
            "indices": np.asarray([[0, 0, 0], [0, 3, 1], [99, np.nan, np.inf]])
                if indices is None else indices,
            "seq_len": np.asarray([3, 3, np.nan]) if lengths is None else lengths,
            "labels": np.asarray([0, 1, np.nan]) if labels is None else labels,
        }
        for name, array in values.items():
            np.save(self.root / f"{name}.npy", array)

    def test_non_training_source_geometry_is_not_loaded_or_repaired(self):
        _, primitives, waveforms = self.step._budget_resources(self.context, {0})
        self.assertEqual([primitive.activity_index for primitive in primitives], [0, 0])
        self.assertEqual(set(waveforms), {0})
        np.testing.assert_array_equal(waveforms[0], self.power)

    def test_two_column_indices_remain_supported(self):
        self.write_arrays(indices=np.asarray([[0, 0], [0, 3]]),
                          lengths=np.asarray([3, 3]), labels=np.asarray([0, 1]))
        _, primitives, _ = self.step._budget_resources(self.context, {0})
        self.assertEqual(len(primitives), 2)

    def test_invalid_index_dimensions_have_explicit_error(self):
        for indices in (np.array(0), np.zeros(3), np.zeros((3, 1)), np.zeros((3, 4))):
            with self.subTest(shape=indices.shape):
                self.write_arrays(indices=indices)
                with self.assertRaisesRegex(ValueError, "indices must have shape"):
                    self.step._budget_resources(self.context, {0})

    def test_nonfinite_or_fractional_activity_ids_are_not_int_truncated(self):
        for activity_id in (0.5, np.nan, np.inf):
            with self.subTest(activity_id=activity_id):
                self.write_arrays(indices=np.asarray([[activity_id, 0, 0], [0, 3, 1], [99, 0, 0]]))
                with self.assertRaisesRegex(ValueError, "activity ID must be a finite integer"):
                    self.step._budget_resources(self.context, {0})

    def test_invalid_training_coordinates_labels_and_lengths_are_rejected(self):
        for name, invalid in (("start", 0.5), ("start", np.inf),
                              ("length", 3.5), ("length", np.nan),
                              ("label", 0.5), ("index_label", np.inf)):
            with self.subTest(field=name, invalid=invalid):
                indices = np.asarray([[0, 0, 0], [0, 3, 1], [99, 0, 0]], dtype=float)
                lengths, labels = np.asarray([3., 3., 0.]), np.asarray([0., 1., 0.])
                if name == "start":
                    indices[0, 1] = invalid
                elif name == "length":
                    lengths[0] = invalid
                elif name == "label":
                    labels[0] = invalid
                else:
                    indices[0, 2] = invalid
                self.write_arrays(indices, lengths, labels)
                with self.assertRaisesRegex(ValueError, "must be a finite integer"):
                    self.step._budget_resources(self.context, {0})

    def test_missing_training_file_id_raises_value_error_not_index_error(self):
        self.write_arrays(indices=np.asarray([[7, 0, 0], [0, 3, 1], [99, 0, 0]]))
        with self.assertRaisesRegex(ValueError, "training activity 7 has no extracted file"):
            self.step._budget_resources(self.context, {7})

    def test_training_power_nan_infinity_and_nonnumeric_values_are_not_filled(self):
        for invalid in (np.nan, np.inf, "not_power", 1e100):
            with self.subTest(power=invalid):
                self.write_power([10, invalid, 10, 20, 20, 20])
                with self.assertRaisesRegex(ValueError, "non-finite or non-numeric source power"):
                    self.step._budget_resources(self.context, {0})

    def test_out_of_bounds_and_nonpositive_slices_are_rejected(self):
        for start, length in ((-1, 3), (0, 7), (3, 4), (0, 0), (0, -1)):
            with self.subTest(start=start, length=length):
                self.write_arrays(
                    indices=np.asarray([[0, start, 0], [0, 3, 1], [99, 0, 0]]),
                    lengths=np.asarray([length, 3, 0]))
                with self.assertRaisesRegex(ValueError, "outside source length 6"):
                    self.step._budget_resources(self.context, {0})


class MeasurementAuditTests(unittest.TestCase):
    def test_required_missing_audit_fails_before_dataset_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pair.csv"
            path.touch()
            step = NilmDatasetStep("tag", str(path), require_additive_measurement=True)
            with self.assertRaisesRegex(FileNotFoundError, "requires measurement audit"):
                step.run({})
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_missing_optional_audit_is_unknown_not_certified(self):
        step = NilmDatasetStep("tag", "pair.csv")
        with tempfile.TemporaryDirectory() as directory:
            audit = step._measurement_audit(Path(directory) / "pair.csv")
            self.assertIsNone(audit["measurement_compatible_for_additive_synthesis"])
            self.assertEqual(audit["mains_power_type"], "unknown")

    def test_required_audit_rejects_incompatible_or_mismatched_pair(self):
        for mains_type, compatible, output in [
                ("apparent", False, "same"), ("active", False, "same"),
                ("apparent", True, "same"), ("active", True, "other")]:
            with self.subTest(mains=mains_type, compatible=compatible, output=output), \
                    tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "pair.csv"
                payload = {"mains_power_type": mains_type, "appliance_power_type": "active",
                           "measurement_compatible_for_additive_synthesis": compatible,
                           "output_path": str(path if output == "same" else path.with_name("other.csv"))}
                Path(str(path) + ".audit.json").write_text(json.dumps(payload))
                step = NilmDatasetStep("tag", str(path), require_additive_measurement=True)
                with self.assertRaises(ValueError):
                    step._measurement_audit(path)

    def test_required_compatible_audit_records_types_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pair.csv"
            payload = {"mains_power_type": "active", "appliance_power_type": "active",
                       "measurement_compatible_for_additive_synthesis": True,
                       "output_path": str(path)}
            sidecar = Path(str(path) + ".audit.json")
            sidecar.write_text(json.dumps(payload))
            step = NilmDatasetStep("tag", str(path), require_additive_measurement=True)
            audit = step._measurement_audit(path)
            self.assertTrue(audit["measurement_compatible_for_additive_synthesis"])
            self.assertTrue(audit["output_path_matches"])
            self.assertEqual(audit["audit_path"], str(sidecar.resolve()))


if __name__ == "__main__":
    unittest.main()
