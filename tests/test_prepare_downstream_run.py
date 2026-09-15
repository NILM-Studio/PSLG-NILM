"""Creation of isolated downstream manifests without changing source artifacts."""
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_downstream_run import UPSTREAM_STEPS, prepare


class PrepareDownstreamRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.log = self.root / "log"
        self.source = self.log / "upstream"
        self.source.mkdir(parents=True)
        self.signal = self.source / "activities" / "activity.csv"
        self.signal.parent.mkdir()
        self.signal.write_text("timestamp,power\n0,0\n6,20\n", encoding="utf-8")
        external = self.root / "external_features.npy"
        external.write_bytes(b"existing upstream bytes")
        self.manifest = {
            "run_id": "upstream", "appliance": "washing_machine",
            "variants": {"feature_model": "detsec", "cluster_method": "kmeans"},
            "config": {"extract_active_data": {"threshold": 10}},
            "config_path": "config/config_ukdale_detsec.yaml",
            "config_sha256": "original-config-digest",
            "status": "old_downstream_finished",
            "steps": {
                "extract_active_data": {"variant": "simple", "artifacts": {
                    "segments_dir": "activities"}},
                "time_segmentation": {"artifacts": {"indices": "segment/indices.npy"}},
                "feature_extract": {"artifacts": {"features": str(external)}},
                "time_clustering": {"results": {
                    "kmeans_k4": {"artifacts": {"labels": "cluster/labels.npy"}},
                    "kmeans_k4_merged": {"artifacts": {
                        "state_sequences": "state_merge/merged/state_sequences.json",
                        "indices": "state_merge/merged/indices.npy"}},
                }},
                "state_merge": {"artifacts": {"summary": "state_merge/summary.json"}},
                **{name: {"artifacts": {"stale": "old_output.json"}} for name in (
                    "temporal_holdout", "cycle_classification", "cycle_validation",
                    "cycle_split", "primitive_synthesis", "nilm_dataset",
                    "nilm_continuous_dataset", "nilm_seq2point", "unrecognized_step")},
            },
        }
        self.source_manifest = self.source / "run_manifest.json"
        self.source_manifest.write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_only_upstream_is_linked_with_absolute_nested_artifacts_and_provenance(self):
        original_manifest = self.source_manifest.read_bytes()
        original_signal = self.signal.read_bytes()
        original_tree = sorted(path.relative_to(self.source) for path in self.source.rglob("*"))
        summary = prepare("upstream", "new_run", self.log)
        destination = Path(summary["manifest_path"])
        result = json.loads(destination.read_text(encoding="utf-8"))
        self.assertEqual(result["run_id"], "new_run")
        self.assertEqual(set(result["steps"]), set(UPSTREAM_STEPS))
        self.assertNotIn("status", result)
        self.assertEqual(result["variants"], self.manifest["variants"])
        self.assertEqual(result["config"], self.manifest["config"])
        self.assertEqual(result["config_sha256"], self.manifest["config_sha256"])
        self.assertEqual(result["steps"]["extract_active_data"]["artifacts"]["segments_dir"],
                         str(self.signal.parent))
        clusters = result["steps"]["time_clustering"]["results"]
        self.assertEqual(clusters["kmeans_k4"]["artifacts"]["labels"],
                         str(self.source / "cluster/labels.npy"))
        self.assertEqual(clusters["kmeans_k4_merged"]["artifacts"]["state_sequences"],
                         str(self.source / "state_merge/merged/state_sequences.json"))
        self.assertEqual(result["steps"]["feature_extract"]["artifacts"],
                         self.manifest["steps"]["feature_extract"]["artifacts"])
        inherited = result["inherited_upstream"]
        self.assertEqual(inherited["source_manifest"], str(self.source_manifest))
        self.assertEqual(inherited["source_manifest_sha256"],
                         hashlib.sha256(original_manifest).hexdigest())
        self.assertEqual(inherited["ownership"], "inherited_senior_work")
        self.assertEqual(inherited["fit_scope"], "not_verified")
        self.assertFalse(inherited["artifacts_copied"])
        self.assertFalse(inherited["artifact_immutability_guaranteed"])
        self.assertEqual([path.name for path in destination.parent.iterdir()],
                         ["run_manifest.json"])
        self.assertEqual(self.source_manifest.read_bytes(), original_manifest)
        self.assertEqual(self.signal.read_bytes(), original_signal)
        self.assertEqual(sorted(path.relative_to(self.source) for path in self.source.rglob("*")),
                         original_tree)

    def test_existing_destination_and_same_run_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "must differ"):
            prepare("upstream", "upstream", self.log)
        destination = self.log / "existing"
        destination.mkdir()
        with self.assertRaises(FileExistsError):
            prepare("upstream", "existing", self.log)
        prepare("upstream", "once", self.log)
        original = (self.log / "once/run_manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            prepare("upstream", "once", self.log)
        self.assertEqual((self.log / "once/run_manifest.json").read_bytes(), original)

    def test_traversal_and_invalid_ids_are_rejected_before_writes(self):
        invalid = ("", ".", "..", "../escape", "nested/run", "nested\\run",
                   "/absolute", "C:\\absolute", "run\nname", " spaced ")
        before = set(self.log.iterdir())
        for run_id in invalid:
            for source_bad in (True, False):
                with self.subTest(run_id=run_id, source_bad=source_bad):
                    with self.assertRaisesRegex(ValueError, "single non-empty path component"):
                        prepare(run_id if source_bad else "upstream",
                                "new_run" if source_bad else run_id, self.log)
        self.assertEqual(set(self.log.iterdir()), before)

    def test_broken_symlink_destination_is_not_followed(self):
        destination = self.log / "linked"
        destination.symlink_to(self.root / "absent", target_is_directory=True)
        with self.assertRaises(FileExistsError):
            prepare("upstream", "linked", self.log)
        self.assertTrue(destination.is_symlink())
        self.assertFalse((self.root / "absent").exists())

    def test_invalid_source_artifact_does_not_create_destination(self):
        self.manifest["steps"]["feature_extract"]["artifacts"]["features"] = 42
        self.source_manifest.write_text(json.dumps(self.manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "invalid inherited artifact path"):
            prepare("upstream", "bad_input", self.log)
        self.assertFalse((self.log / "bad_input").exists())

    def test_cli_emits_reference_only_summary(self):
        script = Path(__file__).resolve().parents[1] / "scripts/prepare_downstream_run.py"
        completed = subprocess.run(
            [sys.executable, str(script), "--source-run-id", "upstream", "--run-id", "cli_run"],
            cwd=self.root, check=True, text=True, capture_output=True)
        summary = json.loads(completed.stdout)
        self.assertEqual(summary["run_id"], "cli_run")
        self.assertIn("no upstream data were copied or modified", summary["note"])
        self.assertIn("does not make the referenced artifacts immutable", summary["note"])


if __name__ == "__main__":
    unittest.main()
