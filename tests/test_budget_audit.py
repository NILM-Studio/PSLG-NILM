import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_budget_dataset import GROUPS, audit


class BudgetAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dataset = Path(self.temp.name) / "dataset"
        self.continuous = Path(self.temp.name) / "continuous"
        self.dataset.mkdir()
        self.continuous.mkdir()
        for name in ("real.npz", "traditional.npz", "synthetic.npz"):
            (self.dataset / name).touch()
        (self.continuous / "off.npz").touch()
        self.row = {
            "real_ratio": 0.01, "selected_real_activity_ids": ["1"],
            "synthesis_fit_activity_ids": ["1"], "selected_real_count": 1,
            "synthesis_fit_count": 1, "selected_traditional_count": 1,
            "selected_generated_count": 1,
            GROUPS[0]: ["real.npz"],
            GROUPS[1]: ["real.npz", "traditional.npz"],
            GROUPS[2]: ["real.npz", "synthetic.npz"],
        }
        self.provenance = {"01pct": [{
            "source_activity_id": "1", "primitive_source_activity_ids": [1],
            "budget_activity_ids": [1], "file": "synthetic.npz",
            "blocks": [{"sources": [{"activity_index": 1}]}],
        }]}
        self.timeline_row = {"synthesis_fit_activity_ids": ["1"]}
        for index, key in enumerate(GROUPS):
            self.timeline_row[key] = ["../dataset/" + name for name in self.row[key]]
            self.timeline_row[key] += ["off.npz"] * (1 if index == 0 else 2)

    def run_audit(self):
        for root, name, payload in (
            (self.dataset, "nilm_dataset_manifest.json", {
                "synthesis_scope": "budget_local", "experiments": {"01pct": self.row}}),
            (self.dataset, "budget_synthesis_manifest.json", self.provenance),
            (self.continuous, "nilm_dataset_manifest.json", {
                "synthesis_scope": "budget_local", "experiments": {"01pct": self.timeline_row}}),
        ):
            (root / name).write_text(json.dumps(payload))
        return audit(self.dataset, self.continuous)

    def test_accepts_matching_sources_and_shared_backgrounds(self):
        self.assertTrue(self.run_audit()["budget_provenance_passed"])

    def test_catches_hidden_source_even_when_summary_says_valid(self):
        self.provenance["01pct"][0]["blocks"][0]["sources"][0]["activity_index"] = 99
        result = self.run_audit()
        self.assertFalse(result["budget_provenance_passed"])
        self.assertTrue(any("out-of-budget" in error for error in result["errors"]))

    def test_catches_stale_continuous_input(self):
        self.timeline_row[GROUPS[2]] = self.timeline_row[GROUPS[1]]
        self.assertFalse(self.run_audit()["budget_provenance_passed"])

    def test_catches_missing_file(self):
        (self.dataset / "synthetic.npz").unlink()
        self.assertFalse(self.run_audit()["budget_provenance_passed"])
