"""Reconcile an independently valid holdout lost solely to class-level gating."""
import csv
import json

import pytest

from scripts.audit_cycle_cohort import audit_cohort
from scripts.run_primitive_composition import write_json
from src.steps.temporal_holdout_step import TemporalHoldoutStep


@pytest.fixture
def audit_run(tmp_path):
    cohort = TemporalHoldoutStep.cohort_window(None, "1970-01-01T00:06:40Z")
    steps = {}

    def add(step, key, value, filename):
        steps.setdefault(step, {"artifacts": {}})["artifacts"][key] = filename
        if filename.endswith(".csv"):
            with (tmp_path / filename).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(value[0]))
                writer.writeheader()
                writer.writerows(value)
        else:
            write_json(tmp_path / filename, value)

    splits = ["train", "train", "validation", "test", "outside_cohort"]
    add("temporal_holdout", "assignments", [
        {"activity_id": str(i), "file": f"{i}.csv", "start_timestamp": i * 100,
         "end_timestamp": i * 100 + 40, "split": split} for i, split in enumerate(splits)], "temporal.csv")
    add("temporal_holdout", "summary", {"cohort": cohort, "counts": {
        "train": 2, "validation": 1, "test": 1}}, "temporal.json")
    steps["temporal_holdout"]["extra"] = {"cohort": cohort}
    activities = {str(i): {"class_id": int(i > 0), "source_split": split,
                          "validation_mode_id": 0} for i, split in enumerate(splits[:4])}
    add("cycle_classification", "cycle_classes", {"fit_scope": "train_only", "activities": activities,
        "classes": [{"class_id": 0, "member_ids": ["0"], "fit_member_ids": ["0"]},
                    {"class_id": 1, "member_ids": ["1", "2", "3"], "fit_member_ids": ["1"]}]}, "classes.json")
    add("cycle_validation", "cycle_report", [
        {"activity_id": str(i), "class_id": int(i > 0), "mode_id": 0,
         "passes_hard_checks": "True", "is_representative_signature": "True",
         "is_valid_member": "True", "rejection_reasons": ""} for i in range(4)], "validity.csv")
    add("cycle_validation", "class_summary", [
        {"class_id": 0, "status": "valid_full"}, {"class_id": 1, "status": "uncertain"}], "class_summary.csv")
    add("cycle_validation", "whitelist", {"valid_class_ids": [0], "valid_activity_ids": ["0"]}, "whitelist.json")
    add("cycle_validation", "validated_cycle_classes", {"validation": {"fit_scope": "train_only"},
        "activities": {"0": activities["0"]}}, "catalog.json")
    add("cycle_split", "assignments", [{"activity_id": "0", "class_id": 0, "mode_id": 0,
                                        "split": "train", "file": "0.csv"}], "split.csv")
    add("cycle_split", "summary", {"structure_fit_scope": "train_only", "counts": {
        "train": 1, "validation": 0, "test": 0}}, "split.json")
    write_json(tmp_path / "run_manifest.json", {"run_id": "fixture", "steps": steps})
    return tmp_path


def test_valid_members_lost_to_class_gate_are_not_treated_as_corrupt_data(audit_run):
    before = {path: path.read_bytes() for path in audit_run.iterdir()}
    report = audit_cohort(audit_run)
    assert report["metadata_integrity_passed"] and not report["evaluation_ready"]
    assert report["status"] == "evaluation_blocked"
    for split in ("validation", "test"):
        row = report["stages"][split]
        assert row["individually_valid"] == row["valid_but_class_excluded"] == 1
        assert row["retained"] == 0 and row["member_rejections"] == {}
        assert row["excluded_valid_by_class"] == {"1": 1}
    assert report["outside_cohort"] == 1
    assert all(path.read_bytes() == data for path, data in before.items())


def test_holdout_must_not_be_declared_a_fit_member(audit_run):
    path = audit_run / "classes.json"
    classes = json.loads(path.read_text())
    classes["classes"][1]["fit_member_ids"].append("2")
    write_json(path, classes)
    report = audit_cohort(audit_run)
    assert not report["metadata_integrity_passed"]
    assert any("fitted or mapped outside" in error for error in report["errors"])


def test_outside_cohort_cannot_reenter_classifier(audit_run):
    path = audit_run / "classes.json"
    classes = json.loads(path.read_text())
    classes["activities"]["4"] = {"class_id": 0, "source_split": "outside_cohort"}
    write_json(path, classes)
    report = audit_cohort(audit_run)
    assert not report["metadata_integrity_passed"]
    assert "classification membership differs from retained temporal cohort" in report["errors"]


def test_cohort_summary_mismatch_is_reported(audit_run):
    path = audit_run / "temporal.json"
    summary = json.loads(path.read_text())
    summary["cohort"]["end_timestamp"] = 10000
    write_json(path, summary)
    report = audit_cohort(audit_run)
    assert not report["metadata_integrity_passed"]
    assert "cohort summary and manifest differ" in report["errors"]


def test_duplicate_activity_rows_are_rejected(audit_run):
    path = audit_run / "temporal.csv"
    text = path.read_text()
    path.write_text(text + text.splitlines()[1] + "\n")
    with pytest.raises(ValueError, match="duplicate activity IDs"):
        audit_cohort(audit_run)
