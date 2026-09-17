"""Cohort-before-split protocol and real CPU server entry-point regression."""
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import numpy as np
import pandas as pd
import pytest

from scripts.audit_cycle_cohort import audit_cohort
from scripts.prepare_downstream_run import prepare
from scripts.run_primitive_composition import run_study, write_json
from src.framework.run_manifest import RunManifest
from src.steps.temporal_holdout_step import TemporalHoldoutStep


@pytest.fixture
def cohort_project(tmp_path):
    project = Path(__file__).resolve().parents[1]
    source = tmp_path / "log/inherited"
    segments = source / "segments"
    segments.mkdir(parents=True)
    sequences, indices, labels = {}, [], []
    for activity_id in range(100):
        signature = [0, 1, 2, 1, 2, 0] if activity_id < 80 else [3, 1, 3, 1, 3, 1]
        levels = [0, 100 + activity_id % 7, 1000 + 10 * (activity_id % 7), 500]
        power = np.repeat([levels[state] for state in signature], 12)
        pd.DataFrame({"timestamp": 1000 + activity_id * 600 + np.arange(72) * 6,
                      "power": power}).to_csv(segments / f"activity_{activity_id:05d}.csv", index=False)
        sequences[str(activity_id)] = [
            {"state_label": state, "start": position * 12, "end": (position + 1) * 12,
             "length_samples": 12} for position, state in enumerate(signature)]
        indices.extend([[activity_id, position * 12, state] for position, state in enumerate(signature)])
        labels.extend(signature)
    np.save(source / "indices.npy", indices)
    np.save(source / "labels.npy", labels)
    np.save(source / "seq_len.npy", np.full(len(labels), 12))
    write_json(source / "state_sequences.json", sequences)
    write_json(source / "run_manifest.json", {"run_id": "inherited", "steps": {
        "extract_active_data": {"artifacts": {"segments_dir": "segments"}},
        "time_clustering": {"results": {"kmeans_k4_merged": {"artifacts": {
            "indices": "indices.npy", "labels": "labels.npy", "seq_len": "seq_len.npy",
            "state_sequences": "state_sequences.json"}}}},
    }})
    # Execute the actual wrapper in a disposable project. Imports remain the real
    # code, while all run directories and output archives stay under tmp_path.
    (tmp_path / "scripts").mkdir()
    for name in ("run_composition_validation.sh", "run_cohort_composition_validation.sh"):
        shutil.copy2(project / "scripts" / name, tmp_path / "scripts" / name)
    (tmp_path / "main.py").symlink_to(project / "main.py")
    (tmp_path / "config").symlink_to(project / "config", target_is_directory=True)
    (tmp_path / ".git").symlink_to(project / ".git", target_is_directory=True)  # read-only rev-parse
    return tmp_path


def wrapper(root, run_id="bounded", **overrides):
    project = Path(__file__).resolve().parents[1]
    environment = {**os.environ, "PYTHONPATH": str(project), "PYTHONDONTWRITEBYTECODE": "1",
                   "PYTHON_BIN": sys.executable, "SOURCE_RUN_ID": "inherited", "RUN_ID": run_id,
                   "COHORT_START": "", "COHORT_END": "1970-01-01T13:36:40Z",
                   "RATIOS": "1", "SEEDS": "42", "MAX_ANCHORS": "2", "REPORT_CASES": "1",
                   **overrides}
    return subprocess.run(["bash", "scripts/run_cohort_composition_validation.sh"],
                          cwd=root, env=environment, text=True, capture_output=True, timeout=60)


def test_real_server_wrapper_selects_era_before_fitting_and_packages(cohort_project):
    source = cohort_project / "log/inherited"
    original = {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
    result = wrapper(cohort_project)
    assert result.returncode == 0, result.stdout + result.stderr
    root = cohort_project / "log/bounded"
    audit = json.loads((root / "cycle_cohort_audit.json").read_text())
    assert audit["evaluation_ready"] and audit["errors"] == []
    assert "downstream_config" in audit["input_sha256"]
    assert audit["source_cycles"] == 100 and audit["outside_cohort"] == 20
    assert {split: row["temporal_assigned"] for split, row in audit["stages"].items()} == {
        "train": 56, "validation": 8, "test": 16}
    assert all(row["retained"] > 0 for row in audit["stages"].values())
    manifest = RunManifest.load_or_create(str(root / "run_manifest.json"))
    classes = json.loads(Path(manifest.artifact_path("cycle_classification", "cycle_classes")).read_text())
    assert set(classes["activities"]) == {str(i) for i in range(80)}
    assert set(classes["cohort_excluded_activity_ids"]) == {str(i) for i in range(80, 100)}
    assert all(int(key) < 56 for row in classes["classes"] for key in row["fit_member_ids"])
    summary = json.loads((root / "composition/composition_summary.json").read_text())
    assert summary["status"] == "ready_for_descriptive_review"
    assert summary["validation_availability"]["cases_with_matching_validation"] == 2
    assert summary["source_protocol"]["cohort"]["end_timestamp"] == 49000
    assert summary["distinct_anchor_cycles"] == 2
    assert (root / "composition_review/report.md").is_file()
    archive = cohort_project / "log/bounded_diagnostics.tar.gz"
    with tarfile.open(archive) as bundle:
        names = bundle.getnames()
        assert any(name.endswith("cycle_cohort_audit.json") for name in names)
        assert any(name.endswith("composition_manifest.json") for name in names)
        assert any(name.endswith("downstream_config.yaml") for name in names)
        assert not any(name.endswith((".npy", ".npz")) for name in names)
        log = bundle.extractfile("./composition_execution.log").read().decode()
        assert "COMPOSITION_EXIT_STATUS=0" in log
    assert all(path.read_bytes() == content for path, content in original.items())
    repeat = wrapper(cohort_project)
    assert repeat.returncode != 0 and "destination run already exists" in repeat.stderr


def test_wrapper_preserves_failure_reports_and_stops_before_composition(cohort_project):
    # A tiny cohort cannot satisfy unchanged min_class_support=30.
    result = wrapper(cohort_project, COHORT_END="1970-01-01T01:56:40Z")
    assert result.returncode == 2, result.stdout + result.stderr
    root = cohort_project / "log/bounded"
    audit = json.loads((root / "cycle_cohort_audit.json").read_text())
    assert audit["metadata_integrity_passed"] and not audit["evaluation_ready"]
    assert "empty_validation_after_filtering" in audit["blockers"]
    assert not (root / "composition").exists()
    assert (cohort_project / "log/bounded_diagnostics.tar.gz").is_file()


def test_cohort_window_keeps_ids_and_excludes_whole_crossing_intervals(cohort_project):
    # Lower bound excludes activity0 including its existing context. Upper bound
    # intersects activity9, which must be excluded rather than trimmed.
    prepare("inherited", "window", cohort_project / "log")
    root = cohort_project / "log/window"
    context = {"manifest": RunManifest.load_or_create(str(root / "run_manifest.json")),
               "log_root": str(root)}
    step = TemporalHoldoutStep("kmeans_k4_merged", cohort_start="1970-01-01T00:26:40Z",
                               cohort_end="1970-01-01T01:50:00Z")
    step.run(context)
    with open(context["manifest"].artifact_path("temporal_holdout", "assignments")) as stream:
        rows = list(csv.DictReader(stream))
    assert {int(row["activity_id"]) for row in rows if row["split"] in ("train", "validation", "test")} == set(range(1, 9))
    assert len(rows) == 100
    assert next(row for row in rows if row["activity_id"] == "9")["split"] == "outside_cohort"
    assert next(row for row in rows if row["activity_id"] == "1")["file"] == "activity_00001.csv"
    assert all(int(row["start_timestamp"]) >= 1000 for row in rows)
    assert all(int(row["end_timestamp"]) >= 1426 for row in rows)


def test_fractional_interval_end_is_not_rounded_into_cohort(cohort_project):
    source = cohort_project / "log/inherited/segments/activity_00000.csv"
    frame = pd.read_csv(source)
    frame["timestamp"] = frame["timestamp"].astype(float)
    frame.loc[frame.index[-1], "timestamp"] = 1426.75
    frame.to_csv(source, index=False)
    prepare("inherited", "fractional", cohort_project / "log")
    root = cohort_project / "log/fractional"
    context = {"manifest": RunManifest.load_or_create(str(root / "run_manifest.json")),
               "log_root": str(root)}
    step = TemporalHoldoutStep("kmeans_k4_merged", cohort_end="1970-01-01T00:23:46.5Z")
    with pytest.raises(ValueError, match="no activities wholly inside cohort"):
        step.run(context)
    with open(next(root.glob("temporal_holdout_*/temporal_holdout_assignments.csv"))) as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["end_timestamp"] == "1426.75"
    assert rows[0]["split"] == "outside_cohort"


@pytest.mark.parametrize("start,end", [
    (None, "2015-09-08"), (None, "not-a-date"), (None, ""),
    ("2016-01-01T00:00:00Z", "2015-01-01T00:00:00Z"),
    ("2015-01-01T00:00:00Z", "2015-01-01T00:00:00Z"), (None, "NaT"),
])
def test_ambiguous_or_invalid_cohort_bounds_are_rejected(start, end):
    with pytest.raises(ValueError):
        TemporalHoldoutStep("kmeans_k4_merged", cohort_start=start, cohort_end=end)


def test_empty_cohort_leaves_explicit_summary(cohort_project):
    result = wrapper(cohort_project, COHORT_END="1970-01-01T00:00:00Z")
    assert result.returncode != 0 and "no activities wholly inside cohort" in result.stdout
    root = cohort_project / "log/bounded"
    summary = json.loads(next(root.glob("temporal_holdout_*/temporal_holdout_summary.json")).read_text())
    assert summary["cohort_activities_before_boundary_purge"] == 0
    assert summary["cohort_excluded_count"] == 100
    assert not (root / "composition").exists()
    assert (cohort_project / "log/bounded_diagnostics.tar.gz").is_file()
