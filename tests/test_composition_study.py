"""Real manifest/CSV/NPY fixture for the complete CPU composition workflow."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from scripts.audit_primitive_composition import audit_composition
from scripts.prepare_downstream_run import prepare
from src.generation.primitive_composition import METHODS, LEGACY_METHODS
from scripts.run_primitive_composition import (
    run_study, write_json, study_status, study_exit_code, validation_availability,
)
from src.steps.temporal_holdout_step import TemporalHoldoutStep


@pytest.fixture
def source_run(tmp_path):
    root = tmp_path / "log" / "source"
    segments = root / "segments"
    segments.mkdir(parents=True)
    activities, indices, labels, sequences, assignments, temporal = {}, [], [], {}, [], []
    for activity_id in range(7):
        split = "train" if activity_id < 5 else "validation" if activity_id == 5 else "test"
        power = np.array([100 + activity_id] * 4 + [1000 + activity_id * 10] * 4, dtype=float)
        filename = f"activity_{activity_id:05d}.csv"
        t = activity_id * 100 + np.arange(8) * 6
        pd.DataFrame({"timestamp": t, "power": power}).to_csv(segments / filename, index=False)
        blocks = [{"state_label": state, "start": state * 4, "end": state * 4 + 4,
                   "length_samples": 4} for state in (0, 1)]
        sequences[str(activity_id)] = blocks
        activities[str(activity_id)] = {"class_id": 0, "validation_mode_id": 0,
                                        "source_split": split, "blocks": blocks}
        assignments.append({"activity_id": str(activity_id), "class_id": 0, "mode_id": 0,
                            "split": split, "file": filename})
        temporal.append({"activity_id": activity_id, "start_timestamp": int(t[0]),
                         "end_timestamp": int(t[-1]), "split": split})
        indices.extend([[activity_id, state * 4, state] for state in (0, 1)])
        labels.extend([0, 1])
    np.save(root / "indices.npy", np.asarray(indices))
    np.save(root / "labels.npy", np.asarray(labels))
    np.save(root / "seq_len.npy", np.full(14, 4))
    write_json(root / "state_sequences.json", sequences)
    for split in ("train", "validation", "test"):
        write_json(root / f"{split}_catalog.json", {
            "source_split": {"name": split, "structure_fit_scope": "train_only"},
            "activities": {key: value for key, value in activities.items() if value["source_split"] == split},
        })
    pd.DataFrame(assignments).to_csv(root / "assignments.csv", index=False)
    pd.DataFrame(temporal).to_csv(root / "temporal.csv", index=False)
    write_json(root / "run_manifest.json", {"run_id": "source", "steps": {
        "extract_active_data": {"artifacts": {"segments_dir": "segments"}},
        "time_clustering": {"results": {"kmeans_k4_merged": {"artifacts": {
            "indices": "indices.npy", "labels": "labels.npy", "seq_len": "seq_len.npy",
            "state_sequences": "state_sequences.json"}}}},
        "temporal_holdout": {"artifacts": {"assignments": "temporal.csv"}},
        "cycle_split": {"extra": {"cluster_tag": "kmeans_k4_merged", "structure_fit_scope": "train_only"},
                        "artifacts": {"assignments": "assignments.csv", **{
                            f"{split}_catalog": f"{split}_catalog.json" for split in ("train", "validation", "test")}}},
    }})
    return root


def study(root, name="study", **options):
    return run_study(root, root / name, ratios=[1.0], max_anchors=3, **options)


def test_complete_cpu_study_all_arms_and_readback_audit(source_run):
    original = (source_run / "run_manifest.json").read_bytes()
    summary = study(source_run)
    assert summary["status"] == "ready_for_descriptive_review"
    assert summary["paired_cases"] == 3 and summary["cases_with_supported_transitions"] == 3
    assert summary["audit"]["passed"] and summary["audit"]["waveform_files_checked"] == 3 * len(METHODS)
    manifest = json.loads((source_run / "study/composition_manifest.json").read_text())
    for case in manifest["cases"]:
        anchor = case["anchor_activity_id"]
        assert anchor not in case["reference_model"]["fit_activity_ids"]
        assert set(case["donor_activity_ids"]) <= set(range(5))
        for row in case["results"].values():
            assert anchor not in row["source_activity_ids"]
            assert set(row["source_activity_ids"]) <= set(range(5))
            with np.load(source_run / "study" / row["file"]) as payload:
                assert len(payload["appliance"]) == 8
                assert "mains" not in payload
    assert {row["validation_cycles"] for row in summary["validation_diagnostics"]} == {1}
    assert (source_run / "run_manifest.json").read_bytes() == original


def test_validation_and_test_power_never_fit_or_select_candidates(source_run):
    first = study(source_run, "first")
    for index in (5, 6):
        path = source_run / "segments" / f"activity_{index:05d}.csv"
        frame = pd.read_csv(path)
        frame["power"] *= 20
        frame.to_csv(path, index=False)
    second = study(source_run, "second")
    a = json.loads((source_run / "first/composition_manifest.json").read_text())
    b = json.loads((source_run / "second/composition_manifest.json").read_text())
    for left, right in zip(a["cases"], b["cases"]):
        assert left["candidates"] == right["candidates"]
        assert left["reference_model"] == right["reference_model"]
        for method in left["results"]:
            assert left["results"][method]["metrics"] == right["results"][method]["metrics"]
            assert left["results"][method]["candidate_indices"] == right["results"][method]["candidate_indices"]
    assert first["input_fingerprint"] != second["input_fingerprint"]
    assert first["validation_diagnostics"] != second["validation_diagnostics"]


def test_nested_budgets_skip_singleton_for_every_arm(source_run):
    result = run_study(source_run, source_run / "nested", ratios=[0.2, 0.4, 1.0], max_anchors=0)
    a, b, c = result["budgets"]
    assert set(a["selected_activity_ids"]) <= set(b["selected_activity_ids"]) <= set(c["selected_activity_ids"])
    assert result["skipped_cases"] == 1
    assert result["paired_cases"] == 7
    assert result["audit"]["passed"]
    rows = [row for row in result["paired_comparisons"] if "40pct" in row["case_id"]]
    assert all(row["supported_edges"] == 0 and row["fallback_edges"] == 1 for row in rows)


def test_empty_study_is_explicit_not_a_successful_method_comparison(source_run):
    result = run_study(source_run, source_run / "empty", ratios=[0.01])
    assert result["status"] == "no_paired_cases" and result["paired_cases"] == 0
    assert result["skipped_cases"] == 1


def test_previous_outputs_cannot_be_overwritten(source_run):
    study(source_run)
    marker = (source_run / "study/composition_summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        study(source_run)
    assert (source_run / "study/composition_summary.json").read_bytes() == marker


def test_output_waveform_tampering_is_detected(source_run):
    study(source_run)
    manifest = json.loads((source_run / "study/composition_manifest.json").read_text())
    path = source_run / "study" / manifest["cases"][0]["results"]["random"]["file"]
    with np.load(path) as payload:
        data = dict(payload)
    data["appliance"][0] += 100
    np.savez_compressed(path, **data)
    audit = audit_composition(source_run / "study")
    assert not audit["passed"]
    assert any("block differs" in error for error in audit["errors"])


def test_candidate_provenance_tampering_is_detected(source_run):
    study(source_run)
    path = source_run / "study/composition_manifest.json"
    manifest = json.loads(path.read_text())
    case = manifest["cases"][0]
    case["candidates"][0][0]["activity_id"] = case["anchor_activity_id"]
    write_json(path, manifest)
    assert not audit_composition(source_run / "study")["passed"]


def test_fallback_provenance_tampering_is_detected(source_run):
    study(source_run)
    path = source_run / "study/composition_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["cases"][0]["results"]["transition_dp"]["transition_fallback_edges"] = [0]
    write_json(path, manifest)
    result = audit_composition(source_run / "study")
    assert not result["passed"]
    assert any("fallback reporting" in error for error in result["errors"])


def test_study_configuration_tampering_is_detected(source_run):
    study(source_run)
    path = source_run / "study/composition_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["configuration"]["max_warp"] = 99
    write_json(path, manifest)
    result = audit_composition(source_run / "study")
    assert not result["passed"]
    assert "configuration differs from input identity" in result["errors"]


def test_stale_source_timestamps_are_rejected(source_run):
    path = source_run / "segments/activity_00000.csv"
    frame = pd.read_csv(path)
    frame["timestamp"] += 1000
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="timestamps no longer match"):
        study(source_run)


def test_misaligned_source_filename_is_rejected(source_run):
    path = source_run / "assignments.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "file"] = "activity_00001.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="filename no longer matches"):
        study(source_run)


@pytest.mark.parametrize("options", [
    {"sample_period": 0}, {"max_warp": float("nan")}, {"min_fit_cycles": 1},
    {"candidates": 0}, {"neighbors": 0}, {"seeds": [-1]},
    {"target_weight": -1}, {"target_weight": float("nan")},
])
def test_bad_parameters_fail_before_writing(source_run, options):
    with pytest.raises(ValueError, match="invalid composition"):
        study(source_run, **options)
    assert not (source_run / "study").exists()


def test_unpurged_overlap_fails_before_creating_study(source_run):
    path = source_run / "temporal.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "end_timestamp"] = 500
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="temporal overlap"):
        study(source_run)
    assert not (source_run / "study").exists()


def test_legacy_non_train_only_structure_is_rejected(source_run):
    path = source_run / "run_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["steps"]["cycle_split"]["extra"]["structure_fit_scope"] = "all_validated_cycles"
    write_json(path, manifest)
    with pytest.raises(ValueError, match="train-only matching"):
        study(source_run)


def test_no_validation_is_generation_only_with_nonzero_exit(source_run):
    path = source_run / "assignments.csv"
    rows = pd.read_csv(path)
    rows[rows["split"] != "validation"].to_csv(path, index=False)
    catalog_path = source_run / "validation_catalog.json"
    catalog = json.loads(catalog_path.read_text())
    catalog["activities"] = {}
    write_json(catalog_path, catalog)
    summary = study(source_run)
    assert summary["audit"]["passed"] and summary["paired_cases"] == 3
    assert summary["status"] == "generation_only_no_validation"
    assert summary["validation_availability"]["cases_without_matching_validation"] == 3
    assert summary["source_protocol"]["retained_split_counts"] == {"train": 5, "validation": 0, "test": 1}
    assert study_exit_code(summary) == 2


def test_nonempty_but_disjoint_validation_is_not_ready(source_run):
    path = source_run / "assignments.csv"
    rows = pd.read_csv(path)
    rows.loc[rows["split"] == "validation", "class_id"] = 99
    rows.to_csv(path, index=False)
    path = source_run / "validation_catalog.json"
    catalog = json.loads(path.read_text())
    catalog["activities"]["5"]["class_id"] = 99
    write_json(path, catalog)
    summary = study(source_run)
    assert summary["status"] == "generation_only_no_matching_validation"
    assert summary["validation_availability"]["validation_cycles"] == 1
    assert study_exit_code(summary) == 2


def test_partial_validation_coverage_is_explicit():
    from types import SimpleNamespace
    cases = [{"class_mode": [0, 0]}, {"class_mode": [1, 0]}]
    coverage = validation_availability(cases, [SimpleNamespace(group=(0, 0))])
    assert coverage["cases_with_matching_validation"] == 1
    assert coverage["groups_without_validation"] == [[1, 0]]
    assert study_status({"passed": True}, cases, 2, coverage) == "partial_validation_coverage"
    assert study_status({"passed": False}, cases, 2, coverage) == "integrity_failed"


def test_composition_independently_checks_source_cohort_timestamps(source_run):
    path = source_run / "run_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["steps"]["temporal_holdout"]["extra"] = {
        "cohort": TemporalHoldoutStep.cohort_window(None, "1970-01-01T00:00:50Z")}
    write_json(path, manifest)
    with pytest.raises(ValueError, match="outside declared cohort"):
        study(source_run)
    assert not (source_run / "study").exists()


def test_missing_training_state_coverage_is_not_silently_repaired(source_run):
    path = source_run / "seq_len.npy"
    lengths = np.load(path)
    lengths[0] = 3
    np.save(path, lengths)
    with pytest.raises(ValueError, match="inherited interface failed"):
        study(source_run)


def test_cli_runs_against_real_fixture_without_tensorflow(source_run):
    project = Path(__file__).resolve().parents[1]
    import os
    environment = {**os.environ, "PYTHONPATH": str(project), "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run([
        sys.executable, "-m", "scripts.run_primitive_composition", "--run-id", "source",
        "--ratios", "1", "--max-anchors", "2", "--output-dir", str(source_run / "cli")],
        cwd=source_run.parents[1], env=environment, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"paired_cases": 2' in result.stdout


def test_actual_downstream_cli_feeds_composition_without_upstream_retraining(tmp_path):
    """Exercise production catalog schemas, not only hand-built study inputs."""
    root = tmp_path / "log" / "inherited"
    segments = root / "segments"
    segments.mkdir(parents=True)
    sequences, indices, labels = {}, [], []
    signature = [0, 1, 2, 1, 2, 0]
    for activity_id in range(80):
        levels = [0, 100 + activity_id % 7, 1000 + 10 * (activity_id % 7)]
        power = np.repeat([levels[state] for state in signature], 12)
        pd.DataFrame({"timestamp": 1000 + activity_id * 600 + np.arange(72) * 6,
                      "power": power}).to_csv(
            segments / f"activity_{activity_id:05d}.csv", index=False)
        sequences[str(activity_id)] = [
            {"state_label": state, "start": position * 12,
             "end": (position + 1) * 12, "length_samples": 12}
            for position, state in enumerate(signature)]
        indices.extend([[activity_id, position * 12, state]
                        for position, state in enumerate(signature)])
        labels.extend(signature)
    np.save(root / "indices.npy", indices)
    np.save(root / "labels.npy", labels)
    np.save(root / "seq_len.npy", np.full(len(labels), 12))
    write_json(root / "state_sequences.json", sequences)
    write_json(root / "run_manifest.json", {"run_id": "inherited", "steps": {
        "extract_active_data": {"artifacts": {"segments_dir": "segments"}},
        "time_clustering": {"results": {"kmeans_k4_merged": {"artifacts": {
            "indices": "indices.npy", "labels": "labels.npy", "seq_len": "seq_len.npy",
            "state_sequences": "state_sequences.json"}}}},
    }})
    original = (root / "run_manifest.json").read_bytes()
    prepare("inherited", "new", tmp_path / "log")
    project = Path(__file__).resolve().parents[1]
    import os
    environment = {**os.environ, "PYTHONPATH": str(project), "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run([
        sys.executable, str(project / "main.py"), "--config",
        str(project / "config/config_ukdale_detsec.yaml"), "--run-id", "new",
        "--cluster-tag", "kmeans_k4_merged", "--segment-method", "prim-glr",
        "--feature-model", "detsec", "--steps",
        "temporal_holdout,cycle_classify,cycle_validate,cycle_split"],
        cwd=tmp_path, env=environment, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    target = tmp_path / "log" / "new"
    summary = run_study(target, target / "composition", ratios=[1.0], max_anchors=2)
    assert summary["paired_cases"] == 2
    assert summary["cases_with_supported_transitions"] == 2
    assert summary["audit"]["passed"]
    assert (root / "run_manifest.json").read_bytes() == original
    manifest = json.loads((target / "run_manifest.json").read_text())
    assert manifest["steps"]["extract_active_data"]["artifacts"]["segments_dir"] == str(segments)
    assert not {"time_segmentation", "feature_extract", "state_merge"} & set(manifest["steps"])


def test_new_evaluation_is_held_out_and_support_stratified(source_run):
    summary = study(source_run)
    evaluation = summary["evaluation"]
    assert evaluation["coverage"]
    assert {row["method"] for row in evaluation["cycle_validation"]} == set(METHODS)
    assert {row["support_status"] for row in evaluation["transition_validation"]} == {"full"}
    assert {row["validation_cycles"] for row in evaluation["transition_validation"]} == {1}
    assert all(row["left_state"] == 0 and row["right_state"] == 1
               for row in evaluation["transition_validation"])
    json.dumps(evaluation, allow_nan=False)


@pytest.mark.parametrize("field", ["duration_target_cost", "unit_selection_objective",
                                  "largest_donor_sample_fraction", "transition_observations"])
def test_new_selection_and_diagnostic_tampering_is_detected(source_run, field):
    study(source_run)
    path = source_run / "study/composition_manifest.json"
    manifest = json.loads(path.read_text())
    row = manifest["cases"][0]["results"]["unit_selection"]
    if field == "transition_observations":
        row[field][0]["signed_jump_watts"] += 20
    else:
        row[field] += 1
    write_json(path, manifest)
    assert not audit_composition(source_run / "study")["passed"]


def test_audit_still_accepts_legacy_four_method_schema(source_run):
    study(source_run)
    path = source_run / "study/composition_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["schema_version"] = 1
    manifest["methods"] = list(LEGACY_METHODS)
    manifest.pop("unit_selection")
    for case in manifest["cases"]:
        case["results"].pop("unit_selection")
        for row in case["results"].values():
            for key in ("duration_target_cost", "unit_selection_objective", "largest_donor_sample_fraction",
                        "nearest_donor_activity_id", "transition_observations"):
                row.pop(key)
    write_json(path, manifest)
    audit = audit_composition(source_run / "study")
    assert audit["passed"] and audit["waveform_files_checked"] == 3 * len(LEGACY_METHODS)
