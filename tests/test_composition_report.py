"""Read-only saved-output review, including legacy studies and failure guards."""
import json
from pathlib import Path

import pytest

from test_composition_study import source_run  # Shared realistic CSV/NPY fixture.
from scripts.report_primitive_composition import build_report, _representatives
from scripts.run_primitive_composition import run_study, write_json
from src.generation.primitive_composition import METHODS, LEGACY_METHODS


def make_study(root, **kwargs):
    directory = root / "study"
    run_study(root, directory, ratios=kwargs.get("ratios", [1.0]), max_anchors=2)
    return directory


def test_report_plots_saved_five_arm_outputs_without_changing_them(source_run, tmp_path):
    directory = make_study(source_run)
    originals = {path: path.read_bytes() for path in directory.rglob("*") if path.is_file()}
    output = tmp_path / "review"
    report = build_report(directory, output, max_cases=1)
    assert report["methods"] == list(METHODS)
    assert report["case_method_rows"] == 2 * len(METHODS)
    assert len(report["images"]) == 1
    assert Path(report["images"][0]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    text = (output / "report.md").read_text()
    assert "unit_selection" in text and "low_support" in text
    assert "evaluation.transition_validation" in text
    assert "真实训练锚点" in text
    assert "largest_donor_sample_fraction" in (output / "cases.csv").read_text()
    assert all(path.read_bytes() == content for path, content in originals.items())
    with pytest.raises(FileExistsError):
        build_report(directory, output)
    assert all(path.read_bytes() == content for path, content in originals.items())


def test_legacy_four_arm_report_without_inventing_new_metrics(source_run, tmp_path):
    directory = make_study(source_run)
    manifest = json.loads((directory / "composition_manifest.json").read_text())
    manifest["schema_version"] = 1
    manifest["methods"] = list(LEGACY_METHODS)
    manifest.pop("unit_selection")
    for case in manifest["cases"]:
        case.pop("anchor_reference")
        case["results"].pop("unit_selection")
        for row in case["results"].values():
            for key in ("duration_target_cost", "unit_selection_objective", "largest_donor_sample_fraction",
                        "nearest_donor_activity_id", "transition_observations"):
                row.pop(key)
    write_json(directory / "composition_manifest.json", manifest)
    summary = json.loads((directory / "composition_summary.json").read_text())
    summary.pop("evaluation")
    summary["validation_diagnostics"] = [row for row in summary["validation_diagnostics"]
                                          if row["method"] in LEGACY_METHODS]
    write_json(directory / "composition_summary.json", summary)
    report = build_report(directory, tmp_path / "legacy_review", max_cases=1)
    assert report["methods"] == list(LEGACY_METHODS)
    assert report["case_method_rows"] == 2 * len(LEGACY_METHODS)
    assert Path(report["images"][0]).is_file()
    assert "旧版 validation_diagnostics" in Path(report["report"]).read_text()


def test_report_refuses_failed_audit_and_mismatched_summary(source_run, tmp_path):
    directory = make_study(source_run)
    summary_path = directory / "composition_summary.json"
    summary = json.loads(summary_path.read_text())
    original = dict(summary)
    summary["input_fingerprint"] = "stale-study"
    write_json(summary_path, summary)
    with pytest.raises(ValueError, match="fingerprint"):
        build_report(directory, tmp_path / "stale")
    assert not (tmp_path / "stale").exists()
    write_json(summary_path, original)
    manifest = json.loads((directory / "composition_manifest.json").read_text())
    waveform = directory / manifest["cases"][0]["results"]["random"]["file"]
    waveform.write_bytes(waveform.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="audit failed"):
        build_report(directory, tmp_path / "tampered")
    assert not (tmp_path / "tampered").exists()


def test_report_cannot_write_under_waveform_case(source_run):
    directory = make_study(source_run)
    manifest = json.loads((directory / "composition_manifest.json").read_text())
    output = directory / manifest["cases"][0]["case_id"] / "review"
    with pytest.raises(ValueError, match="source waveform"):
        build_report(directory, output)
    assert not output.exists()


def test_empty_study_still_has_coverage_report(source_run, tmp_path):
    directory = make_study(source_run, ratios=[0.01])
    report = build_report(directory, tmp_path / "empty")
    assert report["paired_cases"] == 0 and report["images"] == []
    assert "no_paired_cases" in Path(report["report"]).read_text()


def test_report_warns_when_legacy_ready_status_has_no_validation(source_run, tmp_path):
    directory = make_study(source_run)
    path = directory / "composition_summary.json"
    summary = json.loads(path.read_text())
    # Old releases could label this ready despite all validation references missing.
    summary["status"] = "ready_for_descriptive_review"
    for row in summary["validation_diagnostics"]:
        row["validation_cycles"] = 0
    write_json(path, summary)
    report = build_report(directory, tmp_path / "no_validation", max_cases=0)
    text = Path(report["report"]).read_text()
    assert "同组验证覆盖：0/2" in text
    assert "不能据此比较真实分布或评定方法优劣" in text


def test_plot_sampling_covers_budgets_without_score_selection():
    manifest = {"budgets": [
        {"seed": 42, "budget_tag": tag, "real_ratio": ratio}
        for tag, ratio in [("5pct", 0.05), ("100pct", 1.0)]], "cases": []}
    for tag in ("5pct", "100pct"):
        for group in range(8):
            manifest["cases"].append({"seed": 42, "budget_tag": tag, "class_mode": [group, 0],
                                      "anchor_activity_id": group, "case_id": f"{tag}_{group}"})
    chosen = _representatives(manifest, 2)
    assert {case["budget_tag"] for case in chosen} == {"5pct", "100pct"}
    assert all(case["class_mode"] == [0, 0] for case in chosen)
