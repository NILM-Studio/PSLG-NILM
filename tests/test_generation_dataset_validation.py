"""Numerical and provenance validation of the actual downstream NPZ inputs."""
import hashlib
import json

import numpy as np
import pytest

from scripts.audit_budget_dataset import GROUPS
from scripts.validate_generation_run import dataset_report
from scripts.train_nilm_seq2point import resolve_dataset_root


@pytest.fixture
def generated_run(tmp_path):
    dataset, continuous = tmp_path / "dataset", tmp_path / "continuous"
    dataset.mkdir()
    continuous.mkdir()

    def pair(path, start=12, power=100):
        np.savez_compressed(path, timestamp=np.array([start, start + 6]),
                            mains=np.array([200., 200.]), appliance=np.array([power, power]))

    for name in ("real", "traditional", "synthetic"):
        pair(dataset / f"{name}.npz", start=0 if name == "synthetic" else 12)
    pair(continuous / "off.npz", start=0, power=0)
    pair(continuous / "validation.npz", start=60)
    pair(continuous / "test.npz", start=120)
    row = {
        "real_ratio": 0.01, "selected_real_activity_ids": ["1"],
        "synthesis_fit_activity_ids": ["1"], "selected_real_count": 1,
        "synthesis_fit_count": 1, "selected_traditional_count": 1,
        "selected_generated_count": 1,
        GROUPS[0]: ["real.npz"], GROUPS[1]: ["real.npz", "traditional.npz"],
        GROUPS[2]: ["real.npz", "synthetic.npz"],
    }
    cycle = {
        "synthesis_scope": "budget_local", "aligned_series": str(tmp_path / "aligned.csv"),
        "mains_power_type": "active", "appliance_power_type": "active",
        "measurement_compatible_for_additive_synthesis": True,
        "budget_conditioning": {"method": "cycle_neighbors"},
        "experiments": {"01pct": row, "full": {"D_full_real": ["real.npz"]}},
    }
    provenance = {"01pct": [{
        "source_activity_id": "1", "primitive_source_activity_ids": [1],
        "budget_activity_ids": [1], "file": "synthetic.npz",
        "blocks": [{"sources": [{"activity_index": 1}]}],
    }]}
    timeline_row = {"synthesis_fit_activity_ids": ["1"]}
    for index, key in enumerate(GROUPS):
        timeline_row[key] = ["../dataset/" + name for name in row[key]]
        timeline_row[key] += ["off.npz"] * (1 if index == 0 else 2)
    timeline = {
        "synthesis_scope": "budget_local", "aligned_series": cycle["aligned_series"],
        "sample_period_seconds": 6, "active_threshold_watts": 10,
        "temporal_boundaries": {"validation_start": 60, "test_start": 120},
        "experiments": {"01pct": timeline_row, "full": {
            "D_full_real": ["../dataset/real.npz", "off.npz"],
            "validation": ["validation.npz"], "test": ["test.npz"]}},
    }
    run_manifest = {"steps": {
        "nilm_dataset": {"artifacts": {"dataset_manifest": "dataset/nilm_dataset_manifest.json"}},
        "nilm_continuous_dataset": {"artifacts": {"dataset_manifest": "continuous/nilm_dataset_manifest.json"}},
    }}

    def save():
        cycle_path = dataset / "nilm_dataset_manifest.json"
        cycle_path.write_text(json.dumps(cycle))
        timeline["source_cycle_manifest_sha256"] = hashlib.sha256(cycle_path.read_bytes()).hexdigest()
        (dataset / "budget_synthesis_manifest.json").write_text(json.dumps(provenance))
        (continuous / "nilm_dataset_manifest.json").write_text(json.dumps(timeline))
        (tmp_path / "run_manifest.json").write_text(json.dumps(run_manifest))

    save()
    return tmp_path, cycle, timeline, pair, save


def test_actual_dataset_passes_and_exposes_method_and_statistics(generated_run):
    root, *_ = generated_run
    result = dataset_report(root)
    assert result["errors"] == []
    assert result["budget_provenance_passed"]
    assert result["numeric_and_temporal_checks_passed"]
    assert result["synthesis_method"] == "cycle_neighbors"
    assert result["budget_source_diversity"]["01pct"]["anchor_only_fraction"] == 1
    assert result["training_distributions"]["01pct/A_real_only"]["appliance_energy_wh"] > 0


def test_apparent_mains_is_not_certified(generated_run):
    root, cycle, _, _, save = generated_run
    cycle["mains_power_type"] = "apparent"
    cycle["measurement_compatible_for_additive_synthesis"] = False
    save()
    result = dataset_report(root)
    assert not result["numeric_and_temporal_checks_passed"]
    assert any("active/active" in error for error in result["errors"])


def test_wrong_off_pool_is_rejected(generated_run):
    root, _, _, pair, _ = generated_run
    pair(root / "continuous/off.npz", start=0, power=50)
    assert any("OFF background contains" in error for error in dataset_report(root)["errors"])


def test_traditional_train_must_not_reach_validation(generated_run):
    root, _, _, pair, _ = generated_run
    pair(root / "dataset/traditional.npz", start=54)
    assert any("crosses validation" in error for error in dataset_report(root)["errors"])


def test_stale_continuous_source_manifest_rejected(generated_run):
    root, cycle, *_ = generated_run
    cycle["budget_conditioning"]["method"] = "independent"
    (root / "dataset/nilm_dataset_manifest.json").write_text(json.dumps(cycle))
    assert any("hash is missing or stale" in error for error in dataset_report(root)["errors"])


def test_continuous_aligned_input_must_match(generated_run):
    root, _, timeline, _, save = generated_run
    timeline["aligned_series"] = "wrong.csv"
    save()
    assert any("aligned_series differ" in error for error in dataset_report(root)["errors"])


def test_training_cli_prefers_actual_continuous_manifest(generated_run):
    root, *_ = generated_run
    assert resolve_dataset_root(root) == root / "continuous"
    assert resolve_dataset_root(root, str(root / "dataset")) == root / "dataset"
    path = root / "run_manifest.json"
    manifest = json.loads(path.read_text())
    del manifest["steps"]["nilm_continuous_dataset"]
    path.write_text(json.dumps(manifest))
    assert resolve_dataset_root(root) == root / "dataset"
