"""Physical units, held-out stratification and honest sample-support reporting."""
import copy
import json

import numpy as np
import pytest

from src.generation.composition_evaluation import evaluate_composition, transition_observations
from src.generation.primitive_composition import StateBlock, StateCycle, waveform_metrics


def cycle(activity_id, states=(0, 1, 0), levels=(100, 1000, 100), group=(0, 0)):
    blocks = tuple(StateBlock(activity_id, index, state, index * 3,
                              np.full(3, level, dtype=float))
                   for index, (state, level) in enumerate(zip(states, levels)))
    return StateCycle(activity_id, *group, blocks)


def case(anchor, *, supported=2, seed=42, budget="100pct", observations=True):
    lengths = [len(block.power) for block in anchor.blocks]
    result = {
        "transition_supported_edges": supported,
        "transition_fallback_edges": list(range(supported, len(lengths) - 1)),
        "single_donor_cycle": True,
        "nearest_donor_resampled_nrmse": 0.1,
        "largest_donor_sample_fraction": 1.0,
        "duration_target_cost": 0.0,
        "metrics": waveform_metrics(anchor.power, lengths, 6),
    }
    if observations:
        result["transition_observations"] = transition_observations(
            anchor.power, lengths, [block.state for block in anchor.blocks])
    return {"case_id": f"seed{seed}_{budget}_anchor{anchor.activity_id}",
            "seed": seed, "budget_tag": budget, "anchor_activity_id": anchor.activity_id,
            "class_mode": list(anchor.group),
            "template": [{"state_label": block.state, "length_samples": len(block.power)}
                         for block in anchor.blocks],
            "candidates": [[{}, {}] for _ in anchor.blocks],
            "results": {"random": copy.deepcopy(result), "transition_dp": copy.deepcopy(result)}}


def test_observations_keep_real_steps_and_slopes_in_physical_units():
    rows = transition_observations([88, 94, 100, 1000, 1012, 1024, 100], [3, 3, 1], [0, 1, 0])
    assert rows[0] == {"left_state": 0, "right_state": 1, "signed_jump_watts": 900,
                       "left_slope_watts_per_second": 1, "right_slope_watts_per_second": 2}
    assert rows[1]["signed_jump_watts"] == -924
    assert rows[1]["left_slope_watts_per_second"] == 2
    assert rows[1]["right_slope_watts_per_second"] == 0
    assert transition_observations([1], [1], [0]) == []


def test_local_windows_never_cross_state_boundaries():
    rows = transition_observations([0, 0, 0, 10, 20, 1000, 1006, 1012, 2000],
                                  [5, 4], [0, 1], window=3)
    assert rows[0]["left_slope_watts_per_second"] == pytest.approx(10 / 6)
    assert rows[0]["right_slope_watts_per_second"] == 1
    assert rows[0]["signed_jump_watts"] == 980


@pytest.mark.parametrize("overrides", [
    {"power": [1, np.nan]}, {"power": [1, -1]}, {"power": [[1, 2]]},
    {"lengths": [1, 0]}, {"lengths": [1.0, 1]}, {"lengths": [1, 2]},
    {"states": [0]}, {"states": [0, 1.2]}, {"sample_period": 0},
    {"sample_period": np.inf}, {"window": 0}, {"window": 1.5},
])
def test_invalid_measurement_inputs_are_rejected(overrides):
    arguments = dict(power=[1, 2], lengths=[1, 1], states=[0, 1])
    arguments.update(overrides)
    with pytest.raises(ValueError):
        transition_observations(**arguments)


def test_transition_validation_keeps_up_and_down_pairs_separate():
    result = evaluate_composition([case(cycle(0))], [], [], [cycle(i) for i in (10, 11, 12)], 6)
    rows = result["transition_validation"]
    assert len(rows) == 4  # two methods, two directed state pairs
    for row in rows:
        expected = 900 if (row["left_state"], row["right_state"]) == (0, 1) else -900
        assert row["signed_jump_watts"]["generated"]["median"] == expected
        assert row["signed_jump_watts"]["validation"]["median"] == expected
        assert row["signed_jump_watts"]["wasserstein"] == 0
        assert row["generated_cycles"] == 1
        assert row["validation_cycles"] == 3
        assert row["status"] == "descriptive_only"


def test_small_validation_support_counts_cycles_not_repeated_edges():
    repeated = cycle(10, states=(0, 1, 0, 1, 0, 1), levels=(100, 1000, 100, 1000, 100, 1000))
    report = evaluate_composition([case(cycle(0))], [], [], [repeated], 6)
    row = next(row for row in report["transition_validation"] if row["left_state"] == 0)
    assert row["signed_jump_watts"]["validation"]["count"] == 3
    assert row["validation_cycles"] == 1
    assert row["status"] == "low_support"
    assert all(row["status"] == "low_support" for row in report["cycle_validation"])


def test_class_modes_and_fallback_support_are_not_pooled():
    records = [case(cycle(0), supported=2), case(cycle(1, levels=(100, 500, 100)), supported=1),
               case(cycle(2, levels=(100, 200, 100)), supported=0),
               case(cycle(3, group=(1, 0), levels=(50, 5000, 50)), supported=2),
               case(cycle(4, group=(0, 1)), supported=2)]
    report = evaluate_composition(records, [], [], [cycle(i) for i in (10, 11, 12)], 6)
    assert report["coverage"][0]["support_cases"] == {"full": 3, "partial": 1, "none": 1, "unknown": 0}
    assert len(report["method_summary"]) == 10
    full = next(row for row in report["transition_validation"]
                if row["class_mode"] == [0, 0] and row["support_status"] == "full"
                and row["left_state"] == 0)
    assert full["signed_jump_watts"]["generated"]["count"] == 1
    assert full["signed_jump_watts"]["generated"]["median"] == 900
    other_groups = [row for row in report["transition_validation"] if row["class_mode"] != [0, 0]]
    assert all(row["status"] == "no_same_group_transition_validation" for row in other_groups)
    assert all(row["signed_jump_watts"]["wasserstein"] is None for row in other_groups)


def test_matching_group_with_no_matching_transition_is_explicit():
    reference = cycle(10, states=(1, 2, 1))
    report = evaluate_composition([case(cycle(0))], [], [], [reference], 6)
    assert all(row["status"] == "no_same_group_transition_validation"
               for row in report["transition_validation"])
    assert all(row["status"] == "low_support" for row in report["cycle_validation"])


def test_missing_legacy_observations_are_not_fabricated():
    old = case(cycle(0), observations=False)
    report = evaluate_composition([old], [], [], [cycle(10)], 6)
    assert all(row["transition_observations_missing_cases"] == 1 for row in report["method_summary"])
    assert all(row["status"] == "no_generated_observations" for row in report["transition_validation"])
    assert all(row["left_slope_watts_per_second"]["generated"]["count"] == 0
               for row in report["transition_validation"])


def test_coverage_retains_failed_budgets_and_nontrivial_choice_count():
    records = [case(cycle(0)), case(cycle(1), seed=7, budget="5pct")]
    records[1]["candidates"] = [[{}, {}], [{}], [{}, {}]]
    skipped = [{"seed": 7, "budget_tag": "5pct", "reason": "missing_donor"},
               {"seed": 7, "budget_tag": "10pct", "reason": "singleton"}]
    budgets = [{"seed": 7, "budget_tag": "5pct", "requested_anchors": 2},
               {"seed": 7, "budget_tag": "10pct", "requested_anchors": 1}]
    report = evaluate_composition(records, skipped, budgets, [], 6)
    rows = {(row["seed"], row["budget_tag"]): row for row in report["coverage"]}
    assert rows[(7, "5pct")]["completed"] == 1
    assert rows[(7, "5pct")]["skipped"] == 1
    assert rows[(7, "5pct")]["requested"] == 2
    assert rows[(7, "5pct")]["cases_with_at_least_three_blocks"] == 1
    assert rows[(7, "5pct")]["cases_with_nontrivial_global_choice"] == 0
    assert rows[(42, "100pct")]["cases_with_nontrivial_global_choice"] == 1
    assert rows[(7, "10pct")]["completed"] == 0
    assert rows[(7, "10pct")]["skip_reasons"] == {"singleton": 1}


def test_optional_replay_metrics_are_omitted_when_not_stored():
    record = case(cycle(0))
    for row in record["results"].values():
        del row["duration_target_cost"]
        del row["largest_donor_sample_fraction"]
    report = evaluate_composition([record], [], [], [], 6)
    assert all("duration_target_cost" not in row for row in report["method_summary"])
    assert all("largest_donor_sample_fraction" not in row for row in report["method_summary"])
    assert all(row["single_donor_cycle_fraction"] == 1 for row in report["method_summary"])


def test_empty_evaluation_and_missing_references_are_strict_json():
    report = evaluate_composition([], [], [{"seed": 42, "budget_tag": "5pct", "requested_anchors": 0}], [], 6)
    assert report["coverage"][0]["completed"] == 0
    assert report["method_summary"] == report["cycle_validation"] == report["transition_validation"] == []
    json.dumps(report, allow_nan=False)
    report = evaluate_composition([case(cycle(0))], [], [], [], 6)
    json.dumps(report, allow_nan=False)
    assert all(row["energy_wh"]["validation"] == {"count": 0, "median": None, "p10": None, "p90": None}
               for row in report["cycle_validation"])
    assert all(definition["descriptive_only"] for definition in report["metric_definitions"].values())


def test_reference_fallback_and_nonfinite_reports_are_handled_honestly():
    record = case(cycle(0))
    for row in record["results"].values():
        del row["transition_supported_edges"]
        del row["transition_fallback_edges"]
    record["reference_model"] = {"transitions": {"0->1": {"supported": True}, "1->0": {"supported": False}}}
    report = evaluate_composition([record], [], [], [], 6)
    assert report["coverage"][0]["support_cases"]["partial"] == 1
    record["results"]["random"]["nearest_donor_resampled_nrmse"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        evaluate_composition([record], [], [], [], 6)


def test_duplicate_validation_ids_and_inconsistent_state_observations_are_rejected():
    with pytest.raises(ValueError, match="duplicate validation"):
        evaluate_composition([], [], [], [cycle(10), cycle(10)], 6)
    record = case(cycle(0))
    record["results"]["random"]["transition_observations"][0]["right_state"] = 2
    with pytest.raises(ValueError, match="state template"):
        evaluate_composition([record], [], [], [], 6)
