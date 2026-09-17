"""Controlled pools, signed transition learning and exact candidate search."""
import itertools

import numpy as np
import pytest

from src.generation.primitive_composition import (
    LEGACY_METHODS, METHODS, Candidate, StateBlock, StateCycle, TransitionReference, boundary_costs,
    candidate_lattice, compose, lattice_digest, shortest_path, transition_features,
    waveform_metrics,
)


def cycle(activity_id, levels=(100, 1000), group=(0, 0), lengths=(4, 4)):
    cursor, blocks = 0, []
    for index, (level, length) in enumerate(zip(levels, lengths)):
        blocks.append(StateBlock(activity_id, index, index, cursor,
                                 np.full(length, level, dtype=float), (activity_id * 10 + index,)))
        cursor += length
    return StateCycle(activity_id, *group, tuple(blocks))


def test_candidate_pools_are_deterministic_length_matched_and_exclude_anchor():
    anchor = cycle(0, lengths=(6, 3))
    donors = [cycle(index, (100 + index, 1000 + index)) for index in range(1, 5)]
    a, reason = candidate_lattice(anchor, donors, seed=42, candidates=3)
    b, _ = candidate_lattice(anchor, list(reversed(donors)), seed=42, candidates=3)
    assert reason is None and lattice_digest(a) == lattice_digest(b)
    assert [len(options) for options in a] == [3, 3]
    for template, options in zip(anchor.blocks, a):
        for item in options:
            assert item.source.activity_id != 0
            assert item.source.state == template.state
            assert len(item.power) == len(template.power)


@pytest.mark.parametrize("donors", [[cycle(0)], [cycle(2, group=(1, 0))], [cycle(2, group=(0, 1))]])
def test_anchor_and_cross_class_or_mode_donors_are_rejected(donors):
    with pytest.raises(ValueError, match="exclude anchor"):
        candidate_lattice(cycle(0), donors)


def test_singletons_are_skipped_not_silently_self_sampled():
    lattice, reason = candidate_lattice(cycle(0), [])
    assert lattice == [] and "no_non_anchor" in reason


def test_duration_support_is_checked_before_any_arm_runs():
    lattice, reason = candidate_lattice(cycle(0, lengths=(100, 4)), [cycle(1)], max_warp=2)
    assert lattice == [] and "block_0" in reason


def test_reference_learns_signed_transition_not_zero_jump():
    training = [cycle(index, (100 + index, 1000 + 10 * index)) for index in range(1, 5)]
    model = TransitionReference(training, min_cycles=3)
    left = [Candidate(cycle(8).blocks[0], np.full(4, 102.))]
    right = [Candidate(cycle(9).blocks[1], np.full(4, 103.)),
             Candidate(cycle(10).blocks[1], np.full(4, 1020.))]
    boundary = boundary_costs(left, right)
    learned, fallback = model.costs(left, right)
    assert not fallback
    assert boundary[0, 0] < boundary[0, 1]
    assert learned[0, 1] < learned[0, 0]
    output = compose([left, right], model)
    assert output["boundary_dp"]["candidate_indices"] == [0, 0]
    assert output["transition_dp"]["candidate_indices"] == [0, 1]
    assert output["transition_dp"]["power"][4] == 1020  # no smoothing away the real step


def test_low_support_falls_back_explicitly_to_boundary_dp():
    donors = [cycle(1), cycle(2)]
    lattice, _ = candidate_lattice(cycle(0), donors)
    model = TransitionReference(donors, min_cycles=3)
    output = compose(lattice, model)
    assert output["transition_dp"]["transition_fallback_edges"] == [0]
    assert output["transition_dp"]["transition_supported_edges"] == 0
    np.testing.assert_array_equal(output["transition_dp"]["power"], output["boundary_dp"]["power"])


def test_support_counts_independent_cycles_not_repeated_edges():
    original = cycle(1)
    repeated = StateCycle(1, 0, 0, original.blocks * 20)
    model = TransitionReference([repeated], min_cycles=3)
    assert model.summary()["transitions"]["0->1"]["observations"] == 20
    assert not model.supported((0, 1))


def test_reference_cannot_mix_class_modes():
    with pytest.raises(ValueError, match="mix class/mode"):
        TransitionReference([cycle(1), cycle(2, group=(0, 1))])


def test_dp_matches_exhaustive_search_with_fixed_first_block():
    rng = np.random.default_rng(2)
    costs = [rng.uniform(0, 10, (3, 2)), rng.uniform(0, 10, (2, 4))]
    path, cost = shortest_path(costs, [3, 2, 4], first_index=1)
    brute = [(costs[0][1, a] + costs[1][a, b], [1, a, b])
             for a, b in itertools.product(range(2), range(4))]
    expected, expected_path = min(brute)
    assert cost == pytest.approx(expected) and path == expected_path


def test_all_arms_share_first_candidate_and_are_deterministic():
    donors = [cycle(index, (100 + index, 1000 + 5 * index)) for index in range(1, 8)]
    lattice, _ = candidate_lattice(cycle(0), donors)
    reference = TransitionReference(donors)
    output = compose(lattice, reference, seed=3)
    repeated = compose(lattice, reference, seed=3)
    assert set(output) == set(METHODS)
    assert len({row["candidate_indices"][0] for row in output.values()}) == 1
    for method, row in output.items():
        assert row["candidate_indices"] == repeated[method]["candidate_indices"]
        assert len(row["power"]) == 8
    assert output["boundary_dp"]["boundary_objective"] <= output["boundary_greedy"]["boundary_objective"] + 1e-9
    assert output["transition_dp"]["transition_objective"] <= output["boundary_dp"]["transition_objective"] + 1e-9


@pytest.mark.parametrize("power", [[np.nan], [-1], [], [[1, 2]]])
def test_invalid_signals_are_rejected(power):
    with pytest.raises(ValueError, match="state power"):
        transition_features(power, [10])


def test_metrics_preserve_signed_jumps_and_energy():
    metrics = waveform_metrics(np.array([10, 10, 100, 100, 0, 0]), [2, 2, 2], 6)
    assert metrics["signed_boundary_jumps_watts"] == [90, -100]
    assert metrics["energy_wh"] == pytest.approx(220 * 6 / 3600)
    assert metrics["duration_seconds"] == 36


def test_dp_with_unary_costs_matches_exhaustive_search():
    rng = np.random.default_rng(19)
    sizes = [3, 2, 4, 3]
    edges = [rng.uniform(0, 10, (a, b)) for a, b in zip(sizes, sizes[1:])]
    nodes = [rng.uniform(0, 10, size) for size in sizes]
    path, cost = shortest_path(edges, sizes, first_index=2, node_costs=nodes)
    brute = []
    for suffix in itertools.product(*(range(size) for size in sizes[1:])):
        candidate = [2, *suffix]
        objective = sum(edge[a, b] for edge, a, b in zip(edges, candidate, candidate[1:]))
        objective += sum(node[index] for node, index in zip(nodes, candidate))
        brute.append((objective, candidate))
    expected, expected_path = min(brute)
    assert path == expected_path
    assert cost == pytest.approx(expected)


@pytest.mark.parametrize("nodes", [[], [[0, 1]], [[0], [0]], [[0, 1], [np.nan]],
                                   [[0, 1], [np.inf]], [[[0, 1]], [0]]])
def test_dp_rejects_invalid_unary_costs(nodes):
    with pytest.raises(ValueError, match="node cost"):
        shortest_path([np.zeros((2, 1))], [2, 1], 0, node_costs=nodes)


def test_single_block_dp_includes_cost_of_fixed_first_candidate():
    assert shortest_path([], [3], 1, node_costs=[[0., 2., 1.]]) == ([1], 2.)


@pytest.mark.parametrize("weight", [-1, np.nan, np.inf, -np.inf])
def test_composition_rejects_invalid_target_weights(weight):
    lattice, _ = candidate_lattice(cycle(0), [cycle(1)])
    with pytest.raises(ValueError, match="target_weight"):
        compose(lattice, TransitionReference([]), target_weight=weight)


@pytest.mark.parametrize("supported", [False, True])
def test_zero_target_weight_exactly_matches_transition_dp(supported):
    donors = [cycle(index, (90 + 20 * index, 950 + 40 * index),
                    lengths=(3 + index % 4, 3 + (index + 1) % 4))
              for index in range(1, 8)]
    lattice, _ = candidate_lattice(cycle(0), donors)
    reference = TransitionReference(donors if supported else [])
    output = compose(lattice, reference, target_weight=0)
    a, b = output["unit_selection"], output["transition_dp"]
    assert a["candidate_indices"] == b["candidate_indices"]
    assert a["transition_objective"] == b["transition_objective"]
    assert a["unit_selection_objective"] == b["unit_selection_objective"]
    assert a["duration_target_cost"] == b["duration_target_cost"]
    assert a["transition_fallback_edges"] == b["transition_fallback_edges"]
    np.testing.assert_array_equal(a["power"], b["power"])


def test_unit_selection_trades_join_cost_against_duration_distortion():
    left = [Candidate(cycle(1).blocks[0], np.full(4, 100.))]
    right = [Candidate(cycle(2, lengths=(4, 2)).blocks[1], np.full(4, 100.)),
             Candidate(cycle(3).blocks[1], np.full(4, 101.))]
    output = compose([left, right], TransitionReference([]), target_weight=2.)
    transition, unit = output["transition_dp"], output["unit_selection"]
    assert transition["candidate_indices"] == [0, 0]
    assert unit["candidate_indices"] == [0, 1]
    assert unit["transition_objective"] > transition["transition_objective"]
    assert unit["duration_target_cost"] < transition["duration_target_cost"]
    assert unit["unit_selection_objective"] < transition["unit_selection_objective"]
    for row in output.values():
        expected_duration = sum(abs(np.log(len(item.power) / len(item.source.power)))
                                for item in row["selected"])
        assert row["duration_target_cost"] == pytest.approx(expected_duration)
        assert row["unit_selection_objective"] == pytest.approx(
            row["transition_objective"] + 2. * expected_duration)
        assert row["candidate_indices"][0] == 0
        # A nonzero chosen step remains intact, with no seam smoothing.
        np.testing.assert_array_equal(row["power"], np.concatenate(
            [item.power for item in row["selected"]]))
    assert unit["power"][4] - unit["power"][3] == 1.


def test_anchor_power_is_not_used_by_unit_selection():
    donors = [cycle(index, lengths=(3 + index % 4, 4)) for index in range(1, 7)]
    ordinary, _ = candidate_lattice(cycle(0), donors)
    changed_power, _ = candidate_lattice(cycle(0, levels=(0, 100000)), donors)
    assert lattice_digest(ordinary) == lattice_digest(changed_power)
    reference = TransitionReference(donors)
    a = compose(ordinary, reference)["unit_selection"]
    b = compose(changed_power, reference)["unit_selection"]
    assert a["candidate_indices"] == b["candidate_indices"]
    assert a["unit_selection_objective"] == b["unit_selection_objective"]
    np.testing.assert_array_equal(a["power"], b["power"])


def test_legacy_methods_keep_existing_selection_and_objectives():
    anchor = StateCycle(0, 0, 0, tuple(
        StateBlock(0, i, i, 4 * i, np.full(4, 100 + 300 * i)) for i in range(3)))
    donors = [StateCycle(j, 0, 0, tuple(
        StateBlock(j, i, i, 4 * i, np.array([
            100 + 300 * i + 20 * j, 100 + 300 * i + 10 * j,
            110 + 300 * i + 15 * j, 120 + 300 * i + 5 * j], dtype=float))
        for i in range(3))) for j in range(1, 8)]
    lattice, _ = candidate_lattice(anchor, donors, candidates=5)
    output = compose(lattice, TransitionReference(donors), seed=3)
    # Frozen before adding unary costs to the solver or the fifth method.
    expected = {
        "random": ([4, 3, 1], 11.447170203904761, 0.7993896126027737),
        "boundary_greedy": ([4, 2, 0], 11.27385576310576, 1.6392164830693856),
        "boundary_dp": ([4, 2, 0], 11.27385576310576, 1.6392164830693856),
        "transition_dp": ([4, 0, 3], 11.858470702866196, 0.3531377078312339),
    }
    assert tuple(expected) == LEGACY_METHODS
    assert METHODS == LEGACY_METHODS + ("unit_selection",)
    for method, (path, boundary, transition) in expected.items():
        assert output[method]["candidate_indices"] == path
        assert output[method]["boundary_objective"] == pytest.approx(boundary)
        assert output[method]["transition_objective"] == pytest.approx(transition)
