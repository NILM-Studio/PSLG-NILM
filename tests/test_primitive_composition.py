"""Controlled pools, signed transition learning and exact candidate search."""
import itertools

import numpy as np
import pytest

from src.generation.primitive_composition import (
    METHODS, Candidate, StateBlock, StateCycle, TransitionReference, boundary_costs,
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
