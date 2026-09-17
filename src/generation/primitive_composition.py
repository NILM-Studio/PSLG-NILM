"""Paired state-block composition with a fixed candidate lattice.

This is a real-waveform composition prototype, not a neural generator. A block
is a contiguous inherited state episode, possibly comprising several original
primitives. All arms use the same linearly length-adapted candidates, with no
seam smoothing. Learned costs describe observed *signed* transitions rather
than imposing continuity on genuine appliance power steps.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np


METHODS = ("random", "boundary_greedy", "boundary_dp", "transition_dp")
FEATURE_NAMES = (
    "log_left_mean", "log_right_mean", "log_left_std", "log_right_std",
    "signed_log_jump", "signed_log_left_slope", "signed_log_right_slope",
    "log_duration_ratio",
)


def _signal(power):
    value = np.asarray(power, dtype=np.float64)
    if value.ndim != 1 or not len(value) or not np.isfinite(value).all() or np.any(value < 0):
        raise ValueError("state power must be a nonempty finite nonnegative vector")
    return value


@dataclass(frozen=True)
class StateBlock:
    activity_id: int
    block_index: int
    state: int
    start: int
    power: np.ndarray
    primitive_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class StateCycle:
    activity_id: int
    class_id: int
    mode_id: int
    blocks: tuple[StateBlock, ...]

    @property
    def group(self):
        return self.class_id, self.mode_id

    @property
    def power(self):
        return np.concatenate([block.power for block in self.blocks])


@dataclass(frozen=True)
class Candidate:
    source: StateBlock
    power: np.ndarray

    def provenance(self):
        return {
            "activity_id": self.source.activity_id,
            "source_block_index": self.source.block_index,
            "source_start": self.source.start,
            "source_length": len(self.source.power),
            "source_primitive_ids": list(self.source.primitive_ids),
            "state_label": self.source.state,
            "target_length": len(self.power),
            "duration_ratio": len(self.power) / len(self.source.power),
            "power_sha256": hashlib.sha256(
                np.asarray(self.power, dtype="<f8").tobytes()).hexdigest(),
        }


def signed_log(value):
    return np.sign(value) * np.log1p(np.abs(value))


def transition_features(left, right, window=5):
    left, right = _signal(left), _signal(right)
    if window < 1:
        raise ValueError("boundary window must be positive")
    a, b = left[-window:], right[:window]
    return np.asarray([
        np.log1p(left.mean()), np.log1p(right.mean()),
        np.log1p(left.std()), np.log1p(right.std()),
        signed_log(right[0] - left[-1]),
        signed_log(float(np.mean(np.diff(a))) if len(a) > 1 else 0),
        signed_log(float(np.mean(np.diff(b))) if len(b) > 1 else 0),
        np.log(len(right) / len(left)),
    ], dtype=np.float64)


class TransitionReference:
    """Robust nearest-neighbor reference fitted to real adjacent state pairs.

    Support is measured in independent source cycles, not number of edges.
    This empirical one-class model is an optimization objective, not an
    independent quality evaluator or a probability of physical validity.
    """
    def __init__(self, cycles, min_cycles=3, neighbors=3, window=5):
        if min_cycles < 2 or neighbors < 1 or window < 1:
            raise ValueError("min_cycles >= 2, neighbors >= 1 and window >= 1 required")
        self.min_cycles, self.neighbors, self.window = min_cycles, neighbors, window
        cycles = sorted(cycles, key=lambda cycle: cycle.activity_id)
        if len({cycle.group for cycle in cycles}) > 1:
            raise ValueError("transition references must not mix class/mode groups")
        self.fit_ids = sorted({cycle.activity_id for cycle in cycles})
        self.models = {}
        rows, sources = {}, {}
        for cycle in cycles:
            for left, right in zip(cycle.blocks, cycle.blocks[1:]):
                key = left.state, right.state
                rows.setdefault(key, []).append(transition_features(left.power, right.power, window))
                sources.setdefault(key, set()).add(cycle.activity_id)
        for key, features in rows.items():
            matrix = np.stack(features)
            center = np.median(matrix, axis=0)
            scale = np.percentile(matrix, 75, axis=0) - np.percentile(matrix, 25, axis=0)
            # A declared floor prevents nearly constant train features from
            # overwhelming all other coordinates. No held-out statistics used.
            scale = np.maximum(scale, 0.1)
            self.models[key] = {"center": center, "scale": scale,
                                "matrix": (matrix - center) / scale,
                                "activity_ids": sorted(sources[key])}

    def supported(self, key):
        return key in self.models and len(self.models[key]["activity_ids"]) >= self.min_cycles

    def costs(self, left_candidates, right_candidates):
        key = left_candidates[0].source.state, right_candidates[0].source.state
        if not self.supported(key):
            return boundary_costs(left_candidates, right_candidates), True
        model = self.models[key]
        features = np.stack([
            transition_features(a.power, b.power, self.window)
            for a in left_candidates for b in right_candidates])
        normalized = (features - model["center"]) / model["scale"]
        # Work row by row to avoid a candidates² × references × features tensor.
        costs = []
        for query in normalized:
            distance = np.mean(np.abs(model["matrix"] - query), axis=1)
            k = min(self.neighbors, len(distance))
            costs.append(float(np.partition(distance, k - 1)[:k].mean()))
        return np.asarray(costs).reshape(len(left_candidates), len(right_candidates)), False

    def summary(self):
        return {
            "method": "robust_transition_knn", "feature_names": list(FEATURE_NAMES),
            "fit_activity_ids": self.fit_ids, "minimum_source_cycles": self.min_cycles,
            "neighbors": self.neighbors, "boundary_window_samples": self.window,
            "scale_floor": 0.1,
            "transitions": {f"{a}->{b}": {
                "source_activity_ids": value["activity_ids"],
                "observations": len(value["matrix"]), "supported": self.supported((a, b)),
                "center": value["center"].tolist(), "scale": value["scale"].tolist(),
            } for (a, b), value in sorted(self.models.items())},
        }


def candidate_lattice(anchor, donors, seed=42, candidates=8, max_warp=2.0):
    """Build once, share across arms. Anchor and cross-group sources are errors."""
    if candidates < 1 or not np.isfinite(max_warp) or max_warp < 1 or seed < 0:
        raise ValueError("candidates >= 1, max_warp >= 1 and nonnegative seed required")
    donors = sorted(donors, key=lambda cycle: cycle.activity_id)
    if any(cycle.activity_id == anchor.activity_id or cycle.group != anchor.group for cycle in donors):
        raise ValueError("donors must exclude anchor and stay in the same class/mode")
    lattice = []
    for block_index, template in enumerate(anchor.blocks):
        length = len(template.power)
        pool = [block for cycle in donors for block in cycle.blocks
                if block.state == template.state
                and 1 / max_warp <= length / len(block.power) <= max_warp]
        if not pool:
            return [], f"no_non_anchor_candidate_for_block_{block_index}_state_{template.state}"
        # Independent of selection strategy and earlier chosen primitive lengths.
        rng = np.random.default_rng(np.random.SeedSequence([seed, anchor.activity_id, block_index, 9127]))
        indices = rng.choice(len(pool), min(candidates, len(pool)), replace=False)
        options = []
        for index in indices:
            source = pool[int(index)]
            signal = _signal(source.power)
            adapted = np.interp(np.linspace(0, 1, length), np.linspace(0, 1, len(signal)), signal)
            options.append(Candidate(source, adapted))
        lattice.append(options)
    return lattice, None


def lattice_digest(lattice):
    payload = [[candidate.provenance() for candidate in options] for options in lattice]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def boundary_costs(left, right):
    # Compress large jumps; this is not a calibrated equivalence to the learned
    # distance. Both boundary arms use the same monotonically transformed
    # endpoint discrepancy (greedy ordering equals raw jump ordering).
    return np.log1p(np.abs(np.asarray([a.power[-1] for a in left])[:, None]
                           - np.asarray([b.power[0] for b in right])[None, :]))


def shortest_path(costs, sizes, first_index):
    """Exact layered dynamic programming, with a common fixed first candidate."""
    if not sizes or len(costs) != len(sizes) - 1 or not 0 <= first_index < sizes[0]:
        raise ValueError("invalid candidate lattice")
    current = np.full(sizes[0], np.inf)
    current[first_index] = 0.0
    parents = []
    for index, edge in enumerate(costs):
        if edge.shape != (sizes[index], sizes[index + 1]) or not np.isfinite(edge).all():
            raise ValueError("invalid transition cost matrix")
        total = current[:, None] + edge
        parent = np.argmin(total, axis=0)
        current = total[parent, np.arange(sizes[index + 1])]
        parents.append(parent)
    last = int(np.argmin(current))
    result = [last]
    for parent in reversed(parents):
        last = int(parent[last])
        result.append(last)
    return list(reversed(result)), float(np.min(current))


def compose(lattice, reference, seed=42, anchor_id=0):
    if not lattice or any(not options for options in lattice):
        raise ValueError("nonempty candidate lattice required")
    rng = np.random.default_rng(np.random.SeedSequence([seed, anchor_id, 83119]))
    random_path = [int(rng.integers(len(options))) for options in lattice]
    first = random_path[0]
    boundary = [boundary_costs(a, b) for a, b in zip(lattice, lattice[1:])]
    transition, fallbacks = [], []
    for a, b in zip(lattice, lattice[1:]):
        costs, fallback = reference.costs(a, b)
        transition.append(costs)
        fallbacks.append(fallback)
    greedy = [first]
    for costs in boundary:
        greedy.append(int(np.argmin(costs[greedy[-1]])))
    sizes = [len(options) for options in lattice]
    boundary_path, _ = shortest_path(boundary, sizes, first)
    transition_path, _ = shortest_path(transition, sizes, first)
    paths = {"random": random_path, "boundary_greedy": greedy,
             "boundary_dp": boundary_path, "transition_dp": transition_path}
    result = {}
    for method, path in paths.items():
        selected = [options[index] for options, index in zip(lattice, path)]
        result[method] = {
            "power": np.concatenate([item.power for item in selected]),
            "selected": selected, "candidate_indices": path,
            "boundary_objective": float(sum(edge[a, b] for edge, a, b in zip(boundary, path, path[1:]))),
            "transition_objective": float(sum(edge[a, b] for edge, a, b in zip(transition, path, path[1:]))),
            "transition_fallback_edges": [index for index, fallback in enumerate(fallbacks) if fallback],
            "transition_supported_edges": sum(not fallback for fallback in fallbacks),
        }
    return result


def waveform_metrics(power, lengths, sample_period):
    power = _signal(power)
    if not np.isfinite(sample_period) or sample_period <= 0 or sum(lengths) != len(power):
        raise ValueError("invalid period or template lengths")
    boundaries = np.cumsum(lengths)[:-1]
    jumps = power[boundaries] - power[boundaries - 1]
    return {
        "samples": len(power), "duration_seconds": len(power) * sample_period,
        "energy_wh": float(np.sum(power) * sample_period / 3600),
        "mean_watts": float(power.mean()), "peak_watts": float(power.max()),
        "mean_absolute_boundary_jump_watts": float(np.abs(jumps).mean()) if len(jumps) else None,
        "signed_boundary_jumps_watts": jumps.tolist(),
    }
