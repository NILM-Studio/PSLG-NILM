"""Physical-profile neighborhoods for cycle-conditioned primitive sampling."""
from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np


def cycle_profile(power: np.ndarray, blocks: Iterable[dict],
                  states: Iterable[int]) -> np.ndarray:
    """Describe a cycle using global and state-level physical quantities."""
    values = np.clip(np.nan_to_num(
        np.asarray(power, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0),
        0.0, None)
    total = max(len(values), 1)
    state_values: Dict[int, List[np.ndarray]] = {int(state): [] for state in states}
    state_lengths: Dict[int, int] = {int(state): 0 for state in states}
    cursor = 0
    for block in blocks:
        state = int(block["state_label"])
        length = max(0, int(block.get("length_samples", 0)))
        end = min(len(values), cursor + length)
        if state in state_values and end > cursor:
            state_values[state].append(values[cursor:end])
            state_lengths[state] += end - cursor
        cursor += length

    profile = [
        np.log1p(len(values)),
        np.log1p(float(np.sum(values))),
        np.log1p(float(np.mean(values)) if len(values) else 0.0),
        np.log1p(float(np.max(values)) if len(values) else 0.0),
    ]
    for state in sorted(state_values):
        chunks = state_values[state]
        state_power = np.concatenate(chunks) if chunks else np.zeros(1)
        profile.extend([
            float(state_lengths[state] / total),
            np.log1p(float(np.mean(state_power))),
        ])
    return np.asarray(profile, dtype=np.float64)


class CycleNeighborIndex:
    """Find robustly standardized physical neighbors inside one class/mode."""

    def __init__(self, profiles: Dict[int, np.ndarray], neighbor_count: int = 5,
                 exclude_anchor: bool = True):
        if not profiles:
            raise ValueError("cycle neighbor index requires at least one profile")
        self.neighbor_count = max(1, int(neighbor_count))
        self.exclude_anchor = bool(exclude_anchor)
        self.activity_ids = sorted(int(value) for value in profiles)
        matrix = np.vstack([profiles[value] for value in self.activity_ids])
        center = np.median(matrix, axis=0)
        q25, q75 = np.percentile(matrix, [25, 75], axis=0)
        scale = q75 - q25
        standard_deviation = np.std(matrix, axis=0)
        scale = np.where(scale > 1e-9, scale,
                         np.where(standard_deviation > 1e-9,
                                  standard_deviation, 1.0))
        self.matrix = (matrix - center) / scale

    def neighbors(self, activity_id: int) -> List[dict]:
        activity_id = int(activity_id)
        if activity_id not in self.activity_ids:
            raise KeyError(f"activity {activity_id} has no cycle profile")
        position = self.activity_ids.index(activity_id)
        distances = np.linalg.norm(self.matrix - self.matrix[position], axis=1)
        order = np.argsort(distances, kind="stable")
        result = []
        for index in order:
            candidate = self.activity_ids[int(index)]
            if self.exclude_anchor and candidate == activity_id:
                continue
            result.append({
                "activity_id": int(candidate),
                "distance": float(distances[int(index)]),
            })
            if len(result) >= self.neighbor_count:
                break
        if not result:
            result.append({"activity_id": activity_id, "distance": 0.0})
        return result
