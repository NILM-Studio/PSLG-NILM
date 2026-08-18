"""Build and sample a state-conditioned library of real power primitives."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass(frozen=True)
class Primitive:
    """One power primitive and its source provenance."""

    primitive_id: int
    state_label: int
    activity_index: int
    start: int
    power: np.ndarray


class PrimitiveLibrary:
    """State label -> source primitives, kept entirely in memory."""

    def __init__(self, primitives: Iterable[Primitive]):
        self._by_state: Dict[int, List[Primitive]] = {}
        for primitive in primitives:
            if primitive.power.size == 0:
                continue
            self._by_state.setdefault(int(primitive.state_label), []).append(primitive)
        if not self._by_state:
            raise ValueError("primitive library is empty")

    @property
    def states(self) -> List[int]:
        return sorted(self._by_state)

    def count(self, state: int) -> int:
        return len(self._by_state.get(int(state), []))

    def summary(self) -> dict:
        result = {}
        for state in self.states:
            lengths = np.asarray([len(p.power) for p in self._by_state[state]], dtype=np.int64)
            result[str(state)] = {
                "count": int(len(lengths)),
                "length_min": int(lengths.min()),
                "length_median": float(np.median(lengths)),
                "length_max": int(lengths.max()),
            }
        return result

    def sample(self, state: int, rng: np.random.Generator) -> Primitive:
        candidates = self._by_state.get(int(state), [])
        if not candidates:
            raise KeyError(f"no primitives available for state {state}")
        return candidates[int(rng.integers(0, len(candidates)))]


class RealPrimitiveSampler:
    """Fill a requested state duration by resampling real primitives."""

    method = "real_resample"

    def __init__(self, library: PrimitiveLibrary):
        self.library = library

    def sample_block(self, state: int, target_length: int,
                     rng: np.random.Generator) -> Tuple[np.ndarray, List[dict]]:
        target_length = max(1, int(target_length))
        chunks, provenance = [], []
        remaining = target_length
        while remaining > 0:
            primitive = self.library.sample(state, rng)
            take = min(remaining, len(primitive.power))
            chunks.append(np.asarray(primitive.power[:take], dtype=np.float32))
            provenance.append({
                "primitive_id": int(primitive.primitive_id),
                "activity_index": int(primitive.activity_index),
                "source_start": int(primitive.start),
                "source_length": int(len(primitive.power)),
                "used_length": int(take),
            })
            remaining -= take
        return np.concatenate(chunks), provenance
