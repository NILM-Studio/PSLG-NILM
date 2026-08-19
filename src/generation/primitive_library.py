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

    def candidates(self, state: int) -> List[Primitive]:
        candidates = self._by_state.get(int(state), [])
        if not candidates:
            raise KeyError(f"no primitives available for state {state}")
        return candidates


class RealPrimitiveSampler:
    """Fill a requested state duration by resampling real primitives."""

    method = "real_resample"

    def __init__(self, library: PrimitiveLibrary, candidate_pool: int = 32,
                 within_state_smooth_samples: int = 3,
                 boundary_smooth_samples: int = 3):
        self.library = library
        self.candidate_pool = max(1, int(candidate_pool))
        self.within_state_smooth_samples = max(0, int(within_state_smooth_samples))
        self.boundary_smooth_samples = max(0, int(boundary_smooth_samples))

    def _select(self, state: int, previous_end: float | None,
                rng: np.random.Generator) -> Primitive:
        candidates = self.library.candidates(state)
        if previous_end is None:
            return candidates[int(rng.integers(0, len(candidates)))]
        pool_size = min(self.candidate_pool, len(candidates))
        indices = rng.choice(len(candidates), size=pool_size, replace=False)
        return min((candidates[int(i)] for i in indices),
                   key=lambda p: abs(float(p.power[0]) - previous_end))

    @staticmethod
    def _smooth_start(chunk: np.ndarray, previous_end: float,
                      smooth_samples: int) -> np.ndarray:
        n = min(max(0, int(smooth_samples)), len(chunk))
        if n == 0:
            return chunk
        adjusted = np.array(chunk, dtype=np.float32, copy=True)
        blend = np.linspace(0.0, 1.0, n, dtype=np.float32)
        adjusted[:n] = ((1.0 - blend) * float(previous_end)
                        + blend * adjusted[:n])
        np.maximum(adjusted[:n], 0.0, out=adjusted[:n])
        return adjusted

    def sample_block(self, state: int, target_length: int,
                     rng: np.random.Generator,
                     initial_power: float | None = None) -> Tuple[np.ndarray, List[dict]]:
        target_length = max(1, int(target_length))
        chunks, provenance = [], []
        remaining = target_length
        previous_end = initial_power
        while remaining > 0:
            is_first_chunk = not chunks
            primitive = self._select(state, previous_end, rng)
            take = min(remaining, len(primitive.power))
            raw = np.asarray(primitive.power[:take], dtype=np.float32)
            raw_start = float(raw[0])
            smooth_samples = (self.boundary_smooth_samples if is_first_chunk
                              else self.within_state_smooth_samples)
            chunk = (self._smooth_start(raw, previous_end, smooth_samples)
                     if previous_end is not None else raw)
            chunks.append(chunk)
            provenance.append({
                "primitive_id": int(primitive.primitive_id),
                "activity_index": int(primitive.activity_index),
                "source_start": int(primitive.start),
                "source_length": int(len(primitive.power)),
                "used_length": int(take),
                "raw_start_power": raw_start,
                "adjusted_start_power": float(chunk[0]),
                "join_jump_before": (None if previous_end is None
                                     else float(abs(raw_start - previous_end))),
                "join_jump_after": (None if previous_end is None
                                    else float(abs(float(chunk[0]) - previous_end))),
                "join_type": (None if previous_end is None else
                              ("state_boundary" if is_first_chunk else "within_state")),
                "smooth_samples": int(smooth_samples if previous_end is not None else 0),
            })
            previous_end = float(chunk[-1])
            remaining -= take
        return np.concatenate(chunks), provenance
