"""Empirical and first-order Markov models for merged appliance states."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np


END = "__end__"


def _weighted_choice(counter: Counter, rng: np.random.Generator):
    keys = list(counter)
    weights = np.asarray([counter[k] for k in keys], dtype=np.float64)
    weights /= weights.sum()
    return keys[int(rng.choice(len(keys), p=weights))]


class StateTransitionModel:
    """Learn state order and duration distributions from merged activities."""

    def __init__(self, sequences: Iterable[List[dict]]):
        self.sequences = [list(seq) for seq in sequences if seq]
        if not self.sequences:
            raise ValueError("state sequence collection is empty")

        self.initial = Counter()
        self.transitions: Dict[int, Counter] = defaultdict(Counter)
        self.durations: Dict[int, List[int]] = defaultdict(list)
        for sequence in self.sequences:
            labels = [int(block["state_label"]) for block in sequence]
            self.initial[labels[0]] += 1
            for block, label in zip(sequence, labels):
                length = int(block.get("length_samples", 0))
                if length > 0:
                    self.durations[label].append(length)
            for left, right in zip(labels[:-1], labels[1:]):
                self.transitions[left][right] += 1
            self.transitions[labels[-1]][END] += 1

    def sample_duration(self, state: int, rng: np.random.Generator) -> int:
        values = self.durations.get(int(state), [])
        if not values:
            raise KeyError(f"no duration observations for state {state}")
        return int(values[int(rng.integers(0, len(values)))])

    def sample_empirical(self, rng: np.random.Generator) -> List[Tuple[int, int]]:
        sequence = self.sequences[int(rng.integers(0, len(self.sequences)))]
        return [(int(block["state_label"]), int(block["length_samples"]))
                for block in sequence]

    def sample_markov(self, rng: np.random.Generator, min_blocks: int = 1,
                      max_blocks: int = 20) -> List[Tuple[int, int]]:
        min_blocks = max(1, int(min_blocks))
        max_blocks = max(min_blocks, int(max_blocks))
        state = int(_weighted_choice(self.initial, rng))
        sampled = []
        for _ in range(max_blocks):
            sampled.append((state, self.sample_duration(state, rng)))
            choices = Counter(self.transitions.get(state, {}))
            if not choices:
                break
            if len(sampled) < min_blocks:
                choices.pop(END, None)
                if not choices:
                    break
            nxt = _weighted_choice(choices, rng)
            if nxt == END:
                break
            state = int(nxt)
        return sampled

    def sample(self, method: str, rng: np.random.Generator,
               min_blocks: int = 1, max_blocks: int = 20) -> List[Tuple[int, int]]:
        method = str(method).lower()
        if method == "empirical":
            return self.sample_empirical(rng)
        if method == "markov":
            return self.sample_markov(rng, min_blocks=min_blocks, max_blocks=max_blocks)
        raise ValueError(f"unknown sequence sampling method: {method}")

    @staticmethod
    def _probabilities(counter: Counter) -> dict:
        total = float(sum(counter.values()))
        return {str(k): float(v / total) for k, v in sorted(counter.items(), key=lambda x: str(x[0]))}

    def to_dict(self) -> dict:
        duration_summary = {}
        for state, values in sorted(self.durations.items()):
            a = np.asarray(values, dtype=np.int64)
            duration_summary[str(state)] = {
                "count": int(len(a)),
                "min": int(a.min()),
                "median": float(np.median(a)),
                "max": int(a.max()),
            }
        return {
            "n_sequences": int(len(self.sequences)),
            "initial_probabilities": self._probabilities(self.initial),
            "transition_probabilities": {
                str(state): self._probabilities(counter)
                for state, counter in sorted(self.transitions.items())
            },
            "duration_samples": {str(k): [int(v) for v in values]
                                 for k, values in sorted(self.durations.items())},
            "duration_summary": duration_summary,
        }
