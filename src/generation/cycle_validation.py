"""Robust, data-driven validation of classified appliance work cycles."""
from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List

import numpy as np


def robust_z_scores(values: Iterable[float]) -> np.ndarray:
    """Return absolute modified z-scores, with stable handling of zero MAD."""
    data = np.asarray(list(values), dtype=np.float64)
    if data.size == 0:
        return data
    median = float(np.median(data))
    mad = float(np.median(np.abs(data - median)))
    if mad <= np.finfo(np.float64).eps:
        return np.where(np.isclose(data, median), 0.0, np.inf)
    return np.abs(data - median) / (1.4826 * mad)


def infer_cycle_grammar(classes: List[dict], min_class_support: int,
                        core_state_min_prevalence: float,
                        terminal_state_min_prevalence: float) -> dict:
    """Infer common required states and terminal states from supported classes.

    State labels are cluster ids with no fixed physical meaning. Weighting each
    representative signature by class support discovers the common cycle
    grammar without assuming that, for example, state 0 is always heating.
    """
    reference = [entry for entry in classes
                 if int(entry.get("support", 0)) >= int(min_class_support)]
    if not reference:
        reference = list(classes)
    total = float(sum(int(entry.get("support", 0)) for entry in reference))
    if total <= 0:
        return {"required_core_states": [], "allowed_terminal_states": [],
                "state_prevalence": {}, "terminal_prevalence": {}}

    state_support: Counter = Counter()
    terminal_support: Counter = Counter()
    for entry in reference:
        support = int(entry["support"])
        signature = [int(v) for v in entry.get("representative_signature", [])]
        for state in set(signature):
            state_support[state] += support
        if signature:
            terminal_support[signature[-1]] += support

    state_prevalence = {int(k): float(v / total)
                        for k, v in sorted(state_support.items())}
    terminal_prevalence = {int(k): float(v / total)
                           for k, v in sorted(terminal_support.items())}
    required = [state for state, ratio in state_prevalence.items()
                if ratio >= float(core_state_min_prevalence)]
    terminals = [state for state, ratio in terminal_prevalence.items()
                 if ratio >= float(terminal_state_min_prevalence)]
    if not terminals and terminal_prevalence:
        terminals = [max(terminal_prevalence, key=terminal_prevalence.get)]
    return {
        "required_core_states": required,
        "allowed_terminal_states": terminals,
        "state_prevalence": state_prevalence,
        "terminal_prevalence": terminal_prevalence,
        "reference_support": int(total),
    }


def signature_purity(entry: dict) -> float:
    counts = [int(v) for v in (entry.get("signature_counts") or {}).values()]
    support = max(int(entry.get("support", 0)), 1)
    return float(max(counts, default=0) / support)

