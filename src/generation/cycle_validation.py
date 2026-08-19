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


def discover_metric_modes(matrix: np.ndarray, max_modes: int = 3,
                          min_mode_support: int = 15,
                          bic_min_gain: float = 10.0,
                          random_state: int = 42) -> tuple[np.ndarray, dict]:
    """Discover supported physical-program modes with a BIC-selected GMM.

    Columns are positive cycle metrics (duration, energy, mean and peak power).
    A log transform limits scale skew, then standardization prevents one metric
    from dominating. More complex models are accepted only when they improve
    BIC by ``bic_min_gain`` and every component has enough observations.
    Returned mode ids are ordered by median duration for stable interpretation.
    """
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("mode discovery requires a non-empty 2-D metric matrix")
    if len(values) == 1:
        return np.zeros(1, dtype=np.int64), {
            "selected_modes": 1,
            "candidate_bic": {},
            "mode_support": {"0": 1},
        }
    transformed = np.log1p(np.clip(values, 0.0, None))
    scaled = StandardScaler().fit_transform(transformed)
    max_supported = max(1, min(int(max_modes),
                               len(values) // max(1, int(min_mode_support))))

    candidates = []
    for n_components in range(1, max_supported + 1):
        model = GaussianMixture(
            n_components=n_components, covariance_type="full",
            reg_covar=1e-5, n_init=5, random_state=int(random_state))
        labels = model.fit_predict(scaled)
        counts = np.bincount(labels, minlength=n_components)
        candidates.append({
            "n_components": n_components,
            "bic": float(model.bic(scaled)),
            "labels": labels,
            "counts": counts,
        })

    selected = candidates[0]
    for candidate in candidates[1:]:
        if int(candidate["counts"].min()) < int(min_mode_support):
            continue
        if float(candidate["bic"]) <= float(selected["bic"]) - float(bic_min_gain):
            selected = candidate

    raw_labels = np.asarray(selected["labels"], dtype=np.int64)
    order = sorted(range(int(selected["n_components"])),
                   key=lambda label: float(np.median(values[raw_labels == label, 0])))
    remap = {old: new for new, old in enumerate(order)}
    labels = np.asarray([remap[int(label)] for label in raw_labels], dtype=np.int64)
    diagnostics = {
        "selected_modes": int(selected["n_components"]),
        "candidate_bic": {
            str(item["n_components"]): float(item["bic"])
            for item in candidates
        },
        "mode_support": {
            str(mode): int(np.sum(labels == mode))
            for mode in sorted(set(labels.tolist()))
        },
    }
    return labels, diagnostics
