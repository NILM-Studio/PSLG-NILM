"""Read-only, post-composition diagnostics against held-out validation cycles.

These summaries never fit a selector. They retain physical units and condition
on Class/Mode, transition identity, and the training reference's support status.
Shared donors mean distinct generated anchors are not independent replicates.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from numbers import Integral

import numpy as np
from scipy.stats import wasserstein_distance

from src.generation.primitive_composition import waveform_metrics


TRANSITION_METRICS = (
    "signed_jump_watts", "left_slope_watts_per_second", "right_slope_watts_per_second",
)
CYCLE_METRICS = ("energy_wh", "mean_watts", "peak_watts")


def _integer(value, name, minimum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _finite(value, name):
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def transition_observations(power, lengths, states, sample_period=6, window=5):
    """Measure signed seams and within-block local slopes, without smoothing.

    ``states`` contains one integer label per block. A slope is the mean of
    consecutive differences within the last/first ``window`` block samples,
    divided by sample period; a one-sample window/block has slope zero.
    """
    signal = np.asarray(power, dtype=np.float64)
    if (signal.ndim != 1 or not len(signal) or not np.isfinite(signal).all()
            or np.any(signal < 0)):
        raise ValueError("power must be a nonempty finite nonnegative vector")
    period = _finite(sample_period, "sample_period")
    if period <= 0:
        raise ValueError("sample_period must be positive")
    window = _integer(window, "window", 1)
    lengths = [_integer(value, "block length", 1) for value in lengths]
    states = [_integer(value, "state") for value in states]
    if not lengths or len(lengths) != len(states) or sum(lengths) != len(signal):
        raise ValueError("lengths and states must describe every sample exactly once")
    edges = np.cumsum([0, *lengths])
    rows = []
    for index in range(len(lengths) - 1):
        boundary = edges[index + 1]
        left = signal[max(edges[index], boundary - window):boundary]
        right = signal[boundary:min(edges[index + 2], boundary + window)]
        rows.append({
            "left_state": states[index], "right_state": states[index + 1],
            "signed_jump_watts": _finite(signal[boundary] - signal[boundary - 1], "jump"),
            "left_slope_watts_per_second": _finite(
                np.mean(np.diff(left)) / period if len(left) > 1 else 0, "left slope"),
            "right_slope_watts_per_second": _finite(
                np.mean(np.diff(right)) / period if len(right) > 1 else 0, "right slope"),
        })
    return rows


def _distribution(values):
    values = np.asarray([_finite(value, "metric") for value in values], dtype=np.float64)
    if not len(values):
        return {"count": 0, "median": None, "p10": None, "p90": None}
    quantiles = np.percentile(values, [50, 10, 90])
    return {"count": len(values), **{
        name: _finite(value, name) for name, value in zip(("median", "p10", "p90"), quantiles)}}


def _comparison(generated, validation):
    return {
        "generated": _distribution(generated), "validation": _distribution(validation),
        "wasserstein": (_finite(wasserstein_distance(generated, validation), "wasserstein")
                        if len(generated) and len(validation) else None),
    }


def _support_status(case):
    """Older manifests store support on results; newer ones may expose only the reference."""
    total = max(len(case.get("template", [])) - 1, 0)
    results = case.get("results", {})
    preferred = results.get("transition_dp", {})
    rows = [preferred, *results.values()]
    supported = None
    for row in rows:
        if "transition_supported_edges" in row:
            supported = _integer(row["transition_supported_edges"], "supported edges", 0)
            break
        if "transition_fallback_edges" in row:
            supported = total - len(row["transition_fallback_edges"])
            break
    if supported is None:
        reference = case.get("reference_model", {}).get("transitions")
        if reference is None:
            return "unknown"
        template = case.get("template", [])
        supported = sum(bool(reference.get(f'{a["state_label"]}->{b["state_label"]}', {})
                             .get("supported", False)) for a, b in zip(template, template[1:]))
    if not 0 <= supported <= total:
        raise ValueError("supported edge count disagrees with template")
    return "none" if supported == 0 else "full" if supported == total else "partial"


def _stratum(case, method):
    return (int(case["seed"]), str(case["budget_tag"]),
            tuple(int(value) for value in case["class_mode"]), _support_status(case), method)


def _identity(key):
    seed, budget, group, support, method = key
    return {"seed": seed, "budget_tag": budget, "class_mode": list(group),
            "support_status": support, "method": method}


def _status(generated_ids, validation_ids, missing):
    if not validation_ids:
        return missing
    if not generated_ids:
        return "no_generated_observations"
    return "low_support" if len(validation_ids) < 3 else "descriptive_only"


def _metric_definitions():
    definitions = {
        "coverage": {
            "units": "cycles", "direction": "report_all_requested_anchors",
            "definition": "Per seed/budget: requested anchors, completed paired cases, skipped cases and reasons. Support full/partial/none describes supported training-reference edges, shared across methods; unknown means an older record lacks support metadata.",
        },
        "cases_with_at_least_three_blocks": {
            "units": "cases", "direction": "descriptive",
            "definition": "Completed cases whose fixed template contains at least three state blocks.",
        },
        "cases_with_nontrivial_global_choice": {
            "units": "cases", "direction": "descriptive",
            "definition": "At least three blocks, with >1 candidate at an interior position after the fixed first block. This permits, but does not ensure, greedy and global boundary selection to differ.",
        },
        "support_status": {
            "units": "category", "direction": "stratify_do_not_rank",
            "definition": "full: every edge has enough independent training donor cycles; partial: some do; none: every edge falls back. It describes training support, not validation quality.",
        },
        "single_donor_cycle_fraction": {
            "units": "fraction", "direction": "descriptive_replay_diagnostic",
            "definition": "Fraction of reported cases whose selected blocks all come from one donor cycle; not proof of byte-identical replay or physical validity.",
        },
        "nearest_donor_resampled_nrmse": {
            "units": "dimensionless", "direction": "descriptive_replay_diagnostic",
            "definition": "Minimum whole-cycle RMSE to a linearly length-resampled available donor, divided by max(donor peak watts, 1 watt). Near zero suggests replay; larger is not automatically better.",
        },
        "largest_donor_sample_fraction": {
            "units": "fraction", "direction": "descriptive_replay_diagnostic",
            "definition": "Largest fraction of output samples supplied by one donor, after shared length adaptation. One equals a single-donor composition; smaller does not guarantee useful diversity.",
        },
        "duration_target_cost": {
            "units": "dimensionless", "direction": "optimization_diagnostic_only",
            "definition": "Sum over selected blocks of abs(log(target_length/source_length)). This target matching objective is not independent evidence of waveform quality.",
        },
        "signed_jump_watts": {
            "units": "W", "direction": "compare_same_state_pair_distribution",
            "definition": "First right-block sample minus final left-block sample. No log transform or smoothing; zero is not a universal physical target.",
        },
        "left_slope_watts_per_second": {
            "units": "W/s", "direction": "compare_same_state_pair_distribution",
            "definition": "Mean consecutive power difference in the last five samples within the left block, divided by sample period; zero for one sample.",
        },
        "right_slope_watts_per_second": {
            "units": "W/s", "direction": "compare_same_state_pair_distribution",
            "definition": "Mean consecutive power difference in the first five samples within the right block, divided by sample period; zero for one sample.",
        },
        "energy_wh": {
            "units": "Wh", "direction": "compare_same_class_mode_distribution",
            "definition": "Sum of complete-cycle power samples times sample period in seconds divided by 3600.",
        },
        "mean_watts": {
            "units": "W", "direction": "compare_same_class_mode_distribution",
            "definition": "Arithmetic mean of complete-cycle power samples.",
        },
        "peak_watts": {
            "units": "W", "direction": "compare_same_class_mode_distribution",
            "definition": "Maximum complete-cycle power sample.",
        },
        "wasserstein": {
            "units": "same_as_compared_metric", "direction": "smaller_distribution_discrepancy_only",
            "definition": "Unscaled one-dimensional empirical Wasserstein-1 distance. Transition distributions weight observations equally, cycle distributions weight cycles equally. Small distance is not a validity certificate; repeated edges from one cycle are dependent. No significance tests or aggregate ranking.",
        },
        "distribution": {
            "units": "same_as_compared_metric", "direction": "descriptive",
            "definition": "count is the number of scalar observations; median/p10/p90 are empirical percentiles. Empty distributions have count 0 and null percentiles.",
        },
        "generated_cycles": {
            "units": "distinct_anchor_cycles", "direction": "report_support",
            "definition": "Number of distinct anchor activity IDs in a stratum, not number of edges. Shared donor pools mean these are not statistically independent replicates.",
        },
        "validation_cycles": {
            "units": "distinct_validation_cycles", "direction": "report_support",
            "definition": "Number of distinct validation activity IDs supplying this group/transition. Repeated transitions within a cycle count once. <3 produces low_support; no reference gives an explicit missing-validation status.",
        },
        "transition_observations_missing_cases": {
            "units": "cases", "direction": "report_missing_measurements",
            "definition": "Cases without stored transition observations (e.g. older artifacts); absent slopes are never imputed from seam jumps.",
        },
    }
    for definition in definitions.values():
        definition["descriptive_only"] = True
    return definitions


def evaluate_composition(cases, skipped, budgets, validation, period):
    """Summarize already-selected outputs; validation is never used for fitting.

    Reference local slopes use five samples, matching the measurement default.
    No test-set input is accepted. Empty and small strata remain explicit.
    """
    period = _finite(period, "period")
    if period <= 0:
        raise ValueError("period must be positive")
    cases, skipped, budgets, validation = map(list, (cases, skipped, budgets, validation))
    completed, omissions = defaultdict(list), defaultdict(list)
    for case in cases:
        completed[(int(case["seed"]), str(case["budget_tag"]))].append(case)
    for row in skipped:
        omissions[(int(row["seed"]), str(row["budget_tag"]))].append(row)
    budget_map = {(int(row["seed"]), str(row["budget_tag"])): row for row in budgets}
    coverage = []
    for seed, budget in sorted(set(budget_map) | set(completed) | set(omissions)):
        key = seed, budget
        done, missed = completed[key], omissions[key]
        counts = Counter(_support_status(case) for case in done)
        coverage.append({
            "seed": seed, "budget_tag": budget,
            "requested": int(budget_map.get(key, {}).get("requested_anchors", len(done) + len(missed))),
            "completed": len(done), "skipped": len(missed),
            "skip_reasons": dict(sorted(Counter(row["reason"] for row in missed).items())),
            "support_cases": {status: counts[status] for status in ("full", "partial", "none", "unknown")},
            "cases_with_at_least_three_blocks": sum(len(case.get("template", [])) >= 3 for case in done),
            "cases_with_nontrivial_global_choice": sum(
                len(case.get("template", [])) >= 3 and any(len(options) > 1
                    for options in case.get("candidates", [])[1:-1]) for case in done),
        })

    validation_metrics, validation_edges = defaultdict(list), defaultdict(list)
    seen_validation = set()
    for cycle in validation:
        if cycle.activity_id in seen_validation:
            raise ValueError("duplicate validation cycle activity ID")
        seen_validation.add(cycle.activity_id)
        lengths = [len(block.power) for block in cycle.blocks]
        metrics = waveform_metrics(cycle.power, lengths, period)
        validation_metrics[cycle.group].append((cycle.activity_id, metrics))
        for observation in transition_observations(
                cycle.power, lengths, [block.state for block in cycle.blocks], period):
            validation_edges[(cycle.group, observation["left_state"], observation["right_state"])].append(
                (cycle.activity_id, observation))

    generated, generated_edges = defaultdict(list), defaultdict(list)
    transition_keys = set()
    for case in cases:
        for method, row in case["results"].items():
            key = _stratum(case, method)
            generated[key].append((case, row))
            template = case.get("template", [])
            expected_pairs = [(int(a["state_label"]), int(b["state_label"]))
                              for a, b in zip(template, template[1:])]
            for left, right in expected_pairs:
                transition_keys.add((*key, left, right))
            observations = row.get("transition_observations")
            if observations is None:
                continue
            observed_pairs = [(item["left_state"], item["right_state"]) for item in observations]
            if observed_pairs != expected_pairs:
                raise ValueError("transition observations disagree with case state template")
            for observation in observations:
                edge_key = (*key, int(observation["left_state"]), int(observation["right_state"]))
                generated_edges[edge_key].append((case["anchor_activity_id"], observation))

    method_summary, cycle_validation, transition_validation = [], [], []
    for key, rows in sorted(generated.items()):
        ids = {case["anchor_activity_id"] for case, _ in rows}
        replay = [bool(row["single_donor_cycle"]) for _, row in rows if "single_donor_cycle" in row]
        summary = {**_identity(key), "cases": len(rows), "generated_cycles": len(ids),
                   "single_donor_reported_cases": len(replay),
                   "single_donor_cycle_fraction": sum(replay) / len(replay) if replay else None,
                   "transition_observations_missing_cases": sum(
                       row.get("transition_observations") is None for _, row in rows)}
        for name in ("nearest_donor_resampled_nrmse", "largest_donor_sample_fraction", "duration_target_cost"):
            values = [row[name] for _, row in rows if row.get(name) is not None]
            if values:
                summary[name] = _distribution(values)
        method_summary.append(summary)
        reference = validation_metrics[key[2]]
        reference_ids = {activity_id for activity_id, _ in reference}
        entry = {**_identity(key), "generated_cycles": len(ids), "validation_cycles": len(reference_ids),
                 "status": _status(ids, reference_ids, "no_same_group_validation")}
        for name in CYCLE_METRICS:
            a = [row["metrics"][name] for _, row in rows if name in row.get("metrics", {})]
            b = [metrics[name] for _, metrics in reference]
            entry[name] = _comparison(a, b)
        cycle_validation.append(entry)

    for key in sorted(transition_keys):
        rows = generated_edges[key]
        reference = validation_edges[(key[2], key[-2], key[-1])]
        ids = {activity_id for activity_id, _ in rows}
        reference_ids = {activity_id for activity_id, _ in reference}
        entry = {**_identity(key[:5]), "left_state": key[-2], "right_state": key[-1],
                 "generated_cycles": len(ids), "validation_cycles": len(reference_ids),
                 "status": _status(ids, reference_ids, "no_same_group_transition_validation")}
        for name in TRANSITION_METRICS:
            entry[name] = _comparison([row[name] for _, row in rows], [row[name] for _, row in reference])
        transition_validation.append(entry)
    return {"coverage": coverage, "method_summary": method_summary,
            "cycle_validation": cycle_validation, "transition_validation": transition_validation,
            "metric_definitions": _metric_definitions()}
