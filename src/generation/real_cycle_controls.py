"""Post-hoc real-cycle controls; no fitting, synthesis or test-set argument."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json

import numpy as np
from scipy.stats import wasserstein_distance

from src.generation.composition_evaluation import _distribution, _support_status, transition_observations
from src.generation.primitive_composition import waveform_metrics

CYCLE_METRICS = ('duration_seconds', 'energy_wh', 'mean_watts', 'peak_watts')
EDGE_METRICS = ('signed_jump_watts', 'left_slope_watts_per_second', 'right_slope_watts_per_second')


def measure_cycles(cycles, period):
    records = {}
    for cycle in cycles:
        if cycle.activity_id in records:
            raise ValueError('duplicate real cycle activity ID')
        lengths = [len(block.power) for block in cycle.blocks]
        records[cycle.activity_id] = {
            'activity_id': cycle.activity_id, 'class_mode': list(cycle.group),
            'power_sha256': hashlib.sha256(np.asarray(cycle.power, dtype='<f8').tobytes()).hexdigest(),
            'metrics': waveform_metrics(cycle.power, lengths, period),
            'transitions': transition_observations(cycle.power, lengths,
                                                  [block.state for block in cycle.blocks], period),
        }
    return records


def _status(n_source, n_validation):
    if not n_validation:
        return 'no_matching_validation'
    if not n_source:
        return 'no_source_observations'
    return 'low_support' if min(n_source, n_validation) < 3 else 'descriptive_only'


def _values(records, metric, pair=None):
    if pair is None:
        return [record['metrics'][metric] for record in records], len(records)
    values, cycle_ids = [], set()
    for record in records:
        for row in record['transitions']:
            if (row['left_state'], row['right_state']) == pair:
                values.append(row[metric])
                cycle_ids.add(record['activity_id'])
    return values, len(cycle_ids)


def _distance(a, b):
    return float(wasserstein_distance(a, b)) if len(a) and len(b) else None


def _comparison(records, reference, metric, pair=None):
    a, na = _values(records, metric, pair)
    b, nb = _values(reference, metric, pair)
    return {'distribution': _distribution(a), 'source_cycles': na, 'validation_cycles': nb,
            'wasserstein': _distance(a, b), 'status': _status(na, nb)}


def _draws(pool_ids, count, repeats, sampling_seed, key):
    if count > len(pool_ids):
        raise ValueError('matched sample exceeds budget-local real pool')
    # Neither held-out values nor incoming record order affect the sample IDs.
    payload = json.dumps([sampling_seed, *key], sort_keys=True).encode()
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:16], 'big')
    rng = np.random.default_rng(seed)
    return [sorted(map(int, rng.choice(sorted(pool_ids), count, replace=False))) for _ in range(repeats)]


def evaluate_real_controls(manifest, training, validation, *, repeats=100, sampling_seed=20260917):
    """Keep seed/budget/Class/Mode/reference-support strata and directed pairs.

    Matched controls sample WHOLE unchanged real cycles without replacement
    within a draw; draws may overlap. Pools include anchors, transparently.
    Their quantiles describe sampling variability, not confidence intervals.
    """
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError('repeats must be a positive integer')
    if isinstance(sampling_seed, bool) or not isinstance(sampling_seed, int) or sampling_seed < 0:
        raise ValueError('sampling_seed must be a non-negative integer')
    period = float(manifest['configuration']['sample_period'])
    if not np.isfinite(period) or period <= 0:
        raise ValueError('sample period must be finite and positive')
    training, validation = list(training), list(validation)
    real = measure_cycles(training, period)
    heldout = measure_cycles(validation, period)
    if set(real) & set(heldout):
        raise ValueError('training/validation overlap')
    by_group, validation_by_group = defaultdict(list), defaultdict(list)
    for record in real.values():
        by_group[tuple(record['class_mode'])].append(record)
    for record in heldout.values():
        validation_by_group[tuple(record['class_mode'])].append(record)
    budgets = {}
    for row in manifest['budgets']:
        key = row['seed'], row['budget_tag']
        ids = row['selected_activity_ids']
        if key in budgets or len(set(ids)) != len(ids) or not set(ids) <= set(real):
            raise ValueError('invalid or non-training budget membership')
        budgets[key] = set(ids)
    strata, case_ids = defaultdict(list), set()
    for case in manifest['cases']:
        if case['case_id'] in case_ids:
            raise ValueError('duplicate case ID')
        case_ids.add(case['case_id'])
        key = case['seed'], case['budget_tag'], tuple(case['class_mode']), _support_status(case)
        anchor = case['anchor_activity_id']
        if (anchor not in budgets[key[:2]] or tuple(real[anchor]['class_mode']) != key[2]
                or set(case['results']) != set(manifest['methods'])):
            raise ValueError('anchor or method matrix mismatch')
        source = next(cycle for cycle in training if cycle.activity_id == anchor)
        template = [{'state_label': block.state, 'length_samples': len(block.power)} for block in source.blocks]
        if template != case['template']:
            raise ValueError('real anchor template differs from saved composition')
        strata[key].append(case)
    results = []
    for key, cases in sorted(strata.items()):
        cases = sorted(cases, key=lambda case: case['case_id'])
        anchor_ids = [case['anchor_activity_id'] for case in cases]
        if len(set(anchor_ids)) != len(anchor_ids):
            raise ValueError('repeated anchor within a stratum')
        full = sorted(by_group[key[2]], key=lambda row: row['activity_id'])
        pool = [row for row in full if row['activity_id'] in budgets[key[:2]]]
        reference = sorted(validation_by_group[key[2]], key=lambda row: row['activity_id'])
        draws = _draws([r['activity_id'] for r in pool], len(cases), repeats, sampling_seed, key)
        draw_records = [[real[aid] for aid in draw] for draw in draws]
        controls = {'full_train_context': full, 'budget_pool_context': pool,
                    'original_anchors': [real[aid] for aid in anchor_ids]}
        methods = {}
        for method in manifest['methods']:
            methods[method] = [{'activity_id': case['anchor_activity_id'],
                                'metrics': case['results'][method]['metrics'],
                                'transitions': case['results'][method]['transition_observations']}
                               for case in cases]
            expected = [real[aid]['metrics']['duration_seconds'] for aid in anchor_ids]
            if [row['metrics']['duration_seconds'] for row in methods[method]] != expected:
                raise ValueError('generated duration no longer matches original anchor')

        def metrics_report(names, pair=None):
            output = {}
            for metric in names:
                ref_values, ref_cycles = _values(reference, metric, pair)
                real_controls = {name: _comparison(rows, reference, metric, pair)
                                 for name, rows in controls.items()}
                sampled = [_comparison(rows, reference, metric, pair) for rows in draw_records]
                distances = [row['wasserstein'] for row in sampled]
                available = [value for value in distances if value is not None]
                center = float(np.median(available)) if available else None
                real_controls['matched_budget_real'] = {
                    'sample_cycles': len(cases), 'repeats': repeats,
                    'available_draws': len(available), 'missing_draws': repeats-len(available),
                    'wasserstein_by_draw': distances,
                    'wasserstein_distribution': _distribution(available),
                    'source_cycles_by_draw': [row['source_cycles'] for row in sampled],
                    'status_by_draw': [row['status'] for row in sampled],
                }
                measured = {name: _comparison(rows, reference, metric, pair) for name, rows in methods.items()}
                anchor_distance = real_controls['original_anchors']['wasserstein']
                for row in measured.values():
                    value = row['wasserstein']
                    row['delta_vs_original_anchors'] = (value-anchor_distance
                        if value is not None and anchor_distance is not None else None)
                    row['delta_vs_matched_real_median'] = (value-center
                        if value is not None and center is not None else None)
                output[metric] = {'validation': _distribution(ref_values), 'validation_cycles': ref_cycles,
                                  'controls': real_controls, 'methods': measured}
            return output

        pairs = sorted({(edge['left_state'], edge['right_state'])
                        for record in full+reference+sum(methods.values(), []) for edge in record['transitions']})
        results.append({
            'seed': key[0], 'budget_tag': key[1], 'class_mode': list(key[2]), 'support_status': key[3],
            'case_ids': [case['case_id'] for case in cases], 'anchor_activity_ids': anchor_ids,
            'full_train_activity_ids': [row['activity_id'] for row in full],
            'budget_pool_activity_ids': [row['activity_id'] for row in pool],
            'validation_activity_ids': [row['activity_id'] for row in reference],
            'matched_sample_activity_ids': draws, 'unique_matched_subsets': len({tuple(draw) for draw in draws}),
            'status': _status(len(cases), len(reference)),
            'cycle_metrics': metrics_report(CYCLE_METRICS),
            'transition_metrics': [{'left_state': pair[0], 'right_state': pair[1],
                                    'metrics': metrics_report(EDGE_METRICS, pair)} for pair in pairs],
        })
    missing = sum(len(row['case_ids']) for row in results if not row['validation_activity_ids'])
    count = len(manifest['cases'])
    return {
        'schema_version': 1,
        'status': ('no_completed_cases' if not count else 'no_matching_validation' if missing == count
                   else 'partial_validation_coverage' if missing else 'ready_for_descriptive_controls'),
        'configuration': {'repeats': repeats, 'sampling_seed': sampling_seed, 'sample_period': period,
                          'sampling': 'same_budget_class_mode_whole_cycles_without_replacement_per_draw'},
        'metric_units': {'duration_seconds': 's', 'energy_wh': 'Wh', 'mean_watts': 'W',
                         'peak_watts': 'W', 'signed_jump_watts': 'W',
                         'left_slope_watts_per_second': 'W/s', 'right_slope_watts_per_second': 'W/s'},
        'paired_cases': count, 'cases_without_validation': missing,
        'distinct_anchors': len({c['anchor_activity_id'] for c in manifest['cases']}),
        'methods': manifest['methods'], 'strata': results,
        'real_measurements': {'train': list(real.values()), 'validation': list(heldout.values())},
        'limitations': [
            'No fitting, synthesis, interpolation, smoothing or test performance evaluation.',
            'Full train and budget pool controls have unequal sample counts; context only, not method rankings.',
            'Matched draws include eligible original anchors and may overlap each other and donor pools.',
            'Draw p10/p50/p90 are sampling-variability summaries, not confidence intervals or p-values.',
            'Generated cases share anchors/donors across budgets; repeats do not add independent data.',
            'Support strata describe the original transition model, not the real control measurements.',
            'State-pair metrics weight edges equally; equal cycle counts do not imply equal edge counts.',
            'Differences between Wasserstein distances are diagnostic contrasts, not causal decompositions.',
            'Inherited representation and class/mode discovery are outside the declared waveform budget.',
        ],
    }
