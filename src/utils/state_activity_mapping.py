"""Pure mapping of final state blocks to activity cores; no few-shot dependency."""
from __future__ import annotations
import numpy as np


def contiguous_groups(rows, starts, lengths):
    groups, current, end = [], [], None
    for row in sorted(rows, key=lambda r: int(starts[r])):
        start, length = int(starts[row]), int(lengths[row])
        if length <= 0 or (end is not None and start < end):
            raise ValueError('Nonpositive or overlapping primitive spans')
        if current and start != end:
            groups.append(current)
            current = []
        current.append(row)
        end = start + length
    if current:
        groups.append(current)
    return groups


def map_blocks(t, activities, blocks, dt):
    """Map CSV-relative offsets exactly; conflicting contexts remain masked."""
    n = len(t)
    state, bid = np.full(n, -100, np.int64), np.full(n, -1, np.int64)
    covered, conflict = np.zeros(n, bool), np.zeros(n, bool)
    sequences = []
    by_file = {a['csv_idx']: a for a in activities}
    for b in blocks:
        a = by_file[b['csv_idx']]
        start = a['context_start_index'] + b['start']
        end = a['context_start_index'] + b['end']
        if b['end'] - b['start'] != b['length_samples'] or not 0 <= start < end <= n:
            raise ValueError('Invalid block or block crossing discarded primitive support')
        if end > a['context_end_index'] or not np.all(np.diff(t[start:end]) == dt):
            raise ValueError('State block crosses recording/time boundary')
        label = int(b['state_label'])
        if label < 0:
            continue
        conflict[start:end] |= covered[start:end] & (state[start:end] != label)
        # Multiple different block identities (even same label) are ambiguous for BP.
        conflict[start:end] |= covered[start:end] & (bid[start:end] != b['block_id'])
        empty = ~covered[start:end]
        state[start:end][empty], bid[start:end][empty] = label, b['block_id']
        covered[start:end] = True
        ca, cb = max(start, a['core_start_index']), min(end, a['core_end_index'])
        if ca < cb:
            sequences.append(dict(record_id=a['record_id'], activity_id=a['activity_id'],
                block_id=b['block_id'], state_label=label, start=ca, end=cb,
                start_timestamp=int(t[ca]), end_timestamp_exclusive=int(t[cb-1] + dt),
                duration_seconds=(cb-ca)*dt, member_rows=b['member_rows'],
                left_censored=bool(ca != start or (ca == a['core_start_index'] and a['left_censored'])),
                right_censored=bool(cb != end or (cb == a['core_end_index'] and a['right_censored']))))
    state[conflict], bid[conflict] = -100, -1
    return dict(state_full_merge=state, block_id=bid, state_covered=covered, context_conflict=conflict), sequences


def sequence_statistics(sequences, observed, conflict, k, dt, window_length):
    transitions = np.zeros((k, k), np.int64)
    durations = {str(i): [] for i in range(k)}
    previous = None
    for b in sorted(sequences, key=lambda b: (b['activity_id'], b['start'])):
        clean = bool(observed[b['start']:b['end']].all() and not conflict[b['start']:b['end']].any())
        b['interpolated_points'] = int((~observed[b['start']:b['end']]).sum())
        b['conflict_points'] = int(conflict[b['start']:b['end']].sum())
        b['complete_observed'] = clean and not b['left_censored'] and not b['right_censored']
        if b['complete_observed']:
            durations[str(b['state_label'])].append(b['duration_seconds'])
        if (previous and previous['activity_id'] == b['activity_id'] and previous['end'] == b['start']
                and clean and previous['_clean']):
            transitions[previous['state_label'], b['state_label']] += 1
        b['_clean'] = clean
        previous = b
    for b in sequences:
        b.pop('_clean', None)
    totals = transitions.sum(axis=1, keepdims=True)
    probs = np.divide(transitions, totals, out=np.zeros_like(transitions, dtype=float), where=totals > 0)
    return dict(counts=transitions.tolist(), probabilities=probs.tolist(), hard_constraints=False), durations
