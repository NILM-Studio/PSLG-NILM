"""Original activity detector, explicit core/context bounds and observation masks.

This module deliberately does not import project ``src``: the Torch worker uses
NILMFormer's upstream package of that name in a separate process.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd


def teacher_grid(t, y, cfg):
    """Same resampling/interpolation rule as ExtractActiveDataStep."""
    fs = float(cfg.get('resample_fs', 0) or 0)
    if fs <= 0 or len(t) < 2:
        return np.asarray(t), np.asarray(y, dtype=float).copy()
    dt = int(round(1 / fs))
    s = pd.Series(y, index=pd.to_datetime(t, unit='s')).resample(f'{dt}s', origin='epoch').mean()
    times = s.index.asi8 // 10**9
    missing = s.isna().to_numpy()
    keep = np.zeros(len(s), bool)
    drop = float(cfg.get('t_drop', 0) or 0)
    edges = np.diff(np.r_[False, missing, False].astype(int))
    if drop > 0:
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if a == 0 or b == len(s) or times[b] - times[a - 1] > drop:
                keep[a:b] = True
    s = s.interpolate(method='time', limit_area='inside')
    s.iloc[np.flatnonzero(keep)] = np.nan
    return times, s.to_numpy(dtype=float)


def detect_activities(t, y, cfg, record_id='record', dt=None):
    project = str(Path(__file__).resolve().parents[2])
    if project not in sys.path:
        sys.path.append(project)
    from models.extract_active_data.simple_threshold import SimpleThresholdDetector
    if cfg.get('method', 'simple') != 'simple':
        raise ValueError('Strict NILM activity labels currently require the aligned simple detector')
    tt, teacher = teacher_grid(t, y, cfg)
    dt = int(dt or round(1 / float(cfg.get('resample_fs') or cfg.get('fs', 1))))
    if not np.array_equal(tt, t) or (len(tt) > 1 and not np.all(np.diff(tt) == dt)):
        raise ValueError('NILM reference requires the frozen sampling grid')
    dc = dict(cfg)
    dc['fs'] = float(cfg.get('resample_fs') or cfg.get('fs', 1))
    detector = SimpleThresholdDetector('OriginalActivityReference', dc)
    intervals = detector.detect(teacher, tt)
    activities = []
    observed = np.isfinite(y)
    for i, interval in enumerate(intervals):
        a = int(np.searchsorted(tt, interval['start_time']))
        b = int(np.searchsorted(tt, interval['end_time'])) + 1
        context_t = interval['data'].timestamp.to_numpy()
        ca, cb = int(np.searchsorted(tt, context_t[0])), int(np.searchsorted(tt, context_t[-1])) + 1
        # Conservative evidence horizon: no complete boundary claim near missing data.
        margin = max(1, int(np.ceil(float(cfg.get('t_drop', 0)) / dt)))
        activities.append(dict(record_id=record_id, activity_id=i,
            core_start=int(tt[a]), core_end_exclusive=int(tt[b - 1] + dt),
            context_start=int(tt[ca]), context_end_exclusive=int(tt[cb - 1] + dt),
            core_start_index=a, core_end_index=b, context_start_index=ca, context_end_index=cb,
            left_censored=bool(a < margin or not observed[max(0, a-margin):a+1].all()),
            right_censored=bool(b + margin > len(tt) or not observed[b-1:min(len(tt), b+margin)].all()),
            observed_support=int(observed[ca:cb].sum()), interpolated_support=int((~observed[ca:cb]).sum())))
    return activities, intervals


def activity_projection(n, activities, observed):
    """Core is half-open, includes internal low power; context is never ON."""
    labels = np.zeros(n, np.int64)
    ids = np.full(n, -1, np.int64)
    conflict = np.zeros(n, bool)
    for a in activities:
        start, end = a['core_start_index'], a['core_end_index']
        if not 0 <= start < end <= n:
            raise ValueError('Invalid activity boundary')
        conflict[start:end] |= ids[start:end] >= 0
        labels[start:end] = 1
        ids[start:end] = a['activity_id']
    valid = np.asarray(observed, bool) & ~conflict
    labels[~valid] = -100
    ids[conflict] = -1
    return labels, ids, valid
