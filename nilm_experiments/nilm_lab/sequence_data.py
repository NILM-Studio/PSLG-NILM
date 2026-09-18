"""Prepare all control labels on one shared source-only final-block support."""
from __future__ import annotations
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from .activity_labels import activity_projection
from .common import signature


def fit_label_bundles(records, activities, mappings, k, seed=42, include_shape=False):
    if not records or any(r.split != 'train' for r in records):
        raise ValueError('Dictionaries/normalization may only fit source training')
    bundles, point_values, block_values, block_keys, shape_values = {}, [], [], [], []
    for r in records:
        observed = np.isfinite(r.y)
        ox = np.isfinite(r.x) if r.x_observed is None else r.x_observed
        o, aid, activity_valid = activity_projection(len(r.t), activities[r.id], observed & ox)
        m = mappings[r.id]
        if len(m['state_full_merge']) != len(r.t):
            raise ValueError('State timeline mismatch')
        valid = activity_valid & ~m['context_conflict'] & ((o == 0) | (m['state_full_merge'] >= 0))
        p = np.where(o == 1, m['state_full_merge'] + 1, 0)
        p[~valid] = -100
        if np.any(p[valid] > k):
            raise ValueError('Dictionary K mismatch')
        bundles[r.id] = dict(onoff=o, primitive=p, valid=valid, activity_valid=activity_valid,
            activity_id=aid, block_id=m['block_id'], regression_observed=observed,
            state_full_merge=m['state_full_merge'], context_conflict=m['context_conflict'])
        active_observed = activity_valid & (o == 1)
        point_values.extend(r.y[active_observed].tolist())
        for bid in np.unique(m['block_id'][valid & (o == 1)]):
            if bid < 0:
                continue
            mask = valid & (o == 1) & (m['block_id'] == bid)
            block_keys.append((r.id, int(bid)))
            block_values.append(float(r.y[mask].mean()))
            ids = np.flatnonzero(mask)
            values = r.y[ids].astype(float)
            times = r.t[ids]
            adjacent = np.diff(ids) == 1
            rates = np.abs(np.diff(values)[adjacent] / np.diff(times)[adjacent])
            span = float(times[-1]-times[0])
            shape_values.append([float(values.mean()), float(values.std()),
                float(np.percentile(values, 90)-np.percentile(values, 10)),
                float((values[-1]-values[0])/span) if span else 0.,
                float(np.median(rates)) if len(rates) else 0.,
                float(np.log1p(span)), float(not len(rates))])
    if len(set(point_values)) < k or len(set(block_values)) < k:
        raise ValueError('Insufficient distinct source power/block means for requested K')
    bmodel = KMeans(k, n_init=30, random_state=seed).fit(np.asarray(point_values).reshape(-1, 1))
    bpmodel = KMeans(k, n_init=30, random_state=seed).fit(np.asarray(block_values).reshape(-1, 1))
    block_classes = dict(zip(block_keys, bpmodel.labels_ + 1))
    if include_shape:
        scaler = StandardScaler().fit(shape_values)
        smodel = KMeans(k, n_init=30, random_state=seed).fit(scaler.transform(shape_values))
        if len(set(smodel.labels_)) != k:
            raise ValueError('Insufficient distinct shape classes')
        shape_classes = dict(zip(block_keys, smodel.labels_ + 1))
    for r in records:
        z = bundles[r.id]
        active = z['valid'] & (z['onoff'] == 1)
        z['bins'], z['segment_bins'] = np.zeros(len(r.t), np.int64), np.zeros(len(r.t), np.int64)
        if include_shape:
            z['shape_bins'] = np.zeros(len(r.t), np.int64)
            z['bins_full'] = np.zeros(len(r.t), np.int64)
            full = z['activity_valid'] & (z['onoff'] == 1)
            z['bins_full'][full] = bmodel.predict(r.y[full, None].astype(np.float64)) + 1
            z['bins_full'][~z['activity_valid']] = -100
        if active.any():
            z['bins'][active] = bmodel.predict(r.y[active, None].astype(np.float64)) + 1
        for bid in np.unique(z['block_id'][active]):
            mask = active & (z['block_id'] == bid)
            z['segment_bins'][mask] = block_classes[(r.id, int(bid))]
            if include_shape:
                z['shape_bins'][mask] = shape_classes[(r.id, int(bid))]
        for key in ('bins', 'segment_bins'):
            z[key][~z['valid']] = -100
        if include_shape:
            z['shape_bins'][~z['valid']] = -100
    scale = max(1., float(np.percentile(point_values, 99)))
    meta = dict(k=k, scale=scale, label_definition='original_activity_core_v1',
        bin_centers=bmodel.cluster_centers_.ravel().tolist(),
        block_bin_centers=bpmodel.cluster_centers_.ravel().tolist(),
        train_signature=signature(sorted((r.id, r.fingerprint) for r in records)),
        BP_definition='observed mean within final P block intersected with core; conditional control')
    if include_shape:
        meta['shape'] = dict(features=['mean', 'std', 'q90_q10', 'endpoint_rate', 'median_abs_adjacent_rate',
            'log_observed_span', 'no_adjacent_pair'], mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
            centers=smodel.cluster_centers_.tolist(), seed=seed, n_init=30,
            duration_definition='span between first/last observed core point; not complete-state duration')
        meta['optional_support_cases'] = ['O_common', 'B_full'] if any(
            np.any(z['activity_valid'] != z['valid']) for z in bundles.values()) else []
    return bundles, meta
