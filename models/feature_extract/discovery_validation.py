"""Pure source-only activity-grouped temporal holdout and normalization."""
import numpy as np


def temporal_split(indices, activities, records, fraction=.8, gap_seconds=3600):
    cuts = {r['id']: int(r['start_timestamp'] + fraction *
            (r['end_timestamp_exclusive']-r['start_timestamp'])) for r in records}
    by_file = {a['csv_idx']: a for a in activities}
    train, val, excluded = [], [], []
    for i, row in enumerate(indices):
        a = by_file[int(row[0])]
        cut = cuts[a['record_id']]
        if a['context_end_exclusive'] <= cut-gap_seconds/2:
            train.append(i)
        elif a['context_start'] >= cut+gap_seconds/2:
            val.append(i)
        else:
            excluded.append(i)
    if not train or not val:
        raise ValueError('Temporal discovery validation has no training/validation activities')
    if set(indices[train, 0]) & set(indices[val, 0]):
        raise ValueError('An activity crossed discovery split')
    return dict(train_ids=train, val_ids=val, excluded_ids=excluded, cutoff_by_record=cuts,
                gap_seconds=gap_seconds, fraction=fraction, train_csv_ids=sorted(set(indices[train, 0].tolist())),
                val_csv_ids=sorted(set(indices[val, 0].tolist())))


def training_minmax(X, lengths, ids):
    lengths = np.asarray(lengths).ravel()
    mask = np.arange(X.shape[1])[None] < lengths[:, None]
    fitmask = mask.copy()
    fitmask[~np.isin(np.arange(len(X)), ids)] = False
    limits = np.stack([np.percentile(X[:, :, j][fitmask], [1, 99]) for j in range(X.shape[2])])
    result = X.astype(np.float32).copy()
    for j, (lo, hi) in enumerate(limits):
        result[:, :, j] = (np.clip(result[:, :, j], lo, hi)-lo)/(hi-lo+1e-7)
    result[~mask] = 0
    return result, limits
