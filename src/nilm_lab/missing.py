"""Offline isolated-hole context and observation-only primitive support."""
import numpy as np


def context_arrays(x, y):
    x, y = np.asarray(x).copy(), np.asarray(y)
    ox, oy = np.isfinite(x), np.isfinite(y)
    def holes(observed):
        return (~observed & np.r_[False, observed[:-1]] & np.r_[observed[1:], False])
    hx, hy = holes(ox), holes(oy)
    ix = np.flatnonzero(hx)
    x[ix] = x[ix - 1]
    return x, (ox | hx) & (oy | hy), ox


def observed_intervals(r, threshold, segmenter, censor=False):
    from .data import ranges
    observed = np.isfinite(r.x) & np.isfinite(r.y)
    if r.x_observed is not None:
        observed &= r.x_observed
    intervals, excluded = [], 0
    for a, b in ranges(observed & (r.y > threshold)):
        values = r.y[a:b]
        cp = sorted(set(int(i) for i in segmenter(values) if 0 < int(i) < len(values)))
        bounds = [0] + cp + [len(values)]
        left_censored = a == 0 or not observed[a-1]
        right_censored = b == len(r.y) or not observed[b]
        for s, e in zip(bounds[:-1], bounds[1:]):
            if censor and ((s == 0 and left_censored) or (e == len(values) and right_censored)):
                excluded += 1
                continue
            intervals.append(dict(record_id=r.id, start=a+s, end=a+e))
    return intervals, excluded
