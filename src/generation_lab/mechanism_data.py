"""State-mechanism data contracts. No holdout is implicitly opened."""
import hashlib
import json
from pathlib import Path
import numpy as np

CASES = ('window_u', 'window_p', 'block_p', 'block_u', 'block_shuffle', 'block_robust')
BACKBONES = ('masked_transformer', 'masked_conv')


def matrix():
    rows = []
    for phase, backbones, seeds in [('pilot', BACKBONES[:1], (101, 102)),
                                     ('formal', BACKBONES, range(5))]:
        for backbone in backbones:
            for case in CASES:
                for seed in seeds:
                    rows.append(dict(phase=phase, backbone=backbone, case=case, seed=seed,
                                     data='real', noise=0., duration='real'))
    for case in CASES[:3]:
        for noise in (0., .1, .3):
            for duration in ('fixed', 'variable'):
                for seed in range(3):
                    rows.append(dict(phase='mechanism', backbone=BACKBONES[0], case=case,
                                     seed=seed, data='synthetic', noise=noise, duration=duration))
    return rows


def runs(z):
    cuts = np.r_[0, np.flatnonzero(np.diff(z)) + 1, len(z)]
    return list(zip(cuts[:-1], cuts[1:]))


def synthetic(split, variable, count=120, n_states=5):
    """Known clean structure: ramps, periodic motion, pulses, and plateaus.

    Split RNGs are disjoint; noise is added later, not to the evaluation target.
    """
    rng = np.random.default_rng({'train': 913, 'dev': 1913, 'test': 2913}[split])
    result = []
    for aid in range(count):
        ys, zs = [], []
        for k in rng.permutation(np.arange(1, n_states + 1)):
            n = int(rng.integers(24, 121)) if variable else 64
            t = np.arange(n) * 6.
            phase = np.arange(n) / max(n - 1, 1)
            amp = rng.uniform(.85, 1.15)
            shapes = {1: np.full(n, 80.), 2: 300 + 120 * np.sin(2 * np.pi * t / 72),
                      3: 1200 + 300 * phase, 4: 450 + 180 * (np.sin(2 * np.pi * t / 48) > 0),
                      5: 200 + 350 * np.exp(-phase * 4)}
            ys.append((amp * shapes[int(k)]).astype(np.float32))
            zs.append(np.full(n, k, np.int64))
        y, z = np.concatenate(ys), np.concatenate(zs)
        result.append(dict(id=f'{split}_{aid}', y=y, z=z))
    return result


def load_dictionary(folder, splits=('train', 'dev')):
    folder = Path(folder)
    meta = json.loads((folder / 'manifest.json').read_text())
    n_states = int(meta.get('k', 5))
    result = {s: [] for s in splits}
    for r in meta['records']:
        if r['split'] not in result:
            continue
        p = (folder / r['file']).resolve()
        if not p.is_relative_to(folder.resolve()):
            raise ValueError('Dictionary record escapes its directory')
        if hashlib.sha256(p.read_bytes()).hexdigest() != r['sha256']:
            raise ValueError(f'Changed dictionary record: {p}')
        with np.load(p, allow_pickle=False) as a:
            y, z = a['power'].astype(np.float32), a['P'].astype(np.int64)
        if y.ndim != 1 or y.shape != z.shape:
            raise ValueError('Expected aligned one-dimensional power and P')
        if np.any((z > n_states) | ((z >= 0) & (z < 1))):
            raise ValueError('Labels outside declared dictionary classes')
        valid = np.isfinite(y) & (z >= 1) & (z <= n_states)
        cuts = np.diff(np.r_[False, valid, False].astype(int))
        for lo, hi in zip(np.flatnonzero(cuts == 1), np.flatnonzero(cuts == -1)):
            result[r['split']].append(dict(id=str(r['activity_id']), y=y[lo:hi], z=z[lo:hi]))
    if any(not rows for rows in result.values()):
        raise ValueError('Empty requested split')
    ids = [set(r['id'] for r in rows) for rows in result.values()]
    if any(ids[i] & ids[j] for i in range(len(ids)) for j in range(i)):
        raise ValueError('Activities overlap across splits')
    return result, meta


def examples(records, case, length=128, seed=0, noise=0., scale=1500., n_states=5):
    """Native-time blocks, tiled without stretching; padding is masked.

    All methods cover the same valid points, including short tails. Block duration
    and relative position describe the full block even when it exceeds length.
    Labels are randomized per block (not a global name permutation).
    """
    rng = np.random.default_rng(seed)
    out = []
    for record in records:
        y, z = record['y'] / scale, record['z']
        state_pos, state_duration = np.zeros(len(y),np.float32),np.zeros(len(y),np.float32)
        for a,b in runs(z):
            state_pos[a:b]=np.arange(b-a)/max(b-a-1,1)
            state_duration[a:b]=np.log1p((b-a)*6)/10
        y = y + rng.normal(0, noise, len(y))
        # Sparse impulsive contamination accompanies Gaussian sensor noise.
        if noise:
            y = y + (rng.random(len(y)) < .01) * rng.normal(0, noise * 5, len(y))
        blocks = runs(z) if case.startswith('block') else [(0, len(y))]
        for lo, hi in blocks:
            label = int(rng.integers(1, n_states + 1)) if case == 'block_shuffle' else None
            for start in range(lo, hi, length):
                end = min(start + length, hi)
                n = end - start
                yy, zz, mask = np.zeros(length, np.float32), np.zeros(length, np.int64), np.zeros(length, bool)
                yy[:n], zz[:n], mask[:n] = y[start:end], z[start:end], True
                if case in ('window_u', 'block_u'):
                    zz[:n] = n_states + 1
                if label is not None:
                    zz[:n] = label
                pos, duration = np.zeros(length, np.float32), np.zeros(length, np.float32)
                if case.startswith('block'):
                    pos[:n] = np.arange(start - lo, end - lo) / max(hi - lo - 1, 1)
                    duration[:n] = np.log1p((hi - lo) * 6) / 10
                out.append(dict(y=yy, z=zz, mask=mask, pos=pos, duration=duration,
                                id=record['id'], start=start, end=end))
    return out
