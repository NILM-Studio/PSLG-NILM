"""Verify the cached FLUSS adapter against direct project API on a real input."""
import os
from pathlib import Path
import numpy as np
import review
from models.time_segmentation.comparison import run, Unsupported

plan = review.load(review.PLAN)
obj = next(o for o in plan['objects'] if o['object'] == 'ukdale_h2_washing_machine')
activity = obj['activities'][0]
assert review.sha(activity['npz']) == activity['npz_sha256']
with np.load(activity['npz']) as z:
    x = z['power'].copy()
cache = {}
checks = []
for config in obj['grids']['fluss-threshold']:
    p = {k: v for k, v in config['params'].items() if not k.endswith('_seconds_target')}
    if p['threshold'] not in (.45, .95):
        continue
    try:
        expected, _ = run(x, 'fluss-threshold', p)
    except Unsupported as error:
        try:
            review.cached_fluss(x, p, cache)
        except Unsupported:
            checks.append(dict(config_id=config['config_id'], status='same_unsupported'))
            continue
        raise AssertionError('Cached implementation accepted unsupported input') from error
    actual, _ = review.cached_fluss(x, p, cache)
    assert actual == expected, (config, actual, expected)
    checks.append(dict(config_id=config['config_id'], status='exact_cuts', cuts=actual))
assert len(checks) == 6
review.save(Path(os.environ['PSLG_EXECUTION_DIR'])/'analysis/validation.json',
            dict(status='passed', input_sha256=activity['npz_sha256'], checks=checks))
print('FLUSS cache/direct real-input equivalence: 6 conditions passed')
