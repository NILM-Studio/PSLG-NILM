"""Validate the optimization against every available exact official result."""
from pathlib import Path
import time
import numpy as np
import review
from pelt_prefix import run_cached
from models.time_segmentation.comparison import Unsupported

plan = review.load(review.PLAN)
checks, records = [], []
max_error = 0.
start_time = time.monotonic()
for obj in plan['objects']:
    sources = sorted((review.EXP/'executions'/('review_'+obj['object']+'_pelt_rbf')).glob('*/analysis/records.json'))
    reference = {(r['activity_id'], r['config_id']): r for r in review.load(sources[-1])}
    for a in obj['activities']:
        with np.load(a['npz']) as z:
            x = z['power'].astype(np.float64)
        cache = {}
        for config in obj['grids']['pelt-rbf']:
            row = dict(object=obj['object'], activity_id=a['activity_id'], split=a['split'],
                       input_sha256=a['npz_sha256'], config_id=config['config_id'], params=config['params'], seed=0)
            try:
                cuts, info = run_cached(x, config['params'], cache)
                row.update(status='success', cuts=cuts, metadata=info)
            except Unsupported as error:
                row.update(status='unsupported', reason=str(error))
            if (a['activity_id'], config['config_id']) in reference:
                old = reference[a['activity_id'], config['config_id']]
                assert old['status'] == row['status'], (obj['object'], a['activity_id'], config)
                if row['status'] == 'success':
                    assert old['cuts'] == row['cuts'], (obj['object'], a['activity_id'], config, old['cuts'], row['cuts'])
                    assert old['metadata']['gamma'] == row['metadata']['gamma']
                checks.append(dict(object=obj['object'], activity_id=a['activity_id'], config_id=config['config_id'], status='exact'))
            records.append(row)
            review.save(review.OUT/'analysis/records.json', records)
            print('VALIDATED', obj['object'], a['activity_id'], config['config_id'], len(checks), flush=True)
        if 'cost' in cache:
            cost = cache['cost']
            rng = np.random.default_rng(0)
            for _ in range(32):
                lo = int(rng.integers(0, len(x)-1)); hi = int(rng.integers(lo+1, len(x)+1))
                expected = float(cost.official.error(lo, hi)); actual = float(cost.error(lo, hi))
                max_error = max(max_error, abs(expected-actual))
                assert abs(expected-actual) <= 1e-7 + 1e-10*abs(expected)
review.save(review.OUT/'analysis/validation.json', dict(status='passed', exact_existing_results=len(checks),
            total_conditions=len(records), max_cost_abs_error=max_error, elapsed_seconds=time.monotonic()-start_time))
print('PASSED', len(checks), 'exact official comparisons,', len(records), 'complete conditions', flush=True)
