"""Complete the last PELT gallery using validated official-kernel sum caching."""
import copy
import shutil
from pathlib import Path
import numpy as np
import review

plan = review.load(review.PLAN)
review.provenance()
files = [Path(__file__), Path(__file__).with_name('pelt_prefix.py'), Path(__file__).with_name('validate_pelt_prefix.py')]
for p in files:
    shutil.copy2(p, review.OUT/'analysis/source_code'/p.name)
review.save(review.OUT/'analysis/pelt_cache_provenance.json',
            [{'path': str(p), 'sha256': review.sha(p)} for p in files])
obj = next(o for o in plan['objects'] if o['object'] == 'darck_fridge')
validation = Path((review.EXP/'data/pelt_cache_execution.txt').read_text().strip())
assert review.load(validation/'run.json')['exit_code'] == 0
proof = review.load(validation/'analysis/validation.json')
assert proof['status'] == 'passed' and proof['exact_existing_results'] >= 250 and proof['total_conditions'] == 256
cached = {(r['activity_id'], r['config_id']): r for r in review.load(validation/'analysis/records.json')
          if r['object'] == obj['object']}
original = sorted((review.EXP/'executions/review_darck_fridge_pelt_rbf').glob('*/analysis/records.json'))[-1]
prior = {(r['activity_id'], r['config_id']): r for r in review.load(original)}
records, figures = [], []
for a in obj['activities']:
    current = []
    assert review.sha(a['npz']) == a['npz_sha256']
    for config in obj['grids']['pelt-rbf']:
        k = (a['activity_id'], config['config_id'])
        if k in prior:
            assert prior[k]['status'] == cached[k]['status']
            assert prior[k].get('cuts', []) == cached[k].get('cuts', [])
            row = copy.deepcopy(prior[k])
            row['reused_records'] = str(original)
            row['reused_records_sha256'] = review.sha(original)
        else:
            row = copy.deepcopy(cached[k])
            row['reused_records'] = str(validation/'analysis/records.json')
            row['reused_records_sha256'] = review.sha(validation/'analysis/records.json')
            row['core_cuts'] = sum(a['core_a'] < c < a['core_b'] for c in row.get('cuts', []))
        row['method'] = 'pelt-rbf'
        row['record_index'] = len(records)
        records.append(row)
        current.append(row)
    figures.append(review.draw(obj, 'pelt-rbf', a, obj['grids']['pelt-rbf'], current))
review.save(review.OUT/'analysis/records.json', records)
summary = dict(object=obj['object'], method='pelt-rbf', config_count=8, seeds=[0],
               plan_sha256=review.sha(review.PLAN), figures=figures, source_execution=str(review.OUT),
               records_sha256=review.sha(review.OUT/'analysis/records.json'),
               status_counts={s: sum(r['status'] == s for r in records) for s in ('success', 'unsupported', 'failed')},
               cache_validation_execution=str(validation), official_existing_comparisons=proof['exact_existing_results'])
assert not summary['status_counts']['failed']
review.save(review.OUT/'analysis/gallery_manifest.json', summary)
review.save(review.REPORT/'manifests/darck_fridge_pelt_rbf.json', summary)
review.save(review.REPORT/'pelt_cache_validation.json', proof)
review.save(review.OUT/'analysis/validation.json', dict(status='passed', figures=2, proof=proof,
            reused_direct_conditions=len(prior), added_cached_conditions=len(records)-len(prior)))
print('Completed final two PELT figures; existing official outputs retained, missing outputs use validated identical RBF formula')
