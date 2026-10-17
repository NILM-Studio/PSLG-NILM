"""Independent input, source-record and SVG-coordinate audit of the complete gallery."""
import collections
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import review

plan = review.load(review.PLAN)
bundle = review.load(review.REPORT/'figure_manifest.json')
assert bundle['figure_count'] == 160 and not bundle['missing']
assert bundle['plan_sha256'] == review.sha(review.PLAN)
activity_keys = collections.Counter()
config_count = 0
panels = 0
unsupported = collections.Counter()
reasons = []
for obj in plan['objects']:
    refs = {r['activity_id']: r for r in review.load(obj['primglr_records'])}
    default_clasp = next(c['config_id'] for c in obj['grids']['clasp'] if c['is_baseline'])
    manifests = {m: review.load(review.REPORT/'manifests'/f"{obj['object']}_{m.replace('-', '_')}.json")
                 for m in review.METHODS}
    clasp_records = review.load(Path(manifests['clasp']['source_execution'])/'analysis/records.json')
    default_rows = {r['activity_id']: r for r in clasp_records if r['config_id'] == default_clasp}
    for a in obj['activities']:
        assert a['split'] in ('development', 'validation')
        assert review.sha(a['npz']) == a['npz_sha256']
        ref = refs[a['activity_id']]
        assert ref['input_sha256'] == a['npz_sha256']
        assert ref['full_context_cuts'] == a['primglr_reference']['full_context_cuts']
        with np.load(a['npz']) as z:
            ts = z['timestamp'].copy()
        for m, manifest in manifests.items():
            config_count += len(obj['grids'][m])
            fig = next(f for f in manifest['figures'] if f['activity_id'] == a['activity_id'])
            assert fig['split'] == a['split'] and fig['input_sha256'] == a['npz_sha256']
            records = review.load(Path(manifest['source_execution'])/'analysis/records.json')
            tree = ET.parse(review.REPORT/fig['svg'])
            actual_svg_ids = {e.attrib['id'] for e in tree.iter() if e.attrib.get('id', '').startswith('cut_')}
            expected_svg_ids = set()
            for p in fig['panels']:
                r = records[p['source_record_index']]
                assert (r['object'], r['method'], r['activity_id'], r['config_id'], r['seed']) == (
                    obj['object'], m, a['activity_id'], p['config_id'], p['seed'])
                assert r['input_sha256'] == fig['input_sha256'] and r['status'] == p['status']
                assert r.get('cuts', []) == p['cuts']
                assert p['cut_timestamps'] == [float(ts[c]) for c in p['cuts']]
                assert p['cut_elapsed_seconds'] == [float(ts[c]-ts[0]) for c in p['cuts']]
                expected_svg_ids.update(f"cut_{p['panel']}_{c}" for c in p['cuts'])
                if m == 'clap' and p['status'] == 'success':
                    initial = p['metadata']['initializer_cuts']
                    assert default_rows[a['activity_id']]['status'] == 'success'
                    assert initial == default_rows[a['activity_id']]['cuts']
                    assert set(p['cuts']).issubset(initial)
                if p['status'] == 'unsupported':
                    unsupported[m] += 1
                    reasons.append(dict(object=obj['object'], method=m, activity_id=a['activity_id'],
                                        config_id=p['config_id'], seed=p['seed'], reason=p['reason']))
                panels += 1
            assert actual_svg_ids == expected_svg_ids
            activity_keys[obj['object'], a['split'], a['activity_id'], a['npz_sha256']] += 1
assert len(activity_keys) == 32 and set(activity_keys.values()) == {5}
assert panels == 1952
summary = dict(status='passed', figures=160, formats=['png', 'svg', 'pdf'], panels=panels,
               unique_inputs=32, models_per_input=5, evaluation_inputs=0,
               complete_grid_coverage=True, exact_record_svg_cut_timestamps=True,
               exact_default_Clasp_CLaP_initializer_consistency=True,
               unsupported_panel_counts=dict(unsupported), parameters_frozen=False,
               plan_sha256=review.sha(review.PLAN), gallery_manifest_sha256=review.sha(review.REPORT/'figure_manifest.json'))
review.save(review.OUT/'analysis/validation.json', summary)
review.save(review.REPORT/'independent_qa.json', summary)
review.save(review.REPORT/'unsupported_combinations.json', reasons)
print(json.dumps(summary), flush=True)
