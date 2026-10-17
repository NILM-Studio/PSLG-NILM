"""Add the complete PrimGLR grid on the same two existing activities per object."""
import argparse
import concurrent.futures
import copy
import csv
import itertools
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import review
from create_execution import register
from models.time_segmentation.comparison import SafePrimGLR, run

ADD = review.EXP/'configs/prim_glr_addition.json'


def grid(obj, spec):
    dt = obj['nominal_seconds']
    h20 = obj['object'] == 'refit_h20_tumble_dryer'
    windows = spec['W_seconds'] + (spec['h20_extra_W_seconds'] if h20 else [])
    gaps = spec['min_gap_seconds'] + (spec['h20_extra_min_gap_seconds'] if h20 else [])
    for w, gap, alpha, sigma in itertools.product(windows, gaps, spec['alpha'], spec['dur_sigma_finalists']):
        p = dict(W=max(3, math.ceil(w/dt)), min_gap=max(2, math.ceil(gap/dt)), alpha=alpha,
                 dur_sigma=sigma, g_wf=16, g_env=8, n_clusters=None,
                 W_seconds_target=w, min_gap_seconds_target=gap)
        yield dict(config_id=review.key(p), params=p,
                   is_baseline=(w == 120 and gap == 30 and alpha == .9 and sigma == .35))


def prepare():
    assert not ADD.exists()
    original = review.REPORT/'figure_manifest.json'
    bundle = review.load(original)
    assert bundle['figure_count'] == 160
    source = review.ROOT/'runs/20261016_four_object_segmentation_hyperparameters/configs/search_space.json'
    spec = review.load(source)['prim_glr']
    objects = []
    for obj in review.load(review.PLAN)['objects']:
        configs = list(grid(obj, spec))
        assert len(configs) == (240 if obj['object'] == 'refit_h20_tumble_dryer' else 144)
        assert sum(c['is_baseline'] for c in configs) == 1
        objects.append(dict(object=obj['object'], configurations=configs))
    cfg = dict(base_plan_sha256=review.sha(review.PLAN), original_gallery_manifest_sha256=review.sha(original),
               search_space_path=str(source), search_space_sha256=review.sha(source), prim_glr_space=spec,
               sigma_scope='full_cartesian_product_not_finalists_only',
               original_figures=160, added_figures=32, final_figures=192, added_panels=4800,
               execution_backend='cpu', max_workers=6, selected_reference=dict(W_seconds=120, gap_seconds=30, alpha=.9, dur_sigma=.35),
               objects=objects)
    review.save(ADD, cfg)
    shutil.copy2(review.REPORT/'report.md', review.OUT/'analysis/report_before_prim_glr.md')
    review.save(review.OUT/'analysis/validation.json', dict(status='passed', configurations=sum(len(o['configurations']) for o in objects),
                added_figures=32, added_panels=4800, configuration_sha256=review.sha(ADD)))
    print('PREPARED: 32 extra figures / 4800 panels / 2400 object-configurations')


def cached_run(x, params, cache):
    model = SafePrimGLR(config={k:v for k,v in params.items() if not k.endswith('_seconds_target')})
    captured = {}
    def features(sample, width):
        if width not in cache:
            matrix = SafePrimGLR._build_features(sample, width)
            scores = SafePrimGLR._glr_score(matrix, width)
            matrix.flags.writeable = False
            scores.flags.writeable = False
            cache[width] = (matrix, scores)
        return cache[width][0]
    def score(matrix, width):
        captured['width'] = width
        return cache[width][1]
    model._build_features = features
    model._glr_score = score
    cuts = model.train(x)
    return cuts, dict(source='unchanged SafePrimGLR.train with cached features and GLR scores',
                      effective_W_points=captured.get('width'), internal_random_state=0,
                      resolved={k:getattr(model,k) for k in ('W','min_gap','alpha','dur_sigma','g_wf','g_env','n_clusters')})


def worker(ident):
    cfg = review.load(ADD)
    assert cfg['base_plan_sha256'] == review.sha(review.PLAN)
    assert cfg['original_gallery_manifest_sha256'] == review.sha(review.REPORT/'figure_manifest.json')
    obj = next(o for o in review.load(review.PLAN)['objects'] if o['object'] == ident)
    configs = next(o['configurations'] for o in cfg['objects'] if o['object'] == ident)
    review.provenance()
    for p in [Path(__file__), review.ROOT/'models/time_segmentation/prim_glr.py']:
        shutil.copy2(p, review.OUT/'analysis/source_code'/p.name)
    review.save(review.OUT/'inputs.json', dict(addition_sha256=review.sha(ADD), object=ident,
                activity_hashes={str(a['activity_id']):a['npz_sha256'] for a in obj['activities']},
                model_sha256=review.sha(review.ROOT/'models/time_segmentation/prim_glr.py')))
    records, figures, direct_checks = [], [], []
    check_ids = {configs[0]['config_id'], configs[-1]['config_id'], next(c['config_id'] for c in configs if c['is_baseline'])}
    for activity in obj['activities']:
        assert activity['split'] in ('development','validation')
        assert review.sha(activity['npz']) == activity['npz_sha256']
        with np.load(activity['npz']) as z:
            x = z['power'].astype(np.float64)
        cache, current = {}, []
        for c in configs:
            st = time.monotonic()
            cuts, metadata = cached_run(x, c['params'], cache)
            assert cuts == sorted(set(cuts)) and all(0 < k < len(x) for k in cuts)
            if c['is_baseline']:
                assert cuts == activity['primglr_reference']['full_context_cuts'], (ident, activity['activity_id'])
                metadata['matches_provided_reference'] = True
            if c['config_id'] in check_ids:
                raw_params = {k:v for k,v in c['params'].items() if not k.endswith('_seconds_target')}
                direct, _ = run(x, 'prim-glr', raw_params, seed=0)
                assert direct == cuts, (ident, activity['activity_id'], c['config_id'])
                direct_checks.append(dict(activity_id=activity['activity_id'], config_id=c['config_id'], exact=True))
            r = dict(object=ident, method='prim-glr', split=activity['split'], activity_id=activity['activity_id'],
                     input_sha256=activity['npz_sha256'], config_id=c['config_id'], params=c['params'], seed=0,
                     record_index=len(records), status='success', cuts=cuts, metadata=metadata,
                     core_cuts=sum(activity['core_a']<k<activity['core_b'] for k in cuts), wall_seconds=time.monotonic()-st)
            records.append(r);current.append(r)
            if len(current)%12 == 0 or len(current) == len(configs):
                review.save(review.OUT/'analysis/records.json', records)
                review.save(review.OUT/'analysis/progress.json', dict(object=ident, completed=len(records), total=2*len(configs)))
                print('PROGRESS',ident,len(records),'/',2*len(configs),flush=True)
        figures.append(review.draw(obj, 'prim-glr', activity, configs, current))
    summary = dict(object=ident, method='prim-glr', config_count=len(configs), figures=figures,
                   addition_sha256=review.sha(ADD), source_execution=str(review.OUT),
                   records_sha256=review.sha(review.OUT/'analysis/records.json'), direct_checks=direct_checks)
    review.save(review.OUT/'analysis/gallery_manifest.json', summary)
    review.save(review.REPORT/'manifests'/f'{ident}_prim_glr.json', summary)
    review.save(review.OUT/'analysis/validation.json', dict(status='passed', figures=2, panels=2*len(configs),
                direct_checks=len(direct_checks), selected_reference_matches=2))


def validate():
    cfg = review.load(ADD)
    old = review.load(review.REPORT/'figure_manifest.json')
    assert review.sha(review.REPORT/'figure_manifest.json') == cfg['original_gallery_manifest_sha256']
    # Verify the old 160 figures remain byte-identical as the increment is added.
    for f in old['figures']:
        for ext in ('png','svg','pdf'):
            assert review.sha(review.REPORT/f[ext]) == f[ext+'_sha256']
    plan = review.load(review.PLAN)
    figures, manifests, choices = [], [], []
    direct_checks = 0
    for obj in plan['objects']:
        configs = next(o['configurations'] for o in cfg['objects'] if o['object'] == obj['object'])
        manifest_path = review.REPORT/'manifests'/f"{obj['object']}_prim_glr.json"
        m = review.load(manifest_path)
        assert m['addition_sha256'] == review.sha(ADD) and len(m['figures']) == 2
        execution = Path(m['source_execution'])
        assert review.load(execution/'run.json')['exit_code'] == 0
        rows = review.load(execution/'analysis/records.json')
        direct_checks += len(m['direct_checks'])
        for f in m['figures']:
            a = next(a for a in obj['activities'] if a['activity_id'] == f['activity_id'])
            assert f['input_sha256'] == a['npz_sha256'] and f['split'] == a['split']
            assert {p['config_id'] for p in f['panels']} == {c['config_id'] for c in configs}
            assert len(f['panels']) == len(configs)
            with np.load(a['npz']) as z: t = z['timestamp'].copy()
            for ext in ('png','svg','pdf'):
                assert review.sha(review.REPORT/f[ext]) == f[ext+'_sha256']
            with Image.open(review.REPORT/f['png']) as im: im.verify()
            tree = ET.parse(review.REPORT/f['svg'])
            svg_ids = {e.attrib['id'] for e in tree.iter() if e.attrib.get('id','').startswith('cut_')}
            expected = set()
            for p in f['panels']:
                r = rows[p['source_record_index']]
                assert r['cuts'] == p['cuts'] and r['config_id'] == p['config_id']
                assert p['cut_timestamps'] == [float(t[k]) for k in p['cuts']]
                assert p['cut_elapsed_seconds'] == [float(t[k]-t[0]) for k in p['cuts']]
                expected.update(f"cut_{p['panel']}_{k}" for k in p['cuts'])
                if p['is_baseline']: assert p['cuts'] == a['primglr_reference']['full_context_cuts']
            assert svg_ids == expected
            figures.append(f)
        manifests.append(dict(path=str(manifest_path.relative_to(review.REPORT)), sha256=review.sha(manifest_path)))
        for c in configs:
            choices.append(dict(object=obj['object'], config_id=c['config_id'], params=__import__('json').dumps(c['params'],sort_keys=True),
                                selected_reference=c['is_baseline']))
    assert len(figures)==32 and sum(len(f['panels']) for f in figures)==4800 and direct_checks==96
    review.save(review.REPORT/'prim_glr_figure_manifest.json', dict(status='passed', figures=figures, manifests=manifests,
                added_figures=32, added_panels=4800, original_figures_unchanged=160, direct_checks=direct_checks,
                selected_reference_matches=32, addition_sha256=review.sha(ADD)))
    with (review.REPORT/'prim_glr_parameter_choices.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(choices[0]));writer.writeheader();writer.writerows(choices)
    review.save(review.OUT/'analysis/validation.json', dict(status='passed', new_figures=32, new_panels=4800,
                original_figures_byte_identical=160, original_activity_hashes_matched=True, direct_source_checks=96,
                selected_reference_matches=32, combined_figures=192))
    print('QA PASSED: +32 figures, 4800 panels, original 160 figures unchanged, 96 exact direct-source checks')


def coordinate():
    cfg = review.load(ADD)
    def launch(ident):
        ex=register(review.EXP.name,'review_'+ident+'_prim_glr_complete_grid',dict(addition_sha256=review.sha(ADD),object=ident))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1')
        code=subprocess.call([sys.executable,str(review.ROOT/'scripts/run_execution.py'),str(ex),'--',sys.executable,str(Path(__file__).resolve()),'--object',ident],cwd=review.ROOT,env=env)
        return dict(object=ident,execution=str(ex),exit_code=code)
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        fs=[pool.submit(launch,o['object']) for o in cfg['objects']]
        for f in concurrent.futures.as_completed(fs):
            results.append(f.result());review.save(review.EXP/'data/prim_glr_progress.json',dict(completed=len(results),total=16,jobs=results))
    assert all(r['exit_code']==0 for r in results),results
    for condition, command in [('prim_glr_increment_independent_qa',[str(Path(__file__).resolve()),'--validate']),
                               ('render_markdown_with_prim_glr',[str(Path(__file__).with_name('build_markdown_report.py'))])]:
        ex=register(review.EXP.name,condition,dict(addition_sha256=review.sha(ADD)))
        code=subprocess.call([sys.executable,str(review.ROOT/'scripts/run_execution.py'),str(ex),'--',sys.executable,*command],cwd=review.ROOT)
        assert code==0,ex
    review.save(review.OUT/'analysis/validation.json',dict(status='passed',new_figures=32,total_figures=192,jobs=results))
    print('COMPLETE: Markdown report now includes 192 figures',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--coordinate',action='store_true');p.add_argument('--validate',action='store_true');p.add_argument('--object')
    a=p.parse_args()
    if a.prepare:prepare()
    elif a.coordinate:coordinate()
    elif a.validate:validate()
    else:assert a.object;worker(a.object)
