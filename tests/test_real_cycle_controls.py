"""Unchanged-waveform controls, deterministic paired sampling and honest limits."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import numpy as np
import pandas as pd
import pytest

from test_composition_evaluation import case, cycle
from test_composition_study import source_run
from scripts.run_primitive_composition import run_study
from scripts.run_real_cycle_controls import build_controls
from src.generation.real_cycle_controls import evaluate_real_controls


def inputs():
    training = [cycle(i, levels=(100+i*10, 1000+i*10, 100+i*10)) for i in range(6)]
    validation = [cycle(10+i) for i in range(3)]
    cases = [case(training[0]), case(training[1])]
    manifest = {'configuration': {'sample_period': 6}, 'methods': ['random', 'transition_dp'],
                'budgets': [{'seed': 42, 'budget_tag': '100pct', 'selected_activity_ids': list(range(6))}],
                'cases': cases}
    return manifest, training, validation


def test_equal_size_real_draws_are_budget_local_reproducible_and_unmodified():
    manifest, train, val = inputs()
    original = [c.power.copy() for c in train+val]
    result = evaluate_real_controls(manifest, train, val, repeats=12)
    row = result['strata'][0]
    assert len(row['matched_sample_activity_ids']) == 12
    assert all(len(ids)==len(set(ids))==2 and set(ids)<=set(range(6)) for ids in row['matched_sample_activity_ids'])
    assert row['status'] == 'low_support'  # two generated cycles, three validation
    for metric, values in row['cycle_metrics'].items():
        assert values['methods']['random']['wasserstein'] == values['controls']['original_anchors']['wasserstein']
        assert values['methods']['random']['delta_vs_original_anchors'] == 0
        assert values['controls']['matched_budget_real']['available_draws'] == 12
    for before, after in zip(original, train+val):
        np.testing.assert_array_equal(before, after.power)
    reverse = copy.deepcopy(manifest)
    reverse['cases'].reverse()
    reverse['budgets'][0]['selected_activity_ids'].reverse()
    again = evaluate_real_controls(reverse, reversed(train), reversed(val), repeats=12)
    assert again['strata'] == result['strata']
    altered_val = [cycle(10+i, levels=(10,2000,10)) for i in range(3)]
    changed = evaluate_real_controls(manifest, train, altered_val, repeats=12)
    assert changed['strata'][0]['matched_sample_activity_ids'] == row['matched_sample_activity_ids']
    assert changed['strata'][0]['cycle_metrics'] != row['cycle_metrics']
    other_seed = evaluate_real_controls(manifest, train, val, repeats=12, sampling_seed=9)
    assert other_seed['strata'][0]['matched_sample_activity_ids'] != row['matched_sample_activity_ids']


def test_complete_budget_sample_is_one_unique_subset_not_independent_replicates():
    manifest, train, val = inputs()
    manifest['budgets'][0]['selected_activity_ids'] = [0,1]
    row = evaluate_real_controls(manifest, train, val, repeats=10)['strata'][0]
    assert row['unique_matched_subsets'] == 1
    assert row['matched_sample_activity_ids'] == [[0,1]]*10
    assert len(row['full_train_activity_ids']) == 6 and len(row['budget_pool_activity_ids']) == 2


def test_distances_have_known_physical_units_without_resampling_real_cycles():
    manifest, train, val = inputs()
    result = evaluate_real_controls(manifest, train, val, repeats=5)
    row = result['strata'][0]
    energy = row['cycle_metrics']['energy_wh']
    assert energy['controls']['full_train_context']['wasserstein'] == pytest.approx(0.375)
    assert energy['controls']['original_anchors']['wasserstein'] == pytest.approx(0.075)
    expected = [sum(ids)*0.075 for ids in row['matched_sample_activity_ids']]
    assert energy['controls']['matched_budget_real']['wasserstein_by_draw'] == pytest.approx(expected)
    assert result['metric_units']['energy_wh'] == 'Wh'
    longer = [cycle(10, states=(0,1,0,1,0), levels=(100,1000,100,1000,100))]
    row = evaluate_real_controls(manifest, train, longer, repeats=3)['strata'][0]
    for metric, expected in [('duration_seconds',36), ('energy_wh',5.425), ('mean_watts',55), ('peak_watts',5)]:
        assert row['cycle_metrics'][metric]['controls']['original_anchors']['wasserstein'] == pytest.approx(expected)


def test_state_pair_absent_from_some_draws_is_not_zero_or_silently_dropped():
    manifest, train, val = inputs()
    train[2:] = [cycle(i, states=(3,4,3)) for i in range(2,6)]
    row = evaluate_real_controls(manifest, train, val, repeats=30)['strata'][0]
    pair = next(r for r in row['transition_metrics'] if (r['left_state'],r['right_state']) == (0,1))
    sampled = pair['metrics']['signed_jump_watts']['controls']['matched_budget_real']
    absent = [not set(ids)&{0,1} for ids in row['matched_sample_activity_ids']]
    assert 0 < sum(absent) < 30
    assert sampled['missing_draws'] == sum(absent)
    assert sampled['wasserstein_distribution']['count'] == 30-sum(absent)
    for index, missing in enumerate(absent):
        assert (sampled['wasserstein_by_draw'][index] is None) == missing
        assert sampled['source_cycles_by_draw'][index] == len(set(row['matched_sample_activity_ids'][index])&{0,1})
        assert sampled['status_by_draw'][index] == ('no_source_observations' if missing else 'low_support')


def test_empty_case_list_is_not_reported_ready():
    manifest, train, val = inputs()
    manifest['cases'] = []
    result = evaluate_real_controls(manifest, train, val, repeats=3)
    assert result['status'] == 'no_completed_cases'
    assert result['strata'] == [] and result['paired_cases'] == 0


def test_missing_validation_and_missing_state_pairs_remain_null():
    manifest, train, val = inputs()
    result = evaluate_real_controls(manifest, train, [], repeats=3)
    assert result['status'] == 'no_matching_validation'
    row = result['strata'][0]['cycle_metrics']['energy_wh']
    assert row['methods']['random']['wasserstein'] is None
    assert row['methods']['random']['delta_vs_original_anchors'] is None
    assert row['controls']['matched_budget_real']['missing_draws'] == 3
    different = [cycle(10, states=(3,4,3))]
    result = evaluate_real_controls(manifest, train, different, repeats=3)
    edges = result['strata'][0]['transition_metrics']
    a = next(r for r in edges if (r['left_state'],r['right_state']) == (0,1))
    assert a['metrics']['signed_jump_watts']['methods']['random']['status'] == 'no_matching_validation'
    b = next(r for r in edges if (r['left_state'],r['right_state']) == (3,4))
    assert b['metrics']['signed_jump_watts']['methods']['random']['status'] == 'no_source_observations'
    json.dumps(result, allow_nan=False)


def test_support_and_modes_are_not_pooled_and_edges_count_distinct_cycles():
    manifest, train, val = inputs()
    manifest['cases'][1] = case(train[1], supported=0)
    train.append(cycle(6, group=(0,1)))
    manifest['cases'].append(case(train[-1]))
    manifest['budgets'][0]['selected_activity_ids'].append(6)
    val = [cycle(10, states=(0,1,0,1,0), levels=(100,1000,100,1000,100))]
    result = evaluate_real_controls(manifest, train, val, repeats=4)
    assert len(result['strata']) == 3
    row = next(r for r in result['strata'] if r['class_mode']==[0,0] and r['support_status']=='full')
    jump = next(e for e in row['transition_metrics'] if e['left_state']==0)['metrics']['signed_jump_watts']
    assert jump['validation']['count'] == 2 and jump['validation_cycles'] == 1
    assert jump['methods']['random']['status'] == 'low_support'
    assert result['status'] == 'partial_validation_coverage'


@pytest.mark.parametrize('kind', ['overlap', 'budget', 'duplicate_case', 'template', 'duration', 'duplicate_real'])
def test_inconsistent_membership_or_templates_rejected(kind):
    manifest, train, val = inputs()
    if kind == 'overlap': val.append(train[0])
    if kind == 'budget': manifest['budgets'][0]['selected_activity_ids'].append(10)
    if kind == 'duplicate_case': manifest['cases'].append(copy.deepcopy(manifest['cases'][0]))
    if kind == 'template': manifest['cases'][0]['template'][0]['length_samples'] += 1
    if kind == 'duration': manifest['cases'][0]['results']['random']['metrics']['duration_seconds'] += 6
    if kind == 'duplicate_real': train.append(train[0])
    with pytest.raises(ValueError): evaluate_real_controls(manifest, train, val)


@pytest.mark.parametrize('options', [{'repeats':0},{'repeats':1.5},{'repeats':True},{'sampling_seed':-1}])
def test_invalid_control_parameters(options):
    manifest, train, val = inputs()
    with pytest.raises(ValueError): evaluate_real_controls(manifest, train, val, **options)


def test_complete_reader_checks_sources_writes_separate_report_and_refuses_overwrite(source_run, tmp_path):
    study_dir = source_run/'composition'
    run_study(source_run, study_dir, ratios=[0.4,1], max_anchors=2)
    originals = {p:p.read_bytes() for p in source_run.rglob('*') if p.is_file()}
    output = tmp_path/'controls'
    result = build_controls(source_run, study_dir, output, repeats=5)
    assert result['paired_cases'] == 4
    assert result['provenance']['source_data_verified_unchanged']
    assert result['provenance']['raw_anchor_references_checked'] == 4
    assert '不是置信区间' in (output/'report.md').read_text()
    measurements = json.loads((output/'real_cycle_measurements.json').read_text())
    assert set(measurements) == {'train','validation'}
    assert [r['activity_id'] for r in measurements['train']] == list(range(5))
    assert [r['activity_id'] for r in measurements['validation']] == [5]
    assert all(p.read_bytes()==before for p,before in originals.items())
    with pytest.raises(FileExistsError): build_controls(source_run, study_dir, output)
    with pytest.raises(ValueError,match='outside'): build_controls(source_run, study_dir, study_dir/'controls')


def test_reader_refuses_changed_raw_source_or_saved_waveform(source_run, tmp_path):
    study_dir = source_run/'composition'
    run_study(source_run, study_dir, ratios=[1], max_anchors=2)
    path = source_run/'segments/activity_00000.csv'
    original = path.read_bytes()
    data = pd.read_csv(path)
    data.loc[0,'power'] += 20
    data.to_csv(path,index=False)
    with pytest.raises(ValueError,match='source changed'): build_controls(source_run,study_dir,tmp_path/'changed')
    assert not (tmp_path/'changed').exists()
    path.write_bytes(original)
    wave = next(study_dir.glob('*/random.npz'))
    wave.write_bytes(wave.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='audit failed'): build_controls(source_run,study_dir,tmp_path/'tampered')
    assert not (tmp_path/'tampered').exists()


def test_actual_server_wrapper_packages_success_and_failure(source_run):
    project = Path(__file__).resolve().parents[1]
    work = source_run.parents[1]
    (work/'scripts').mkdir()
    shutil.copyfile(project/'scripts/run_real_cycle_controls.sh',work/'scripts/run_real_cycle_controls.sh')
    (work/'.git').symlink_to(project/'.git',target_is_directory=True)
    study_dir = source_run/'composition'
    run_study(source_run,study_dir,ratios=[1],max_anchors=2)
    env = {**os.environ,'PYTHONPATH':str(project),'PYTHONDONTWRITEBYTECODE':'1',
           'PYTHON_BIN':sys.executable,'RUN_ID':'source','CONTROL_TAG':'controls','CONTROL_REPEATS':'3'}
    def run():
        return subprocess.run(['bash','scripts/run_real_cycle_controls.sh'],cwd=work,env=env,
                              text=True,capture_output=True,timeout=60)
    result = run()
    assert result.returncode == 0, result.stdout+result.stderr
    archive = work/'log/source_controls_diagnostics.tar.gz'
    with tarfile.open(archive) as bundle:
        names = bundle.getnames()
        assert './results/real_cycle_controls.json' in names
        assert './results/source_composition_manifest.json' in names
        assert not any(name.endswith(('.csv','.npz','.npy')) for name in names)
        assert 'REAL_CONTROLS_EXIT_STATUS=0' in bundle.extractfile('./controls_execution.log').read().decode()
    before = archive.read_bytes()
    assert run().returncode != 0 and archive.read_bytes() == before
    wave = next(study_dir.glob('*/random.npz'))
    wave.write_bytes(wave.read_bytes()+b'changed')
    env['CONTROL_TAG'] = 'failed_controls'
    assert run().returncode != 0
    with tarfile.open(work/'log/source_failed_controls_diagnostics.tar.gz') as bundle:
        log = bundle.extractfile('./controls_execution.log').read().decode()
        assert 'source composition audit failed' in log and 'REAL_CONTROLS_EXIT_STATUS=1' in log
    assert not (source_run/'failed_controls/results').exists()
