"""Add real-cycle controls to a saved study without rerunning generation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import shutil

import numpy as np
import scipy

from scripts.audit_primitive_composition import audit_composition
from scripts.prepare_downstream_run import validate_run_id
from scripts.run_primitive_composition import load_inputs, write_json
from scripts.validate_generation_run import sha256_file
from src.generation.real_cycle_controls import CYCLE_METRICS, evaluate_real_controls


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _report(result):
    def number(value):
        return 'NA' if value is None else f'{value:.4g}'

    lines = [
        '# 真实完整周期对照', '',
        f"状态：`{result['status']}`。原拼接案例 {result['paired_cases']} 组，"
        f"不同锚点 {result['distinct_anchors']} 个，无同组验证案例 {result['cases_without_validation']} 组。", '',
        '本轮只补评价：不重新划分、拟合或拼接，不修改原始波形，不计算测试集性能。', '',
        '## 如何读表', '',
        '- 数值为到同组验证集的Wasserstein-1距离，保留指标物理单位；越小仅代表该项分布差距越小。',
        '- 训练全集、预算池：真实数据分布背景，样本数不等，不能直接排名。',
        '- 原始锚点：使用与生成案例相同的原始真实周期，便于对照模板选择与拼接后的变化。',
        '- 等量真实抽样：同预算、同Class/Mode，无放回抽取与本层生成数相同的完整周期；允许抽中锚点。',
        '- 等量栏为中位数 [P10, P90]，只是重复抽样波动，**不是置信区间或显著性检验**。',
        '- `low_support` 表示生成或验证周期不足3个；NA不能填0。full/partial/none只是原转移模型支持状态。',
        '- 五种方法共享原锚点时长，因此时长行在方法之间相同，不是五次独立证据。', '',
    ]
    labels = {'duration_seconds': '时长(s)', 'energy_wh': '能量(Wh)',
              'mean_watts': '均值(W)', 'peak_watts': '峰值(W)'}
    for row in result['strata']:
        lines += [f"## seed {row['seed']} / {row['budget_tag']} / Class-Mode {row['class_mode']} / {row['support_status']}", '',
                  f"状态：{row['status']}；生成/锚点 {len(row['case_ids'])}，验证 {len(row['validation_activity_ids'])}，"
                  f"真实训练 {len(row['full_train_activity_ids'])}，预算池 {len(row['budget_pool_activity_ids'])}。",
                  f"{result['configuration']['repeats']} 次等量抽样共有 {row['unique_matched_subsets']} 个不同子集。", '',
                  '| 指标 | 训练全集 | 预算池 | 原始锚点 | 等量真实抽样中位数 [P10,P90] | '
                  + ' | '.join(result['methods']) + ' |',
                  '| ' + ' | '.join(['---']*(5+len(result['methods']))) + ' |']
        for metric in CYCLE_METRICS:
            values = row['cycle_metrics'][metric]
            controls = values['controls']
            sample = controls['matched_budget_real']['wasserstein_distribution']
            cells = [labels[metric], *[number(controls[name]['wasserstein'])
                     for name in ('full_train_context', 'budget_pool_context', 'original_anchors')],
                     f"{number(sample['median'])} [{number(sample['p10'])}, {number(sample['p90'])}]",
                     *[number(values['methods'][name]['wasserstein']) for name in result['methods']]]
            lines.append('| ' + ' | '.join(cells) + ' |')
        lines += ['', '真实训练 / 原始锚点 / 验证的指标中位数：', '']
        for metric in CYCLE_METRICS:
            values = row['cycle_metrics'][metric]
            medians = [values['controls'][name]['distribution']['median'] for name in ('full_train_context', 'original_anchors')]
            medians.append(values['validation']['median'])
            lines.append(f"- {labels[metric]}：" + ' / '.join(map(number, medians)))
        lines.append('')
    lines += ['## 进一步复核', '',
              '完整逐状态对跳变/斜率对照、实际抽样ID、样本量、缺失抽样及相对锚点/抽样中位数的距离差，'
              '见 [real_cycle_controls.json](real_cycle_controls.json)。没有将不同状态对汇成总分。', '',
              '训练与验证真实周期的原始测量汇总见 [real_cycle_measurements.json](real_cycle_measurements.json)。'
              '未导出测试周期指标，也未生成新波形。', '',
              '## 限制', '', *['- '+item for item in result['limitations']], '']
    return '\n'.join(lines)


def build_controls(run_root, study_dir, output_dir, *, repeats=100, sampling_seed=20260917):
    run_root, study_dir, output = (Path(value).resolve() for value in (run_root, study_dir, output_dir))
    if output == study_dir or study_dir in output.parents:
        raise ValueError('controls must be outside the source composition directory')
    if output.exists():
        raise FileExistsError(f'controls output already exists: {output}')
    manifest = _read(study_dir/'composition_manifest.json')
    identity = _read(study_dir/'input_identity.json')
    source_summary = _read(study_dir/'composition_summary.json')
    audit = audit_composition(study_dir)
    if not audit['passed']:
        raise ValueError(f'source composition audit failed: {audit["errors"][:8]}')
    if source_summary['input_fingerprint'] != manifest['input_fingerprint']:
        raise ValueError('source summary fingerprint differs')
    configuration = manifest['configuration']
    print('[real controls] saved waveform audit passed; verifying unchanged source data', flush=True)
    training, validation, upstream, hashes, protocol = load_inputs(
        run_root, configuration['cluster_tag'], configuration.get('device_change_date'))
    for name, observed in [('inputs', hashes), ('source_protocol', protocol),
                           ('source_signal_digest', upstream['source_digest']),
                           ('upstream_artifact_sha256', upstream['artifact_sha256'])]:
        if identity.get(name) != observed:
            raise ValueError(f'saved study source changed: {name}; do not mix old output with new inputs')
    # Anchors must be exactly the raw training waveforms, not merely matching IDs.
    by_id = {cycle.activity_id: cycle for cycle in training}
    for case in manifest['cases']:
        if 'anchor_reference' not in case:
            raise ValueError('saved anchor references required; use the five-arm study')
        with np.load(study_dir/case['anchor_reference']['file'], allow_pickle=False) as payload:
            if not np.array_equal(payload['appliance'], by_id[case['anchor_activity_id']].power):
                raise ValueError(f"real anchor mismatch: {case['case_id']}")
    print('[real controls] measuring unchanged train/validation cycles and matched real samples', flush=True)
    result = evaluate_real_controls(manifest, training, validation, repeats=repeats, sampling_seed=sampling_seed)
    measurements = result.pop('real_measurements')
    project = Path(__file__).resolve().parents[1]
    provenance = {
        'source_run': str(run_root), 'source_study': str(study_dir),
        'source_fingerprint': manifest['input_fingerprint'], 'source_protocol': protocol,
        'source_audit': audit, 'current_inputs': hashes,
        'source_data_verified_unchanged': True, 'raw_anchor_references_checked': len(manifest['cases']),
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
        'test_use': 'Inherited loader checks all-source integrity and interval isolation only; no test performance metrics.',
        'code_sha256': {name: sha256_file(project/name) for name in (
            'src/generation/real_cycle_controls.py', 'scripts/run_real_cycle_controls.py',
            'scripts/run_real_cycle_controls.sh',
            'scripts/run_primitive_composition.py', 'scripts/audit_primitive_composition.py',
            'src/generation/composition_evaluation.py', 'src/generation/primitive_composition.py',
            'src/steps/nilm_dataset_step.py')},
        'source_files_sha256': {name: sha256_file(study_dir/name) for name in (
            'composition_manifest.json', 'composition_summary.json', 'input_identity.json')},
    }
    result['provenance'] = provenance
    # Reserve only after all checks/calculations; never modify an old report.
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'real_cycle_controls.json', result)
    write_json(output/'real_cycle_measurements.json', measurements)
    write_json(output/'input_audit.json', provenance)
    (output/'report.md').write_text(_report(result), encoding='utf-8')
    for name in provenance['source_files_sha256']:
        shutil.copyfile(study_dir/name, output/('source_'+name))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--study-name', default='composition')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=100)
    parser.add_argument('--sampling-seed', type=int, default=20260917)
    args = parser.parse_args()
    root = Path('log')/validate_run_id(args.run_id)
    result = build_controls(root, root/validate_run_id(args.study_name), args.output_dir,
                            repeats=args.repeats, sampling_seed=args.sampling_seed)
    print(json.dumps({key: result[key] for key in (
        'status', 'paired_cases', 'cases_without_validation', 'distinct_anchors', 'configuration')}, indent=2))
    print(f'Return report: {args.output_dir / "report.md"}')
    raise SystemExit(2 if result['status'] in ('no_completed_cases', 'no_matching_validation') else 0)


if __name__ == '__main__':
    main()
