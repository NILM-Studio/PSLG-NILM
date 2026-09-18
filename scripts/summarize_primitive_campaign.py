"""Create the predeclared decision report from completed artifacts, without retraining."""
from pathlib import Path
import argparse
import json
import statistics


def read(path):
    return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-id',default='primitive_source_20260915_v2')
    args=p.parse_args()
    import sys
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project))
    from src.framework.run_paths import run_directories
    root = run_directories(args.run_id, project)[0]
    if not (root/'nilm_evaluate/results.json').exists():
        lines=['# 源侧执行结果：最终测试未开展','',
               '训练作业已结束，但没有完整的最终测试结果。以下为保留的源侧记录；不代表基元下游效用已得到验证。','']
        for model in ['NILMFormer','FCN']:
            folder=root/'nilm_train'/model
            finished=list(folder.glob('*/finished.json'))
            lines += [f'## {model}','',f'已完成训练数：{len(finished)}。','']
            if (folder/'source_gate.json').exists():
                gate=read(folder/'source_gate.json')
                lines += [f"源门槛通过：{gate['passed']}。",f"常数 MAE：{gate.get('baseline_mae')}。",'']
            if (folder/'source_selection.json').exists():
                selection=read(folder/'source_selection.json')
                lines += ['| 组 | 选定权重 | H5开发种子平均MAE W |','| --- | ---: | ---: |']
                for row in selection['selected']:
                    lines.append(f"| {row['case']} | {row['weight']} | {row['mean_mae']:.4f} |")
                lines.append('')
            if (folder/'worker.log').exists():
                tail=(folder/'worker.log').read_text()[-3000:]
                lines += ['```text',tail,'```','']
        dest=root/'nilm_report'
        dest.mkdir(exist_ok=True)
        (dest/'SOURCE_ONLY_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
        print('Final test not completed; source-only report saved.')
        return
    report=read(root/'nilm_report/report.json')
    if report['test'] is None:
        raise ValueError('Final results not available')
    source=read(root/'nilm_train/source_selection.json')
    freeze=read(root/'nilm_train/formal_freeze.json')
    details=report['detailed_metrics']
    rows=details['per_seed']
    results=[]
    table=[]
    def values(model,case,metric):
        return {r['seed']:r[metric] for r in rows if r['model']==model and r['case']==case}
    def comparison(model,proposed,control,metric='mae'):
        a,b=values(model,proposed,metric),values(model,control,metric)
        if a.keys()!=b.keys() or any(v is None for v in list(a.values())+list(b.values())):
            return None
        ma,mb=statistics.mean(a.values()),statistics.mean(b.values())
        return dict(model=model,proposed=proposed,control=control,metric=metric,
            proposed_mean=ma,control_mean=mb,relative_improvement=(mb-ma)/mb if mb else None,
            mean_delta=ma-mb,wins=sum(a[s]<b[s] for s in a),seeds=len(a),
            paired_deltas=[dict(seed=s,delta=a[s]-b[s]) for s in sorted(a)])
    for model in source['simple_control']:
        basic=comparison(model,'P','R')
        simple=comparison(model,'P',source['simple_control'][model])
        sequence=[comparison(model,'P_gru',c) for c in ['P_direct','P_point','H_gru']]
        best_sequence=comparison(model,'P_gru',source['sequence_control'][model])
        active=[comparison(model,'P',c,'active_MAE') for c in ['BP','S']]
        inactive_constraint=[comparison(model,'P',c) for c in ['BP','S']]
        passed=lambda item,threshold: item is not None and item['relative_improvement'] is not None and item['relative_improvement']>=threshold
        results.append(dict(model=model,basic_effect=basic,unique_effect=simple,sequence_effects=sequence,
            source_selected_sequence=best_sequence,active_effects=active,
            basic_threshold_met=passed(basic,.05) and basic['wins']>=4,
            unique_threshold_met=passed(simple,.03),
            sequence_threshold_met=all(s and s['mean_delta']<0 for s in sequence) and passed(best_sequence,.03),
            active_only_threshold_met=all(passed(s,.05) for s in active) and all(passed(s,-.01) for s in inactive_constraint)))
    metadata=dict(run_id=args.run_id,test_opened=True,final_models=len(freeze['runs']),decisions=results,
        limitations=['Single held-out household; seeds are not households.',
            'Activity targets are detector pseudo-labels; not independent functional semantics.',
            'Only 60-minute window auxiliary training, not full-program decoding.'])
    out=root/'nilm_report/decision_report.json'
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf-8')
    lines=['# 基元状态标注与直接 NILM：最终实验结果','',f'运行：`{args.run_id}`。正式模型数：{len(freeze["runs"])}。',
        '', 'H1 90 日训练、H5 60 日开发、H2 全部可用记录测试；采样 6 秒，窗口 60 分钟。',
        '以下为 5 个配对种子的结果，不构成多个独立住宅的统计证据。','',
        '| 骨干 | 组 | MAE均值 W | 种子标准差 | 活动内MAE W | 活动外MAE W | 活动召回 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    def fmt(value):
        return 'NA' if value is None else f'{value:.4f}'
    for r in details['household_means']:
        lines.append(f"| {r['model']} | {r['case']} | {fmt(r['mae']['mean'])} | {fmt(r['mae']['seed_std'])} | {fmt(r['active_MAE']['mean'])} | {fmt(r['inactive_MAE']['mean'])} | {fmt(r['activity_recall']['mean'])} |")
    lines+=['','## 预登记门槛判定','']
    for item in results:
        lines.append(f"### {item['model']}")
        lines.append('')
        for label,key in [('P 对 R（MAE改善≥5%，至少4/5种子同向）','basic_threshold_met'),
            ('P 对源侧最强简单组（MAE改善≥3%）','unique_threshold_met'),
            ('序列对三个关键活动头控制','sequence_threshold_met'),
            ('活动内条件性改善门槛','active_only_threshold_met')]:
            lines.append(f"- {label}：{'达到' if item[key] else '未达到'}。")
        lines.append('')
        for key in ['basic_effect','unique_effect','source_selected_sequence']:
            comp=item[key]
            if comp:
                relative=comp['relative_improvement']
                lines.append(f"{comp['proposed']} 对 {comp['control']}：MAE 差 {comp['mean_delta']:.4f} W，相对改善 {relative*100:.2f}%，改善种子 {comp['wins']}/{comp['seeds']}。" if relative is not None else '参照误差为零，比例无定义。')
                lines.append('')
    lines+=['## 解释边界','',
        '未达到门槛的比较应如实报告；不能用活动识别改善替代功率分解改善，也不能将辅助训练收益自动归因于功能语义。',
        '完整观测状态块为零，完整程序时长结论仍不成立。更多住宅、跨数据集和独立语义验证均未包含在本实验。','']
    (root/'nilm_report/FINAL_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(metadata,indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
