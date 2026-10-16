"""Create a separate all-activity experiment, preserving the previous campaign."""
from pathlib import Path
import ast
import hashlib
import json
import re
import shutil

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'runs/20261015_multidataset_primglr_dtw_kmeans'
EXP=ROOT/'runs/20261016_public_all_activities_primglr_dtw_kmeans'
if EXP.exists(): raise SystemExit('New experiment already exists; never overwrite it')
for name in ['scripts','configs','data','executions','reports']:(EXP/name).mkdir(parents=True,exist_ok=True)

source=PARENT/'scripts/public_pipeline.py'
code=source.read_text(encoding='utf-8')
def change(old,new):
    global code
    if code.count(old)!=1:raise RuntimeError('Expected one source match: '+old[:100])
    code=code.replace(old,new)

change('Public-only experiment adapter. Raw H5 inputs are always read-only.','All-activity experiment adapter. No modeling populations are partitioned. Raw H5 inputs are read-only.')
change('# Calibration evidence only: first three days, inside the training prefix.','# The first three calendar days estimate cadence only; every activity remains eligible for modeling.')
change("gap_policy='training-prefix p95+1 or twice median; REFIT explicit maximum 16s ZOH support'","gap_policy='first-three-calendar-days p95+1 or twice median; REFIT explicit maximum 16s ZOH support'")
start=code.index('    # All source dates retained. Chronological split')
end=code.index("    (base/'supports').mkdir",start)
code=code[:start]+'''    audit.update(effective_end=float(effective_end),
                 input_scope='full selected household appliance meter, all stored rows',
                 modeling_scope='all_activities',population_partition='none',
                 sampling='8s epoch ZOH time-weighted, coverage>=0.8' if target['dataset']=='refit' else 'native timestamp/point order retained')
    activities=[]; rejected=[]; arrays={}
'''+code[end:]
start=code.index('            split=None')
end=code.index('            aid=len(activities)',start)
code=code[:start]+"            ca,cb=a['ca'],a['cb']\n"+code[end:]
change("            a.update(activity_id=aid,split=split,context_start=float(tt[0]),context_end=float(ends[-1]),\n                     core_a=int(a['a']-ca),core_b=int(a['b']-ca),n_points=len(tt),\n                     context_clipped_at_split=ca!=a['ca'] or cb!=a['cb'])",
"            a.update(activity_id=aid,context_start=float(tt[0]),context_end=float(ends[-1]),\n                     core_a=int(a['a']-ca),core_b=int(a['b']-ca),n_points=len(tt))")
start=code.index('    audit.update(activity_count=len(activities),activities_by_split=')
end=code.index('    observed_seconds=',start)
code=code[:start]+'''    audit.update(activity_count=len(activities),
                 rejected_short_count=len(rejected),support_intervals=len(supports),
                 n_uncensored_activities=sum(not a['left_censored'] and not a['right_censored'] for a in activities),
                 descriptive_cycle_support=sum(not a['left_censored'] and not a['right_censored'] for a in activities)>=20)
'''+code[end:]
change("save(base/'rejected_short.json',rejected); save(base/'excluded_cross_split.json',split_dropped)","save(base/'rejected_short.json',rejected)")
change("activity_id=ordered[i]['activity_id'],split=ordered[i]['split'],duration_seconds=ordered[i]['duration_seconds']","activity_id=ordered[i]['activity_id'],duration_seconds=ordered[i]['duration_seconds']")
change("'PREPARED',audit['activities_by_split']","'PREPARED_ALL',len(activities)")
change("for a in [v for v in activities if v['split']=='train'][:20]:","for a in activities[:20]:")
change("save(base/'training_window_sensitivity.json',dict(selected_seconds=60,selection='predeclared 60s baseline; diagnostics are training-only, no held-out selection',results=sensitivity))","save(base/'window_sensitivity.json',dict(selected_seconds=60,selection='predeclared 60s baseline; first 20 full-population activities for descriptive sensitivity',results=sensitivity))")
change("row=dict(row=len(rows),activity_id=aid,split=a['split'],start=int(l)","row=dict(row=len(rows),activity_id=aid,start=int(l)")
change("axs[0].set_title(f'k={k}; activity {aid}; split={chosen[\"split\"]}')","axs[0].set_title(f'k={k}; activity {aid}; all-activity fit')")
start=code.index("    train=np.array([i for i,r in enumerate(rows) if r['split']=='train'],int)")
end=code.index('    if len(rows)>10000:',start)
code=code[:start]+"    fit_rows=np.arange(len(rows),dtype=int)\n"+code[end:]
# This is a new source fork. Inherited model.train is an API method, not a data partition.
start=code.index('def prototype_rows(');end=code.index('\ndef metrics(',start)
piece=re.sub(r'\btrain\b','eligible',code[start:end]);code=code[:start]+piece+code[end:]
start=code.index('def run_target(');end=code.index('\ndef main():',start)
piece=code[start:end]
piece=re.sub(r'\btrain\b','fit_rows',piece)
piece=piece.replace("fit_scope='train_only'","fit_scope='all_activities',n_fit_rows=len(rows)")
piece=piece.replace('clip_heldout=False','clip_values=False')
piece=piece.replace('Representative unpadded power waveforms, chosen using training distances only.','Representative unpadded power waveforms from the full fitted population.')
piece=piece.replace('            mm={}\n            for split in [\'fit_rows\',\'val\',\'test\']:\n                ix=np.array([i for i,r in enumerate(rows) if r[\'split\']==split],int)\n                mm[split]=metrics(z[ix],labels[ix]) if len(ix) else dict(n_samples=0,status=\'empty_split\')',
                    "            mm={'all_activities':metrics(z,labels)}")
piece=piece.replace("record.update(activity_id=a['activity_id'],split=a['split'],block_id=len(blocks))","record.update(activity_id=a['activity_id'],block_id=len(blocks))")
piece=piece.replace('n_train=len(fit_rows)','n_fit_rows=len(fit_rows)')
piece=piece.replace("record=dict(seed=seed,k=k,status='completed',metrics=mm,merge=merge_metrics)","record=dict(seed=seed,k=k,status='completed',fit_scope='all_activities',n_fit_rows=len(rows),metrics=mm,merge=merge_metrics,n_iter=int(km.n_iter_))")
# Avoid repeated O(number of events * number of primitives) scans in every k/mode.
piece=piece.replace('    all_labels={}; records=[]',"    all_labels={}; records=[]\n    activity_rows={}\n    for i,r in enumerate(rows):activity_rows.setdefault(r['activity_id'],[]).append(i)\n    np.save(base/'fit_row_indices.npy',fit_rows)")
piece=piece.replace("ix=[i for i,r in enumerate(rows) if r['activity_id']==a['activity_id']]","ix=activity_rows.get(a['activity_id'],[])")
code=code[:start]+piece+code[end:]
# Plot discontinuous grids without visually connecting across missing intervals.
change("        ax.plot(pd.to_datetime(tt[mask],unit='s',utc=True), yy[mask], lw=.7, label='8 s time-weighted grid')",
"        xx=tt[mask]; yp=yy[mask]\n        breaks=np.flatnonzero(np.diff(xx)>8.01)+1\n        if len(breaks):\n            yp=np.insert(yp,breaks,np.nan); xx=np.insert(xx,breaks,xx[breaks])\n        ax.plot(pd.to_datetime(xx,unit='s',utc=True), yp, lw=.7, label='8 s time-weighted grid')")
for bad in ["['split']","'train'","'val'","'test'",'train_only','heldout','split_bounds','split_dropped','training_window']:
    if bad in code:raise RuntimeError('Residual partition logic: '+bad)
ast.parse(code)
(EXP/'scripts/public_pipeline.py').write_text(code,encoding='utf-8')
shutil.copy2(PARENT/'data/public_h5_manifest.json',EXP/'data/public_h5_manifest.json')
config=json.loads((PARENT/'configs/approved_native_refit8s.json').read_text(encoding='utf-8'))
config.update(status='authorized_all_activities_no_partition',experiment=EXP.name,
    input_scope='All stored rows of all seven confirmed household/appliance channels',
    modeling_population='all accepted activities and all their valid primitives',
    population_partition='none',window='all valid source dates; no population cut boundaries or activity exclusions from partitioning',
    algorithm='all activities -> point PrimGLR -> all-population DTW prototypes and scaling -> all-population KMeans -> activity-local merge',
    parent_experiment=PARENT.name,
    controlled_change='remove population partition and refit all transforms/models on all activities; retain other settings including 60s context')
config['dtw'].pop('train_only_fit',None);config['dtw']['fit_scope']='all_activities'
config['resources'].update(max_parallel_targets=7,threads_per_target=2)
config['resources']['process_scope']='one fresh registered process per object; no reuse of partitioned fits'
config['stability_support']='at least 20 uncensored activity cores; descriptive flag only, never excludes a target'
config['randomness']=dict(prototype_and_final_kmeans_seeds=[0,1,2],prim_glr_internal_seed=0,silhouette_sampling_seed=42,manual_audit_sampling_seed=20261016)
(EXP/'configs/all_activities.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
provenance=dict(parent_experiment=PARENT.name,parent_code_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    new_code_sha256=hashlib.sha256((EXP/'scripts/public_pipeline.py').read_bytes()).hexdigest(),
    canonical_lf_sha256=hashlib.sha256(code.encode()).hexdigest(),controlled_change=config['controlled_change'])
(EXP/'data/parent_code_provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(EXP/'proposal.md').write_text('''# 全量活动 PrimGLR → DTW → KMeans → 时间状态合并

2026-10-16，用户已授权取消建模人群的训练/验证/测试划分，重新注册实验并执行全量流程。

七对象沿用已确认UK-DALE House2洗衣机meter12、微波炉meter15；REDD House2洗碗机meter10；REFIT building7/meter7、building9/meter5、building18/meter8、building20/meter3。REFIT后两者原住宅为House19/House21。

所有指定通道完整时间记录只读。采样中断在活动提取处切开，UK-DALE/REDD保留原点；REFIT统一8秒、最大观测支持16秒、覆盖率≥80%。活动历时、低功率间歇、上下文均按真实时间；PrimGLR按活动内点索引处理。

区别：没有人口时间边界，不再排除跨集合活动，不再裁剪集合边界上下文。DTW原型、功率缩放、特征标准化和KMeans全部拟合全部有效活动原语。全部k=2–8、种子0/1/2及raw/RLE/full保留。

仅取消建模划分；不擅自改变活动阈值、t_drop、最小时长、60秒上下文等参数，以便归因。已知微波炉上下文、面包机低功率间歇风险仍保留；活动语义验收不能由聚类分数代替。

本实验只作全量结构发现及内部一致性评价，不报告独立测试泛化性能。验收要求全量meter行数/首末时间一致、全部有效活动保留、拟合行覆盖全部原语、原型可追溯、RLE不改标签、合并不跨活动中断及完整覆盖。
''',encoding='utf-8')
(EXP/'protocol.md').write_text('''# 冻结协议

- 用户授权：不设训练/验证/测试人群，直接全量活动完成流程；每对象独立登记并行，最多2 CPU线程。
- 数据/超参数以configs/all_activities.json为准；原H5只读，沿用已验证文件身份并检查大小和全量行数/端点。
- 全部活动不因人为日期分界排除；仅保留采样中断、无效观测、设备最小时长等既定规则。上下文只裁剪到实际观测支持范围。
- 所有拟合对象和拟合行索引均标all_activities；参考波形最多16个，来自全量活动分层随机抽样，不来自KMeans簇中心。
- DTW距离本身无随机性；种子0/1/2控制原型抽样和最终KMeans初始化；PrimGLR内部随机种子固定0。
- 标准化和中心在全体样本上拟合，内部指标仅作描述。保留全部k，不按分数仅存最好k。
- 活动审核结果保持pending，不伪造语义真值。低功率内容审计是功率代理，不宣称实际关闭误检率。
- 按根README及guides/server-usage.md登记运行，PSLG_EXECUTION_DIR隔离产物，旧实验/执行不覆盖。CPU实现不占GPU，物理GPU1/2当前被其他任务使用。
''',encoding='utf-8')
(EXP/'README.md').write_text('''# 全量活动无划分实验

- 用户已授权：[方案](proposal.md)、[协议](protocol.md)
- [冻结配置](configs/all_activities.json)
- [报告索引](reports/README.md)
- 七通道全量活动拟合，无训练/验证/测试划分；按电器并行，k=2–8、seed=0/1/2。
- 旧实验20261015_multidataset_primglr_dtw_kmeans保持原样。
''',encoding='utf-8')
(EXP/'reports/README.md').write_text('# 报告索引\n\n全量活动无划分实验已注册，计算结果将登记于此。\n',encoding='utf-8')
(EXP/'executions/index.json').write_text('[]\n',encoding='utf-8')
registry=ROOT/'runs/README.md'
with registry.open('a',encoding='utf-8') as f:f.write(f'\n| [{EXP.name}]({EXP.name}/README.md) | 用户授权七对象全量活动拟合，无训练/验证/测试划分，按对象并行；状态以执行索引为准 |\n')
print(EXP)
