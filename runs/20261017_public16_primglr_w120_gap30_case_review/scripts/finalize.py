import os,sys,json,pathlib,shutil,datetime,hashlib,collections
import numpy as np
from PIL import Image
R=pathlib.Path(__file__).resolve().parents[3];E=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(E/'scripts'))
from pipeline import save,sha,PLAN,grid
out=pathlib.Path(os.environ['PSLG_EXECUTION_DIR']);D=pathlib.Path(PLAN['dataset_root']);name='final_'+out.name
report=E/'reports'/name;(report/'figures').mkdir(parents=True,exist_ok=False)

def last(condition):
 for p in sorted((E/'executions'/condition).glob('*/run.json'),reverse=True):
  if json.loads(p.read_text()).get('exit_code')==0:return p.parent
 raise RuntimeError(f'No successful execution: {condition}')

items=[];figures=[];checks=[];total=0
for obj in PLAN['objects']:
 ident=obj['id'];ex=last('extract_'+ident);pp=last('plot_'+ident)
 summary=json.loads((ex/'analysis/summary.json').read_text());im=json.loads(pathlib.Path(summary['raw_manifest']).read_text())
 rows=json.loads(pathlib.Path(summary['activity_manifest']).read_text())['records'];copy=json.loads(pathlib.Path(summary['copy_QA']).read_text())
 assert copy['status']=='passed' and len(rows)<=obj['cap']
 raw_ns=np.memmap(im['raw']['timestamp_ns']['path'],dtype=np.int64,mode='r');raw_p=np.memmap(im['raw']['power']['path'],dtype=np.float64,mode='r')
 assert len(raw_ns)==len(raw_p)==im['rows'] and np.all(np.diff(raw_ns)>0)
 assert raw_ns[0]>=im['effective_start_ns'] and raw_ns[-1]<im['effective_end_exclusive_ns']
 for meta in copy['files']:assert sha(meta['path'])==meta['sha256']
 group_split={};projection_cache_key=None;projection_cache=None
 for row in rows:
  assert row['context_start_ns']>=im['effective_start_ns'] and row['context_end_exclusive_ns']<=im['effective_end_exclusive_ns']
  assert row['main_csv_exact'] and sha(row['csv'])==sha(row['main_step_output_csv'])
  assert group_split.setdefault(row['time_group'],row['split'])==row['split']
  with np.load(row['npz']) as z:
   assert len(z['power'])==row['n_points'] and np.isfinite(z['power']).all()
   assert np.all(np.diff(z['timestamp_ns'])>0) and 0<=int(z['core_a'])<int(z['core_b'])<=len(z['power'])
   if obj['source']['dataset']=='refit':
    key=tuple(row['raw_support_index_range'])
    if key!=projection_cache_key:
     ra,rb=key;projection_cache=grid.time_weighted_grid(raw_ns[ra:rb].astype(np.float64)/1e9,raw_p[ra:rb],8.,16.,.8);projection_cache_key=key
    gt,gp,_=projection_cache;ix=np.searchsorted(gt,z['timestamp'])
    assert np.all(ix<len(gt)) and np.array_equal(gt[ix],z['timestamp']) and np.array_equal(gp[ix],z['power']),'REFIT raw-to-8s reconstruction differs'
 total+=len(rows)
 plot=json.loads((pp/'analysis/cases.json').read_text());assert all(c['split']!='evaluation' for c in plot['cases'])
 for kind in ['png','svg']:
  f=pathlib.Path(plot[kind]);shutil.copy2(f,report/'figures'/f.name)
  assert sha(f)==sha(report/'figures'/f.name)
  if kind=='png':
   with Image.open(f) as image:image.verify()
  else:
   import xml.etree.ElementTree as ET
   root=ET.parse(f).getroot();assert root.tag.endswith('svg')
 assert len(plot['cases'])<=10
 summary['review_png']=str(report/'figures'/pathlib.Path(plot['png']).name);summary['review_svg']=str(report/'figures'/pathlib.Path(plot['svg']).name);summary['case_count']=len(plot['cases'])
 items.append(summary);figures.append(plot);checks.append(dict(object=ident,status='passed',activities=len(rows),source_rows=len(raw_ns),refit_independent_raw_to_8s_exact=True if obj['source']['dataset']=='refit' else None,split_counts=dict(collections.Counter(r['split'] for r in rows)),case_count=len(plot['cases'])))

manifest=dict(experiment=E.name,generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),dataset_root=str(D),status='data_validated_parameters_await_human_review',object_count=len(items),activity_count=total,time_encoding='Aware UK-DALE/REFIT timestamps are UTC epoch ns; their displayed effective_start/end strings are UTC. Naive SmartNIALMeter/DARCK timestamps preserve source calendar values, timezone unspecified. source_timezone records original source metadata, not the display timezone.',main_step_source=str(R/'src/steps/extract_active_data_step.py'),main_step_sha256=sha(R/'src/steps/extract_active_data_step.py'),objects=items,checks=checks)
assert len(items)==16
save(D/'manifest.json',manifest);save(report/'data_manifest.json',manifest);save(report/'figure_manifest.json',figures);save(out/'analysis/QA.json',dict(status='passed',objects=checks,activity_count=total))
lines=['# 16对象原始区间与活动状态片段使用指引','',f'数据已验证；PrimGLR参数仍待用户十例图审。活动状态为主项目提取步骤输出的有上下文支持片段。共16对象，'+str(total)+'条入选片段。','',f'机器入口：`{D}/manifest.json`。下表source与activities均是独立物理文件副本，不依赖runs软链接。','', 'source保存连续范围内全部原始有功功率，包括NaN和时间间断；其间可能有未入选活动。activities.json列出按上限实际入选的片段、时间块划分、核心和上下文索引、来源主步骤CSV、各文件SHA256。REFIT活动片段采用8秒支撑网格，source仍为原生点。','', '原始序列读取：`np.memmap(source/timestamp_ns.i64,dtype=np.int64,mode="r")`和`np.memmap(source/power.f64,dtype=np.float64,mode="r")`；长度和时区见source/selected_interval.json。timestamp_ns为纳秒，未标注时区的源不改称UTC。','', '活动读取：用activities/activities.json逐条读取npz或csv；NPZ包含timestamp、timestamp_ns、power、end_timestamp、core_a、core_b，核心为[core_a:core_b]。CSV为主步骤实际输出的逐字节副本。不得把上下文中的相邻活动算入本段核心。','', '|序号|对象|有效范围（起含终不含）|入选片段数|原始序列目录|活动状态目录|图审|','|---:|---|---|---:|---|---|---|']
for x in items:
 im=x['selected_interval'];zone='UTC' if im.get('source_timezone') else '源时区未标注';lines.append(f'|{x["number"]}|{x["object"]}|[{im["effective_start"]}, {im["effective_end_exclusive"]}) {zone}|{im["selected_activity_count"]}|`{x["raw_directory"]}`|`{x["activity_directory"]}`|[十例图]({x["review_png"]})|')
lines+=['','时间解释：UK-DALE／REFIT原记录带时区，数值及表中日期统一用UTC；selected_interval.json中的source_timezone描述原记录时区。SmartNIALMeter／DARCK未标注时区，保留源日历口径，不宣称UTC。','', '咖啡机／冰箱在连续范围内保留所有满足支撑条件的主步骤片段；其他14对象按固定时间块划分后在日期层内确定性选样至上限。主步骤与历史筛选脚本不同，数量可能变化。请求与实际生效范围、调整前后计数均保存在各source/selected_interval.json。','',f'完整执行与代码归档：`{E}`。PrimGLR W=120秒、gap=30秒、alpha=.90、dur_sigma=.35；短序列按原生模型规则缩小实际W，逐例值见报告figure_manifest.json。图审不使用evaluation侧。']
guide='\n'.join(lines)+'\n';(D/'DATA_GUIDE.md').write_text(guide);(D/'README.md').write_text('# 活动状态数据包\n\n见[使用指引](DATA_GUIDE.md)及[机器清单](manifest.json)。\n')
(report/'report.md').write_text(guide+f'\n核验：16对象原始时间范围、片段核心／上下文、实际主步骤输出、原生数组对应、独立复制哈希、同时间组划分、PNG和SVG可读均通过。详见`{out}/analysis/QA.json`。\n')
with (E/'reports/README.md').open('a') as f:f.write(f'\n- [{name}]({name}/report.md)：16对象数据双副本与十例图审；{total}条活动状态，参数待人审。\n')
save(E/'data/delivery.json',dict(dataset_root=str(D),manifest=str(D/'manifest.json'),guide=str(D/'DATA_GUIDE.md'),report=str(report/'report.md'),object_count=16,activity_count=total))
with (R/'runs/EXPERIMENT_LOG.md').open('a') as f:f.write(f'\n- {manifest["generated_at"]} `{E.name}`：16对象两组数据独立复制与QA完成，入选{total}条；16张十例图供人审。数据清单`{D}/manifest.json`，未自动冻结PrimGLR论文最终参数。\n')
print('DELIVERY_READY',D/'DATA_GUIDE.md','ACTIVITIES',total,flush=True)
