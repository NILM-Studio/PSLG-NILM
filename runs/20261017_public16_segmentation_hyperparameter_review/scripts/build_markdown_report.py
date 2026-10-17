"""A single Markdown review report embedding every existing overview figure."""
import review

plan = review.load(review.PLAN)
bundle = review.load(review.REPORT/'figure_manifest.json')
assert bundle['figure_count'] == 160 and not bundle['missing']
addition_path = review.REPORT/'prim_glr_figure_manifest.json'
addition = review.load(addition_path) if addition_path.exists() else None
if addition:
    assert addition['status'] == 'passed' and len(addition['figures']) == 32
figures = bundle['figures'] + (addition['figures'] if addition else [])
methods = (('prim-glr',) if addition else ()) + review.METHODS
names = {'prim-glr': 'PrimGLR', 'pelt-rbf': 'PELT-RBF', 'fluss-threshold': 'FLUSS-threshold',
         'clasp': 'Clasp', 'clap': 'CLaP', 'e2usd': 'E2USD'}
lines = ['# 切分模型参数图审', '',
         f'数据：`public16_activity_states_20261017_v2`。16个对象、{len(methods)}种方法，每组开发/验证各1张，共{len(figures)}张总图，全部采用CPU结果。', '',
         '每张总图包含全部候选参数组合。黑线为原始功率，红虚线为切点，灰区为上下文；不适用组合在图内说明。点击总图可打开SVG放大查看。', '',
         '审核时反馈“对象＋方法＋图中参数或配置编号”即可。CLaP暂用默认Clasp初始化，改选Clasp后需复核对应CLaP结果。', '']
if addition:
    lines += ['PrimGLR完整展开W、gap、α、σ：普通对象144组，H20为240组。蓝框SELECTED标出已选W120s/gap30s、α=.90、σ=.35参考配置；图内effective W为短活动适配后的实际窗口。', '']
count = 0
for obj in plan['objects']:
    lines += [f"## {obj['object']}", '']
    for method in methods:
        number = (len(next(f for f in figures if f['object'] == obj['object'] and f['method'] == method)['panels'])
                  if method == 'prim-glr' else len(obj['grids'][method]))
        lines += [f"### {names[method]}（{number}组参数）", '']
        for split, label in [('development', '开发活动'), ('validation', '验证活动')]:
            f = next(f for f in figures if f['object'] == obj['object'] and f['method'] == method and f['split'] == split)
            assert (review.REPORT/f['png']).is_file() and (review.REPORT/f['svg']).is_file()
            alt = f"{obj['object']} {names[method]} {label} {f['activity_id']} 完整参数总图"
            lines += [f"**{label} #{f['activity_id']}**", '', f"[![{alt}]({f['png']})]({f['svg']})", '']
            count += 1
assert count == len(figures)
report = review.REPORT/'report.md'
report.write_text('\n'.join(lines), encoding='utf-8')
(review.REPORT/'index.md').write_text('# 参数审核\n\n打开[图审报告](report.md)，直接查看全部总图。\n', encoding='utf-8')
inputs = dict(figure_manifest_sha256=review.sha(review.REPORT/'figure_manifest.json'), plan_sha256=review.sha(review.PLAN))
if addition:
    inputs['prim_glr_manifest_sha256'] = review.sha(addition_path)
review.save(review.OUT/'inputs.json', inputs)
review.save(review.OUT/'analysis/validation.json', dict(status='passed', embedded_overview_figures=count,
            image_and_enlarge_links_checked=2*count, report_sha256=review.sha(report)))
print(f'Markdown review report: {count} embedded overview figures, {2*count} valid image/enlarge references')
