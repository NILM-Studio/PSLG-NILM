"""Build a local-file gallery viewer for the existing scientific SVG figures."""
import json
from pathlib import Path
import review

bundle = review.load(review.REPORT/'figure_manifest.json')
assert bundle['figure_count'] == 160 and not bundle['missing']
items = [{k: f[k] for k in ('object', 'method', 'split', 'activity_id', 'png', 'svg', 'pdf', 'nominal_seconds')}
         for f in bundle['figures']]
for item in items:
    for ext in ('png', 'svg', 'pdf'):
        assert (review.REPORT/item[ext]).is_file()
html = '''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>public16：完整参数图审</title>
<style>
body{font-family:system-ui,sans-serif;margin:0;color:#20313b;background:#f5f7f8}
header{padding:18px 24px;background:white;border-bottom:1px solid #d5dfe4}
h1{font-size:22px;margin:0 0 8px}p{margin:7px 0;font-size:14px;line-height:1.6}
nav{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin:14px 0 0}
select,button{font:inherit;padding:6px;border:1px solid #afbec7;background:white;border-radius:4px}
a{color:#1257a5;margin-right:12px}main{padding:16px 24px}.canvas{overflow:auto;background:white;border:1px solid #d5dfe4}
#plot{display:block;width:100%;max-width:none}#meta{font-size:14px;margin:0 0 12px}label{font-size:14px}
</style>
<header><h1>public16 v2：五种切分模型完整参数图审</h1>
<p>16对象 × 5方法 × 2活动 = 160张总图。每张图完整展示参数组合；开发、验证各1条，五种方法使用相同输入。
黑线为功率，红虚线为切点，灰区为上下文。BASE表示登记基线。参数等待人工审核。</p>
<p>CLaP暂用官方默认Clasp初始化；改选Clasp后须重绘CLaP。E2USD每组参数展示三个种子。
不适用组合保留原因；本图库的两条活动不构成完整验证或最终排名。</p>
<nav><label>对象 <select id="obj"></select></label><label>方法 <select id="method"></select></label>
<label>活动 <select id="split"><option value="development">开发集</option><option value="validation">验证集</option></select></label>
<label>缩放 <select id="zoom"><option value="100">适应宽度</option><option value="150">150%</option><option value="200">200%</option><option value="300">300%</option></select></label>
<a href="index.md">文字索引</a><a href="all_parameter_choices.csv">全部候选与审核表</a></nav></header>
<main><p id="meta"></p><p><a id="svg" target="_blank">打开SVG</a><a id="png" target="_blank">打开PNG</a><a id="pdf" target="_blank">打开PDF</a></p>
<div class="canvas"><img id="plot" alt="完整参数网格"></div></main>
<script>const items=__ITEMS__;
const el=id=>document.getElementById(id);
for(const id of ['obj','method']){const key=id==='obj'?'object':'method';for(const value of [...new Set(items.map(f=>f[key]))]){const option=document.createElement('option');option.value=value;option.textContent=value;el(id).append(option);}}
function show(){const f=items.find(f=>f.object===el('obj').value&&f.method===el('method').value&&f.split===el('split').value);
el('plot').src=f.svg;el('plot').style.width=el('zoom').value+'%';el('meta').textContent=f.object+' / '+f.method+' / '+f.split+' / 活动 #'+f.activity_id+' / 标称采样间隔 '+f.nominal_seconds+' 秒';
for(const ext of ['svg','png','pdf'])el(ext).href=f[ext];}
for(const id of ['obj','method','split','zoom'])el(id).addEventListener('change',show);show();</script></html>
'''.replace('__ITEMS__', json.dumps(items, ensure_ascii=False).replace('</', '<\\/'))
target = review.REPORT/'index.html'
target.write_text(html, encoding='utf-8')
review.save(review.OUT/'analysis/validation.json', dict(status='passed', referenced_figures=len(items),
            referenced_artifacts=len(items)*3, viewer_sha256=review.sha(target)))
print(target, '160 figures, 480 valid artifact references')
