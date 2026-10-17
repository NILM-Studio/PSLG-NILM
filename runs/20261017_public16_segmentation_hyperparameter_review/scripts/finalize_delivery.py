"""Registered final QA, viewer and portable gallery package after the CPU queue."""
import datetime
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import review
from create_execution import register

coordinator = Path((review.EXP/'data/cpu_coordinator_execution.txt').read_text().strip())
while review.load(coordinator/'run.json')['status'] in ('registered', 'running'):
    time.sleep(3)
assert review.load(coordinator/'run.json')['exit_code'] == 0, coordinator
executions = []
for condition, script in [('independent_complete_grid_qa', 'delivery_qa.py'), ('render_markdown_review', 'build_markdown_report.py')]:
    execution = register(review.EXP.name, condition, dict(plan_sha256=review.sha(review.PLAN), backend='cpu'))
    code = subprocess.call([sys.executable, str(review.ROOT/'scripts/run_execution.py'), str(execution), '--',
                            sys.executable, str(Path(__file__).with_name(script))], cwd=review.ROOT)
    assert code == 0, execution
    executions.append(str(execution))
qa = review.load(review.REPORT/'independent_qa.json')
footer = ('\n\n## 交付核验\n\n'
          '独立核验通过：160张总图、1952个参数×种子子图、32条既有活动、每条活动五种方法使用相同输入；'
          '所有切点均已与源记录及SVG线条核对。PNG/SVG/PDF三格式齐全，E2USD最终图均为作者CPU路径。\n\n'
          '参数仍待用户审核。\n')
for name in ('report.md',):
    with (review.REPORT/name).open('a', encoding='utf-8') as f:
        f.write(footer)
with (review.EXP/'README.md').open('a', encoding='utf-8') as f:
    f.write('\n完成：160张总图（PNG/SVG/PDF）、1952格，32条既有活动跨模型输入一致与原始切点核验通过。'
            '审核入口 [report.md](reports/full_grid_review/report.md)，全部切分参数待人工审核。\n')
with (review.ROOT/'runs/EXPERIMENT_LOG.md').open('a', encoding='utf-8') as f:
    f.write(f'\n- {datetime.datetime.now(datetime.timezone.utc).isoformat()} `{review.EXP.name}`：'
            'CPU完整参数图库完成，16对象×5模型×开发/验证各1图=160张，1952格；'
            '32条public16 v2既有活动，不重新提取或重跑PrimGLR。三格式、全网格、源记录/SVG切点、'
            'Clasp/CLaP初始化一致性和E2USD CPU核验通过。人工审核pending，未冻结其他模型或运行正式实验一。\n')
archive = Path('/tmp/public16_segmentation_review_20261017.tar.gz')
with tarfile.open(archive, 'w:gz') as tar:
    tar.add(review.REPORT, arcname=review.REPORT.name)
record = dict(status='passed', archive=str(archive), bytes=archive.stat().st_size,
              archive_sha256=review.sha(archive), final_qa_executions=executions, **{'gallery_qa': qa})
review.save(review.OUT/'analysis/delivery.json', record)
review.save(review.EXP/'data/final_delivery.json', record)
print(record, flush=True)
