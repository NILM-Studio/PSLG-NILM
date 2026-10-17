# public16 v2 切分模型完整参数图库

范围、授权与方法见 [proposal.md](proposal.md)、[protocol.md](protocol.md)。

执行中的进度：`data/coordinator_progress.json` 与已登记execution日志。

最终图库：[reports/full_grid_review/index.md](reports/full_grid_review/index.md)。共160张总图，全部提供PNG/SVG/PDF，参数待人工审核。

当前资源：按用户要求全部使用CPU；E2USD走作者cuda=False路径。新协调器为scripts/resume_cpu.py，进度data/coordinator_progress.json；旧GPU图另归档，不进入160张最终图库。

完成：160张总图（PNG/SVG/PDF）、1952格，32条既有活动跨模型输入一致与原始切点核验通过。浏览入口 [index.html](reports/full_grid_review/index.html)，全部切分参数待人工审核。
