# zhouslab 主线与当前入口

主线恢复提交 `eb85bf532b5ce6db0b13d0b7348d80c7964253b7`，恢复前 121 个源码文件的备份在 `/home/avis/pslg-mainline-merge.959ZeG/server-source-before.tar.gz`。
恢复时 68 个同名文件一致、361 个上游文件缺失并已补齐；服务器新增文件保留。重构前快照与映射在 `archive/20261015_structure_v1/`。

当前规范以项目根 [README](../../../README.md) 为准；新执行使用登记入口，不再创建平铺的 runs/mainline_*。

```bash
cd /home/avis/pslg-nilm
bash scripts/check_mainline_zhouslab.sh
EXEC=$(python3 scripts/create_execution.py 20261015_mainline_restoration smoke)
python3 scripts/run_execution.py "$EXEC" -- .venv-generation-tf/bin/python scripts/smoke_mainline_zhouslab.py
```

zhouslab 不依赖 Slurm；自定义 NILM 配置使用 `runtime.require_slurm: false`。F4 已退出活动树并归档。
验证结果与限制见 `runs/20261015_mainline_restoration/reports/`，下游生成验证见 `runs/20261015_downstream_generation_validation/reports/`。
主线测试启动器会清除 PSLG_EXECUTION_DIR，保证测试夹具在临时根目录独立运行；外层执行登记和控制台日志仍保留。
