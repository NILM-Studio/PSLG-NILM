# PSLG-NILM

工作目录 `/home/avis/pslg-nilm`，连接 `ssh zhouslab`。本说明对应 2026-10-15 重构后的结构。
GitHub 主路线基线为 `eb85bf532b5ce6db0b13d0b7348d80c7964253b7`；保留服务器新增代码，未提交或推送 Git。

```text
pslg-nilm/
├── main.py                         # 主工作流
├── src/
│   ├── framework/, steps/, utils/   # 主线框架与步骤
│   ├── nilm_lab/                   # NILM 适配、标签、训练与评价
│   ├── generation_lab/             # 活动条件生成流程（不含 F4）
│   └── generation_paths.py         # 生成执行路径管理
├── models/
│   ├── extract_active_data/, time_segmentation/, feature_extract/, clustering/
│   ├── third_party/                # 原 nilm_experiments 的上游实现
│   ├── sources.lock.json
│   └── downstream_generation_models/ # 下游生成模型源码、适配器、环境配方
├── config/                         # 公共配置，保持原单数名称
├── datasets/                       # 共享原始数据，不移入模型目录
├── runtime/                        # 环境配方和本机依赖 overlay
│   ├── nilm/, fcn/
│   ├── generation/torch_site/
│   └── envs/timegan/               # 克隆并验证的 Python 3.7 / TF 1.15
├── .venv-generation-*/             # 已有共享环境保持原位
├── scripts/, tests/, visualize/, docs/, guides/
├── slurm/                          # 仅其他服务器使用，zhouslab 不依赖 Slurm
├── runs/
│   ├── README.md, EXPERIMENT_LOG.md, _template/
│   └── YYYYMMDD_name/
│       ├── README.md, proposal.md, protocol.md
│       ├── configs/, scripts/, data/
│       ├── executions/
│       │   ├── index.json
│       │   └── <condition>/<execution_id>/
│       │       ├── run.json, config.resolved.json, command.json
│       │       ├── environment.json, inputs.json, artifacts.json
│       │       └── logs/, checkpoints/, features/, generations/, analysis/
│       └── reports/
│           ├── README.md
│           └── <report_id>/report.md, figures/, tables/
└── archive/                        # 迁移快照、路径映射、退出主线的 F4
```

## 归属与约束

- 实验注册见 [runs/README.md](runs/README.md)，增量登记见 [runs/EXPERIMENT_LOG.md](runs/EXPERIMENT_LOG.md)。方案、报告及图表必须归属实验，不在模型目录保存运行结果。
- `nilm_experiments/` 与根目录 `downstream_generation_models/` 已迁移，不保留旧入口软链接。共享代码不是实验数据，因此分别归入 `src/`、`models/` 和 `runtime/`。
- 历史执行内部文件保持原样，包括记录中的旧绝对路径；按 [迁移映射](archive/20261015_structure_v1/moves.json) 解释。未知的配置、退出码、审核、完成时间写 `unknown`，不补造实验结果。
- F4 源码及 F3/F4 混合历史实验整体保留于 `archive/retired_f4/`，不进入主线、默认测试或后续实验。未永久删除历史数据。
- 模型源码、环境配方、执行元数据、方案、报告、图表和小型汇总表可以纳入版本管理；原始数据、环境、日志、大型中间结果及权重不纳入 Git。
- `run_directories` 与 `resolve_campaign` 保留旧调用的路径兼容逻辑；**新实验必须通过登记入口和 `PSLG_EXECUTION_DIR` 启动**，不要直接调用旧接口创建平铺结果。

## 新实验操作

1. 复制 `runs/_template/` 为 `runs/YYYYMMDD_name/`；日期使用原方案或首次实验日期。填写 proposal、protocol，并登记注册表。
2. 配置放 `configs/`，专用代码放 `scripts/`；输入清单记录来源、采样规则、版本和哈希，不修改外部数据。
3. `python3 scripts/create_execution.py YYYYMMDD_name condition --config <完整JSON配置>` 只登记，输出独立执行目录。
4. `python3 scripts/run_execution.py <执行目录> -- <实际命令及参数>` 启动并保存命令、控制台日志、退出码和产物清单；自动设置 `PSLG_EXECUTION_DIR`。同一执行禁止覆盖重跑。
5. 补全实际运行环境、输入和产物验证；报告保存为 `reports/<report_id>/report.md`，图表及汇总表放同级 `figures/`、`tables/`。
6. 阶段结束（包括失败）必须追加 `runs/EXPERIMENT_LOG.md`，更新报告索引与执行索引。进程退出成功不等于科学结论验证通过。

示例（先按服务器使用情况选择可用 GPU，不预设占用某张卡）：

```bash
cd /home/avis/pslg-nilm
bash scripts/check_mainline_zhouslab.sh
EXEC=$(python3 scripts/create_execution.py 20261015_mainline_restoration smoke)
python3 scripts/run_execution.py "$EXEC" -- .venv-generation-tf/bin/python scripts/smoke_mainline_zhouslab.py
EXEC=$(python3 scripts/create_execution.py 20261015_downstream_generation_validation smoke)
python3 scripts/run_execution.py "$EXEC" -- bash models/downstream_generation_models/scripts/smoke_all.sh
```

主线包含活动状态提取 → 切分 → 特征提取 → 聚类 → 合并 → 状态序列；最小验证使用真实自采数据，不代表完整训练效果。
下游模型说明见 [部署文档](models/downstream_generation_models/README.md)，环境与服务器要求见 [guides/server-usage.md](guides/server-usage.md)。
旧 README 完整保存于 [迁移前说明](archive/20261015_structure_v1/README.before.md)，其中旧路径仅供溯源。
