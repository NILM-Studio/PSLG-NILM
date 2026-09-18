# NILM 与活动生成流程目录

```text
nilm_experiments/
├── nilm_lab/                    # 接入根 main.py 的 NILM 实现
│   ├── models.py               # NILMFormer、FCN、BERT4NILM、SGN 适配器
│   ├── sequence_heads.py       # 状态/活动序列辅助头
│   ├── sequence_losses.py      # 辅助训练目标
│   ├── data.py / sequence_data.py
│   └── cli.py / train.py / workflow.py / campaign.py
├── generation_lab/             # 条件波形生成流程
│   ├── backbones/
│   │   ├── diffusion_ts_adapter.py
│   │   └── conditional_unet_1d.py
│   ├── diffusion.py            # 扩散目标与采样
│   ├── models.py               # 模型导出与自测
│   ├── prepare.py / dictionary.py
│   ├── plan.py / gate.py
│   └── train.py / sample.py / evaluate.py
├── third_party/                # 上游代码唯一副本，保留许可证
│   ├── nilmformer/             # NILMFormer 及共享 FCN 基线
│   ├── bert4nilm/
│   ├── nilmtk_contrib/          # 社区参考实现
│   └── diffusion_ts/
├── runtime/                    # 共享环境与镜像构建配方
│   ├── Dockerfile / Dockerfile.offline
│   ├── requirements-runtime.txt / requirements-lock.txt
│   ├── setup_conda.sh / verify_runtime.py
│   └── tensorflow/            # 发现阶段环境配方与安装任务
├── FCN/                        # runtime/、ENVIRONMENT.md 与 LICENSE
├── scripts/                    # 环境报告工具
├── sequence_workflow_20260915/  # 当前序列协议，仅跟踪 manifest
├── primitive_vs_direct_20260915/# 当前对照协议，仅跟踪 manifest 与 protocol
├── run_paths.py                # 生成结果路径约束
├── sources.lock.json           # 上游来源与版本
└── requirements.txt / pyproject.toml
```

当前配置引用的本地数据目录 `data_review_20260915/`、既有生成结果
`activity_gen_full_20260917_v1/` 与虚拟环境不入 Git。
新实验结果统一写入项目根目录 `runs/`，不写入模型源码目录。

## Slurm 模块边界

`nilm_lab` 与根目录 `main.py` 解耦。`main.py` 只负责发现阶段和冻结
`run_manifest.json`；NILM 的数据准备、标签、训练、选模、评估和报告由
`nilm_lab.standalone` 独立执行，不会再次进入 `main.py`。

```text
slurm/run_nilm_sequence.sh       # 发现阶段 + labels/train/select/report 编排
slurm/run_nilm_module.sh         # 单独提交一个 NILM 阶段
nilm_lab/standalone.py           # Slurm-facing 入口
nilm_lab/workflow.py             # 隔离的数值 worker
```

完整流程：

```bash
sbatch slurm/run_nilm_sequence.sh

# 对已有 source-only run 单独提交阶段
sbatch --export=ALL,COMMAND=train,CONFIG=config/config_nilm_sequence_pilot.yaml,RUN_ID=<run_id> \
  slurm/run_nilm_module.sh
```

`COMMAND` 可取 `data`、`labels`、`train`、`select`、`evaluate`、`report`。
每个模块读取 `runs/<run_id>/run_manifest.json`，只在对应步骤目录写入产物并
更新 manifest。Torch 数值阶段使用 `runtime.torch_python` 对应环境；发现阶段
的 TensorFlow/Numba 环境不进入 Torch worker。生成实验仍由
`slurm/run_activity_gen_*_v2.sh` 直接按 `prepare → dictionary → plan/gate →
train → sample/evaluate` 调用模块。
