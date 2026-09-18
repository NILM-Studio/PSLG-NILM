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
