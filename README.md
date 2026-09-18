# PSLG-NILM

当前流程目录结构：

```text
pslg-nilm/
├── main.py                     # NILM 与基元发现工作流入口
├── config/                     # 发现、NILM、活动生成配置
├── src/
│   ├── framework/              # Step、Workflow、RunManifest 与路径管理
│   ├── steps/                  # 数据、发现、标签、训练、选模、评估、报告
│   └── utils/                  # 指标与状态映射工具
├── models/                     # 分段、特征提取、聚类等发现模型
├── nilm_experiments/
│   ├── nilm_lab/               # NILM 适配器、序列头、损失与训练
│   ├── generation_lab/         # 条件波形生成流程
│   ├── third_party/            # 去重后的上游代码与许可证
│   ├── runtime/                # Conda/Pip 配方、Dockerfile、环境验证
│   ├── FCN/                   # FCN 专用环境配方
│   └── scripts/               # 环境检查工具
├── slurm/                     # 当前集群任务入口
├── scripts/                   # 数据准备、审计与运行状态工具
├── visualize/                 # 基于 manifest 的绘图
├── tests/                     # 当前流程测试
├── docs/                      # 当前流程使用与设计说明
├── requirements.txt           # 基础依赖
├── pytest.ini                 # 主线测试范围
├── input/、datasets/          # 本地输入数据，不入 Git
├── .cache/                    # 本地特征缓存，不入 Git
└── runs/                      # 运行产物不入 Git，仅跟踪目录说明与忽略规则
    ├── <run_id>/              # 工作流 manifest、步骤产物、图表
    ├── generation/<campaign>/ # 生成实验计划、权重、样本和指标
    └── slurm/                 # 作业日志
```

模型和环境目录详见 [nilm_experiments/README.md](nilm_experiments/README.md)。
仅跟踪现役代码、配置、依赖配方和文档；原始数据、实验结果、虚拟环境及镜像不入 Git。
