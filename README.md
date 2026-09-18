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

NILM 数值实验由 `slurm/run_nilm_module.sh` 和
`nilm_experiments/nilm_lab/standalone.py` 按模块调度；发现阶段才使用
`main.py`。活动生成实验由 `slurm/run_activity_gen_*_v2.sh` 直接调用
`generation_lab` 各阶段模块。具体边界见
[nilm_experiments/README.md](nilm_experiments/README.md)。

## 启动主流程

主流程入口是根目录 `main.py`。它负责数据发现、分段、特征提取、聚类和状态
合并；运行结果写入 `runs/<run_id>/run_manifest.json`。推荐通过 Slurm 启动：

```bash
# 完整的“数据冻结 + 发现 + NILM”编排
sbatch slurm/run_nilm_sequence.sh

# 仅运行发现主流程（不进入 NILM 模型）
python main.py --config config/config.yaml \
  --profile discovery --appliance fridge \
  --steps extract,segment,feature,cluster,state_merge,state_sequence
```

可用参数包括 `--config`、`--profile discovery|nilm`、`--steps`、
`--appliance`、`--run-id`、`--segment-method`、`--feature-model` 和
`--cluster-method`。已存在的 `run_id` 会复用其 manifest；正式实验应为每次
协议或配置变化使用新的 `run_id`。

## 主流程结束后调用 NILM 模型

NILM 模型不由 `main.py` 再次派发。序列脚本先用独立 Torch 环境运行
`nilm_lab.standalone data` 冻结输入，然后由 `main.py` 完成发现；发现结束后，
再用 Torch 环境依次执行：

```text
labels → train → select → report
```

完整编排已经包含在：

```bash
sbatch slurm/run_nilm_sequence.sh
```

如需单独启动某一个阶段：

```bash
sbatch --export=ALL,COMMAND=labels,CONFIG=config/config_nilm_sequence_pilot.yaml,RUN_ID=<run_id> \
  slurm/run_nilm_module.sh

sbatch --export=ALL,COMMAND=train,CONFIG=config/config_nilm_sequence_pilot.yaml,RUN_ID=<run_id> \
  slurm/run_nilm_module.sh

sbatch --export=ALL,COMMAND=select,CONFIG=config/config_nilm_sequence_pilot.yaml,RUN_ID=<run_id> \
  slurm/run_nilm_module.sh

sbatch --export=ALL,COMMAND=report,CONFIG=config/config_nilm_sequence_pilot.yaml,RUN_ID=<run_id> \
  slurm/run_nilm_module.sh
```

`COMMAND` 还支持 `data` 和显式的 `evaluate`。模型、训练战役和 checkpoint
由 `nilm_experiments/nilm_lab/` 管理，结果分别写入
`runs/<run_id>/nilm_labels`、`nilm_train`、`nilm_select`、`nilm_report`。
发现阶段使用 `nilm-discovery-tf-v1`，NILM 模型阶段使用
`nilmformer-cu118-v1`。不要在同一个 Python 进程中混用两套环境的依赖。

## 主流程结束后调用生成模型

生成模型不依赖 `main.py`，由 Slurm 按 campaign 分阶段调用
`nilm_experiments.generation_lab`。先指定一个新 campaign：

```bash
CAMPAIGN=runs/generation/washing_machine_v1
```

按以下顺序提交：

```bash
# 1. 原始 CSV → 6 秒网格、活动片段和 train/dev/holdout
sbatch --export=ALL,CAMPAIGN="$CAMPAIGN" \
  slurm/run_activity_gen_prepare_v2.sh

# 2. PrimGLR 片段 → 训练/开发字典
sbatch --export=ALL,CAMPAIGN="$CAMPAIGN" \
  slurm/run_activity_gen_dictionary_v2.sh

# 3. 生成不可变试验矩阵；随后执行 pilot gate
sbatch --export=ALL,CAMPAIGN="$CAMPAIGN",STAGE=plan \
  slurm/run_activity_gen_contracts_v2.sh
sbatch --export=ALL,CAMPAIGN="$CAMPAIGN",STAGE=pilot \
  slurm/run_activity_gen_contracts_v2.sh

# 4. 按 plan 中的数组索引训练
sbatch --array=0-<N-1> --export=ALL,CAMPAIGN="$CAMPAIGN",PHASE=pilot \
  slurm/run_activity_gen_train_v2.sh

# 5. 样本和评估；正式阶段需先完成相应 gate/freeze
sbatch --array=0-<N-1> --export=ALL,CAMPAIGN="$CAMPAIGN" \
  slurm/run_activity_gen_sample_v2.sh
sbatch --export=ALL,CAMPAIGN="$CAMPAIGN" \
  slurm/run_activity_gen_evaluate_v2.sh
```

`STAGE` 可取 `plan`、`pilot`、`freeze`、`summary`；`PHASE` 可取
`pilot`、`development`、`formal`、`fourier`、`sensitivity`。训练和采样的
数组范围必须与对应 `plan_gap*/<phase>.json` 的条目数一致。生成阶段使用
`nilmformer-cu118-v1`；字典阶段使用 `nilm-discovery-tf-v1`。
所有 campaign 产物位于 `runs/generation/<campaign>/`，不会写回源码目录。
