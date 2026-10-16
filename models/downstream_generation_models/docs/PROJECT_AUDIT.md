# `/home/avis/pslg-nilm` 数据、模型与流程审计

更新（2026-10-15）：下文为恢复前审计，旧目录与“缺失”判断仅作历史记录。现已从官方 GitHub 提交 `eb85bf5` 补齐主线，真实微波炉数据全流程已跑通。当前说明见 [主线入口](../../../runtime/nilm/zhouslab/MAINLINE.md) 和 [根 README](../../../README.md)；F4 已归档退出主线。

审计日期：2026-10-15。结论以服务器磁盘实际文件为准，而不是只按 README 的目标目录结构判断。

## 可用数据集

`datasets/` 下当前只有自采集单电器数据：

| 文件 | 大小 | 行数（含表头） | 时间范围/频率 |
|---|---:|---:|---|
| `Air_condition.csv` | 172,583,760 B | 1,468,798 | 2025-08-04 起，约 1 Hz |
| `Microwave.csv` | 169,221,414 B | 1,468,801 | 2025-08-04 起，约 1 Hz |
| `WashMachine.csv` | 171,161,257 B | 1,468,800 | 2025-08-04 起，约 1 Hz |

三者字段相同：`datetime, timestamp, current, voltage, apparent power, active power, status, on/off status`。完整跨度约 17 天。空调少 3 行，正式预处理必须按时间戳重采样，不能假定三张表逐行完全对齐。当前 `datasets/` 未发现 REDD、UK-DALE、REFIT 等公共数据集副本。

## 磁盘上实际存在的模型

活动检测：`AdaptiveClusteringDetector`、`SimpleThresholdDetector`。

切分：PrimGLR、ClaSP/ClaSP ensemble、Espresso、FLUSS，以及 `extension_v2` 的统一切分适配入口。

特征：DTW、普通 AutoEncoder、CNN-AE、LSTM-AE、BiLSTM-AE、带 attention 的 BiLSTM-AE、DETSEC、物理约束 DeTSEC（含时序 TV/非负等损失）。

聚类：DPC-KMeans，含简化 DBCV、轮廓等评价及 K 扫描。

已有生成代码：`nilm_experiments/generation_lab` 中有 Conditional U-Net、Conditional Diffusion-TS 适配器，以及状态扩散 `StateDiffusion`；还存在 prepare/dictionary/plan/gate/train/sample/evaluate 与 F4 生成实验模块。此前已产生 `runs/generation` 和 F4 实验运行目录。

本次新增的报告模型位于 `downstream_generation_models/vendor`：Diffusion-TS、CSDI、ImagenTime、TimeVAE、TimeVQVAE、TimeGAN、Fourier-flows。C-TimeGAN 没有公开源码入口。

## 是否已有完整“活动状态提取—切分—特征—聚类—合并”流程

没有一套与当前 README 命令一致、可直接从头运行的完整流程。

证据：

- README 指定根目录 `main.py`、`src/framework`、`src/steps` 和 `config/config.yaml`，磁盘上均不存在。
- README 指定 `nilm_experiments/nilm_lab`、`slurm/run_nilm_sequence.sh`、`slurm/run_nilm_module.sh`，磁盘上也不存在；当前 `slurm/` 只有 `run_f4_generation.sh`。
- 提取、切分、特征、聚类的算法文件实际存在，但没有 README 所称的主工作流将它们串接起来。
- 未发现独立的 `state_merge` / `state_sequence` 实现与对应工作流步骤，因此“合并”环节不能按 README 命令验证。
- `nilm_experiments/README.md` 同样描述了缺失的 `nilm_lab` 和 Slurm 脚本，说明文档更像完整工程的目标/来源快照，而当前服务器目录是被裁剪后的实验子集。

因此应区分两个结论：算法组件“部分可用”，端到端活动发现主流程“当前不完整”。已有 `generation_lab` 是相对独立的生成实验代码，不等同于活动发现五阶段主流程。

## 建议修复顺序

1. 从可信源恢复 `main.py`、`src/`、完整 `config/`、`nilm_lab/` 和 Slurm 编排，或承认当前是裁剪版并重写 README。
2. 为现有组件建立一个最小 manifest 驱动入口，固定输入/输出 schema：活动区间、切点、特征矩阵、聚类标签、状态合并映射。
3. 明确实现合并规则（小簇合并、相邻同状态合并、持续时间/转移约束），为每阶段增加可恢复中间产物和测试。
4. 用时间隔离的 train/dev/holdout 验证完整链路，再把本目录生成模型接在冻结的活动/状态产物之后；不要直接在完整时间序列上随机切窗造成泄漏。
