# UK-DALE 洗衣机基元生成实验记录

> 最后更新：2026-08-20  
> 当前分支：`feature/primitive-generation`  
> 基准运行：`ukdale_wm_primglr_detsec_3789`  
> 用途：维护论文实验上下文、复现实验流程，并区分已验证结论与待办事项。

## 1. 研究目标

本阶段面向 NILM 小样本问题，从真实电器工作周期中提取基元状态，学习合理周期结构，再重组生成完整工作周期。生成数据最终用于扩充少量真实训练数据，并通过下游 NILM 性能变化判断实际价值。

当前以 UK-DALE House 1 的洗衣机为基准电器：

```text
原始功率序列
  -> 工作周期提取
  -> 周期内部时间分割
  -> DETSEC 特征提取
  -> KMeans 基元聚类
  -> 相邻状态合并
  -> 周期类别发现
  -> 完整周期与物理模式验证
  -> 周期级训练/验证/测试拆分
  -> 按类别/模式构建基元库
  -> 平滑拼接生成完整周期
  -> 生成质量评估
  -> NILM 小样本增益评估
```

状态编号 `0/1/2/3` 是聚类标签，不预先绑定“加热”“漂洗”等人工语义。其含义需要结合中心曲线、功率范围和周期位置解释。

## 2. 数据与环境

### 2.1 数据

- 数据集：UK-DALE，House 1。
- 电器通道：`labels.dat` 中 channel 5，`washing_machine`。
- 原始目录：`/home/scnu2024024563/dataset/house_1/`。
- 转换结果：`input/ukdale_washing_machine_full.csv`。
- 字段：`timestamp,power`。
- 记录数：19,555,935。
- 功率范围：0--3999 W。
- 大于10 W的采样点：946,476。
- 时间戳重复数：0；两个字段的缺失数均为0。
- 采样频率：约1/6 Hz，即 `fs=0.1666667`。

### 2.2 代码与服务器

- 本地仓库：`/Users/kyrie/Documents/Thesis_research_proposal/PSLG-NILM`。
- 服务器仓库：`/home/scnu2024024563/NILM/PSLG-NILM`。
- Git分支：`feature/primitive-generation`。
- Conda环境：`pslg-nilm`。
- TensorFlow 2.16.2；GPU为Slurm分配的RTX 3090。
- GPU环境入口：`slurm/env.sh`，由作业脚本source，不直接提交。
- 洗衣机训练脚本：`slurm/run_ukdale_washing_machine.sh`。

DETSEC训练需要GPU。得到聚类与合并产物后，`cycle_validate`、`synthesize` 和绘图不需要重新训练DETSEC，可复用同一 `run-id`。

## 3. 完整工作流

### 3.1 工作周期提取

配置：`config/config_ukdale_detsec.yaml`。

```yaml
extract_active_data:
  threshold: 10
  t_drop: 150
  t_min_work: 180
  context_seconds: 90
  fs: 0.1666667
```

低功率持续达到 `t_drop` 后才认为一次工作结束，从而减少洗衣过程中的短暂停机被误切为两个周期。

### 3.2 基元切分、特征与聚类

当前基准使用 `prim-glr` 切分周期内部变化点，DETSEC编码不同长度片段，KMeans完成基元聚类。已经比较 `k=3,4,5,6` 的中心图、t-SNE图、叠加曲线和周期重构图；人工检查认为 `k=4` 在区分度与解释性之间最好，当前采用：

```text
kmeans_k4_merged
```

`state_merge` 合并相邻同标签状态和满足条件的相似短片段，输出每个真实周期的状态块序列。重构图表明 `k=4` 能稳定区分约2 kW高功率阶段、间歇高功率阶段和低功率运行阶段。

### 3.3 周期类别发现

周期分类先折叠相邻相同标签，再按离散状态签名分类：

```text
[1,1,0,2,1] -> [1,0,2,1]
```

高频精确签名形成类别锚点。罕见签名只有在编辑距离足够小时才分配到现有类别，否则作为离群周期保留但不进入生成库。

### 3.4 完整周期与物理模式验证

验证步骤只使用与类别代表签名完全一致的周期：

1. 检查缺失值、最短时长和工作段首尾边界。
2. 检查核心状态和常见终止状态。
3. 使用时长、能耗、平均功率和峰值功率构建指标矩阵。
4. 使用BIC选择1--3个GMM物理模式。
5. 在每个模式内部用MAD robust z-score排除指标离群值。
6. 将类别标记为 `valid_full`、`valid_short`、`uncertain` 或 `invalid`。

生成器只接受schema version 3的验证目录，并要求：

```json
{
  "canonical_signatures_only": true,
  "physical_modes_required": true
}
```

旧验证目录会被拒绝，避免不完整类别或签名变体进入最终生成。

### 3.5 条件基元拼接

在拼接前，`cycle_split` 按 `(Class, Mode)` 分层，并在每组内部依据活动时间顺序执行 `70%/10%/20%` 的训练、验证和测试拆分。生成器启用 `require_cycle_split=true` 后只读取 `train_cycle_catalog.json`，测试周期的功率波形和基元不会进入生成库。

当前拆分发生在周期分类与模式验证之后，因此实现的是严格的**波形来源隔离**，但代表签名和GMM模式仍由全部有效周期估计。该方案作为当前可执行基线；最终若要求完全无结构信息泄漏，需要将拆分前移，只在训练集拟合类别和模式，再将测试周期映射到训练模型。

生成器按 `(class_id, mode_id)` 分别建立真实基元库、状态转移模型和采样器。当前基线为 `real_resample + empirical`：周期结构和目标持续时间来自同类别、同模式的有效真实周期；各状态块从对应基元库重采样。候选池选择和短窗口平滑用于降低连接跳变。

该方法不是TimeVAE，也不是复制完整真实周期。当前生成的是来自多个真实工作段的基元重组结果，并保留来源信息。

## 4. 当前实验结果

### 4.1 有效类别和模式

严格验证后保留920个真实周期：

| Class | 代表签名 | 有效周期 | 模式支持量 |
|---|---|---:|---|
| 0 | `1->0->2->1` | 343 | M0=14, M1=121, M2=208 |
| 1 | `3->0->3->1` | 264 | M0=239, M1=25 |
| 2 | `2->0->2->1` | 140 | M0=62, M1=67, M2=11 |
| 3 | `3->0->2->1` | 136 | M0=136 |
| 5 | `3->0->1` | 37 | M0=18, M1=19 |

排除类别：`4,6,7,8,9,10,11`。原因包括缺少常见核心状态、异常终止状态、支持量不足、签名纯度不足或类别为 `uncertain/invalid`。

`Class 0 / Mode 0` 仅14条、`Class 2 / Mode 2` 仅11条，后续必须单独报告质量，避免总体指标掩盖小模式风险。

输入审计文件：

```text
log/ukdale_wm_primglr_detsec_3789/
primitive_synthesis_real_resample_empirical_all_validated_modes_on_kmeans_k4_merged/
synthesis_input_audit.json
```

### 4.2 生成结构检查

当前按类别均衡生成100条，每个有效Class生成20条；模式在类别内部按经验支持量抽样。已确认：

```text
Class 0: 20
Class 1: Mode 0=16, Mode 1=4
Class 2: Mode 0=6, Mode 1=12, Mode 2=2
Class 3: Mode 0=20
Class 5: Mode 0=10, Mode 1=10
```

Class 0本轮可视化样本由Mode 1和Mode 2构成；低支持量Mode 0在经验随机抽样下可能未被抽中。最终精确分布以 `synthesis_manifest.json` 为准。

所有已检查生成周期的状态序列均严格等于对应代表签名，不再出现旧版本中的 `Class 4--11` 或非代表结构。

### 4.3 视觉检查结论

已人工检查最新20张 `Class/Mode` 标注图：

- 状态顺序正确，工作周期结构完整。
- 类别和模式元数据正确写入CSV与图标题。
- 状态边界整体连续，未见明显人工断层。
- 持续时间、尖峰位置和低功率阶段存在样本间变化，不是单一模板复制。
- 约3--4 kW孤立尖峰来自真实基元片段，而非拼接产生；是否保留仍需真实分布对比。
- 波形没有被过度平滑，真实片段的局部跌落和尖峰仍被保留。

因此，“筛选后的条件基元拼接”在结构和视觉层面可行，但视觉判断不能替代论文定量评价。

## 5. 复现命令

### 5.1 更新服务器代码

```bash
cd /home/scnu2024024563/NILM/PSLG-NILM
git checkout feature/primitive-generation
git pull origin feature/primitive-generation
```

### 5.2 重跑验证、周期拆分与训练集生成

```bash
python main.py \
  --config config/config_ukdale_detsec.yaml \
  --steps cycle_validate,cycle_split,synthesize \
  --run-id ukdale_wm_primglr_detsec_3789 \
  --cluster-tag kmeans_k4_merged \
  --cycle-class all
```

最新目录：

```text
cycle_validation_canonical_multimodal_robust_on_kmeans_k4_merged/
cycle_split_chronological_stratified_on_kmeans_k4_merged/
primitive_synthesis_real_resample_empirical_all_train_split_on_kmeans_k4_merged/
```

### 5.3 绘图

```bash
python -m visualize.visualize_cycle_validation \
  --run-id ukdale_wm_primglr_detsec_3789

python -m visualize.visualize_synthetic_cycles \
  --run-id ukdale_wm_primglr_detsec_3789 \
  --max-files 20
```

输出目录：

```text
output/ukdale_wm_primglr_detsec_3789/figure/cycle_validation_modes/
output/ukdale_wm_primglr_detsec_3789/figure/primitive_synthesis/
```

## 6. 论文实验计划

### 6.1 下一步：生成质量定量评价

先检查 `cycle_split_summary.json`，确认每个 `(Class, Mode)` 均保留训练成员和测试成员，再运行质量评价Step。按照920个有效周期和 `70%/10%/20%` 比例，总量应接近训练644、验证92、测试184；精确数量会因逐组取整略有变化。

当前拆分结果已经确认：训练644、验证92、测试184，11个 `(Class, Mode)` 均有训练与测试成员。`synthesis_eval` 已实现以下产物：

- `distribution_metrics.csv`：时长、能耗、平均功率和峰值功率的Wasserstein距离、归一化距离与KS统计量；
- `state_duration_metrics.csv`：各状态持续时间的同类指标；
- `novelty_metrics.csv`：生成周期到最近训练周期、最近其他生成周期的归一化形状RMSE；
- `quality_summary.json`：总体摘要及未覆盖模式列表。
- `real_holdout_shape_baseline.csv`：真实测试周期到最近训练周期、最近生成周期的形状距离，用作自然类内变化和生成覆盖率基线。

首轮完整均衡实验结果（2026-08-20）：训练真实周期644、测试真实周期184、生成周期220，11个 `(Class, Mode)` 全部覆盖；平均归一化Wasserstein距离为0.1057，生成到最近训练周期的形状RMSE为0.3920，生成样本间最近形状RMSE为0.3363。真实测试到训练集的自然形状差异为0.2639，因此生成新颖性比值为1.485；生成覆盖比值为1.544。统计分布总体接近，但生成形状偏离自然类内变化，主要异常为 `Class 5 / Mode 1` 的能耗和平均功率。

运行命令：

```bash
python main.py \
  --config config/config_ukdale_detsec.yaml \
  --steps synthesis_eval \
  --run-id ukdale_wm_primglr_detsec_3789 \
  --cluster-tag kmeans_k4_merged
```

按 `(Class, Mode)` 分组比较真实周期与生成周期：

- 周期时长、总能耗、平均功率和峰值功率；
- 各状态持续时间和周期占比；
- 状态序列正确率和边界跳变量；
- Wasserstein distance或MMD；
- 生成样本之间的DTW距离；
- 生成样本到最近真实周期的距离；
- 重复率、覆盖率和潜在记忆问题。

建议维护两套生成设置：

- `mode_sampling=empirical`：保持真实模式比例，用于NILM数据增强。
- `mode_sampling=balanced`：保证每个模式有足够样本，用于逐模式质量评估。
- `class_sampling=balanced_pairs`：直接均衡全部 `(Class, Mode)` 组合。本实验使用220个生成周期，对11个组合各生成20个，作为质量评价协议；正式数据增强仍使用经验比例。

### 6.2 周期条件基元生成

独立基元重采样只限制基元来自相同 `(Class, Mode)`，不同状态仍可能取自物理尺度差异较大的周期。改进方法为每个生成周期先抽取一个目标训练周期，再根据以下周期画像在同一类别和模式内寻找近邻：

- 周期时长、总能量、平均功率和峰值功率；
- 各状态持续时间占比；
- 各状态平均功率。

画像在组内采用中位数与四分位距稳健标准化，排除目标周期本身，然后只从最近周期邻域重组基元。近邻数经过 `k=3/5/10` 消融后确定为10。该方法保留跨周期组合能力，同时约束完整周期内部的功率与能量关联。每条生成记录保存目标周期、近邻周期、距离和最终基元来源。

严格消融协议将周期结构随机流与波形基元随机流分离，确保不同生成方法使用相同的类别、模式、目标周期、状态顺序和状态时长。质量评价时，目标周期在每个 `(Class, Mode)` 内采用打乱后的无放回轮换；因此小模式不会因为随机重复抽样而产生结构分布偏差。`conditioning_summary.json` 保存目标周期使用次数和随机流隔离标志。

改进方法：

```bash
python main.py \
  --config config/config_ukdale_detsec.yaml \
  --steps synthesize,synthesis_eval \
  --run-id ukdale_wm_primglr_detsec_3789 \
  --cluster-tag kmeans_k4_merged
```

独立采样消融基线：

```bash
python main.py \
  --config config/config_ukdale_detsec.yaml \
  --steps synthesize,synthesis_eval \
  --run-id ukdale_wm_primglr_detsec_3789 \
  --cluster-tag kmeans_k4_merged \
  --synthesis-conditioning independent
```

两种方法使用独立产物目录，分别带有 `cycle_neighbors` 和 `independent` 标签。比较重点是总体和逐模式Wasserstein距离、两个形状基线比值，以及 `Class 5 / Mode 1` 的能耗和平均功率。

严格配对消融结果（2026-08-20，随机种子42，220个生成周期）：

| 指标 | 独立基元采样 | 周期近邻条件采样 | 变化 |
|---|---:|---:|---:|
| 平均归一化Wasserstein | 0.1035 | 0.0972 | -6.1% |
| 生成到最近训练周期RMSE | 0.3934 | 0.3372 | -14.3% |
| 生成到最近测试周期RMSE | 0.5537 | 0.5095 | -8.0% |
| 生成样本间最近RMSE | 0.3619 | 0.3391 | -6.3% |
| 新颖性/真实自然变化比值 | 1.4908 | 1.2779 | 更接近1 |
| 覆盖/真实基线比值 | 1.5685 | 1.4915 | 更接近1 |

两种方法逐 `(Class, Mode)` 的时长Wasserstein差异最大绝对值为0，确认目标周期、状态结构和状态时长完全配对。`Class 5 / Mode 1` 的能耗距离从0.5482降至0.3126，平均功率距离从0.5255降至0.1896。周期条件方法明显改善物理一致性和形状真实性，但生成样本间最近距离下降6.3%，表明局部多样性略有收缩；部分模式的峰值功率距离上升，后续需通过近邻数消融和多随机种子实验确认稳定性。

#### 6.2.1 近邻数与随机种子消融

`--conditioning-neighbors` 和 `--synthesis-seed` 可覆盖配置文件。实验标签包含方法、近邻数和随机种子，例如 `cycle_neighbors_k5_seed42`，因此不同实验不会覆盖。第一阶段固定种子42比较 `k=3/5/10`：

```bash
bash scripts/run_ukdale_synthesis_ablation.sh \
  ukdale_wm_primglr_detsec_3789 42
```

脚本依次运行 `independent_seed42`、`cycle_neighbors_k3_seed42`、`cycle_neighbors_k5_seed42` 和 `cycle_neighbors_k10_seed42`，随后执行：

```bash
python scripts/summarize_synthesis_ablation.py \
  --run-id ukdale_wm_primglr_detsec_3789 \
  --cluster-tag kmeans_k4_merged \
  --neighbors 3 5 10 \
  --seeds 42
```

汇总产物位于：

```text
output/ukdale_wm_primglr_detsec_3789/synthesis_ablation/
├── ablation_runs.csv
├── ablation_aggregate.csv
├── paired_metric_deltas.csv
└── ablation_report.json
```

固定种子42的近邻数消融结果如下：

| 方法 | 平均归一化Wasserstein | 生成到训练RMSE | 生成到测试RMSE | 生成样本间RMSE | 新颖性比值 | 覆盖比值 |
|---|---:|---:|---:|---:|---:|---:|
| Independent | 0.1035 | 0.3934 | 0.5537 | 0.3619 | 1.4908 | 1.5685 |
| Cycle-conditioned, k=3 | 0.0960 | **0.3242** | **0.4993** | 0.3092 | **1.2285** | **1.4805** |
| Cycle-conditioned, k=5 | 0.0972 | 0.3372 | 0.5095 | 0.3391 | 1.2779 | 1.4915 |
| Cycle-conditioned, k=10 | **0.0917** | 0.3445 | 0.5100 | **0.3493** | 1.3055 | 1.4811 |

三个近邻条件方法均优于独立采样。`k=3` 的形状保真、物理邻近性和测试集覆盖最好，但样本间距离最低，存在局部多样性收缩；`k=10` 的整体统计分布最好，并保留更多样本间变化。`k=5` 未在主要指标上形成优势，因此后续稳定性实验只保留 `k=3` 和 `k=10`，最终选择必须依据多随机种子均值与标准差，而不能依据种子42单次结果。

脚本第三个参数可指定待运行的近邻数。使用种子 `43 44 45 46` 补充两种候选设置：

```bash
for seed in 43 44 45 46; do
  bash scripts/run_ukdale_synthesis_ablation.sh \
    ukdale_wm_primglr_detsec_3789 "$seed" "3 10"
done

python scripts/summarize_synthesis_ablation.py \
  --run-id ukdale_wm_primglr_detsec_3789 \
  --cluster-tag kmeans_k4_merged \
  --neighbors 3 10 \
  --seeds 42 43 44 45 46
```

最终报告五个随机种子的均值与标准差。汇总工具会强制检查每组配对实验的周期时长指标完全一致，避免非公平比较进入最终表格。

五随机种子稳定性结果（种子 `42-46`，均值 ± 标准差）：

| 方法 | 归一化Wasserstein | 生成到训练RMSE | 生成到测试RMSE | 生成样本间RMSE | 新颖性比值 | 覆盖比值 |
|---|---:|---:|---:|---:|---:|---:|
| Independent | 0.1052 ± 0.0034 | 0.3983 ± 0.0060 | 0.5540 ± 0.0081 | 0.3642 ± 0.0063 | 1.5094 ± 0.0228 | 1.5908 ± 0.0343 |
| Cycle-conditioned, k=3 | 0.0979 ± 0.0042 | **0.3240 ± 0.0057** | **0.4972 ± 0.0074** | 0.3192 ± 0.0064 | **1.2278 ± 0.0217** | **1.4896 ± 0.0287** |
| Cycle-conditioned, k=10 | **0.0924 ± 0.0027** | 0.3423 ± 0.0058 | 0.5076 ± 0.0088 | **0.3447 ± 0.0074** | 1.2971 ± 0.0218 | 1.5017 ± 0.0422 |

相对独立采样，`k=10` 将平均归一化Wasserstein降低约12.2%，生成到训练和测试周期的形状RMSE分别降低约14.1%和8.4%，同时生成样本间最近距离只降低约5.4%。`k=3` 的单样本形状保真和覆盖更强，但生成样本间距离降低约12.4%，多样性收缩更明显。考虑论文目标是为NILM小样本学习提供既真实又具有变化的增强数据，正式方法选择 `k=10`；`k=3` 作为偏重局部保真的消融方案保留。所有主要指标的跨种子标准差较小，结论未依赖单一随机种子。

### 6.3 核心实验：NILM小样本增益

建议训练组：

```text
A. 少量真实数据
B. 少量真实数据 + 传统扰动增强
C. 少量真实数据 + 本文生成数据
D. 全量真实数据（参考上限）
```

真实数据比例建议为 `5%,10%,20%,50%`。测试集必须由未参与基元库、类别发现、模式估计和生成器拟合的真实周期组成，防止数据泄漏。下游指标建议包括 `MAE、SAE、NDE、F1`。论文主要结论应来自C相对A/B的提升，而不是只依赖曲线视觉相似度。

### 6.4 后续扩展

洗衣机完成全套评价后，再选择至少一种运行逻辑明显不同的电器，例如冰箱或水壶，验证方法是否只适用于多阶段洗衣机。

## 7. 当前结论与限制

### 已完成

- UK-DALE洗衣机全量数据转换与工作周期提取。
- `prim-glr + DETSEC + KMeans(k=4)` 基元分割与聚类。
- 周期重构图人工检查。
- 周期类别发现、多模式验证和MAD离群值过滤。
- 只使用 `valid_full`、精确代表签名和有效模式的严格生成。
- 220条均衡生成周期、11个类别/模式组合及结构与拼接连续性的人工验收。
- 独立基元重采样的留出集统计、形状新颖性和覆盖率基线。

### 尚未证明

- 生成分布与真实分布在统计上足够接近。
- 生成数据具有足够多样性且没有记忆训练样本。
- 生成数据能够提升下游NILM小样本性能。
- 方法可以泛化到其他电器、房屋和数据集。

论文表述必须区分“已完成的工程流程”“视觉可行性”和“经实验验证的研究结论”。

## 8. 代码节点

关键提交：

- `c5498dd`：按周期类别平滑拼接基元。
- `9afa136`：增加稳健周期验证工作流。
- `dc92ab7`：保留多物理模式程序。
- `814f471`：模式验证只使用精确代表签名。
- `abc6b88`：强制验证类别和模式，输出输入审计。
- `91167ba`：统一相邻重复状态的规范签名与生成逻辑。

关键源码：

- `src/steps/temporal_state_merge_step.py`
- `src/steps/cycle_classification_step.py`
- `src/steps/cycle_validation_step.py`
- `src/steps/primitive_synthesis_step.py`
- `src/generation/cycle_patterns.py`
- `src/generation/cycle_validation.py`
- `src/generation/primitive_library.py`
- `visualize/visualize_cycle_validation.py`
- `visualize/visualize_synthetic_cycles.py`
