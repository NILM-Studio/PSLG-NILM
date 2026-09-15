# 生成与下游验证：本轮服务器交接

## 研究边界与本轮交付

切分和聚类属于师兄的已有工作，保持算法和原始产物不变。本人的研究链是：继承状态基元 → 周期组织/条件重组 → 同预算增强对比 → 真实连续时间线 NILM 验证。本轮没有重新训练 DETSEC、改聚类算法，亦没有把工程检查当成论文性能证据。

本轮实现：

- 预算内 `independent` 与 `cycle_neighbors` 两条生成路径；近邻画像、候选库仅来自所选预算内同 Class/Mode 的周期。两方法配对使用相同锚点、块顺序、块长度和种子。非单例排除锚点，单例明确记录 `singleton_self_resample`，不当作跨周期生成成功。
- 原语覆盖、模板位置、源功率有效性检查；已有截尾/缺口会报错，不改写师兄产物。
- 时间边界剔除：较早分组中任何跨到下一分组起点的完整活动标记 `purged`，保留原分组和原因，分类器排除这些活动。剔除导致空分组时停止。
- UK-DALE 官方四列 `mains.dat` 有功字段解析；两列文件默认计量类型未知。有功/有功对齐审计通过后才允许配置中的背景相减。支持日期范围和换机日期统计。
- 数据集目录按方法/邻居数/种子区分；连续集再按实际输入、参数和代码摘要隔离；训练结果必须匹配数据、参数、代码、运行库指纹才可复用。
- CPU 验证脚本按实际 manifest 读取产物，报告覆盖缺口、换机前后来源、真实/增强/OFF 数据分布、预算来源、时间边界、计量类型和单例比例。

历史 `primitive_synthesis → synthesis_evaluation` 结果仍是另一批产物。虽然预算内重组现在调用相同近邻机制，不能把历史质量数值当作本轮 C 组质量。

## 1. 拉取并检查继承接口（先运行这一段）

在服务器已配置好的项目 Python 环境下执行：

```bash
cd /home/scnu2024024563/NILM/PSLG-NILM
git switch feature/primitive-generation
git pull --ff-only origin feature/primitive-generation

python -m scripts.validate_generation_run \
  --run-id ukdale_wm_primglr_detsec_3789 \
  --stage upstream \
  --device-change-date '2015-09-08T00:00:00+01:00' \
  --output log/ukdale_wm_primglr_detsec_3789/upstream_validation_20260915.json
```

第一段只读已有实验，除指定的报告文件外不改产物；不需要 GPU。换机时刻是按日期设定的诊断边界，不是已确认的秒级设备切换事件。

先回传 `upstream_validation_20260915.json`。若 `validation_passed=false`，发完整 `errors` 和 `examples`，不要跳过检查直接训练。若是 `max_seg_len` 截断导致缺口，我们先确定仅下游接口修复的方式，不把修复擅自扩展成重做师兄方法。

## 2. 接口通过后，新建下游实验，不覆盖旧结果

```bash
python -m scripts.prepare_downstream_run \
  --source-run-id ukdale_wm_primglr_detsec_3789 \
  --run-id ukdale_wm_generation_verify_20260915
```

仅创建新 manifest，继承上游五个步骤及其绝对产物路径，不复制大文件，不继承旧时间划分/周期类别/生成/训练结果。新目录已存在会拒绝。记录来源 manifest 哈希和 inherited ownership；绝对引用要求旧目录保留不变。这不是上游端到端拟合隔离认证。

## 3. 使用真实有功总表重新对齐

先确认服务器确有以下 `house_1/mains.dat`；不存在时回报路径和可用数据，不用 `channel_1.dat` 冒充有功总表。

```bash
python -m scripts.prepare_ukdale_nilm_pair \
  --mains /home/scnu2024024563/dataset/house_1/mains.dat \
  --appliance input/ukdale_washing_machine_full.csv \
  --out input/ukdale_house1_active_mains_washing_machine.csv \
  --mains-format ukdale-mains \
  --mains-power-type active \
  --appliance-power-type active \
  --instance-boundary '2015-09-08T00:00:00+01:00' \
  --tolerance-seconds 3.1
```

输出 `.csv` 及 `.csv.audit.json`。新文件名避免混用旧 apparent 对齐结果；若此新文件已存在且属于需保留的实验，先使用另一个新名称并同步配置两处 `aligned_series`。测量类型通过不代表同步误差为零，也不保证每个样本总表都大于支路；检查匹配率、时间覆盖、负值和重用情况。

该命令没有裁剪设备时期。先检查上游报告中的 `device_era_counts` 和本对齐 audit 的覆盖范围。只有确认同一设备时期/任务定义后，才冻结论文主实验；不能仅裁剪对齐CSV却继续沿用全时期的类别、模式来宣称同机结论。`--start` 含起点、`--end` 不含终点，无时区字符串按 UTC 解释。

## 4. 运行 CPU 下游链和检查

```bash
python main.py \
  --config config/config_ukdale_detsec.yaml \
  --run-id ukdale_wm_generation_verify_20260915 \
  --cluster-tag kmeans_k4_merged \
  --segment-method prim-glr --feature-model detsec \
  --steps temporal_holdout,cycle_classify,cycle_validate,cycle_split,nilm_dataset,nilm_continuous \
  --synthesis-conditioning cycle_neighbors \
  --conditioning-neighbors 10 --synthesis-seed 42

python -m scripts.validate_generation_run \
  --run-id ukdale_wm_generation_verify_20260915 \
  --stage all \
  --device-change-date '2015-09-08T00:00:00+01:00' \
  --output log/ukdale_wm_generation_verify_20260915/generation_validation.json
```

这里不运行 `extract/segment/feature/cluster/state_merge`，也不需要先跑全局 `synthesize`。CPU工作量取决于原始时间线规模和预算数；本轮没有用真实 UK-DALE 跑通整个链，不能承诺固定用时。

将这三项发回：

1. `generation_validation.json`，以及新 `run_manifest.json`。
2. `temporal_holdout_summary.json`（路径由 manifest 的 `temporal_holdout.artifacts.summary` 指向）。
3. 有功对齐 `.csv.audit.json`。若生成失败，完整错误日志和对应 activity ID 优先于截图。

不要手动猜旧目录名。CPU report 给出当前 `dataset_dir` 和 `continuous_dir`；训练 CLI 和连续 Slurm 包装也改为从 manifest 解析。

`independent` 消融可在该链通过后，使用相同参数，仅改变 `--synthesis-conditioning independent` 重跑 `nilm_dataset,nilm_continuous`，并将 report 保存为另一个文件。两种目录共存，但 run manifest 只指向最后一次选中的版本。提交训练时应显式固定所需的 `DATASET_DIR`，避免排队期间 manifest 选择改变。

## 5. 通过检查后的 GPU 冒烟（不是最终实验）

先根据回传报告确认计量类型、跨设备范围和单例比例，再做短跑。`DATASET_DIR` 必须替换成 report 内的实际路径：

```bash
RUN_ID=ukdale_wm_generation_verify_20260915 \
DATASET_DIR=/实际报告中的/continuous_dir \
OUTPUT_ROOT=log/ukdale_wm_generation_verify_20260915/smoke_seed42 \
EXPERIMENTS=05pct:A,05pct:B,05pct:C EPOCHS=2 SEED=42 \
sbatch slurm/run_ukdale_seq2point_continuous.sh
```

正式训练另设 `OUTPUT_ROOT`，不可把2 epoch的metrics当成30 epoch完成结果。指纹不符会拒绝复用；不要用 `--force` 绕过身份检查来混写结果。

## 结论边界和接下来的研究任务

当前固定下游任务：在继承同一上游表示、共享类别/模式/OFF资源的条件下，比较 A 少量真实、B 真实+传统增强、C 真实+预算内周期生成，并用 D 全部合格真实周期作为参考。D 不是完整训练时间线；预算分母是通过周期筛选和对齐的训练活动数。应报告真实周期数、单例/跨周期比例、独立来源数、ON/OFF占比。

本轮 CPU 检查通过只能证明受检接口与数据协议满足实现约束，不能证明增强有效、新颖性或可发表性。还需完成：

- 冻结同机/跨机任务定义和上游共享资源口径，避免声称整个流程严格少标签或端到端无泄漏。
- 对实际预算 C 组做质量与重复率检查，确认不是主要靠单例重放。
- 先小预算 A/B/C 跑通，再做配对多种子和生成消融；评价真实连续数据 MAE、能量误差、F1及OFF误报，参数选择仅看验证集。
- 当前窗口在文件/分块边缘仍零填充，尚未实现连续 halo；正式实验前需评估边缘效应。
- 旧测试集若已用于选 k，不能称为全新盲测。新的冻结评估协议需要另行确定。

本地测试验证的是小规模 fixture、实现和错误保护；本地没有 UK-DALE 的实际上游运行产物，也未执行 TensorFlow/GPU训练。另找到的 REDD HDF5 不等于这次 UK-DALE 实验数据，不能替代服务器验证。

本轮本地回归：`158 passed, 6 warnings`；警告为已有的核心数检测和小型重复样本聚类收敛警告。`git diff --check`、连续 Slurm 脚本的 `bash -n` 通过。复现测试命令：`python -m pytest tests -q -p no:cacheprovider`。
