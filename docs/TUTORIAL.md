# PSLG-NILM-ADVANCED 全流程教程

从零部署到跑出图表的完整 walkthrough。以 UK-DALE washing machine 的
100 行小切片为运行示例，真实规模实验只是把配置和数据换掉。

---

## 1. 环境部署

```bash
# Python 3.12
pip install -r requirements.txt
```

部署时注意（都是踩过的坑）：

| 坑 | 处理 |
|---|---|
| **numpy 必须 <2** | pandas 2.2.x 按 numpy 1.x 编译；requirements 已锁 `numpy==1.26.4`。若环境里被别的包装了 numpy 2.x（典型症状：`numpy.core.multiarray failed to import`），降级即可 |
| **nilmtk 不在 PyPI**（py3.12） | requirements 里是 `nilmtk @ git+https://github.com/nilmtk/nilmtk.git`，pip 会从 GitHub 源码装 |
| **不要 pip 安装 claspy** | clasp/fluss/espresso 已 vendored 在 `models/time_segmentation/`，`main.py` 自动加 sys.path。pip 版会拖入 numpy 2.x 破坏环境 |
| **torch 要么是好的、要么别装** | tslearn import 时探测 torch，只捕获 `ImportError`；损坏的 torch（DLL 加载失败）抛 `OSError` 会让 tslearn 整个不可用 |
| tensorflow | 可选。只有神经特征模型（detsec/bilstm_ae/…）需要；`dtw` 特征模型不需要 |
| hdbscan | 可选。只有 `--cluster-method hdbscan` 需要 |

验证部署：

```bash
python tests/test_m1_framework.py   # 引擎 (8)
python tests/test_m2_cache.py       # 缓存 (10)
python tests/test_m3_clustering.py  # 聚类 (7)
python tests/test_m4_downstream.py  # 下游 (3)
```

## 2. 数据准备

### 2.1 从 UK-DALE 截一段测试数据（nilmtk）

`datasets/ukdale/ukdale.h5` 是 nilmtk 格式 HDF5。准备脚本会加载指定电器的
功率序列，找到第一个活动区，截取 N 行导出为 CSV：

```bash
python scripts/prepare_ukdale.py
# 默认: building 1, washing machine, 100 行, >20W 为活动
# 输出: input/ukdale_washing_machine_100.csv  (列: timestamp, power)

# 自定义:
python scripts/prepare_ukdale.py --building 1 --appliance "dish washer" \
    --n-rows 500 --out input/dishwasher_500.csv
```

### 2.2 用自己的数据

`extract` 步骤接受 CSV（`timestamp,power` 或 `datetime,power` 两列）/
`.npy`（N×2）/ 文本（两列），通过 `--raw-series` 或 config 的
`paths.raw_series` 指定。

## 3. 跑通流水线

### 3.1 小切片冒烟测试（本教程示例）

```bash
python main.py --config config/config_ukdale_test.yaml \
    --appliance washing_machine --run-id ukdale_wm_test \
    --steps extract,segment,feature,cluster \
    --raw-series input/ukdale_washing_machine_100.csv \
    --segment-method fluss --feature-model dtw \
    --cluster-method kmeans --n-clusters 2
```

预期输出：

```
extract:  100 行 → 1 个活动段 CSV
segment:  fluss 在 @34/@58 检出变化点 → 3 个基元 (tensor: 3 samples)
feature:  dtw → features (3, 3)
cluster:  kmeans k=2 → 2 clusters (2/1) → result 'kmeans_k2'
```

> 为什么小切片用 fluss 而不是默认 clasp：clasp 分支用 `window_size="suss"`
> 自动估窗，100 点的序列上估出 46，`2*min_seg_size > 100` 会强制单段。
> 真实规模数据（几千点以上）用 clasp 没问题。

### 3.2 真实规模的完整流水线

```bash
python main.py --appliance fridge \
    --steps extract,segment,feature,cluster,fewshot,pam,split \
    --segment-method clasp --feature-model detsec \
    --cluster-method kmeans --n-clusters 3,4,5,6
```

七步流水线：

| 步骤 | 干什么 | 关键产物 |
|---|---|---|
| extract | 原始功率 → 活动段 CSV | `segments/` |
| segment | 活动段 → 基元张量 | `X.npy / lengths.npy / indices.npy` |
| feature | 基元 → 潜特征（**唯一缓存的步骤**） | `features.npy` |
| cluster | 聚类，**每个候选 k 一个结果** | `kmeans_k{k}/cluster_labels.npy + metrics.json` |
| fewshot | 识别少样本簇并导出 | `few_shot_cluster_summary.json` |
| pam | 基元↔活动映射，划分少样本活动 | 9 个 JSON/npy |
| split | 生成 train/test_a/test_b | 三套 branch/mains/mask |

### 3.3 窄范围重跑（manifest 复用）

同一 `--run-id` 重跑部分步骤，上游产物自动从 manifest 解析：

```bash
# 只换聚类方法重跑（特征直接复用）
python main.py --run-id ukdale_wm_test --steps cluster \
    --cluster-method dbscan --config config/config_ukdale_test.yaml

# 只重跑特征 → 观察缓存命中
python main.py --run-id ukdale_wm_test --steps feature \
    --segment-method fluss --feature-model dtw \
    --config config/config_ukdale_test.yaml
# [feature_extract] cache HIT (2a712f314299...) — skip training
```

特征缓存是纯内容寻址的：`SHA256(X 内容 | lengths | 模型名 | 全部超参 | 模型源码指纹)`，
与 run-id、目录无关。上游（数据集/切分方法/切分超参）的任何变化都会改变 X
从而改变 key；模型源码改动同样使缓存失效。

## 4. 产物结构

```
log/<run_id>/
  run_manifest.json                 # 产物路径的唯一事实来源
  extract_active_data_simple/segments/*.csv
  time_segmentation_fluss/{X,lengths,indices}.npy
  FeatureExtract_dtw_on_fluss/features.npy
  TimeClustering_kmeans_on_dtw_on_fluss/
    feature_matrix.npy seq_len.npy kept_rows.npy   # 共享，只存一份
    kmeans_k2/{cluster_labels,indices}.npy metrics.json
```

下游和出图都通过 manifest 按 `(step_type, key)` 解析路径，不要自己拼目录。

## 5. 图表生成

出图与流水线完全解耦，只读 manifest + config 的 `visualization:` 块，
统一输出到 `output/<run_id>/figure/<kind>/`：

```bash
python -m visualize.visualize_segments      --run-id ukdale_wm_test
python -m visualize.visualize_separation    --run-id ukdale_wm_test
python -m visualize.visualize_clustering    --run-id ukdale_wm_test
python -m visualize.visualize_cluster_reconstruction --run-id ukdale_wm_test
python -m visualize.visualize_feature_history        --run-id ukdale_wm_test
```

生成内容：

| 脚本 | 图表 |
|---|---|
| segments | 每个活动段的功率曲线（含切分点标注） |
| separation | 原始信号 vs 各通道重建 |
| clustering | 每簇 item 图 + 簇中心 / stacked / t-SNE（默认对**所有**聚类 tag 出图，`--cluster-tag kmeans_k2` 可指定） |
| cluster_reconstruction | 原始信号上的基元区间 + 簇标签着色 |
| feature_history | 特征模型训练损失曲线 |

## 6. 集群提交（slurm）

纯循环脚本，变体选择全走 CLI 参数，不改 config：

```bash
sbatch slurm/run_pipeline.sh                          # 单次完整流水线
sbatch slurm/run_segmentation_grid.sh                 # 切分方法网格
sbatch slurm/run_feature_models.sh                    # 特征模型网格（缓存去重）
sbatch --export=ALL,RUN_ID=<id> slurm/run_cluster_grid.sh   # 聚类网格

# 覆盖变量:
sbatch --export=ALL,APPLIANCE=kettle,FEATURE_MODEL=bilstm_ae slurm/run_pipeline.sh
```

提交前把 `slurm/env.sh` 里的 `PROJECT_DIR` 和 conda 环境名改成你的。

## 7. 常见问题

**Q: 聚类报 `k=N needs k < n_samples`？**
基元数太少（小切片正常）。减少 `--n-clusters`，或换更大的数据切片。

**Q: 改了切分方法/超参，特征会重训吗？**
会。X 内容变了 → 缓存 key 变 → 自动重训。同一输入重复跑则 cache HIT。

**Q: kmeans-scan 是什么？**
可选诊断：`--cluster-method kmeans-scan` 只输出每个 k 的 SCI/DBI/CHI 和
max-SCI 推荐值（`kmeans_scan.json` + 三联图），**不产生**聚类结果。
正式流程直接用 kmeans 全候选 k。

**Q: 多个聚类结果，下游用哪个？**
`--cluster-tag kmeans_k4`。缺省时：只有一个 tag 自动用，多个则报错并列出可选。

**Q: 哪些东西不进 git？**
见 `.gitignore`：`input/`、`datasets/`、`log/`、`output/`、`.cache/`、
slurm 日志、`__pycache__`。代码、config、models、tests、docs 都进。
