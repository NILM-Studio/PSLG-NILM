# clasp-origin 切分效果对比分析报告

- **日期**：2026-08-08
- **切分模型**：`clasp-origin`（`models/time_segmentation/clasp_origin.py`，原始完整版 ClaSP，纯 CPU，n_jobs=1）
- **流水线**：`extract → segment`（`--segment-method clasp-origin`）
- **并行方式**：6 个数据集各自以独立 Determined command 任务并行运行
- **任务名**：`20260808_clasp-origin_<数据集>`

## 1. 任务明细

| 任务名 | 数据集 | 原始序列行数 | 阈值 | 采样率 fs | 状态 | 图片 |
|---|---|---|---|---|---|---|
| 20260808_clasp-origin_eco | ECO | 19,670,702 | 20W | 1.0 Hz | 成功 | 10 |
| 20260808_clasp-origin_greend | GREEND | 19,886,555 | 5W | 1.0 Hz | 成功 | 10 |
| 20260808_clasp-origin_iawe | IAWE | 23,740 | 5W | 1.0 Hz | 成功 | 10 |
| 20260808_clasp-origin_redd | REDD | 745,878 | 5W | 0.25 Hz | 成功 | 10 |
| 20260808_clasp-origin_refit | REFIT | 6,960,008 | 5W | 0.1429 Hz | 成功 | 10 |
| 20260808_clasp-origin_ukdale | UK-DALE | 19,555,935 | 10W | 0.1667 Hz | 成功 | 10 |

> 全部任务均完成，产物在 `log_det_test/20260808_clasp-origin_<数据集>/`，图片在 `output/20260808_clasp-origin_<数据集>/figure/segments/`。

## 2. 切分结果对比

| 数据集 | 活动段 cycles | 切分基元 primitives | prim/cycle | 基元时长均值 | 中位数 | 标准差 | 最大时长 | 内部变点数 | 变点/cycle | cycles≥2变点 |
|---|---|---|---|---|---|---|---|---|---|---|
| eco | 274 | 2,627 | 9.59 | 528s | 443s | 302s | 2,839s | 2,353 | 8.59 | 266 (97%) |
| greend | 402 | 2,819 | 7.01 | 727s | 333s | 1582s | 28,541s | 2,417 | 6.01 | 285 (71%) |
| iawe | 27 | 72 | 2.67 | 390s | 328s | 177s | 1,022s | 45 | 1.67 | 10 (37%) |
| redd | 24 | 55 | 2.29 | 208s | 187s | 80s | 547s | 31 | 1.29 | 12 (50%) |
| refit | 507 | 559 | 1.10 | 283s | 311s | 108s | 1,648s | 52 | 0.10 | 7 (1.4%) |
| ukdale | 1,490 | 3,824 | 2.57 | 285s | 268s | 94s | 829s | 2,334 | 1.57 | 833 (56%) |

基元时长分位数：

| 数据集 | p10 | p25 | p50 | p75 | p90 |
|---|---|---|---|---|---|
| eco | 335s | 364s | 443s | 567s | 682s |
| greend | 255s | 268s | 333s | 455s | 1,115s |
| iawe | 234s | 284s | 328s | 458s | 546s |
| redd | 140s | 155s | 187s | 252s | 283s |
| refit | 147s | 234s | 311s | 334s | 357s |
| ukdale | 190s | 231s | 268s | 325s | 398s |

## 3. 各数据集切分效果分析

### ECO —— 切分最细
274 个活动段被切成 2,627 个基元（9.59 prim/cycle），97% 的 cycle 含 ≥2 个内部变点。clasp-origin 把每个洗衣机周期细分为 8 个左右的子阶段（进水/加热/洗涤/漂洗/脱水等），粒度最细、最结构化。基元时长分布集中在 335–682s。

### GREEND —— 长周期+重尾
402 个 cycle → 2,819 基元（7.01 prim/cycle）。均值 727s 远大于中位数 333s，标准差高达 1,582s，最大基元 28,541s（约 8 小时）——说明存在极长的"工作段"未被切分干净，重尾明显。clasp-origin 对超长段的分辨率有限，尾部大段值得人工复核。

### IAWE —— 小样本、中等粒度
数据量最小（2.4 万行），仅 27 个 cycle → 72 基元（2.67 prim/cycle）。37% 的 cycle 有 ≥2 变点。粒度适中，基元时长 234–546s，分布较均匀。因样本太少，统计意义有限。

### REDD —— 采样率低、基元最短
0.25 Hz 低采样率下基元最短（均值 208s，p90 仅 283s）。24 个 cycle → 55 基元，50% 的 cycle 被细分。粒度偏细但与物理子阶段匹配尚可。

### REFIT —— 几乎不细分
507 个 cycle → 仅 559 基元（1.10 prim/cycle），仅 1.4% 的 cycle 有内部变点。clasp-origin 在 REFIT（0.143 Hz 极低采样率）下几乎把每个活动段当作一个整体基元，切分最粗糙。基元时长 147–357s，分布窄。

### UK-DALE —— 数据量最大、粒度适中
1,490 个 cycle（最多）→ 3,824 基元（2.57 prim/cycle），56% 的 cycle 被细分。基元时长 190–398s。粒度介于 ECO/REFIT 之间，是数据量最大的数据集，适合作为下游特征/聚类主数据源。

## 4. 关键发现

1. **细粒度排序**：ECO ≫ GREEND > UK-DALE > IAWE > REDD ≫ REFIT。ECO 与 REFIT 分别代表"过度细分"和"几乎不细分"两个极端。
2. **采样率影响显著**：REFIT（0.14 Hz）与 REDD（0.25 Hz）等低频数据基元数量骤减，clasp-origin 在低频下难以识别子阶段边界；ECO/GREEND（1 Hz）切分最细。
3. **活动段提取阈值影响**：ECO 用 20W 阈值排除了待机噪声，只保留强信号段，利于细分；GREEND/IAWE/REDD/REFIT 用 5W 阈值捕获更多弱信号，其中 GREEND 引入超长段。
4. **重尾问题**：GREEND 基元最大 28,541s，需在特征提取/聚类前考虑按最大长度截断或过滤，否则会拉偏聚类中心。
5. **确定性**：clasp-origin 完全确定——同一数据集两次独立运行（`20260808_clasp_*` 与 `20260808_clasp-origin_*`）产物逐字节一致（仅 run_id 不同）。

## 5. 输出与复现

```bash
# 图片
output/20260808_clasp-origin_<数据集>/figure/segments/segment_*.png   # 每数据集 10 张

# 切分产物（X 特征矩阵 / lengths / indices 变点索引）
log_det_test/20260808_clasp-origin_<数据集>/
  extract_active_data_simple/segments/*.csv
  time_segmentation_clasp-origin/{X,lengths,indices}.npy
  run_manifest.json

# 复现（并行提交）
det command run --config-file determined/20260808_clasp-origin_<数据集>.yaml --detach
```

## 6. 建议

- **下游聚类主选**：ECO 与 UK-DALE（细粒度+数据量大），特征维度高、可区分度好。
- **REFIT**：需调低 `n_regimes` / 调大 `window_size` 或改用 `clasp`（小波增强版）以提升细分能力。
- **GREEND**：对超长基元（>2h）做长度过滤，避免拖尾。
- **低采样率数据集**（REFIT/REDD）建议单独评估切分参数，勿与 1Hz 数据集共用默认窗口（window=30）。
