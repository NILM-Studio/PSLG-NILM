# clasp vs clasp-origin 切分效果对比分析报告

- **日期**：2026-08-08
- **对比对象**：`clasp`（BinaryClaSP + db4 小波 + 中值滤波） vs `clasp-origin`（原始完整版 ClaSP）
- **数据集**：ECO / GREEND / IAWE / REDD / REFIT / UK-DALE（washing_machine）
- **任务名**：`20260808_clasp-clasp_<数据集>`（clasp）、`20260808_clasp-origin_<数据集>`（clasp-origin）
- **产出**：每数据集每模型各 10 张切分图（`output/<run_id>/figure/segments/`）

## 1. 算法差异（原因分析基础）

| 环节 | `clasp` | `clasp-origin` |
|---|---|---|
| 信号预处理 | `medfilt(5)` 中值滤波去噪 | 原始信号直接用 |
| ClaSP 调用次数 | **每 cycle 3 次**（原始 + db4 2 层小波分解的低频/高频分量各一次） | 每 cycle 1 次 |
| 小波变换 | db4、level=2，分解 low/high 频段后各自切分 | 无 |
| 变点合成 | `synthesize_changepoints`：以较多变点序列为参考，将对方变点归并到最近邻并求均值 | 无（直接输出） |
| 核心参数 | `BinaryClaSP(window="suss", n_segments="learn", validation="score_threshold", threshold=0.001, distance="znormed_euclidean", n_jobs=-1)` | 同参数但 `n_jobs=1` |

> 两者共用同一个 `BinaryClaSP` 内核，差异完全来自：①中值滤波；②3 次 ClaSP 多尺度探测；③最近邻变点合成。

## 2. 任务明细

| 模型 | 任务名 | 状态 | 图片 |
|---|---|---|---|
| clasp | 20260808_clasp-clasp_{eco,greend,iawe,redd,refit,ukdale} | 全部成功 | 各 10 |
| clasp-origin | 20260808_clasp-origin_{eco,greend,iawe,redd,refit,ukdale} | 全部成功 | 各 10 |

## 3. 切分粒度对比

| 数据集 | cycles | prims_clasp | prims_origin | Δ prims | cp_clasp | cp_origin |
|---|---|---|---|---|---|---|
| eco | 274 | 3,100 | 2,627 | **+473** | 2,826 | 2,353 |
| greend | 402 | 3,501 | 2,819 | **+682** | 3,099 | 2,417 |
| iawe | 27 | 75 | 72 | +3 | 48 | 45 |
| redd | 24 | 50 | 55 | -5 | 26 | 31 |
| refit | 507 | 563 | 559 | +4 | 56 | 52 |
| ukdale | 1,490 | 3,947 | 3,824 | +123 | 2,457 | 2,334 |

> 总体趋势：**clasp 更"细"**（除 redd 外基元普遍多于 clasp-origin），其中 eco/greend/ukdale 最明显；redd 上 clasp 反而更粗。

## 4. 指标对比

### 4.1 F1@k（变点级一致性，容忍 k = %cycle 长度）

| 数据集 | k=0.5% | k=1% | k=2% | k=5% |
|---|---|---|---|---|
| eco | 0.280 | 0.434 | 0.604 | 0.798 |
| greend | 0.062 | 0.117 | 0.210 | 0.358 |
| iawe | 0.064 | 0.079 | 0.182 | 0.363 |
| redd | 0.000 | 0.000 | 0.026 | 0.386 |
| refit | 0.081 | 0.176 | 0.241 | 0.542 |
| ukdale | 0.136 | 0.222 | 0.339 | 0.557 |

**结论**：两模型变点位置一致性普遍很低（F1@k ≤ 0.8 仅 eco@5%），redd 在 ≤1% 容忍度下**完全不一致（F1=0）**。它们"数出来"的变点数量接近，但**位置显著不同**。

### 4.2 Coverage（对真实活动段的覆盖）

| 数据集 | Cov_clasp | Cov_origin |
|---|---|---|
| 全部 6 个 | 1.00 | 1.00 |

**结论**：两个模型都把每个活动段完整划分为基元（boundaries=[0]+cps+[len]），对真实活动段覆盖率恒为 100%，**该指标无区分度**——这正是 NILM 中"分割段天然铺满活动段"的构造性结果，不随切分方法变化。

### 4.3 OSR / USR（以对方为参考的过/欠切分率，tol 1%）

| 数据集 | OSR_clasp | USR_clasp | OSR_origin | USR_origin | 最近邻变点距离 dnn_clasp | dnn_origin |
|---|---|---|---|---|---|---|
| eco | 0.296 | 0.023 | 0.031 | 0.179 | 97s | 66s |
| greend | **1.574** | 0.156 | 0.395 | 0.292 | 1,048s | 243s |
| iawe | 0.379 | 0.066 | 0.095 | 0.178 | 114s | 81s |
| redd | 0.000 | 0.237 | 0.474 | 0.000 | 36s | 53s |
| refit | 0.027 | 0.060 | 0.113 | 0.014 | 51s | 57s |
| ukdale | 0.102 | 0.046 | 0.085 | 0.059 | 49s | 47s |

**解读**（clasp vs clasp-origin）：
- **greend：clasp 严重过切分（OSR=1.574）** —— clasp 平均多检出约 1.57 倍于 origin 的变点；origin 相对更稳定。
- **redd：clasp 欠切分（USR=0.237）、origin 过切分（OSR=0.474）** —— 方向与 greend 相反。
- **eco/iawe**：clasp 轻度过切分。
- **refit/ukdale**：两模型接近平衡。

## 5. 原因分析

### 5.1 为什么 clasp 整体更细、greend 上严重过切分
`clasp` 对每个 cycle 做 **3 次 ClaSP**（原始 + 低频 + 高频分量）。db4 2 层小波把信号拆成平滑的低频趋势与振荡的高频细节，**低频/高频各自会单独探测出原始信号看不到的边界**，合成后变点更多 → 细粒度、倾向过切分。GREEND 周期长、幅度波动大（最长 cycle 28,541s），多尺度分解在长序列上产生大量子边界 → OSR 飙到 1.574。

### 5.2 为什么 redd 上 clasp 反而更粗
`clasp` 的 `synthesize_changepoints` 会把另一组的变点**归并到最近邻参考点**。REDD 采样率低（0.25Hz）、cycle 短（~500s），origin 检出的双边界（如 160s/320s）在 clasp 的小波+合成流程中被合并成单边界（~277s）→ 变点变少、变粗（USR=0.237）。低采样率下多尺度分解的分辨率不足以区分相邻子阶段。

### 5.3 为什么 F1@k 普遍很低
1. **预处理差异**：clasp 先 `medfilt(5)` 再切分，边界位置被平滑偏移；origin 直接在原始信号上找边界。
2. **小波合成**：clasp 的变点是"原始+低频+高频"三组最近邻合成的均值位置，与 origin 的原始变点位置系统性错位。
3. **探测逻辑**：多尺度探测发现的边界往往落在不同的物理位置上。
→ 数量相近、位置迥异，导致 F1@k 低。ECO（1Hz 干净周期、子阶段清晰）两模型收敛最好（F1@5%=0.798）；greend/redd 波动大、采样低，一致性最差。

### 5.4 为什么 Coverage 无区分度
两个模型的基元都完整铺满活动段（`[0]+cps+[len]` 分段），覆盖恒为 1.0。**Coverage 只反映"是否覆盖活动段"，不反映内部切分质量**——这与 NILM 标注粗糙时"分割覆盖活动段"的含义相符，但在方法间对比上没有鉴别力。

## 6. 总结与建议

- **细粒度/子阶段划分**：选 `clasp`（更适合 1Hz 的 ECO/GREEND 这类多阶段清洗周期），但需警惕过切分（尤其 GREEND，建议调低 `n_regimes` 或提高 `threshold`）。
- **粗粒度/稳定性**：选 `clasp-origin`（单次 ClaSP、边界更稳），适合低频（REDD/REFIT）。
- **变点一致性**：两模型在同一活动段上的变点位置差异大（F1@k 低），做跨方法融合/校验时需容忍较大偏移。
- **Coverage 恒为 1.0**：不应用作方法间区分指标；若需"覆盖"类指标，建议改为"基元是否覆盖活动段内 ≥N% 的显著功率区段"。

## 7. 复现

```bash
# clasp 运行
det command run --config-file determined/20260808_clasp-clasp_<数据集>.yaml --detach
# clasp-origin 运行
det command run --config-file determined/20260808_clasp-origin_<数据集>.yaml --detach
# 指标计算（PSLG-NILM conda env）
python /tmp/compare_clasp_vs_origin.py
```

> 注：`run_dataset_segment.sh` 已参数化 `$5 SEGMENT_METHOD`；`clasp` 需 `NUMBA_NUM_THREADS=$(nproc)`（已内置），否则 `n_jobs=-1` 与线程数冲突报 "The number of threads must be between 1 and 1"。
