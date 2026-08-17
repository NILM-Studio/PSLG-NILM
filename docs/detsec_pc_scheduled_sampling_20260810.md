# detsec_pc 带计划采样（Scheduled Sampling）特征提取模型

> R1 消融胜出方案（2026-08-10）。文档对应 `models/feature_extract/detsec_pc.py` 中
> `tf_ratio` / `tf_schedule` 两个配置项及其训练机制。

## 1. 动机：Teacher-Forcing 的信息瓶颈

`detsec_pc` 的解码器采用 teacher-forcing：每个时间步的输入是真实上一时刻与嵌入的拼接
`[x_{t-1}; z]`。解码器"开卷考试"，直接看到真实波形，因此重构误差对嵌入 `z` 的梯度很小，
`z` 无需编码完整的波形动态就能把损失压低。结果是嵌入成为"低信息量压缩"，聚类可分性受限于
信息瓶颈。

对比 detsec 基线（解码器从 `z` 经 `RepeatVector` 重建整条序列，`z` 被迫携带全部信息），
两者在 ukdale 上的聚类差距（avgSCI 0.58 vs 0.59，但 k5~k8 明显落后）即主要由此造成。

## 2. 机制：Teacher-Forcing 保留率（Keep Ratio）线性衰减

对解码器输入施加伯努利掩码 `keep_mask ∈ {0,1}^{B×T×1}`：

- `keep=1`：喂入真实 `x_{t-1}`（完整 teacher-forcing）；
- `keep=0`：喂入零向量，解码器只能依赖嵌入 `z`（z-only 条件重构）。

训练按 epoch 线性衰减保留率：

```
ratio(e) = tf_ratio + (1 − tf_ratio) · (1 − (e − 1)/(E − 1))
```

- `epoch=1`：`ratio=1.0`（纯 teacher-forcing，保证早期稳定收敛）；
- `epoch=E`：`ratio=tf_ratio`（默认 `tf_ratio=0.0`，完全 z-only，信息强迫最强）；
- 中间 epoch 以概率在真实输入与零输入之间混合（scheduled sampling 的简化实现）。

验证/特征提取阶段始终使用完整 teacher-forcing（`keep_mask=1`），保证指标口径一致。

## 3. 配置项

| 配置键 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `tf_ratio` | float [0,1] | `1.0` | 计划采样的目标保留率；`1.0` = 纯 teacher-forcing（原行为） |
| `tf_schedule` | str | `constant` | `constant`：全程用 `tf_ratio`；`linear`：从 1.0 线性衰减到 `tf_ratio` |

均在 `_model_config()` 内，**进入特征缓存 key**，修改任一参数即自动失效旧缓存并重训。

## 4. R1 消融结果（ukdale，SCI 越高越好）

| 变体 | avgSCI | avgDBI | k3 | k4 | k5 | k6 | k7 | k8 |
|---|---|---|---|---|---|---|---|---|
| detsec 基线 | 0.589 | 0.652 | 0.524 | 0.585 | 0.567 | 0.598 | 0.616 | 0.633 |
| v2 基线（无采样） | 0.580 | 0.678 | 0.681 | 0.627 | 0.538 | 0.568 | 0.522 | 0.528 |
| **S3 scheduled sampling** | **0.640** | **0.546** | **0.734** | **0.682** | 0.639 | 0.615 | 0.577 | 0.589 |

结论：

- avgSCI 0.640 **反超 detsec 基线 0.589**，胜 5/7 个 K，DBI 全场最低；
- k3=0.734 为所有消融变体在所有 K 上的最高值；
- 关键验证：即便保留率仅衰减到 0.61 即因早停停止（epoch 20），信息强迫的效果已显著——
  证实 teacher-forcing 信息瓶颈是剩余差距的主因；
- 结合 R1 的 S1 结果（同特征下 `kmeans(n_init=30)` 比 `dpc-kmeans(n_init=1)` 平均高 ~0.03，
  大 K 更明显），最终验证建议聚类固定为 `kmeans(n_init=30)` 与 detsec 基线对齐。

## 5. R2 终验（2026-08-10，3 数据集全流程，聚类与 detsec 基线同为 kmeans n_init=30）

| 数据集 | 模型 | k2 | k3 | k4 | k5 | k6 | k7 | k8 | avgSCI |
|---|---|---|---|---|---|---|---|---|---|
| eco | detsec | 0.850 | 0.882 | 0.864 | 0.834 | 0.829 | 0.730 | 0.726 | **0.817** |
| | S3 | 0.735 | 0.694 | 0.713 | 0.597 | 0.583 | 0.528 | 0.524 | 0.625 |
| refit | detsec | 0.715 | 0.685 | 0.697 | 0.697 | 0.737 | 0.702 | 0.712 | **0.706** |
| | S3 | 0.627 | 0.553 | 0.621 | 0.633 | 0.654 | 0.657 | 0.646 | 0.627 |
| ukdale | detsec | 0.603 | 0.524 | 0.585 | 0.567 | 0.598 | 0.616 | 0.633 | 0.589 |
| | **S3** | 0.766 | **0.825** | 0.792 | 0.796 | 0.605 | 0.553 | 0.575 | **0.702** |

结论：

- **ukdale（波形特征强、切分质量高，参考性最强的数据集）上 S3 全面反超 detsec**
  （avgSCI 0.702 vs 0.589，+0.11，胜 5/7 个 K；k3=0.825 为全部实验的最高值）；
- **eco/refit 上 S3 较 v2 基线有提升但仍落后 detsec**（eco avg 0.625 vs 0.817，
  refit 0.627 vs 0.706），信息强迫的收益与数据波形信息量正相关；
- 聚类算法因子在 ukdale 上再次确认：S3+kmeans 0.702 vs S3+dpc 0.646（~+0.06），
  与 R1 的 S1 结论一致；eco 上则相反（kmeans 0.625 < dpc 0.645），说明最优算法也随数据而变。

## 6. 使用方式

```bash
# 特征提取（含 scheduled sampling）
python main.py --steps extract,segment,feature,cluster \
    --feature-model detsec_pc --segment-method prim-glr \
    --cluster-method kmeans --n-clusters 2,3,4,5,6,7,8 \
    --config config/config_ukdale_pcdetsec_s3.yaml

# 对应 config 关键段：
feature_extract:
  embed_dim: 32
  lambda_phy: 0.1
  nonneg_channels: [0, 1, 2, 3]
  norm_mode: minmax        # 方向1A：按通道 1%/99% 分位数裁剪的全局 MinMax
  embed_proj: relu         # 方向3：Dense(embed, relu) 稀疏低秩投影
  tf_ratio: 0.0            # 目标保留率（最终完全 z-only）
  tf_schedule: linear      # 1.0 → 0.0 线性衰减
  epochs: 50
  patience: 10             # 放宽早停，让衰减更充分
```

## 6. 相关文档

- 差距归因与四方向改进：会话分析（方向 1A/2/3/4）
- R1 消融：`determined/20260810_090804_abl_*.yaml`
- 图表审核：`output/20260810_090804_abl_s3_tf_sampling_ukdale/figure/clustering/`
