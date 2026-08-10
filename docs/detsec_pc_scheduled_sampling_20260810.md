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

## 5. 使用方式

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
