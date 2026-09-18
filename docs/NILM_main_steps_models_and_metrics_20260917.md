# PSLG-NILM 主要步骤：模型原理、对比方案与评价指标

核查日期：2026-09-17。分析对象：通过 `ssh scnu202438025446@scnu-2024` 访问的 `/home/scnu202438025446/pslg-nilm`。

本文以远程当前代码、配置、manifest 和已有实验指标为依据，没有重新训练模型。远程 HEAD 为 `e9e9e27ff7a71f5b725c2e6bec42b59821f473b3`，但工作区有未提交修改，因此本文描述的是核查时的工作区，不是该提交的纯净版本。主要范围是 `main.py` 的 `segment → feature → cluster`，并补充上下游指标。

## 1. 先说明：项目目前选用了什么，“最佳”成立到什么范围

当前正式配置 `config/config_nilm_primitive_formal.yaml` 选择：

| 环节 | 当前主选 | 解决的问题 | 现有证据能支持的结论 |
| --- | --- | --- | --- |
| 切分 `segment` | PrimGLR，CLI 名 `prim-glr` | 找到活动内部功率形态的变化边界 | 已作为主线使用；有历史切分统计，但未见覆盖全部候选方法、统一真实边界标注的最优性证明 |
| 特征 `feature` | DeTSEC-PC 的 S3 变体，CLI 名仍为 `detsec_pc` | 把不同长度片段编码为 32 维向量 | UK-DALE 的历史对比中胜过 DeTSEC；ECO、REFIT 上并非最好 |
| 聚类 `cluster` | 特征 z-score + K-means，`n_init=30` | 把片段分成可复用的形态类型 | UK-DALE 的 S3 特征上有优势；不同数据集可能偏好 DPC 初始化 |
| 簇数 | 现有配置为 `K=5`；本次讨论推荐 UK-DALE 采用 `K=3` | 确定字典粒度 | K=3 的 SCI、DBI 均优于 K=5，类别更少；历史综合排名的等权假设及简化密度指标不足以使 K=5 成为确定最优 |
| 后处理 | `state_merge`，相邻同类合并、短块吸收、相似块合并 | 将片段类型还原为连续状态块 | 是时序规则后处理，不是新的聚类网络 |

所以准确表述应是：**“当前采用 PrimGLR + DeTSEC-PC/S3 + z-score K-means；这是基于已有实验确定的主线，其中 S3 的优势具有数据集条件。”** 不能写成“三个阶段均已证明在所有数据集上最优”。

根据本次讨论，**UK-DALE 的后续推荐簇数为 K=3**，理由见第 5.2 节。本次只整理分析文档，现有运行配置、K=5 字典及实验产物未修改；“现有配置”与“本次推荐”需分别理解。

另一个容易混淆的地方：通用 `config/config.yaml` 没有完整设置当前主选变体，`main.py` 的回退值仍为 `clasp / detsec / kmeans`。复现主线必须读取明确的正式配置或传入 CLI 参数，不能只执行裸命令。

## 2. 数据如何经过主要 steps

```text
目标电器分表功率
  → extract：活动区间
  → segment：活动内部变点、片段张量 X、有效长度 lengths、位置 indices
  → feature：每个片段的固定维度向量 features
  → cluster：每个候选 K 的簇标签和内部评价指标
  → state_merge：活动内连续状态块
  → state_sequence / nilm_labels：序列与逐点辅助监督
  → NILM 训练：部署输入为总表，状态来自模型预测
```

活动提取与内部切分是两件事。正式洗衣机配置先用 `simple` 活动检测，功率阈值 20 W、跌落容忍 150 s、最短工作时间 180 s、上下文 90 s、约 6 s 采样；再在活动内运行 PrimGLR。不能把超过 20 W 的每次瞬时变化都等同于一个新程序阶段。

`segment` 保存 `X.shape=(N,T_max,4)`，短片段补零，同时保存真实长度和原始位置。特别注意：当前 **PrimGLR 分支的四通道实际是 `[power, power, power, zeros]`**，不是四种独立传感器，也不是四种独立物理测量。只有增强版 `clasp` 分支使用原信号、滤波信号、小波低频和小波高频。

基元类别是自动发现的伪标签，不能直接命名为“加热、排水、脱水”等真实功能状态；这类命名需要独立语义标注验证。

## 3. 切分主模型：PrimGLR

来源：`models/time_segmentation/prim_glr.py` 的 `PrimGLRModel`，由 `src/steps/time_segmentation.py` 调用。

### 3.1 原理：先找变化候选，再利用结构与长度约束去掉多余边界

**第一步：构造局部统计通道。** 对一维功率序列构造：

\[
X_t=[p_t,\ p_t-p_{t-1},\ p_t-\operatorname{median}_W(p),\ \operatorname{std}_W(p)].
\]

四项分别描述功率水平、瞬时跳变、偏离局部基线的程度和局部波动。这是 PrimGLR 内部的检测特征，与输出给神经网络的四通道张量不是同一组特征。

**第二步：计算左右窗口的变化得分。** 对位置 t 左右各 W 个点，代码使用对角近似的 GLR 风格统计量：

\[
S(t)=\sum_f\frac{(\mu_{L,f}-\mu_{R,f})^2}
{\sigma_{L,f}^2/W+\sigma_{R,f}^2/W+10^{-12}}.
\]

左右均值差越大、相对背景波动越明显，得分越高。它不是监督分类器；代码也没有从预设显著性水平推导严格检验阈值，而是使用得分分位数。

**第三步：筛选候选边界。** 对正得分取 `alpha=0.80` 分位数作为阈值，按得分从高到低选择候选点，用 `min_gap=25` 点进行非极大值抑制，再在 ±2 点内寻找局部峰值。

**第四步：描述候选片段并做局部初始聚类。** 片段描述包含 6 个功率统计量、16 维分段均值近似 PAA、4 个高低电平特征和 8 维 PAA，共 34 维。所谓 envelope PAA 在当前代码中仍然是对该功率片段执行 PAA，没有独立的包络提取器。

当前 `train()` 路径使用 K-means，状态数按候选片段数 M 自适应为 `min(6,max(2,M//2))`，再标准化描述向量聚类。这是**单个活动内部、用于优化边界的临时聚类**，不等于下游跨片段建立 K=5 字典的 `cluster`。

**第五步：用持续时间先验重新赋予片段状态。** 代码结合到状态中心的方差加权距离、对数正态形式的长度先验，以及保持相同状态的 +2 转移奖励，执行 Viterbi 风格动态规划：

\[
\ell_{dur}(L,c)=-\frac{(\log L-\mu_{\log L,c})^2}{2\sigma_{dur}^2}-\log L.
\]

最后删除相邻同状态片段之间的边界，保留状态变化处。直观上，它避免把同一种持续工作形态切成太多碎片。

### 3.2 实现边界与参数

| 参数/机制 | 实际行为 |
| --- | --- |
| W | 模型默认 50 点，短序列用 `min(50,max(3,n//10))` |
| alpha | 0.80，候选检测的分位数阈值 |
| min_gap | 25 点；约 6 s 采样下相当于约 150 s，换采样率时物理尺度会变化 |
| dur_sigma | 0.35，持续时间先验宽度 |
| GMM/BIC | 辅助函数有实现，但当前 `train()` 已先确定 K，实际走 K-means，不走 BIC 自动选 K |
| HSMM | 是固定候选片段上的带持续时间先验动态规划，不是枚举任意持续时间和边界的完整 HSMM 学习 |
| `time_segmentation.window_size: 30` | 主 step 对 PrimGLR 调用 `PrimGLRModel()`，未把此值传入；不能把它当作 PrimGLR 的 W |
| `max_seg_len: 1536` | step 对过长片段截掉尾部，不是继续细分；可能损失尾部覆盖，分析时需核查 lengths 与原活动范围 |

### 3.3 切分对比方法

| 方法 | 原理与项目中的实际差别 | 当前核查到的状态 |
| --- | --- | --- |
| `clasp-origin` | ClaSP 将候选切点两边的子序列看成两类，利用近邻分类可分性生成分类得分剖面，再递归选边界；不加小波。主 step 显式传入普通欧氏距离，保留幅值差异 | 有多数据集历史运行 |
| `clasp` | 中值滤波后做 db4 二层小波分解，对低频、高频分别执行 ClaSP，再把邻近边界分组求均值；使用 z-normalized Euclidean 距离，偏重形状 | 有历史运行；融合函数接收但没有使用原始信号变点 `orig_cp` |
| `fluss` | Matrix Profile 找相似子序列，依据近邻连接的校正弧线曲线 CAC 谷值寻找状态分界；需要 `window_size/n_regimes/excl_factor` | 代码集成，且有小型冒烟运行；不是当前正式主选 |
| `espresso` | 设计上调用外部 TSSB 的 `ESPRESSO.fit_predict()` | 当前只查到适配器，未找到预期的 `tssb_repo` 实现；不能据此声称已成功参与完整实验，也不能把未核实的论文机制当成项目实现 |
| `none` | 每个活动保留为一个片段，不检测内部变点 | 代码支持，可用作“去掉切分”的对照 |

历史 manifest 中的部分统计如下。这些只说明切分粒度和计算规模，不说明边界准确率；不同运行的采样率、活动阈值和输入还需对齐后才能做严格排名。

| 数据集 | ClaSP-origin 片段数 | PrimGLR 片段数 |
| --- | ---: | ---: |
| ECO | 2,627 | 2,087 |
| REFIT | 559 | 1,717 |
| UK-DALE | 3,824 | 9,624 |

来源分别为 `log_det_test/20260808_clasp-origin_<dataset>/run_manifest.json` 和 `log_det_test/20260809_055315_prim-glr_<dataset>/run_manifest.json`。切得更多或更少都不能单独证明更好。

## 4. 特征主模型：DeTSEC-PC / S3

来源：`models/feature_extract/detsec_pc.py`。S3 是配置变体，不是另一个 CLI 模型名。其目标是无监督学习片段表示，训练时没有真实功能类别，也没有在这里进行端到端聚类优化。

### 4.1 结构：双向 GRU、掩码注意力和门控融合

输入先按通道、在真实有效点上计算全局 1%/99% 分位数，裁剪后归一化到 [0,1]。相比逐片段 z-score，这种方式保留了片段之间的功率水平差异，但会压缩极端尖峰。

双向 GRU 分别提取正向、反向时序表示，各自通过时间注意力汇聚：

\[
\alpha_t=\operatorname{softmax}_t(u^\top\tanh(Wh_t+b)),\qquad
\bar h=\sum_t\alpha_t h_t.
\]

padding 位置的注意力分数设为极小值，不直接参与汇聚。正、反向向量再经两个 sigmoid 门控加权相加，最后通过 `Dense(32, relu)` 得到嵌入 z。

注意：代码给注意力和损失加了掩码，但 `self.encoder(x)` 没有传入循环层 mask。因此不能宣称“完全消除了 padding 对表示的影响”或“完全消除了长度信息”；反向 GRU 仍可能受补零影响。ReLU 投影也不等于显式低秩约束或稀疏惩罚。

### 4.2 训练目标：双向重构与平滑先验

前、后向各有一个 GRU 解码器。每步接收 `[上一真实采样值; z]`，初始隐状态也由 z 提供，分别重构原序列和有效区域反转后的序列。

对指定通道使用 Softplus 输出：

\[
\hat x=\log(1+e^a)\ge0.
\]

重构误差只在有效位置计算，先按每个片段真实长度归一化，再对样本平均，正反两个方向相加：

\[
L_{ae}=\frac1N\sum_i\frac1{T_i}\sum_{t\le T_i,f}
(x_{itf}-\hat x^{fw}_{itf})^2+L_{ae}^{bw}.
\]

物理先验采用 Charbonnier TV，相邻有效点的重构差分受到近似 L1 惩罚：

\[
L_{phy}^{fw}=\frac1N\sum_i\frac1{\max(T_i-1,1)}
\sum_{t=2}^{T_i}\sum_{f\in C_+}
\sqrt{(\hat x_{itf}-\hat x_{i,t-1,f})^2+10^{-6}},
\quad L=L_{ae}+\lambda_{phy}(L_{phy}^{fw}+L_{phy}^{bw}).
\]

正式配置 `lambda_phy=0.1`。这种先验鼓励平台平稳，对大跳变的惩罚约线性增长；它仍然惩罚边缘，不能描述成“边缘不受惩罚”。这里的“物理约束”具体是非负输出和保边平滑，没有实现电路方程、功率守恒或真实功能状态约束。

### 4.3 S3 改进：逐渐遮掉解码器可见的上一真实值

完整 teacher forcing 允许解码器依靠相邻真实输入重构，z 可能不需要携带足够信息。S3 用伯努利掩码，以概率 r 保留上一真实值，否则将它置零：

\[
r(e)=r_{target}+(1-r_{target})\left(1-\frac{e-1}{E-1}\right).
\]

当前 `tf_schedule=linear`、`tf_ratio=0`，计划从 1 线性降到 0，使训练越来越依赖 z。严格说这是 **scheduled-sampling 风格的真实输入遮蔽**：置零，不是用上一时刻模型预测替代真实值。

验证默认使用完整 teacher forcing；严格验证路径还记录 `validation_z_only_loss`。特征本身来自编码器，其计算不依赖解码器的 keep mask。早停可能发生在 r 降到 0 之前，故 `tf_ratio=0` 表示计划终点，不代表实际训练最后一轮已完全 z-only。

主要超参：`embed_dim=32`、`norm_mode=minmax`、`embed_proj=relu`、50 epochs、batch size 8、学习率 1e-4、patience 10。`latent_dim=16` 虽仍出现在配置中，但此模型的输出维数取 `embed_dim`。

### 4.4 其它特征对比模型

下表区分“代码支持的候选”和“已经有直接可核对的主线比较”。仅存在模型文件不能证明该模型参加了相同数据和预算的完整对比。

| CLI 模型 | 原理 | 对照意义 / 证据范围 |
| --- | --- | --- |
| `detsec` | 双向 GRU + 双向时间注意力 + 门控融合 + ReLU 嵌入；将 z RepeatVector 后用双向 GRU 重构，线性输出、MSE | 最主要历史基线；无 PC 的非负解码与 TV；主线历史嵌入为 16 维 |
| `detsec_pc` 原始版 | 掩码注意力、双向非负 teacher-forcing 解码、长度归一化重构与 TV；早期采用逐片段 z-score | 检验物理先验和变长处理；与 DeTSEC 同时改了归一化、维度、解码器，不能把差异都归因于单一约束 |
| `detsec_pc` v2 | 改为全局分位数 MinMax，并加入 ReLU 嵌入投影，仍完整 teacher forcing | 是 S3 消融的直接基线 |
| `lstm_ae` | 单向 LSTM 编码，压缩到潜变量，再重复潜变量并重构序列 | 基础时序自编码器，代码支持 |
| `bilstm_ae` | 双向 LSTM 编码，融合前后文后压缩，再用循环解码器重构 | 检验双向上下文，存在历史测试运行 |
| `bilstm_ae_attention` | 双向 LSTM + 两方向注意力和门控融合 + 自编码重构 | 注意力与循环单元变体，代码支持 |
| `cnn_ae` | Conv1D、池化、全局平均池化、Dense 嵌入；上采样卷积解码 | 强调局部波形，代码支持 |
| `autoencoder` | Flatten 后全连接 128→64→潜变量，再对称重构 | 不显式利用时序结构，存在历史测试运行；其损失没有 lengths 掩码 |
| `dtw` | 选前 `min(latent_dim,N)` 个样本作原型，以到各原型的动态时间规整距离构成向量 | 非神经基线；不是“训练 DTW 网络”，也不是直接把 DTW 当作 K-means 的距离 |

DeTSEC-PC 的历史消融还包括 S1 更换聚类初始化、S2 使用 `softplus_offset`、S3 遮蔽 teacher forcing、S4a 将 TV 权重降到 0.01、S4b 将嵌入降到 16 维。其中 `softplus_offset(a)=softplus(a)-log(2)` 在 a<0 时为负，不能继续称为严格非负输出。

### 4.5 “S3 最佳”的实验依据

以下均从对应 run 的各 K `metrics.json` 重新读取并计算均值，K=2…8，共 7 个候选值；均值是跨 K 的算术平均，不是跨随机种子的均值。

UK-DALE R1 消融：

| 变体 | 聚类 | 平均 SCI ↑ | 平均 DBI ↓ |
| --- | --- | ---: | ---: |
| v2 | DPC-KMeans | 0.579582 | 0.677696 |
| S1：复用 v2 特征改聚类 | K-means | 0.608623 | 0.603587 |
| S2：Softplus offset | DPC-KMeans | 0.582235 | 0.650194 |
| S3：输入遮蔽 | DPC-KMeans | **0.639524** | **0.545594** |
| S4a：lambda=0.01 | DPC-KMeans | 0.604178 | 0.758511 |
| S4b：embed_dim=16 | DPC-KMeans | 0.588234 | 0.647583 |

R2 用相同的 K-means `n_init=30` 比较特征方案：

| 数据集 | DeTSEC 平均 SCI | S3 平均 SCI | S3−DeTSEC | 该组比较的优胜方案 |
| --- | ---: | ---: | ---: | --- |
| ECO | 0.816538 | 0.624913 | −0.191625 | DeTSEC |
| REFIT | 0.706456 | 0.627354 | −0.079102 | DeTSEC |
| UK-DALE | 0.589306 | 0.701771 | +0.112465 | S3 |

UK-DALE 的 S3 在 K=2…6 时 SCI 更高，在 K=7、8 时低于 DeTSEC。因此“UK-DALE 上跨 K 平均表现更好”有证据，“所有 K 全面更好”没有证据。上述实验也不构成不同特征空间上真实状态识别率的证明，内部几何得分需要下游任务和外部标签补充验证。

## 5. 聚类主模型：z-score K-means

来源：`src/steps/time_clustering_step.py`。

先删除存在 NaN/Inf 的特征行，并保持标签、长度、位置索引对齐。对每一特征维做训练数据内标准化：

\[
\tilde z_{ij}=\frac{z_{ij}-\mu_j}{\sigma_j},\qquad
\min_{c_i,\mu_k}\sum_i\|\tilde z_i-\mu_{c_i}\|_2^2.
\]

K-means 交替执行“分到最近中心”和“更新中心为簇内均值”，适合欧氏空间中相对紧凑的簇。项目调用 sklearn 默认 K-means++ 初始化，`n_init=30` 多次初始化，`max_iter=300`、`random_state=42`。多次初始化按 K-means 自身目标选择结果，并非每次都按 SCI 选解。

正常 `kmeans` 步骤为每个候选 K 保存独立结果，如 `kmeans_k3`、`kmeans_k5`；不会自动把其它 K 删掉，只保留所谓最佳 K。下游需要明确的 `cluster_tag`。

### 5.1 聚类对比方案

| 方法 | 原理 | 与主选的区别 |
| --- | --- | --- |
| `dpc-kmeans` | 先计算局部密度 rho、到更高密度点的距离 delta，按归一化乘积 gamma 选 K 个初始中心，再运行 K-means | 改的是初始化，最终仍是欧氏 K-means；显式初始中心导致 `n_init=1` |
| `dbscan` | eps 邻域内达到 min_samples 的点为核心点，通过密度可达扩展簇 | 不预设 K，可输出噪声 −1；对尺度、密度和 eps 敏感 |
| `hdbscan` | 基于互可达距离构建密度层次，提取稳定簇 | 不固定 K，支持不同密度及噪声；项目使用可选 hdbscan 库 |
| `kmeans-scan` | 扫候选 K，按最大 SCI 推荐 | 诊断选 K，不是新的聚类算法，不登记最终簇结果 |
| `dpc-kmeans-scan` | 扫 K，对 DBI、SCI、简化密度指标进行排名求和 | 同样只是诊断；排名和越小越好 |

DPC 的主要公式为：

\[
\rho_i=\sum_{j\ne i}e^{-(d_{ij}/d_c)^2},\qquad
\delta_i=\min_{j\text{ 位于更高密度排序}}d_{ij},\qquad
\gamma_i=\hat\rho_i\hat\delta_i.
\]

`d_c` 取成对距离的约 2% 分位值；按 gamma 降序选中心，可设置最小间距。最高密度点的 delta 取其最大距离。它需要两两距离矩阵，内存/计算成本不能与普通 K-means 混为一谈；也没有“绝不选离群点”的数学保证。

### 5.2 历史为何选择 K=5，以及本次为何推荐 K=3

历史 `output/20260811_072323_compare_cluster_kmeans_vs_dpc/best_k.csv` 的最佳结果为：

| 数据集 | 方法 | 综合排名选 K | SCI ↑ | DBI ↓ | CHI ↑ | 简化 DBCV ↑ | rank_sum ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ECO | K-means | 2 | 0.734873 | 0.374304 | 4737.75 | 0.259161 | 12 |
| ECO | DPC-KMeans | 5 | 0.697316 | 0.443480 | 8343.48 | 0.314628 | 9 |
| REFIT | K-means | 7 | 0.656815 | 0.483913 | 6593.12 | 0.267817 | 2 |
| REFIT | DPC-KMeans | 8 | 0.648462 | 0.519477 | 7498.46 | 0.308571 | 6 |
| UK-DALE | K-means | 5 | 0.796220 | 0.426314 | 110636.83 | 0.131788 | 6 |
| UK-DALE | DPC-KMeans | 3 | 0.824937 | 0.330640 | 106127.62 | 0.085730 | 15 |

这里使用：`rank(DBI 升序) + rank(SCI 降序) + rank(简化 DBCV 降序)`。比较脚本先在**同一数据集的两种方法、全部候选 K** 上一起排名，再在各方法内部取最小排名和。单独调用 `sweep_k()` 则只对该次传入候选集合排名；候选集合不同，排名数值不能直接横比。

UK-DALE 的 K-means 在 K=3 时 SCI=0.824937，高于 K=5 的 0.796220，但历史综合排名选 K=5。当前正式配置固定 K=5，与该选择一致；这不代表当前 H1 数据重新完成了一次 K 搜索，也不意味着 K=5 是五个真实功能状态。

**本次建议采用 K=3。** 用户指出综合排名各部分的权重缺乏确定依据，这与代码中的实际问题一致：

1. `rank_sum` 隐含三项权重均为 1，没有验证这些权重能反映基元质量或下游收益；排序还丢失了原指标差距的大小。
2. 简化 DBCV 是自定义比值，不具备标准 DBCV 的解释，将它与 SCI、DBI 等权合并缺少依据。
3. 在 UK-DALE 的同一组 S3 + K-means 结果中，K=3 的 SCI 为 **0.824937**、DBI 为 **0.330640**；K=5 分别为 0.796220、0.426314。K=3 在这两项常用指标上同时更好，并使用更小的字典。
4. 该次结果中 K=3 的最小簇占比约 **15.70%**，K=5 约 **3.69%**，K=3 的类别支持更充足。这是辅助证据，不表示真实电器状态必须均衡。

建议把选型规则明确为：**在预先给定的 K=2…8 范围内，以 SCI 为主指标，DBI 和类别支持作为诊断，并结合字典简洁性选择 K=3；自定义密度指标单独报告，不再用未验证的等权排名覆盖这一选择。** 这是针对现有历史结果的解释性决策，不能追溯宣称为实验前已注册的规则。

K=3 也不是所有指标的共同最优：例如 K=2 的 DBI=0.328828，略低于 K=3，但 SCI=0.766394，明显更低。采用 K=3 的依据是明确主指标并有辅助证据，而非声称不存在任何取舍。历史簇数尚需在当前源训练支持上核查类别稳定性和下游收益。

后续若将正式流程切换到 K=3，需同步更新 `run.n_clusters`、`run.cluster_tag` 和 `nilm.k`，重建聚类、合并状态、字典和辅助标签，并重新训练对应下游模型；不能将已有 K=5 产物改名作为 K=3 结果。此次文档整理没有执行这些实验变更。

## 6. 项目设计了哪些指标

### 6.1 切分与状态块指标：结构诊断，不是准确率

| 指标 | 定义/目的 | 当前证据位置 |
| --- | --- | --- |
| 活动数、片段数、片段/活动 | 衡量切分规模和粒度 | manifest；历史切分分析文档 |
| 内部变点数、变点/活动、含≥2变点的活动比例 | 检查是否几乎不分段或切分过密 | 历史切分分析文档 |
| 片段长度均值、中位数、标准差、最大值、P10/P25/P75/P90 | 检查碎片、重尾和 padding 计算压力 | `lengths.npy`；历史分析汇总 |
| `n_segments / n_blocks` | 合并前片段数和连续块数 | state_merge 的 `metrics.json` |
| `n_merged_segments / merged_segment_ratio` | **标签发生变化的原片段数量/比例** | 代码变量为 `n_changed`；不是所有被拼接片段的数量 |
| `n_short_absorbed_segments / n_similar_merges` | 短块吸收数量、最终块上相似合并标记的计数 | state_merge 的 `metrics.json` |
| 状态块数、累计长度、来源活动数、完整观测块数 | 检查序列是否可用、状态是否有足够独立活动支持 | 状态合并和 sequence quality / label QC |
| `active_coverage`、缺失/冲突点、低功率活动点监督覆盖 | 防止将缺测、低功率活动、OFF 混淆 | `label_qc.json`、`sequence_quality.json` |

时长必须注明口径：旧切分统计通常是 `lengths/fs`；严格观测审计还会考虑真实时间、缺失和截断。缺乏真实边界标注时，不能把粒度统计当成 boundary precision/recall/F1。人工遮挡后的标签恢复一致性或边界稳定性也只是在测扰动鲁棒性。

### 6.2 特征学习指标

| 指标 | 作用与正确解释 |
| --- | --- |
| `loss / val_loss` | 总训练/验证损失；同一设置内观察收敛和早停。不同归一化、维度、输出约束和损失归一化下，绝对数值未必可比 |
| `l_ae / l_phy` | 分离重构误差与 TV 惩罚，检查平滑项是否主导 |
| `epochs_trained / best_epoch` | 实际训练轮数及恢复的最佳轮次，不代表都训练满 50 轮 |
| `tf_ratio_used` | 每轮实际真实输入保留率，核实 S3 衰减进度 |
| `validation_z_only_loss` | 严格验证路径补充的纯 z 条件重构诊断；并非所有历史 run 都有 |
| PCA `evr6 / var_sum2 / var_sum6` | 前若干主成分解释方差及累计值，检查表示是否集中于少数方向；高或低都不是无条件更好 |
| 下游 SCI、DBI、CHI | 间接评价特征空间的可聚类性，不等于真实功能识别率 |

PCA 诊断由 `scripts/compare_detsec_vs_pcdetsec.py` 实现。DTW 的 history 中 `loss=0` 是占位记录，没有神经网络训练，不能据此说 DTW 重构误差最好。

### 6.3 聚类内部指标：定义、方向、局限

来源：`src/utils/cluster_metrics.py`。普通路径在非噪声样本上计算 SCI、DBI、CHI；不足两个有效簇等退化情形保存 `None`，不是 0。

| 指标 | 公式/含义 | 趋势 |
| --- | --- | --- |
| SCI / Silhouette | 每个点的簇内平均距离 a、到最近其它簇平均距离 b，`s=(b−a)/max(a,b)`，再取均值 | 越大越好，通常范围 [−1,1] |
| DBI / Davies–Bouldin | `mean_i max_(j≠i) (S_i+S_j)/distance(c_i,c_j)`，S 为簇内到中心平均距离 | 越小越好 |
| CHI / Calinski–Harabasz | `[tr(B)/(K−1)] / [tr(W)/(N−K)]`，比较簇间与簇内离散度 | 越大越好；不同样本量/特征空间不能机械横比 |
| 簇数、噪声数、簇大小分布 | 识别单簇退化、丢弃大量噪声、极不均衡分组 | 无统一单调方向 |
| 最小簇占比 | 最小簇样本数/总样本数 | 支持性诊断，不是越均衡越符合电器状态 |
| few-shot 簇数 | 默认大小小于平均簇大小 50%；也支持固定数量阈值 | 检查稀有簇，不能据此断定稀有簇错误 |
| avgSCI / avgDBI | 同一运行跨候选 K 的算术均值 | 汇总候选范围表现，不是统计显著性检验 |
| rank_sum | 多指标各自排序后加和 | 越小越好，依赖候选集合与并列值处理 |

SCI 排除噪声后可能显得很高，必须同时报告 `n_noise`。使用预计算 DTW 距离时，SCI 可以按该距离算，但 DBI/CHI 仍来自传入的向量特征空间，三项未必具有同一几何口径。普通 K-means 则始终对归一化特征用欧氏目标，不能仅修改 `metric: dtw` 就把它变成 DTW K-means。

### 6.4 必须区分两种 DBCV

**标准实现：** `src/utils/cluster_metrics.py::dbcv_score()` 调用 `hdbscan.validity.validity_index`，通过簇内密度稀疏程度和簇间密度分离程度计算密度有效性，通常在 [−1,1]，越大越好。当前 HDBSCAN 分支会尝试记录它，库缺失或退化时可能跳过。

**项目简化指标：** `models/clustering/dpc_kmeans.py::dbcv_simplified()` 的实际公式是：

\[
\rho_k=\operatorname{mean}_{i\in C_k}\frac1{\operatorname{mean}(d_{i,kNN})+\epsilon},
\quad Q=\frac1K\sum_k\max_{l\ne k}\frac{\|c_k-c_l\|}{\rho_k+\rho_l+\epsilon}.
\]

它在 DPC 扫描和历史聚类比较脚本中被命名为 DBCV，但**不是标准 DBCV**，没有相同的值域和标准密度有效性解释。其尺度也敏感：统一放大向量会同时改变中心距离和逆距离密度。因此本文将它称为“简化 DBCV / 自定义密度比值指标”；正式论文应明确命名和公式，不与标准 DBCV 混报。

K-means/DPC-KMeans 的普通 `metrics.json` 本身并不保存这项简化指标；历史比较脚本会另行计算。`dpc-kmeans-scan` 元数据里的 `DBI↑` 文本与实际代码方向不符，真正执行的是 **DBI 升序，数值越小越优**。

### 6.5 合并前后指标的一个实际陷阱

`cluster` 用标准化特征计算指标；`state_merge` 当前调用 `compute_cluster_metrics(feats, seg_labels)`，传入原始嵌入而非标准化嵌入。它计算的也是片段级标签指标，不是聚合后块级特征指标。

实证例子：`log/primitive_source_20260915_v2` 有 403 个片段、32 维特征、5 个簇，合并成 288 个块，标签变化数为 0，但 SCI 从 0.775751 变为 0.812309。**标签没变而分数变了，原因是评价尺度变了，不能把差值当成合并效果。** 同类片段拼接也能减少块数，所以 `merged_segment_ratio=0` 不代表完全没有时间合并。

### 6.6 下游 NILM 指标：检验基元是否最终有用

这些指标属于下游，不是特征自编码器训练指标。实现主要在 `nilm_experiments/nilm_lab/metrics.py`、`activity_metrics.py` 和 `workflow.py`。

| 指标 | 定义/用途 | 方向 |
| --- | --- | --- |
| MAE / RMSE，W | 有效点上的平均绝对误差/均方根误差 | ↓ |
| `threshold_ON_MAE` | 真实功率超过阈值的点上的 MAE | ↓ |
| `active_MAE / inactive_MAE` | 原活动检测器定义的活动内/外 MAE，活动内部低功率也保留 | ↓ |
| SAE | `abs(E_pred−E_true)/E_true` | ↓ |
| MR | `sum(min(y,p))/sum(max(y,p))` | ↑ |
| `energy_wh / predicted_energy_wh` | 真实/预测累计能量，按采样间隔积分 | 报偏差，不单独排名 |
| OFF/非活动虚报能量 | `off_false_energy_wh` 按功率阈值 OFF；`inactive_predicted_wh` 与 `inactive_overprediction_wh` 按活动区间外统计 | ↓；两种 OFF 口径需区分 |
| 功率阈值 Precision/Recall/F1 | 对真实和预测功率是否超过阈值评分 | ↑ |
| `activity_F1 / activity_recall / activity_false_positive_rate` | 活动概率以 0.5 为阈值，相对原活动检测器的伪标签评分 | F1/Recall ↑，FPR ↓ |
| `activity_PR_AUC` | 代码实际调用 `average_precision_score`，是 Average Precision 口径 | ↑ |
| 完整活动 Precision/Recall/F1 | 仅在边界完整、可观测范围内，以 IoU≥0.5 一对一贪心匹配 | ↑；无完整真值机会时不可评 |
| `activity_boundary_MAE_seconds` | 已匹配完整活动的起止边界绝对误差均值 | ↓，需同时报告匹配数 |
| 预测/目标覆盖率、有效点数 | 防止模型因少预测、少评估而看似更优 | 必须随误差共同报告 |
| 训练秒数、参数量 | 衡量计算代价 | 配合精度权衡 |

基础 `metrics()` 还有瞬时功率事件 `event_f1_iou05`，但活动评分函数明确删除这项，另算完整活动事件指标；二者不能混写成同一个“活动 F1”。输入预测先裁剪为非负，非有限点不参与误差评分，能量为零时 SAE 等指标可能为空。

`nilm_select` 的标准 workflow 按源验证的住宅宏平均 MAE 选模，不按目标测试集 SCI 或 MAE 调参。配对 MAE 改善的置信区间工具先在住宅内平均配对种子，再对住宅 bootstrap；少于 3 个独立住宅不报告该 CI，多个种子不能冒充多个独立住宅。

下游配置还设计了 R（纯回归）、O（活动监督）、B（逐点功率等级）、BP（相同状态块边界上的均值分组）、P（深度基元）、简单形状对照，以及直接活动头/GRU 活动头等对照。其目的分别是排除“只需 ON/OFF”“只需功率幅值”“只需边界”“只需简单形状”这些解释。它们不应与本节前面的特征提取候选模型混为一个排行榜。

## 7. 建议用于论文或汇报的表述

> 本项目首先从电器活动区间中，通过局部统计变化检测与持续时间先验修正获取候选基元；随后用双向 GRU、掩码注意力及带非负输出和平滑先验的 DeTSEC-PC 学习片段表示，并通过逐渐遮蔽 teacher-forcing 输入增强嵌入的信息承载；最终在标准化特征空间中用 K-means 建立基元字典，经时间合并生成状态序列。已有 UK-DALE 实验支持 S3 表示和 K-means 的选择，并依据轮廓系数、DBI 与字典简洁性推荐 K=3，但该优势尚不能推广为全部数据集的统一最优。聚类内部指标度量几何结构，基元的语义真实性及对 NILM 的增益需由独立对照与下游评估进一步验证。

若需追加“最优模型”的强结论，应补齐统一数据、活动支持、归一化、训练预算、候选 K 和多种子的比较；切分需独立边界参考，特征/聚类需稳定性与下游收益，同时统一合并前后的评价空间。上述是后续验证建议，不是本次已经执行的实验。

## 8. 代码与实验依据索引

所有代码及结果路径均相对于远程 `/home/scnu202438025446/pslg-nilm`。

| 内容 | 核查来源 |
| --- | --- |
| 当前 step 顺序、默认选择、调用参数 | `main.py` |
| 当前正式主选与超参 | `config/config_nilm_primitive_formal.yaml` |
| 历史 S3 配置 | `config/config_ukdale_pcdetsec_s3.yaml` |
| 切分分支、四通道、截断行为 | `src/steps/time_segmentation.py` |
| PrimGLR 公式与细化 | `models/time_segmentation/prim_glr.py` |
| ClaSP/FLUSS/ESPRESSO 接口 | `models/time_segmentation/{clasp_origin,fluss,espresso}.py` |
| 特征注册、输出和缓存 | `src/steps/feature_extract_step.py` |
| 主特征模型与对比模型 | `models/feature_extract/detsec_pc.py`、`detsec_model.py` 及同目录各 AE/DTW 文件 |
| 聚类、scan 与指标 | `src/steps/time_clustering_step.py`、`src/utils/cluster_metrics.py` |
| DPC 初始化、自定义指标与排名 | `models/clustering/dpc_kmeans.py` |
| 历史 DeTSEC 原始结果 | `log_det_test/20260809_primglr_detsec_<eco/refit/ukdale>/TimeClustering_kmeans_on_detsec_on_prim-glr/kmeans_k*/metrics.json` |
| 历史 S3 原始结果 | `log_det_test/20260810_120812_pcdetsec_s3_<eco/refit/ukdale>/TimeClustering_kmeans_on_detsec_pc_on_prim-glr/kmeans_k*/metrics.json` |
| R1 v2/S1 与各消融 | `log_det_test/20260810_061444_primglr_pcdetsecv2_ukdale/`、`log_det_test/20260810_090804_abl_*_ukdale/` |
| 聚类最佳 K 汇总及生成逻辑 | `output/20260811_072323_compare_cluster_kmeans_vs_dpc/best_k.csv`、`scripts/compare_cluster_methods.py` |
| 特征×聚类控制比较、PCA 诊断 | `scripts/compare_detsec_vs_pcdetsec.py` |
| 合并语义与指标空间 | `src/steps/temporal_state_merge_step.py` |
| 403 片段实际例子 | `log/primitive_source_20260915_v2/run_manifest.json` 及其中登记的聚类、合并、序列和标签 QC 产物 |
| 下游指标与选择规则 | `nilm_experiments/nilm_lab/{metrics,activity_metrics,workflow}.py` |

可结合现有文档阅读：[S3 实验记录](detsec_pc_scheduled_sampling_20260810.md)、[历史切分分析](segment_analysis_20260808_clasp-origin.md)、[当前基元状态定义](NILM_primitive_state_definition_20260917.md)。本文涉及实现细节和结论边界时，以实际执行代码及原始产物为准，不沿用历史说明中的未验证语义推断。
