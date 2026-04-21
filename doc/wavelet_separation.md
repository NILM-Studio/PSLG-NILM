# WaveletSeparationStep 步骤说明文档

`WaveletSeparationStep` 是工作流中的核心信号处理步骤，负责对原始功率信号进行离散小波分解（DWT）、自动切分以及张量化处理，为后续深度学习模型提供规整的输入数据。

---

## 1. 输入 (Input)

该步骤从上一个步骤（通常是 `DataLoader`）的缓存目录中读取数据：
- **来源路径**: `log/{sequence_id}/DataLoader/*.csv`
- **数据格式**: 
    - 必须包含 `power` 列（原始功率采样值）。
    - 可选包含 `timestamp` 列（用于绘图和标签生成）。

---

## 2. 核心处理逻辑 (Processing)

1.  **异常值去除**: 使用中值滤波（Median Filter, kernel_size=5）对原始信号进行平滑处理，生成“清洗后波形”。
2.  **小波分解与重构**:
    - 使用离散小波变换（默认 `db4` 小波，2 层分解）。
    - 重构出**低频分量（Low-frequency）**：反映信号的稳态趋势（如电器运行的平均功率）。
    - 重构出**高频分量（High-frequency）**：反映信号的瞬态特征（如电器开关瞬间的冲击电流）。
3.  **自动切分 (Segmentation)**:
    - 利用 `BinaryClaSPSegmentation` 算法分别在低频和高频分量上寻找变化点（Change Points）。
    - 合并两者的切分点，生成最终的合成切分点 `synthesized_cp`。
4.  **张量化与填充 (Tensorization & Padding)**:
    - 根据合成切分点将长信号切割成多个独立的样本片段（Samples）。
    - 识别批次中最长片段的长度 $T_{max}$。
    - 对所有较短片段进行**零填充（Zero-padding）**，使所有样本具有统一的形状。

---

## 3. 输出 (Output)

### 3.1 持久化文件 (Log Files)
输出文件保存在 `log/{sequence_id}/WaveletSeparation/` 目录下：

-   **`X.npy`**: 
    - **形状**: `(n_samples, timestamp, 4)`
    - **特征通道定义**:
        - **Channel 0**: 原始波形 (Original Signal)
        - **Channel 1**: 清洗后波形 (Cleaned Signal)
        - **Channel 2**: 低频重构波形 (Low-frequency)
        - **Channel 3**: 高频重构波形 (High-frequency)
-   **`lengths.npy`**: 
    - **形状**: `(n_samples, 1)`
    - **含义**: 记录每个样本在填充前的真实长度，用于后续模型处理变长序列。

### 3.2 可视化图表 (Visualizations)
根据配置生成的 PNG 图像：
-   **`wavelet_separation_{file}_{wavelet}.png`**: 展示原始/清洗对比、低频、高频以及最终合成切分点的 4 栏对比图。
-   **`wavelet_heatmap_{file}.png`**: 展示低频信号的连续小波变换（CWT）能量时频图。

### 3.3 上下文传递 (Context)
该步骤会将处理后的张量直接存入 `context['data']`，供下游 Step（如 `ModelTraining`）直接使用：
```python
context['data'] = {
    'X': np_array,       # (n_samples, timestamp, 4)
    'lengths': np_array  # (n_samples, 1)
}
```

---

## 4. 配置参数 (Configuration)

在 [config.yaml](file:///f:/B__ProfessionProject/PSLG-NILM/config/config.yaml) 的 `wavelet_separation` 节点下进行设定：

- **`enabled`**: 
    - **含义**: 步骤开关。
    - **取值**: `true` (运行该步骤) / `false` (跳过该步骤)。
- **`is_shape_dtw`**: 
    - **含义**: 控制切分算法（ClaSPSegmentation）使用的距离度量方式。
    - **取值**: 
        - `false` (默认): 使用 `znormed_euclidean_distance`，速度快，适合常规功率波动。
        - `true`: 使用 `shape_dtw`，能更好地捕捉复杂的波形形状特征，但计算开销较大。
- **`plot_count`**: 
    - **含义**: 可视化控制。
    - **取值**: 
        - `0` (默认): 不生成任何图表，以加快处理速度。
        - `x`: 为当前批次的前 `x` 个 CSV 文件生成分析图表（保存在 Log 目录中）。

---

## 5. 验证工具 (Verification Tool)

为了确保输出的 `.npy` 张量符合深度学习模型的输入要求，可以使用项目根目录下的 [check_outputs.py](file:///f:/B__ProfessionProject/PSLG-NILM/check_outputs.py) 进行检测。

### 5.1 主要功能
- **形状检查**: 自动打印 `X.npy` 的 `(N, T, 4)` 形状以及 `lengths.npy` 的 `(N, 1)` 形状。
- **通道验证**: 预览前 3 个样本的特征值，验证 4 个通道（原始、清洗、低频、高频）的数据是否正确。
- **填充验证**: 检查样本末尾是否正确执行了零填充（Zero-padding）。

### 5.2 使用方法
修改脚本中的 `target_log_dir` 变量为需要检测的 Log 文件夹路径，然后运行：
```bash
python check_outputs.py
```
