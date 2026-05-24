# 特征提取模型库

本目录包含多种用于时序数据特征提取的自编码器模型，适用于 NILM（非侵入式负载监测）任务。

## 模型列表

| 模型名称 | 文件 | 特点 | 适用场景 |
|---------|------|------|----------|
| LSTM 自编码器 | `lstm_ae.py` | 单向 LSTM，结构简单，计算效率高 | 简单时序模式学习 |
| BiLSTM 自编码器 | `bilstm_ae.py` | 双向 LSTM，捕捉前后向依赖 | 需要完整时序上下文的场景 |
| BiLSTM + DETSEC 注意力 | `bilstm_ae_attention.py` | 双向 LSTM + 注意力机制，自动学习时间步重要性 | 复杂时序模式，需要关注关键时间点的场景 |

## 模型架构对比

### 1. LSTM 自编码器

**架构**: 
- 编码器: LSTM(32) + Dense(latent_dim)
- 解码器: RepeatVector + LSTM(32) + TimeDistributed(Dense)

**特点**:
- 结构简单，计算效率高
- 只捕捉时序数据的前向依赖
- 适合处理简单的时序模式

### 2. BiLSTM 自编码器

**架构**:
- 编码器: BiLSTM(16+16) + Concatenate + Dense(latent_dim)
- 解码器: RepeatVector + LSTM(32) + TimeDistributed(Dense)

**特点**:
- 同时捕捉时序数据的前向和后向依赖
- 表达能力比单向 LSTM 更强
- 参数量与单向 LSTM 相当，但信息利用更充分

### 3. BiLSTM + DETSEC 注意力

**架构**:
- 编码器: BiLSTM(32+32) + 分离前向/后向输出 + 分别计算注意力 + 门控融合 + Dense(latent_dim)
- 解码器: RepeatVector + BiLSTM(32+32) + TimeDistributed(Dense)

**特点**:
- 集成了 DETSEC 风格的注意力机制
- 自动学习时序数据中不同时间步的重要性
- 门控机制融合前向和后向注意力结果
- 提取的特征更具针对性，适合复杂时序模式

## 输入输出格式

### 输入
- **数据形状**: `(n_samples, timesteps, n_features)`
  - `n_samples`: 样本数量
  - `timesteps`: 时间步长度（填充后的统一长度）
  - `n_features`: 特征数量（单特征时为 1）
- **配置参数**: 包含以下键的字典
  - `latent_dim`: 提取的特征维度
  - `epochs`: 训练轮数
  - `batch_size`: 批量大小
  - `learning_rate`: 学习率
  - `patience`: 早停耐心值
  - `attention_size`: 注意力隐藏层维度（仅注意力模型）

### 输出
- **返回值**: `(features, training_history)`
  - `features`: 提取的特征，形状为 `(n_samples, latent_dim)`
  - `training_history`: 训练过程的历史信息，包含损失值和训练轮数

## 使用示例

### 基本使用

```python
import numpy as np
from models.feature_extract.lstm_ae import lstm_ae
from models.feature_extract.cnn_ae import cnn_ae
from models.feature_extract.bilstm_ae import bilstm_ae
from models.feature_extract.bilstm_ae_attention import bilstm_ae_attention

# 创建示例数据
data = np.random.rand(100, 50, 1)  # 100个样本，50个时间步，1个特征

# 配置参数
config = {
    "latent_dim": 64,
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.001,
    "patience": 5
}

# 使用 LSTM 自编码器
features_lstm, history_lstm = lstm_ae(data, config)
print(f"LSTM 特征形状: {features_lstm.shape}")

# 使用 CNN 自编码器
features_cnn, history_cnn = cnn_ae(data, config)
print(f"CNN 特征形状: {features_cnn.shape}")

# 使用 BiLSTM 自编码器
features_bilstm, history_bilstm = bilstm_ae(data, config)
print(f"BiLSTM 特征形状: {features_bilstm.shape}")

# 使用 BiLSTM + 注意力
config_attention = config.copy()
config_attention["attention_size"] = 32
features_attention, history_attention = bilstm_ae_attention(data, config_attention)
print(f"BiLSTM + 注意力特征形状: {features_attention.shape}")
```

### 与工作流集成

```python
# 在工作流步骤中使用
from src.steps.feature_extract_step import FeatureExtractStep

# 创建特征提取步骤
feature_step = FeatureExtractStep(
    name="FeatureExtract",
    model_name="bilstm_ae_attention",
    latent_dim=64,
    epochs=50,
    batch_size=32
)

# 在工作流中执行
wf.add_step(feature_step)
```

## 模型选择指南

| 场景 | 推荐模型 | 理由 |
|------|---------|------|
| 计算资源有限，需要快速训练 | `lstm_ae.py` | 结构简单，计算效率高 |
| 希望增强局部模式建模 | `cnn_ae.py` | 卷积对局部波形/短期模式更敏感 |
| 需要捕捉双向时序依赖 | `bilstm_ae.py` | 双向 LSTM 能够同时考虑过去和未来信息 |
| 复杂负载识别，需要关注关键时间点 | `bilstm_ae_attention.py` | 注意力机制能够自动学习时间步的重要性 |

## 性能对比

| 模型 | 计算复杂度 | 特征表达能力 | 训练时间 | 适用场景 |
|------|-----------|-------------|----------|----------|
| LSTM | 低 | 中 | 快 | 简单时序模式 |
| CNN | 低-中 | 中 | 快 | 局部形状模式明显的序列 |
| BiLSTM | 中 | 高 | 中 | 一般时序模式 |
| BiLSTM + 注意力 | 高 | 很高 | 慢 | 复杂时序模式 |

## 注意事项

1. **数据预处理**: 所有模型都内置了数据归一化处理，将数据映射到 [0, 1] 区间
2. **填充值处理**: 使用 Masking 层忽略填充值（0.0）的影响
3. **早停机制**: 所有模型都实现了早停机制，防止过拟合
4. **梯度裁剪**: 使用梯度裁剪防止梯度爆炸
5. **验证集**: 默认使用 20% 的数据作为验证集

## 依赖要求

- TensorFlow 2.x
- NumPy
- Keras

## 扩展指南

要添加新的特征提取模型：

1. 在 `models/feature_extract/` 目录下创建新的模型文件
2. 实现与现有模型相同的接口（函数名和返回值格式）
3. 在 `src/steps/feature_extract_step.py` 中添加对新模型的支持
4. 更新本 README.md 文件

## 版本历史

| 版本 | 模型 | 改进 |
|------|------|------|
| v1 | LSTM, BiLSTM | 基础自编码器实现 |
| v2 | BiLSTM + 注意力 | 添加 DETSEC 风格注意力机制 |