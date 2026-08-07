import os

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, RepeatVector, TimeDistributed, Dense, Masking,
    Bidirectional, Permute, Multiply, Lambda, Layer, Concatenate
)
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import initializers
import tensorflow as tf


class DETSECAttention(Layer):
    """
    参考DETSEC.py实现的注意力机制层
    
    实现细节：
    - 使用可学习的权重矩阵 W_omega 进行特征变换
    - 使用偏置项 b_omega 增加表达能力
    - 使用上下文向量 u_omega 计算注意力分数
    - 数学公式：
      1. v = tanh(inputs @ W_omega + b_omega)  # (batch, time, attention_size)
      2. vu = v @ u_omega                      # (batch, time)
      3. alphas = softmax(vu)                  # (batch, time)
      4. output = sum(inputs * alphas)         # (batch, features)
    
    Args:
        attention_size: 注意力隐藏层维度，默认32
        kernel_initializer: 权重初始化方法
    
    Returns:
        output: 加权聚合后的特征，形状为 (batch_size, nunits)
        alphas: 注意力权重，形状为 (batch_size, timesteps)
    """
    def __init__(self, attention_size=32, kernel_initializer='random_normal', **kwargs):
        super(DETSECAttention, self).__init__(**kwargs)
        self.attention_size = attention_size
        self.kernel_initializer = initializers.get(kernel_initializer)
        
    def build(self, input_shape):
        # input_shape: (batch_size, timesteps, nunits)
        self.nunits = input_shape[-1]
        self.timesteps = input_shape[1]
        
        # W_omega: (nunits, attention_size) - 特征变换矩阵
        self.W_omega = self.add_weight(
            name='W_omega',
            shape=(self.nunits, self.attention_size),
            initializer=self.kernel_initializer,
            trainable=True
        )
        
        # b_omega: (attention_size,) - 偏置项
        self.b_omega = self.add_weight(
            name='b_omega',
            shape=(self.attention_size,),
            initializer='zeros',
            trainable=True
        )
        
        # u_omega: (attention_size,) - 上下文向量
        self.u_omega = self.add_weight(
            name='u_omega',
            shape=(self.attention_size,),
            initializer=self.kernel_initializer,
            trainable=True
        )
        
        super(DETSECAttention, self).build(input_shape)
    
    def call(self, inputs):
        """
        前向传播
        
        Args:
            inputs: 编码器输出，形状为 (batch_size, timesteps, nunits)
        
        Returns:
            output: 注意力加权后的特征，形状为 (batch_size, nunits)
        """
        # inputs形状: (batch_size, timesteps, nunits)
        batch_size = tf.shape(inputs)[0]
        
        # 第一步：计算 v = tanh(inputs @ W_omega + b_omega)
        # inputs @ W_omega: (batch, time, nunits) @ (nunits, attention_size) = (batch, time, attention_size)
        v = tf.tanh(tf.tensordot(inputs, self.W_omega, axes=1) + self.b_omega)
        # v形状: (batch_size, timesteps, attention_size)
        
        # 第二步：计算 vu = v @ u_omega
        # v @ u_omega: (batch, time, attention_size) @ (attention_size,) = (batch, time)
        vu = tf.tensordot(v, self.u_omega, axes=1)
        # vu形状: (batch_size, timesteps)
        
        # 第三步：计算 alphas = softmax(vu)
        alphas = tf.nn.softmax(vu, axis=1)  # 在时间维度上softmax
        # alphas形状: (batch_size, timesteps)
        
        # 第四步：加权求和 output = sum(inputs * alphas)
        # 扩展alphas维度: (batch, time) -> (batch, time, 1)
        alphas_expanded = tf.expand_dims(alphas, -1)
        # 加权: inputs * alphas_expanded = (batch, time, nunits) * (batch, time, 1) = (batch, time, nunits)
        weighted = inputs * alphas_expanded
        # 求和: (batch, time, nunits) -> (batch, nunits)
        output = tf.reduce_sum(weighted, axis=1)
        # output形状: (batch_size, nunits)
        
        # 保存alphas供后续可视化使用
        self.alphas = alphas
        
        return output
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])  # (batch_size, nunits)
    
    def get_config(self):
        config = super(DETSECAttention, self).get_config()
        config.update({
            'attention_size': self.attention_size,
            'kernel_initializer': initializers.serialize(self.kernel_initializer)
        })
        return config


class GatingLayer(Layer):
    """
    门控层，参考DETSEC.py的gate函数实现
    
    使用sigmoid激活的全连接层生成门控掩码，
    用于控制信息流的通过程度。
    
    公式：gate(vec) = sigmoid(W @ vec + b)
    """
    def __init__(self, **kwargs):
        super(GatingLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.dense = Dense(input_shape[-1], activation='sigmoid', name='gate_dense')
        super(GatingLayer, self).build(input_shape)
    
    def call(self, inputs):
        return self.dense(inputs)
    
    def compute_output_shape(self, input_shape):
        return input_shape


def bilstm_ae_attention(data: np.ndarray, config: dict):
    """
    BiLSTM + DETSEC风格全局注意力自编码器特征提取函数
    
    该函数使用双向 LSTM 结合DETSEC风格全局注意力机制的自编码器从时序数据中提取全局特征。
    参考DETSEC.py的实现，采用双向分别计算注意力+门控融合的策略。
    
    模型结构：
    - 编码器：
      - BiLSTM(32+32)：提取双向时序特征
      - Split：分离前向和后向输出
      - DETSECAttention：分别对前向和后向计算注意力
      - GatingLayer：使用门控机制融合双向注意力结果
      - Dense(latent_dim)：降维到目标维度
    - 解码器：RepeatVector + BiLSTM(32+32) + TimeDistributed(Dense)，重构原始时序
    
    Args:
        data (np.ndarray): 输入数据，形状为 (n_samples, timesteps, n_features)
                          - n_samples: 样本数量
                          - timesteps: 时间步长度（填充后的统一长度）
                          - n_features: 特征数量（单特征时为1）
        config (dict): 模型配置字典，包含以下键：
                      - latent_dim (int): 全局特征维度，默认64
                      - epochs (int): 训练轮数，默认50
                      - batch_size (int): 批量大小，默认32
                      - learning_rate (float): 学习率，默认0.001
                      - patience (int): 早停耐心值，默认5
                      - attention_size (int): 注意力隐藏层维度，默认32
    
    Returns:
        tuple: (features, training_history)
            features (np.ndarray): 提取的全局注意力特征，形状为 (n_samples, latent_dim)
            training_history (dict): 训练过程的历史信息，包含：
                - loss: 训练损失
                - val_loss: 验证损失
                - epochs_trained: 实际训练轮数
    
    Example:
        >>> import numpy as np
        >>> data = np.random.rand(100, 50, 1)  # 100个样本，50个时间步，1个特征
        >>> config = {"latent_dim": 64, "epochs": 50, "batch_size": 32, 
        ...           "learning_rate": 0.001, "patience": 5, "attention_size": 32}
        >>> features, history = bilstm_ae_attention(data, config)
        >>> print(features.shape)  # (100, 64)
    
    Note:
        与原版DETSEC的区别：
        - 使用Keras Functional API而非TensorFlow 1.x的低级API
        - 保留了BiLSTM+AE的主干结构
        - 采用DETSEC的注意力计算方式（W_omega, b_omega, u_omega）
        - 添加了门控融合机制
    """
    # ===================== 1. 解析配置参数 =====================
    latent_dim = config.get("latent_dim", 64)  # 全局特征维度
    epochs = config.get("epochs", 50)          # 训练轮数
    batch_size = config.get("batch_size", 32)  # 批量大小
    learning_rate = config.get("learning_rate", 0.001)  # 学习率
    patience = config.get("patience", 5)       # 早停耐心值
    attention_size = config.get("attention_size", 32)  # 注意力隐藏层维度
    
    # ===================== 2. 提取数据维度信息 =====================
    timesteps = data.shape[1]  # 时间步长度
    n_features = data.shape[2]  # 特征数量
    
    # 赋值为模型输入
    X = data
    
    # ===================== 3. 数据归一化 =====================
    # 深度学习对数据量级敏感，原始数据可能高达数千，不归一化极易导致梯度爆炸（NaN）
    # 使用 Min-Max 归一化将数据映射到 [0, 1] 区间
    X_min = X.min()
    X_max = X.max()
    X = (X - X_min) / (X_max - X_min + 1e-7)  # 加上小常数防止除零
    
    # 计算归一化后的 Mask 值（原始填充值 0.0 对应的新值）
    # Masking 层会忽略这个值，避免填充值对模型训练的影响
    scaled_mask_value = (0.0 - X_min) / (X_max - X_min + 1e-7)
    print(f"归一化完成 | 范围: {X.min():.2f} ~ {X.max():.2f} | Mask值: {scaled_mask_value:.4f}")
    
    # ===================== 4. 构建 BiLSTM + DETSEC注意力自编码器模型 =====================
    # 输入层：适配单特征的时序形状 (timesteps, n_features)
    input_layer = Input(shape=(timesteps, n_features), name="input_layer")
    
    # Masking 层：忽略填充值，适配不等长数据
    # 对于填充为 0.0 的位置，Masking 层会将其在计算中忽略
    masking_layer = Masking(mask_value=scaled_mask_value, name="masking_layer")(input_layer)
    
    # ===================== 编码器部分 =====================
    # BiLSTM 编码器：双向 LSTM，提取时序数据的前向+后向依赖
    # - 每个方向 32 个单元，总共 64 个单元
    # - return_sequences=True: 返回每个时间步的输出（保持 3 维结构）
    # - 输出形状: (batch_size, timesteps, 64)
    # 注意：Bidirectional LSTM的输出是前向和后向的拼接 [forward_32, backward_32]
    encoder_bilstm = Bidirectional(
        LSTM(32, activation='tanh', return_sequences=True),
        name="encoder_bilstm"
    )(masking_layer)
    # encoder_bilstm形状: (batch_size, timesteps, 64)
    
    # 分离前向和后向输出
    # 前向LSTM输出：前32维
    forward_output = Lambda(lambda x: x[:, :, :32], name="forward_split")(encoder_bilstm)
    # 后向LSTM输出：后32维
    backward_output = Lambda(lambda x: x[:, :, 32:], name="backward_split")(encoder_bilstm)
    
    # 分别对前向和后向计算DETSEC风格的注意力
    # 前向注意力
    attention_fw = DETSECAttention(
        attention_size=attention_size,
        kernel_initializer=initializers.RandomNormal(stddev=0.1),
        name="attention_forward"
    )(forward_output)
    # attention_fw形状: (batch_size, 32)
    
    # 后向注意力
    attention_bw = DETSECAttention(
        attention_size=attention_size,
        kernel_initializer=initializers.RandomNormal(stddev=0.1),
        name="attention_backward"
    )(backward_output)
    # attention_bw形状: (batch_size, 32)
    
    # 使用门控机制融合前向和后向注意力结果
    # 参考DETSEC.py：encoder = gate(encoder_fw) * encoder_fw + gate(encoder_bw) * encoder_bw
    gate_fw = GatingLayer(name="gate_forward")(attention_fw)
    gate_bw = GatingLayer(name="gate_backward")(attention_bw)
    
    # 门控融合
    gated_fw = Lambda(lambda x: x[0] * x[1], name="gated_forward")([gate_fw, attention_fw])
    gated_bw = Lambda(lambda x: x[0] * x[1], name="gated_backward")([gate_bw, attention_bw])
    
    # 拼接融合后的特征
    encoder_concat = Concatenate(name="encoder_concat")([gated_fw, gated_bw])
    # encoder_concat形状: (batch_size, 64)
    
    # 编码器特征降维：将 64 维特征降维到 latent_dim
    encoder_features = Dense(latent_dim, activation='relu', name="encoder_global_dense")(encoder_concat)
    # encoder_features形状: (batch_size, latent_dim)
    
    # ===================== 解码器部分 =====================
    # 将全局特征复制到每个时间步，作为解码器的输入
    # - RepeatVector: 将 (batch_size, latent_dim) 转换为 (batch_size, timesteps, latent_dim)
    decoder_input = RepeatVector(timesteps, name="repeat_vector")(encoder_features)
    
    # BiLSTM 解码器：从全局特征重构时序数据
    # - 使用双向 LSTM（与编码器对称）
    # - return_sequences=True: 返回每个时间步的输出
    # - 输出形状: (batch_size, timesteps, 64)
    decoder_bilstm = Bidirectional(
        LSTM(32, activation='tanh', return_sequences=True),
        name="decoder_bilstm"
    )(decoder_input)
    
    # 输出层：TimeDistributed + Dense
    # - 对每个时间步独立应用全连接层
    # - 将解码器的输出映射到原始特征空间
    # - 输出形状: (batch_size, timesteps, n_features)
    output_layer = TimeDistributed(
        Dense(n_features, activation='linear'),
        name="output_layer"
    )(decoder_bilstm)
    
    # ===================== 5. 构建完整模型 =====================
    # 自编码器模型：用于训练，输入和输出都是原始数据
    # - 输入形状: (batch_size, timesteps, n_features)
    # - 输出形状: (batch_size, timesteps, n_features)
    lstm_autoencoder = Model(inputs=input_layer, outputs=output_layer, name="bilstm_detsec_attention_ae")
    
    # 编码器模型：用于特征提取，只保留编码器部分
    # - 输出形状: (batch_size, latent_dim)
    # - 这是全局特征，时间维度已被聚合
    lstm_encoder_model = Model(inputs=input_layer, outputs=encoder_features, name="detsec_attention_encoder")
    
    # 注意力可视化模型：用于获取注意力权重
    # 创建辅助模型来获取注意力权重
    attention_model_fw = Model(
        inputs=input_layer,
        outputs=attention_fw
    )
    attention_model_bw = Model(
        inputs=input_layer,
        outputs=attention_bw
    )
    
    # ===================== 6. 编译模型 =====================
    # 使用 Adam 优化器，添加梯度裁剪防止梯度爆炸
    # clipnorm=1.0: 将梯度范数裁剪到 1.0 以内
    lstm_autoencoder.compile(
        optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss='mse'
    )
    
    # ===================== 7. 配置早停回调 =====================
    # 当验证集损失在 patience 个 epoch 内没有改善时，停止训练
    earliest_stop = EarlyStopping(
        monitor='val_loss',      # 监控验证集损失
        patience=patience,       # 耐心值：多少个 epoch 没有改善就停止
        mode='min',              # 损失越小越好
        restore_best_weights=True,  # 恢复到最佳权重的模型
        verbose=1                # 打印停止信息
    )
    
    # ===================== 8. 训练模型 =====================
    # 无监督学习：输入和标签都是原始数据
    # Masking 层会忽略填充值的重构误差
    # 根据样本数量自动调整验证集比例
    n_samples = X.shape[0]
    if n_samples < 5:
        print(f"[FeatureExtract] Warning: Too few samples ({n_samples}) for validation split. Disabling validation.")
        current_val_split = 0.0
        current_callbacks = [] # 样本太少时禁用早停，因为没有验证集
    else:
        current_val_split = 0.2
        current_callbacks = [earliest_stop]
        
    history = lstm_autoencoder.fit(
        X, X,                    # 输入 = 标签（自编码器的特点）
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,             # 每个 epoch 打乱数据顺序
        validation_split=current_val_split,
        callbacks=current_callbacks
    )
    
    # ===================== 9. 提取特征 =====================
    # 使用训练好的编码器提取全局特征
    # 输出形状: (n_samples, latent_dim)
    # 这是全局特征，时间维度已被聚合
    X_global_features = lstm_encoder_model.predict(X)
    
    # ===================== 10. 输出结果 =====================
    # 构建训练历史信息字典
    training_history = {
        'loss': history.history['loss'],
        'val_loss': history.history.get('val_loss', history.history['loss']), # 如果没有 val_loss，用 loss 代替
        'epochs_trained': len(history.history['loss']),
        'model_name': 'BiLSTM_Attention'
    }
    
    return X_global_features, training_history


if __name__ == "__main__":
    # 测试代码
    print("测试DETSEC风格的BiLSTM+Attention模型...")
    
    # 创建测试数据
    test_data = np.random.rand(20, 30, 1).astype(np.float32)
    
    config = {
        "latent_dim": 16,
        "epochs": 5,
        "batch_size": 4,
        "learning_rate": 0.001,
        "patience": 3,
        "attention_size": 16
    }
    
    features, history = bilstm_ae_attention(test_data, config)
    print(f"\n提取的特征形状: {features.shape}")
    print(f"训练轮数: {history['epochs_trained']}")
    print("测试完成！")
