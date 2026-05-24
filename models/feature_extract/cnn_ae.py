import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    UpSampling1D,
    Dense,
    Reshape,
    GlobalAveragePooling1D,
    Lambda,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


def _downsample_steps(steps: int, times: int = 2) -> int:
    """Compute temporal size after repeated MaxPooling1D(pool_size=2, padding='same')."""
    value = int(steps)
    for _ in range(times):
        value = (value + 1) // 2
    return max(1, value)


def cnn_ae(data: np.ndarray, config: dict):
    """
    CNN (Conv1D) 自编码器特征提取函数。

    设计目标：
    1. 使用卷积提取局部时序模式；
    2. 通过全局池化+Dense 产生全局潜特征；
    3. 支持 lengths 驱动的逐时间步 sample_weight，避免 padding 区域主导重构损失。

    Args:
        data (np.ndarray): 输入数据，形状为 (n_samples, timesteps, n_features)
        config (dict): 模型配置字典，支持以下键：
            - latent_dim (int)
            - epochs (int)
            - batch_size (int)
            - learning_rate (float)
            - patience (int)
            - lengths (np.ndarray, optional)

    Returns:
        tuple: (features, training_history)
            - features: (n_samples, latent_dim)
            - training_history: dict(loss/val_loss/epochs_trained/model_name)
    """
    # ===================== 1. 解析配置参数 =====================
    latent_dim = config["latent_dim"]
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    patience = config["patience"]
    lengths = config.get("lengths", None)

    # ===================== 2. 提取数据维度信息 =====================
    timesteps = int(data.shape[1])
    n_features = int(data.shape[2])
    X = data

    # ===================== 3. 数据归一化 =====================
    X_min = X.min()
    X_max = X.max()
    X = (X - X_min) / (X_max - X_min + 1e-7)
    print(f"数据归一化完成 | 范围: {X.min():.2f} ~ {X.max():.2f}")

    # ===================== 3.1 构建逐时间步损失权重 =====================
    sample_weight = np.ones((X.shape[0], timesteps), dtype=np.float32)
    if lengths is not None:
        lengths_arr = np.asarray(lengths).reshape(-1)
        if lengths_arr.shape[0] != X.shape[0]:
            raise ValueError(
                f"Invalid lengths size: {lengths_arr.shape[0]}, expected {X.shape[0]}"
            )
        clipped_lengths = np.clip(lengths_arr.astype(np.int32), 0, timesteps)
        sample_weight = (np.arange(timesteps)[None, :] < clipped_lengths[:, None]).astype(np.float32)

    # ===================== 4. 构建 CNN 自编码器模型 =====================
    # 编码器：Conv -> Pool -> Conv -> Pool -> GAP -> Dense(latent)
    input_layer = Input(shape=(timesteps, n_features), name="cnn_input")

    x = Conv1D(32, kernel_size=5, padding="same", activation="relu", name="enc_conv1")(input_layer)
    x = MaxPooling1D(pool_size=2, padding="same", name="enc_pool1")(x)
    x = Conv1D(64, kernel_size=3, padding="same", activation="relu", name="enc_conv2")(x)
    x = MaxPooling1D(pool_size=2, padding="same", name="enc_pool2")(x)

    x = GlobalAveragePooling1D(name="enc_gap")(x)
    latent_features = Dense(latent_dim, activation="relu", name="embedding")(x)

    # 解码器：Dense -> Reshape -> UpSample -> Conv -> UpSample -> Conv -> 输出
    reduced_steps = _downsample_steps(timesteps, times=2)
    dec = Dense(reduced_steps * 64, activation="relu", name="dec_dense")(latent_features)
    dec = Reshape((reduced_steps, 64), name="dec_reshape")(dec)
    dec = UpSampling1D(size=2, name="dec_up1")(dec)
    dec = Conv1D(64, kernel_size=3, padding="same", activation="relu", name="dec_conv1")(dec)
    dec = UpSampling1D(size=2, name="dec_up2")(dec)
    dec = Conv1D(32, kernel_size=3, padding="same", activation="relu", name="dec_conv2")(dec)
    dec = Conv1D(n_features, kernel_size=3, padding="same", activation="linear", name="reconstruct_raw")(dec)

    # 由于池化/上采样对奇数长度会产生偏差，统一裁剪回原始 timesteps。
    output_layer = Lambda(lambda t: t[:, :timesteps, :], name="reconstruct")(dec)

    # ===================== 5. 构建模型 =====================
    cnn_autoencoder = Model(inputs=input_layer, outputs=output_layer)
    cnn_encoder_model = Model(inputs=input_layer, outputs=latent_features)

    # ===================== 6. 编译模型 =====================
    cnn_autoencoder.compile(
        optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss="mse",
    )

    # ===================== 7. 配置早停 =====================
    earliest_stop = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        mode="min",
        restore_best_weights=True,
        verbose=1,
    )

    # ===================== 8. 训练模型 =====================
    n_samples = X.shape[0]
    if n_samples < 5:
        print(f"[FeatureExtract] Warning: Too few samples ({n_samples}) for validation split. Disabling validation.")
        current_val_split = 0.0
        current_callbacks = []
    else:
        current_val_split = 0.2
        current_callbacks = [earliest_stop]

    history = cnn_autoencoder.fit(
        X,
        X,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_split=current_val_split,
        callbacks=current_callbacks,
        sample_weight=sample_weight,
    )

    # ===================== 9. 提取特征 =====================
    X_cnn_extracted_features = cnn_encoder_model.predict(X)

    # ===================== 10. 输出结果 =====================
    print(f"\n原始单特征时序数据形状: {X.shape}")
    print(f"CNN提取的特征形状: {X_cnn_extracted_features.shape}")

    training_history = {
        "loss": history.history["loss"],
        "val_loss": history.history.get("val_loss", history.history["loss"]),
        "epochs_trained": len(history.history["loss"]),
        "model_name": "CNN",
    }

    return X_cnn_extracted_features, training_history
