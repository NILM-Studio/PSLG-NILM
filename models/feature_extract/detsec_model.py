import os
import numpy as np
import tensorflow as tf
import keras
from keras import ops
from tensorflow.keras.layers import (
    Input, Dense, GRU, Bidirectional, RepeatVector, TimeDistributed,
    Add, Multiply, Layer,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from models.base_model import BaseModel

# ===================== 核心机制 =====================
def gate(vec):
    """门控函数：对特征维做 sigmoid 自门控。vec 形状 (batch, nunits)，nunits 为静态整数。"""
    # 注意：必须传 Python int 作为 units（Keras 3 不接受符号张量）。
    return Dense(vec.shape[-1], activation='sigmoid')(vec)


class AttentionLayer(Layer):
    """DETSEC 注意力层：沿时间轴加权汇聚序列表示 -> (batch, nunits)。"""

    def __init__(self, nunits, attention_size, **kwargs):
        super().__init__(**kwargs)
        self.nunits = nunits
        self.attention_size = attention_size

    def build(self, input_shape):
        self.W_omega = self.add_weight(
            name="W_omega",
            shape=(self.nunits, self.attention_size),
            initializer=tf.keras.initializers.GlorotNormal(),
            trainable=True,
        )
        self.b_omega = self.add_weight(
            name="b_omega",
            shape=(self.attention_size,),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
            trainable=True,
        )
        self.u_omega = self.add_weight(
            name="u_omega",
            shape=(self.attention_size,),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        # inputs shape: (batch, seq_len, nunits)
        v = ops.tanh(ops.add(ops.dot(inputs, self.W_omega), self.b_omega))
        vu = ops.dot(v, self.u_omega)
        alphas = ops.softmax(vu, axis=1)
        output = ops.sum(ops.multiply(inputs, ops.expand_dims(alphas, -1)), axis=1)
        output = ops.reshape(output, [-1, self.nunits])
        return output

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"nunits": self.nunits, "attention_size": self.attention_size})
        return cfg


# ===================== DETSEC 模型类 =====================
class DETSECModel(BaseModel):
    """双向 GRU + 门控注意力自编码器（纯特征提取器，走 CuDNN 融合核）。

    设计与 bilstm_ae / bilstm_ae_attention 对齐：
      - 训练用 model.fit() + sample_weight 做变长掩码（损失侧掩码，
        前向融合核不受影响）。
      - GRU 层保持 CuDNN 可用约束（activation='tanh',
        recurrent_activation='sigmoid', recurrent_dropout=0.0）；
        任一约束被破坏，TF 会静默回退到慢速核。
    """

    def __init__(self, name="DETSEC", config=None):
        super().__init__(name, config)
        self.latent_dim = self.config.get("latent_dim", 64)
        self.nunits = self.config.get("nunits", 128)
        self.attention_size = self.config.get("attention_size", 32)
        self.learning_rate = self.config.get("learning_rate", 0.0001)
        self.batch_size = self.config.get("batch_size", 32)
        self.epochs = self.config.get("epochs", 50)
        self.patience = self.config.get("patience", 5)

        self.autoencoder_model = None  # 训练用 (输入 -> 重构)
        self.encoder_model = None      # 推断用 (输入 -> embedding)
        self._X_min = None             # 归一化参数，extract 时复用
        self._X_max = None

    def _build_model(self, timesteps, n_features):
        """构建双向 GRU 门控注意力自编码器（单一输入）。"""
        inp = Input(shape=(timesteps, n_features), name="inputs")

        # ---- 编码器：双向 GRU，融合 CuDNN，保留前后向两半 ----
        enc = Bidirectional(
            GRU(
                self.nunits,
                activation='tanh',
                recurrent_activation='sigmoid',
                recurrent_dropout=0.0,
                use_bias=True,
                return_sequences=True,
            ),
            merge_mode=None,  # 返回列表 [fw_seq, bw_seq]，各自 (b, T, nunits)
            name="encoder_biGRU",
        )(inp)
        fw_seq, bw_seq = enc

        # ---- 各方向 DETSEC 注意力（汇聚时间轴）----
        encoder_fw = AttentionLayer(self.nunits, self.attention_size, name="attn_fw")(fw_seq)
        encoder_bw = AttentionLayer(self.nunits, self.attention_size, name="attn_bw")(bw_seq)

        # ---- 门控融合（保持 g⊙e_fw + g⊙e_bw 语义）----
        fused = Add(name="gated_fuse")([
            Multiply()([gate(encoder_fw), encoder_fw]),
            Multiply()([gate(encoder_bw), encoder_bw]),
        ])
        embedding = Dense(self.latent_dim, activation='relu', name="embedding")(fused)

        # ---- 解码器：单一双向 GRU + TimeDistributed Dense ----
        dec_in = RepeatVector(timesteps, name="repeat")(embedding)
        dec = Bidirectional(
            GRU(
                self.nunits,
                activation='tanh',
                recurrent_activation='sigmoid',
                recurrent_dropout=0.0,
                use_bias=True,
                return_sequences=True,
            ),
            merge_mode='concat',
            name="decoder_biGRU",
        )(dec_in)
        recon = TimeDistributed(Dense(n_features, activation='linear'), name="recon")(dec)

        self.autoencoder_model = Model(inp, recon, name="DETSEC_ae")
        self.encoder_model = Model(inp, embedding, name="DETSEC_encoder")
        self.autoencoder_model.compile(
            optimizer=Adam(learning_rate=self.learning_rate, clipnorm=1.0),
            loss='mse',
        )

    def _normalize(self, X):
        X_min, X_max = X.min(), X.max()
        return (X - X_min) / (X_max - X_min + 1e-7), X_min, X_max

    def train(self, data):
        """训练 DETSEC 自编码器。data 可为 ndarray 或 {'X': ndarray}。"""
        X = data['X'] if isinstance(data, dict) else data
        X = X.astype(np.float32)
        n_samples, timesteps, n_features = X.shape

        # ---- MinMax 归一化到 [0,1]（参数在 extract 时复用）----
        X_norm, self._X_min, self._X_max = self._normalize(X)

        # ---- 由 lengths 构造 sample_weight（镜像 bilstm_ae）----
        lengths = self.config.get("lengths", None)
        if lengths is None:
            sample_weight = np.ones((n_samples, timesteps), dtype=np.float32)
        else:
            lengths_arr = np.asarray(lengths).reshape(-1)
            if lengths_arr.shape[0] != n_samples:
                raise ValueError(
                    f"lengths 大小 {lengths_arr.shape[0]} != 样本数 {n_samples}"
                )
            clipped = np.clip(lengths_arr.astype(np.int32), 0, timesteps)
            sample_weight = (
                np.arange(timesteps)[None, :] < clipped[:, None]
            ).astype(np.float32)

        if self.autoencoder_model is None:
            self._build_model(timesteps, n_features)

        # ---- EarlyStopping + 验证集，含小样本保护 ----
        if n_samples < 5:
            print(f"[DETSEC] 样本过少 ({n_samples})，关闭验证/早停。")
            val_split, callbacks = 0.0, []
        else:
            val_split = 0.2
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.patience,
                    mode='min',
                    restore_best_weights=True,
                    verbose=1,
                )
            ]

        history = self.autoencoder_model.fit(
            X_norm, X_norm,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True,
            validation_split=val_split,
            callbacks=callbacks,
            sample_weight=sample_weight,
        )

        return {
            'loss': history.history['loss'],
            'val_loss': history.history.get('val_loss', history.history['loss']),
            'epochs_trained': len(history.history['loss']),
            'model_name': self.name,
        }

    def extract_features(self, data):
        """提取 embedding 特征。data 可为 ndarray 或 {'X': ndarray}。"""
        X = data['X'] if isinstance(data, dict) else data
        X = X.astype(np.float32)
        if self._X_min is None or self._X_max is None:
            # load 路径或异常调用：就地重算归一化参数
            X_norm, self._X_min, self._X_max = self._normalize(X)
        else:
            X_norm = (X - self._X_min) / (self._X_max - self._X_min + 1e-7)
        if self.encoder_model is None:
            raise ValueError("DETSECModel: 需先 train() 或 load() 再 extract_features()。")
        return self.encoder_model.predict(X_norm, batch_size=self.batch_size, verbose=0)

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        if self.autoencoder_model:
            self.autoencoder_model.save(os.path.join(path, "detsec_autoencoder.keras"))
        if self.encoder_model:
            self.encoder_model.save(os.path.join(path, "detsec_encoder.keras"))

    def load(self, path: str):
        custom_objects = {"AttentionLayer": AttentionLayer}
        ep = os.path.join(path, "detsec_encoder.keras")
        if os.path.exists(ep):
            self.encoder_model = tf.keras.models.load_model(
                ep, compile=False, custom_objects=custom_objects
            )
        ap = os.path.join(path, "detsec_autoencoder.keras")
        if os.path.exists(ap):
            self.autoencoder_model = tf.keras.models.load_model(
                ap, compile=False, custom_objects=custom_objects
            )


def detsec_ae(np_data, model_config):
    """DETSEC 特征提取包装函数，适配 FeatureExtractStep。

    签名：(ndarray (n, timesteps, dim), dict) -> (features (n, latent_dim), history dict)
    model_config 可能携带 'lengths'（由 feature_extract_step 注入）。
    """
    model = DETSECModel(config=model_config)
    history = model.train(np_data)
    features = model.extract_features(np_data)
    return features, history
