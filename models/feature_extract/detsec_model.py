import os
import time
import numpy as np
import tensorflow as tf
import keras
from keras import ops
from tensorflow.keras.layers import GRUCell, RNN, Dense, Layer
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from models.base_model import BaseModel

# ===================== 工具函数 =====================
def build_mask_batch(batch_seql, max_size):
    """生成批次掩码：1标记有效数据，0标记填充数据"""
    mask_batch = []
    for el in batch_seql:
        mask_batch.append(np.concatenate((np.ones(el), np.zeros(max_size - el))))
    return np.array(mask_batch, dtype=np.float32)

def get_batch(X, i, batch_size, seq_len=None):
    """按批次获取数据"""
    start_id = i * batch_size
    end_id = min((i + 1) * batch_size, X.shape[0])
    batch_x = X[start_id:end_id]
    if seq_len is not None:
        batch_seq = seq_len[start_id:end_id]
        return batch_x, batch_seq
    return batch_x

# ===================== 核心机制 =====================
def gate(vec):
    """门控函数"""
    mask = Dense(ops.shape(vec)[1], activation='sigmoid')(vec)
    return mask

class AttentionLayer(Layer):
    """注意力机制层"""
    def __init__(self, nunits, attention_size, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
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
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs shape: (batch, seq_len, nunits)
        v = ops.tanh(ops.add(ops.dot(inputs, self.W_omega), self.b_omega))
        vu = ops.dot(v, self.u_omega)
        alphas = ops.softmax(vu, axis=1)
        output = ops.sum(ops.multiply(inputs, ops.expand_dims(alphas, -1)), axis=1)
        output = ops.reshape(output, [-1, self.nunits])
        return output

# ===================== DETSEC 模型类 =====================
class DETSECModel(BaseModel):
    def __init__(self, name="DETSEC", config=None):
        super().__init__(name, config)
        self.latent_dim = self.config.get("latent_dim", 64)
        self.nunits = self.config.get("nunits", 512)
        self.attention_size = self.config.get("attention_size", 32)
        self.learning_rate = self.config.get("learning_rate", 0.0001)
        self.batch_size = self.config.get("batch_size", 16)
        self.epochs = self.config.get("epochs", 300)
        self.pretrain_epochs = self.config.get("pretrain_epochs", 50)
        self.n_clusters = self.config.get("n_clusters", 5)
        
        self.pretrain_model = None
        self.combined_model = None
        self.embedd_model = None
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def _build_ae3(self, input_t, seqL, n_dims):
        """构建 AE3 自编码器结构"""
        with tf.name_scope("AE3_Model"):
            # input_t shape: (batch, seq_len, dim)
            n_splits = input_t.shape[1]
            if n_splits is None:
                n_splits = ops.shape(input_t)[1]
            
            x_list = input_t
            x_list_bw = ops.flip(input_t, axis=1)

            # RNN 掩码
            seq_len_for_mask = ops.minimum(ops.cast(seqL, "int32"), n_splits)
            mask_indices = ops.cast(ops.arange(n_splits), "int32")[None, :]
            mask_threshold = ops.cast(seq_len_for_mask, "int32")[:, None]
            rnn_mask = ops.less(mask_indices, mask_threshold)

            # 编码器
            with tf.name_scope("encoderFWL"):
                cellEncoderFW = GRUCell(self.nunits)
                outputsEncLFW = RNN(cellEncoderFW, return_sequences=True)(x_list, mask=rnn_mask)

            with tf.name_scope("encoderBWL"):
                cellEncoderBW = GRUCell(self.nunits)
                outputsEncLBW = RNN(cellEncoderBW, return_sequences=True)(x_list_bw, mask=rnn_mask)

            # 使用 AttentionLayer
            encoder_fw = AttentionLayer(self.nunits, self.attention_size)(outputsEncLFW)
            encoder_bw = AttentionLayer(self.nunits, self.attention_size)(outputsEncLBW)
            
            # 融合编码特征并映射到 latent_dim
            combined_encoder = ops.add(
                ops.multiply(gate(encoder_fw), encoder_fw),
                ops.multiply(gate(encoder_bw), encoder_bw)
            )
            encoder = Dense(self.latent_dim, name="embedding")(combined_encoder)

            # 解码器
            x_list2decode = ops.repeat(ops.expand_dims(encoder, 1), n_splits, axis=1)
            x_list2decode_bw = ops.repeat(ops.expand_dims(encoder, 1), n_splits, axis=1)

            with tf.name_scope("decoderG"):
                cellDecoder = GRUCell(self.nunits)
                outputsDecG = RNN(cellDecoder, return_sequences=True)(x_list2decode, mask=rnn_mask)

            with tf.name_scope("decoderGFW"):
                cellDecoderBW = GRUCell(self.nunits)
                outputsDecGFW = RNN(cellDecoderBW, return_sequences=True)(x_list2decode_bw, mask=rnn_mask)

            # Dense 层可以直接处理 3D 张量
            reconstruct = Dense(n_dims)(outputsDecG)
            reconstruct2 = Dense(n_dims)(outputsDecGFW)

        return reconstruct, reconstruct2, encoder

    def _compile_models(self, input_shape):
        """编译模型"""
        input_t = tf.keras.Input(shape=input_shape[1:], name='inputs')
        seqL = tf.keras.Input(shape=(), name="seqL", dtype=tf.int32)
        
        reconstruct, reconstruct2, embedding = self._build_ae3(input_t, seqL, input_shape[-1])
        
        self.pretrain_model = tf.keras.Model(inputs=[input_t, seqL], outputs=[reconstruct, reconstruct2])
        
        b_centroids = tf.keras.Input(shape=(embedding.shape[1],), name='b_centroids')
        self.combined_model = tf.keras.Model(
            inputs=[input_t, seqL, b_centroids],
            outputs=[reconstruct, reconstruct2, embedding]
        )
        
        self.embedd_model = tf.keras.Model(inputs=[input_t, seqL], outputs=embedding)

    def _custom_recon_loss(self, y_true, y_pred, mask):
        """带掩码的重构损失"""
        # mask shape: (batch, seq_len)
        # y_true, y_pred shape: (batch, seq_len, dim)
        loss = tf.square(y_true - y_pred)
        # 将 mask 扩展到 dim 维度
        mask_expanded = tf.expand_dims(mask, -1)
        loss = loss * mask_expanded
        return tf.reduce_mean(tf.reduce_sum(loss, axis=[1, 2]))

    @tf.function
    def _pretrain_step(self, batch_data, batch_seql, mask_batch):
        with tf.GradientTape() as tape:
            recon1, recon2 = self.pretrain_model([batch_data, batch_seql], training=True)
            loss1 = self._custom_recon_loss(batch_data, recon1, mask_batch)
            loss2 = self._custom_recon_loss(batch_data, recon2, mask_batch)
            total_loss = loss1 + loss2

        gradients = tape.gradient(total_loss, self.pretrain_model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        self.optimizer.apply_gradients(zip(gradients, self.pretrain_model.trainable_variables))
        return total_loss

    @tf.function
    def _combined_step(self, batch_data, batch_seql, mask_batch, batch_centroids):
        with tf.GradientTape() as tape:
            recon1, recon2, emb = self.combined_model([batch_data, batch_seql, batch_centroids], training=True)
            recon_loss = self._custom_recon_loss(batch_data, recon1, mask_batch) + \
                         self._custom_recon_loss(batch_data, recon2, mask_batch)
            crc_loss = tf.reduce_mean(tf.reduce_sum(tf.square(emb - batch_centroids), axis=1))
            total_loss = recon_loss + 0.1 * crc_loss

        gradients = tape.gradient(total_loss, self.combined_model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        self.optimizer.apply_gradients(zip(gradients, self.combined_model.trainable_variables))
        return total_loss, recon_loss, crc_loss

    def train(self, data):
        """
        训练 DETSEC 模型
        data: dict, 包含 'X' (ndarray) 和 'lengths' (ndarray)
        """
        X = data['X'].astype(np.float32)
        seq_lengths = data['lengths'].flatten().astype(np.int32)
        
        # ===================== 数据归一化 =====================
        # 深度学习对数据量级敏感，原始数据可能高达数千（如 3999.0），不归一化极易导致梯度爆炸（NaN）
        # 使用 Min-Max 归一化将数据映射到 [0, 1] 区间
        X_min = X.min()
        X_max = X.max()
        X = (X - X_min) / (X_max - X_min + 1e-7)  # 加上小常数防止除零
        print(f"[DETSEC] 数据归一化完成 | 范围: {X.min():.2f} ~ {X.max():.2f}")

        n_samples, max_seq_len, n_dims = X.shape
        
        if self.pretrain_model is None:
            self._compile_models(X.shape)
            
        history = {'loss': [], 'val_loss': [], 'epochs_trained': 0}
        
        print(f"[DETSEC] Starting training for {self.epochs} epochs (pretrain: {self.pretrain_epochs})")
        
        for e in range(self.epochs):
            start_time = time.time()
            epoch_loss = 0.0
            epoch_crc_loss = 0.0
            
            # 打乱数据
            idx = np.random.permutation(n_samples)
            X_shuffled = X[idx]
            seq_lengths_shuffled = seq_lengths[idx]
            
            iterations = int(np.ceil(n_samples / self.batch_size))
            
            if e < self.pretrain_epochs:
                # 预训练阶段
                for i in range(iterations):
                    batch_x, batch_seq = get_batch(X_shuffled, i, self.batch_size, seq_lengths_shuffled)
                    mask_batch = build_mask_batch(batch_seq, max_seq_len)
                    loss = self._pretrain_step(batch_x, batch_seq, mask_batch)
                    epoch_loss += loss.numpy()
                avg_loss = epoch_loss / iterations
                history['loss'].append(avg_loss)
                print(f"Epoch [{e+1}/{self.epochs}] Pretrain Loss: {avg_loss:.4f} | Time: {time.time()-start_time:.2f}s")
            else:
                # 联合训练阶段
                # 提取特征进行聚类
                features = self.extract_features({'X': X, 'lengths': seq_lengths})
                kmeans = KMeans(n_clusters=self.n_clusters, n_init=10).fit(features)
                centroids = kmeans.cluster_centers_
                labels = kmeans.labels_
                
                # 重新打乱（包含标签）
                idx = np.random.permutation(n_samples)
                X_shuffled = X[idx]
                seq_lengths_shuffled = seq_lengths[idx]
                labels_shuffled = labels[idx]
                
                for i in range(iterations):
                    batch_x, batch_seq = get_batch(X_shuffled, i, self.batch_size, seq_lengths_shuffled)
                    batch_labels = labels_shuffled[i*self.batch_size : (i+1)*self.batch_size]
                    batch_centroids = centroids[batch_labels]
                    mask_batch = build_mask_batch(batch_seq, max_seq_len)
                    
                    total_loss, recon_loss, crc_loss = self._combined_step(batch_x, batch_seq, mask_batch, batch_centroids)
                    epoch_loss += recon_loss.numpy()
                    epoch_crc_loss += crc_loss.numpy()
                
                avg_recon_loss = epoch_loss / iterations
                avg_crc_loss = epoch_crc_loss / iterations
                history['loss'].append(avg_recon_loss)
                print(f"Epoch [{e+1}/{self.epochs}] Recon Loss: {avg_recon_loss:.4f} | Clust Loss: {avg_crc_loss:.4f} | Time: {time.time()-start_time:.2f}s")

        history['epochs_trained'] = self.epochs
        # DETSEC 目前没有独立的验证集逻辑，这里简单复制 loss
        history['val_loss'] = history['loss']
        return history

    def extract_features(self, data):
        """提取特征"""
        X = data['X'].astype(np.float32)
        seq_lengths = data['lengths'].flatten().astype(np.int32)
        
        # ===================== 数据归一化 =====================
        # 必须与训练时的归一化逻辑保持一致
        X_min = X.min()
        X_max = X.max()
        X = (X - X_min) / (X_max - X_min + 1e-7)

        if self.embedd_model is None:
            self._compile_models(X.shape)
            
        features = self.embedd_model.predict([X, seq_lengths], batch_size=self.batch_size, verbose=0)
        return features

    def save(self, path: str):
        """保存模型"""
        if not os.path.exists(path):
            os.makedirs(path)
        if self.embedd_model:
            self.embedd_model.save(os.path.join(path, "detsec_embedd.h5"))
        if self.pretrain_model:
            self.pretrain_model.save(os.path.join(path, "detsec_pretrain.h5"))

    def load(self, path: str):
        """加载模型"""
        # 注意：由于自定义层和结构，直接加载可能需要 custom_objects
        # 这里简单实现
        embedd_path = os.path.join(path, "detsec_embedd.h5")
        if os.path.exists(embedd_path):
            self.embedd_model = tf.keras.models.load_model(embedd_path, compile=False)

def detsec_ae(data, config):
    """
    DETSEC 特征提取包装函数，适配 FeatureExtractStep
    """
    # 将 3D 数据和 lengths 包装进字典
    # 假设 data 是 3D tensor
    input_data = {
        'X': data,
        'lengths': config.get('lengths', np.array([data.shape[1]] * data.shape[0]))
    }
    
    model = DETSECModel(config=config)
    history = model.train(input_data)
    features = model.extract_features(input_data)
    
    return features, history
