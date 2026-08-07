import sys
import os
import numpy as np
import math
from operator import itemgetter, attrgetter, methodcaller
import tensorflow as tf
from tensorflow.keras.layers import GRUCell, RNN, Dropout
import random
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from scipy.spatial import distance
import time
import calendar
import random as rand


# ===================== 工具函数：生成掩码/批次数据 =====================
def buildMaskBatch(batch_seql, max_size):
    """生成批次掩码：1标记有效数据，0标记填充数据"""
    mask_batch = []
    for el in batch_seql:
        mask_batch.append(np.concatenate((np.ones(el), np.zeros(max_size - el))))
    return np.array(mask_batch, dtype=np.float32)  # 强制float32，避免类型冲突


def getBatch(X, Y, i, batch_size):
    """按批次获取数据"""
    start_id = i * batch_size
    end_id = min((i + 1) * batch_size, X.shape[0])
    batch_x = X[start_id:end_id]
    batch_y = Y[start_id:end_id]
    return batch_x, batch_y


# ===================== 核心函数：门控/注意力机制 =====================
def gate(vec):
    """门控函数：生成门控掩码"""
    mask = tf.keras.layers.Dense(vec.shape[1], activation=tf.sigmoid)(vec)
    return mask


def gating(outputs_list, mask):
    """门控机制：过滤无效输出"""
    gating_results = None
    if mask is None:
        for i in range(len(outputs_list)):
            val = outputs_list[i]
            multiplication = val * gate(val)
            if gating_results is None:
                gating_results = multiplication
            else:
                gating_results = gating_results + multiplication
        return gating_results

    for i in range(len(outputs_list)):
        val = outputs_list[i]
        multiplication = val * gate(val)
        multiplication = tf.transpose(multiplication)
        multiplication = multiplication * mask[:, i]
        multiplication = tf.transpose(multiplication)
        if gating_results is None:
            gating_results = multiplication
        else:
            gating_results = gating_results + multiplication

    return gating_results


def attention(outputs_list, nunits, attention_size):
    """注意力机制：计算序列注意力权重"""
    outputs = tf.stack(outputs_list, axis=1)
    # 随机初始化注意力参数（TF2.x 推荐用 tf.random.normal）
    W_omega = tf.Variable(tf.random.normal([nunits, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random.normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random.normal([attention_size], stddev=0.1))

    v = tf.tanh(tf.tensordot(outputs, W_omega, axes=1) + b_omega)
    vu = tf.tensordot(v, u_omega, axes=1)
    alphas = tf.nn.softmax(vu)

    output = tf.reduce_sum(outputs * tf.expand_dims(alphas, -1), 1)
    output = tf.reshape(output, [-1, nunits])
    return output


# ===================== 核心模型：AE3 自编码器（移除mask输入） =====================
def AE3(input_t, seqL, n_dims):
    with tf.name_scope("AE3_Model"):
        n_splits = input_t.shape[1] // n_dims
        x_list = tf.split(input_t, n_splits, axis=1)
        x_list_bw = tf.stack(x_list[::-1], axis=1)
        x_list = tf.stack(x_list, axis=1)

        nunits = 512

        # ========== 核心修改：修正RNN掩码 ==========
        # 1. 计算序列长度（限制为n_splits，避免超过RNN输入长度）
        seq_len_for_mask = tf.math.minimum(seqL // n_dims, n_splits)
        # 2. 生成掩码（长度=min(seqL//n_dims, n_splits)）
        rnn_mask = tf.sequence_mask(seq_len_for_mask, maxlen=n_splits)

        # 编码器：前向GRU（使用修正后的掩码）
        with tf.name_scope("encoderFWL"):
            cellEncoderFW = GRUCell(nunits)
            outputsEncLFW = RNN(cellEncoderFW, return_sequences=True)(
                x_list, mask=rnn_mask
            )

        # 编码器：反向GRU（同样使用修正后的掩码）
        with tf.name_scope("encoderBWL"):
            cellEncoderBW = GRUCell(nunits)
            outputsEncLBW = RNN(cellEncoderBW, return_sequences=True)(
                x_list_bw, mask=rnn_mask
            )

        # ========== 其余代码不变 ==========
        final_list_fw = [outputsEncLFW[:, i, :] for i in range(n_splits)]
        final_list_bw = [outputsEncLBW[:, i, :] for i in range(n_splits)]
        encoder_fw = attention(final_list_fw, nunits, nunits)
        encoder_bw = attention(final_list_bw, nunits, nunits)
        encoder = gate(encoder_fw) * encoder_fw + gate(encoder_bw) * encoder_bw

        x_list2decode = [tf.identity(encoder) for _ in range(n_splits)]
        x_list2decode_bw = [tf.identity(encoder) for _ in range(n_splits)]
        x_list2decode = tf.stack(x_list2decode, axis=1)
        x_list2decode_bw = tf.stack(x_list2decode_bw, axis=1)

        # 解码器：同样使用修正后的掩码
        with tf.name_scope("decoderG"):
            cellDecoder = GRUCell(nunits)
            outputsDecG = RNN(cellDecoder, return_sequences=True)(
                x_list2decode, mask=rnn_mask
            )

        with tf.name_scope("decoderGFW"):
            cellDecoder = GRUCell(nunits)
            outputsDecGFW = RNN(cellDecoder, return_sequences=True)(
                x_list2decode_bw, mask=rnn_mask
            )

        out_list = []
        out_list_bw = []
        for i in range(n_splits):
            temp_cell = outputsDecG[:, i, :]
            tt = tf.keras.layers.Dense(n_dims, activation=None)(temp_cell)
            out_list.append(tt)

            temp_cell2 = outputsDecGFW[:, i, :]
            tt2 = tf.keras.layers.Dense(n_dims, activation=None)(temp_cell2)
            out_list_bw.append(tt2)

        reconstruct = tf.concat(out_list, axis=1)
        reconstruct2 = tf.concat(out_list_bw[::1], axis=1)

    return reconstruct, reconstruct2, encoder


# ===================== 特征提取函数：适配TF2.x模型推理 =====================
def extractFeatures(ts_data, seq_length, mask_val, model):
    """提取嵌入特征（替代TF1.x的sess.run）"""
    batchsz = 1024
    iterations = int(ts_data.shape[0] / batchsz)
    if ts_data.shape[0] % batchsz != 0:
        iterations += 1
    features = None

    for ibatch in range(iterations):
        batch_data, batch_seqL = getBatch(ts_data, seq_length, ibatch, batchsz)
        batch_mask, _ = getBatch(mask_val, mask_val, ibatch, batchsz)
        # TF2.x 用model.predict替代sess.run，直接获取嵌入特征
        partial_features = model.predict([batch_data, batch_seqL], verbose=0)
        if features is None:
            features = partial_features
        else:
            features = np.vstack((features, partial_features))

        del batch_data, batch_seqL, batch_mask
    return features


# ===================== 自定义训练步骤：解决mask符号张量冲突 =====================
def define_train_functions(pretrain_model, combined_model, n_feat, optimizer):
    """定义预训练/联合训练步骤（核心：直接传入mask数值）"""

    # 重构损失函数（mask为数值张量）
    def custom_recon_loss(y_true, y_pred, mask):
        loss = tf.square((y_true - y_pred) * mask)
        return tf.reduce_sum(loss, axis=1)

    # 预训练步骤（仅重构损失）
    @tf.function  # 静态图加速
    def pretrain_step(batch_data, batch_seql, mask_batch):
        with tf.GradientTape() as tape:
            recon1, recon2 = pretrain_model([batch_data, batch_seql], training=True)
            loss1 = tf.reduce_mean(custom_recon_loss(batch_data, recon1, mask_batch))
            loss2 = tf.reduce_mean(custom_recon_loss(batch_data, recon2, mask_batch))
            total_loss = loss1 + loss2

        gradients = tape.gradient(total_loss, pretrain_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, pretrain_model.trainable_variables))
        return total_loss

    # 联合训练步骤（重构+聚类损失）
    @tf.function
    def combined_step(batch_data, batch_seql, mask_batch, batch_centroids):
        with tf.GradientTape() as tape:
            recon1, recon2, emb = combined_model([batch_data, batch_seql, batch_centroids], training=True)
            # 重构损失
            recon_loss = tf.reduce_mean(custom_recon_loss(batch_data, recon1, mask_batch)) + \
                         tf.reduce_mean(custom_recon_loss(batch_data, recon2, mask_batch))
            # 聚类损失
            crc_loss = tf.reduce_mean(tf.reduce_sum(tf.square(emb - batch_centroids), axis=1))
            total_loss = recon_loss + 0.1 * crc_loss  # 聚类损失权重

        gradients = tape.gradient(total_loss, combined_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, combined_model.trainable_variables))
        return total_loss, recon_loss, crc_loss

    return pretrain_step, combined_step, custom_recon_loss


# ===================== 主程序：完整训练逻辑 =====================
if __name__ == "__main__":
    # 0. 基础配置（可根据需求调整）
    tf.random.set_seed(0)
    np.random.seed(0)
    random.seed(0)

    # 1. 命令行参数解析（运行时需传入：数据目录 n_dims n_clusters）
    if len(sys.argv) == 4:
        dirName = sys.argv[1]
        n_dims = int(sys.argv[2])
        n_clusters = int(sys.argv[3])
    else:
        print("使用默认参数运行...")
        dirName = r'./cluster_data'
        n_dims = 1
        n_clusters = 5

    # 2. 加载数据（确保目录下有data.npy和seq_length.npy）
    try:
        dataFileName = os.path.join(dirName, "data.npy")
        seqLFileName = os.path.join(dirName, "seq_length.npy")
        data = np.load(dataFileName)
        if len(data.shape) == 3 and data.shape[-1] == 1:
            data = np.squeeze(data, axis=-1)
        data = data.astype(np.float32)
        seqLength = np.load(seqLFileName)
    except FileNotFoundError as e:
        print(f"数据文件缺失：{e}")
        print("请确保指定目录下有 data.npy（特征数据）和 seq_length.npy（序列长度）")
        sys.exit(1)

    orig_data = data.copy()
    orig_seqLength = seqLength.copy()
    n_feat = data.shape[1]
    max_length = data.shape[1]
    batchsz = 16
    hm_epochs = 300
    th = 50  # 预训练轮数

    # 3. 构建TF2.x模型（移除mask Input层）
    input_t = tf.keras.Input(shape=(n_feat,), name='inputs')
    seqL = tf.keras.Input(shape=(), name="seqL", dtype=tf.int32)
    # AE3模型：仅输入input_t + seqL，无mask
    reconstruction, reconstruction2, embedding = AE3(input_t, seqL, n_dims)

    # 预训练模型（仅重构输出）
    pretrain_model = tf.keras.Model(inputs=[input_t, seqL], outputs=[reconstruction, reconstruction2])
    # 联合训练模型（含聚类损失）
    b_centroids = tf.keras.Input(shape=(embedding.shape[1],), name='b_centroids')
    combined_model = tf.keras.Model(
        inputs=[input_t, seqL, b_centroids],
        outputs=[reconstruction, reconstruction2, embedding]
    )
    # 嵌入特征提取模型（推理用）
    embedd_model = tf.keras.Model(inputs=[input_t, seqL], outputs=embedding)

    # 4. 定义优化器和训练函数
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    pretrain_step, combined_step, _ = define_train_functions(pretrain_model, combined_model, n_feat, optimizer)

    # 5. 训练循环（预训练+联合训练）
    print("开始训练...")
    for e in range(hm_epochs):
        start_time = time.time()
        data, seqLength = shuffle(data, seqLength, random_state=e)  # 每轮打乱数据
        costT = 0.0  # 重构损失累计
        costT2 = 0.0  # 聚类损失累计
        iterations = int(data.shape[0] / batchsz)
        if data.shape[0] % batchsz != 0:
            iterations += 1

        # 阶段1：预训练（仅重构损失）
        if e < th:
            for ibatch in range(iterations):
                batch_data, batch_seql = getBatch(data, seqLength, ibatch, batchsz)
                mask_batch = buildMaskBatch(batch_seql, batch_data.shape[1])
                # 类型转换：匹配模型输入类型
                batch_seql = batch_seql.astype(np.int32)
                batch_data = batch_data.astype(np.float32)

                # 调用自定义预训练步骤（直接传入mask数值）
                loss = pretrain_step(batch_data, batch_seql, mask_batch)
                costT += loss.numpy()  # 转为numpy数值累计

                del batch_data, batch_seql, mask_batch

        # 阶段2：联合训练（重构+聚类损失）
        else:
            # 提取当前嵌入特征用于KMeans聚类
            mask_val = buildMaskBatch(seqLength, max_length)
            features = extractFeatures(data, seqLength, mask_val, embedd_model)

            # KMeans聚类
            kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=rand.randint(1, 10000000)).fit(features)
            new_centroids = kmeans.cluster_centers_
            kmeans_labels = kmeans.labels_
            data, seqLength, kmeans_labels = shuffle(data, seqLength, kmeans_labels, random_state=e)

            # 批次训练
            for ibatch in range(iterations):
                batch_data, batch_seql = getBatch(data, seqLength, ibatch, batchsz)
                batch_mask = buildMaskBatch(batch_seql, batch_data.shape[1])
                batch_km_labels, _ = getBatch(kmeans_labels, kmeans_labels, ibatch, batchsz)
                batch_centroids = np.array([new_centroids[el] for el in batch_km_labels], dtype=np.float32)

                # 类型转换
                batch_seql = batch_seql.astype(np.int32)

                # 调用联合训练步骤
                total_loss, recon_loss, crc_loss = combined_step(batch_data, batch_seql, batch_mask, batch_centroids)
                costT += recon_loss.numpy()
                costT2 += crc_loss.numpy()

                del batch_data, batch_seql, batch_mask, batch_centroids

        # 打印训练日志
        epoch_time = time.time() - start_time
        print(
            f"Epoch [{e + 1}/{hm_epochs}] | Recon Loss: {costT / iterations:.4f} | Clust Loss: {costT2 / iterations:.4f} | Time: {epoch_time:.2f}s")

    # 6. 结果保存
    output_dir = os.path.join(dirName, "detsec512_results")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 提取最终嵌入特征
    mask_val = buildMaskBatch(orig_seqLength, max_length)
    final_embedd = extractFeatures(orig_data, orig_seqLength, mask_val, embedd_model)
    # 最终聚类
    kmeans_final = KMeans(n_clusters=n_clusters, random_state=0).fit(final_embedd)

    # 保存结果
    np.save(os.path.join(output_dir, "detsec_features.npy"), final_embedd)
    np.save(os.path.join(output_dir, "detsec_clust_assignment.npy"), kmeans_final.labels_)
    print(f"训练完成！结果已保存至：{output_dir}")