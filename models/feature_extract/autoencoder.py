import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from models.base_model import BaseModel

class AutoEncoderModel(BaseModel):
    """
    普通自编码器特征提取类
    
    该类使用全连接层自编码器从时序数据中提取全局特征。
    继承自 BaseModel 以符合项目开发规范。
    """
    def __init__(self, name="AutoEncoder", config=None):
        super().__init__(name, config)
        self.latent_dim = self.config.get("latent_dim", 64)
        self.epochs = self.config.get("epochs", 50)
        self.batch_size = self.config.get("batch_size", 32)
        self.learning_rate = self.config.get("learning_rate", 0.001)
        self.patience = self.config.get("patience", 5)
        
        self.autoencoder_model = None
        self.encoder_model = None

    def _build_model(self, timesteps, n_features):
        """构建模型结构"""
        input_layer = Input(shape=(timesteps, n_features))
        
        # 编码器部分
        flatten = Flatten()(input_layer)
        encoder1 = Dense(128, activation='relu')(flatten)
        encoder2 = Dense(64, activation='relu')(encoder1)
        latent_features = Dense(self.latent_dim, activation='relu')(encoder2)
        
        # 解码器部分
        decoder1 = Dense(64, activation='relu')(latent_features)
        decoder2 = Dense(128, activation='relu')(decoder1)
        decoder3 = Dense(timesteps * n_features, activation='linear')(decoder2)
        output_layer = Reshape((timesteps, n_features))(decoder3)
        
        self.autoencoder_model = Model(inputs=input_layer, outputs=output_layer)
        self.encoder_model = Model(inputs=input_layer, outputs=latent_features)
        
        self.autoencoder_model.compile(
            optimizer=Adam(learning_rate=self.learning_rate, clipnorm=1.0), 
            loss='mse'
        )

    def train(self, data):
        """
        训练自编码器模型
        data: np.ndarray, 形状为 (n_samples, timesteps, n_features)
        """
        X = data
        if isinstance(data, dict):
            X = data.get('X')
            
        n_samples, timesteps, n_features = X.shape
        
        # 数据归一化
        X_min = X.min()
        X_max = X.max()
        X_norm = (X - X_min) / (X_max - X_min + 1e-7)
        
        if self.autoencoder_model is None:
            self._build_model(timesteps, n_features)
            
        earliest_stop = EarlyStopping(
            monitor='val_loss',
            patience=self.patience,
            mode='min',
            restore_best_weights=True,
            verbose=1
        )
        
        history = self.autoencoder_model.fit(
            X_norm, X_norm,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True,
            validation_split=0.2,
            callbacks=[earliest_stop]
        )
        
        training_history = {
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss'],
            'epochs_trained': len(history.history['loss']),
            'model_name': self.name
        }
        return training_history

    def extract_features(self, data):
        """使用编码器提取特征"""
        X = data
        if isinstance(data, dict):
            X = data.get('X')
            
        # 归一化（应使用训练时的参数，这里简化处理）
        X_min = X.min()
        X_max = X.max()
        X_norm = (X - X_min) / (X_max - X_min + 1e-7)
        
        if self.encoder_model is None:
            raise ValueError("Model must be trained or loaded before extracting features.")
            
        return self.encoder_model.predict(X_norm)

    def save(self, path: str):
        """保存模型"""
        if not os.path.exists(path):
            os.makedirs(path)
        if self.autoencoder_model:
            self.autoencoder_model.save(os.path.join(path, f"{self.name}_autoencoder.h5"))
        if self.encoder_model:
            self.encoder_model.save(os.path.join(path, f"{self.name}_encoder.h5"))

    def load(self, path: str):
        """加载模型"""
        encoder_path = os.path.join(path, f"{self.name}_encoder.h5")
        if os.path.exists(encoder_path):
            self.encoder_model = tf.keras.models.load_model(encoder_path, compile=False)
        
        ae_path = os.path.join(path, f"{self.name}_autoencoder.h5")
        if os.path.exists(ae_path):
            self.autoencoder_model = tf.keras.models.load_model(ae_path, compile=True)

def autoencoder(data: np.ndarray, config: dict):
    """
    自编码器特征提取包装函数，适配 FeatureExtractStep
    """
    model = AutoEncoderModel(config=config)
    training_history = model.train(data)
    features = model.extract_features(data)
    return features, training_history
