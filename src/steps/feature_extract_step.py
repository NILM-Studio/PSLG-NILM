
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from src.framework.step import Step
from models.feature_extract.cnn_ae import cnn_ae
from models.feature_extract.lstm_ae import lstm_ae
from models.feature_extract.bilstm_ae import bilstm_ae
from models.feature_extract.bilstm_ae_attention import bilstm_ae_attention
from models.feature_extract.detsec_model import detsec_ae


class FeatureExtractStep(Step):
    """
    Step for extracting features from time series data using autoencoders.
    
    Supports three feature extraction methods:
    1. LSTM Autoencoder (lstm_ae): Extracts global latent space features
    2. BiLSTM Autoencoder (bilstm_ae): Extracts global latent space features (bidirectional)
    3. CNN Autoencoder (cnn_ae): Extracts global latent space features via Conv1D encoder
    4. BiLSTM + Attention Autoencoder (bilstm_ae_attention): Extracts time-step level features
    
    Data input priority:
    1. data_path / seq_len_path (if specified)
    2. context['data'] with tensor key 'X' (if available)
    3. Raise error (if neither is available)

    Input contract:
    - External file input from data_path must be a 3D tensor with shape (num, len, dim)
    - Context input expects context['data']['X'] with the same 3D shape

    Output contract:
    - context['features'] is a numpy.ndarray (not a dict)
    """
    def __init__(
        self,
        name="FeatureExtract",
        model_name="bilstm_ae",
        latent_dim=64,
        epochs=50,
        batch_size=32,
        learning_rate=0.001,
        patience=5,
        attention_size=32,
        column_name='cleaned_power',
        save_to_file=True,
        appliance_name="",
        data_path="",
        seq_len_path=""
    ):
        super().__init__(name)
        self.model_name = model_name
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.attention_size = attention_size
        self.column_name = column_name
        self.save_to_file = save_to_file
        self.appliance_name = appliance_name
        self.data_path = data_path
        self.seq_len_path = seq_len_path

    def restore(self, context: dict) -> dict:
        log_dir = self.get_log_dir(context)
        feature_dir = os.path.join(log_dir, 'extracted_features')
        if not os.path.isdir(feature_dir):
            return context

        candidates = [
            os.path.join(feature_dir, f)
            for f in os.listdir(feature_dir)
            if f.lower().endswith('.npy')
        ]
        if not candidates:
            return context

        candidates.sort(key=lambda p: os.path.getmtime(p))
        feature_path = candidates[-1]
        extracted_features = np.load(feature_path)
        context['features'] = extracted_features
        context['feature_extract_config'] = {
            'model_name': self.model_name,
            'latent_dim': self.latent_dim,
            'restored_from': feature_path,
            'output_feature_shape': tuple(extracted_features.shape),
        }
        return context

    def visualize_training_history(self, training_history, model_name, result_dir, file_name):
        """
        Visualize training history and save as JSON file
        
        Args:
            training_history (dict): Training history information
            model_name (str): Model name
            result_dir (str): Result save directory
            file_name (str): File name prefix
        """
        plt.figure(figsize=(10, 6))
        
        plt.plot(training_history['loss'], label='Training Loss')
        plt.plot(training_history['val_loss'], label='Validation Loss')
        
        plt.title(f'Training History - {model_name} ({file_name})')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True)
        
        # Save visualization results
        save_dir = os.path.join(result_dir, 'training_history')
        os.makedirs(save_dir, exist_ok=True)
        
        # Save PNG file
        save_path = os.path.join(save_dir, f'{model_name}_{file_name}_training_history.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[FeatureExtract] Training history visualization saved to: {save_path}")
        
        plt.close()
        
        # Save training history as JSON file
        json_save_path = os.path.join(save_dir, f'{model_name}_{file_name}_training_history.json')
        with open(json_save_path, 'w', encoding='utf-8') as f:
            json.dump(training_history, f, indent=4, ensure_ascii=False)
        print(f"[FeatureExtract] Training history JSON saved to: {json_save_path}")

    def _load_file(self, file_path):
        """通用文件加载方法"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.npy':
            return np.load(file_path)
        elif ext == '.csv':
            import pandas as pd
            return pd.read_csv(file_path).values
        elif ext == '.txt':
            return np.loadtxt(file_path)
        else:
            raise ValueError(f"Unsupported format: {ext}. Supported: .npy, .csv, .txt")
    
    def load_data_from_file(self, file_path, lengths_file_path=None):
        """
        Load data from specified file path.
        
        Args:
            file_path (str): Path to the data file (supports .npy, .csv, .txt)
            lengths_file_path (str, optional): Path to the lengths file
            
        Returns:
            dict: Loaded data dictionary with 'X' and 'lengths'
        """
        data = self._load_file(file_path)

        # Enforce unified tensor contract: (num, len, dim)
        if len(data.shape) != 3:
            raise ValueError(
                f"Invalid input shape: {data.shape}. "
                "External data_path must be a 3D tensor with shape (num, len, dim)."
            )

        X = data
        default_len = data.shape[1]
        
        # 加载或生成 lengths
        if lengths_file_path:
            lengths = self._load_file(lengths_file_path)
            lengths = lengths.reshape(-1, 1) if len(lengths.shape) == 1 else lengths
        else:
            lengths = np.array([default_len] * X.shape[0]).reshape(-1, 1)
        
        return {'X': X, 'lengths': lengths}
    
    def run(self, context: dict) -> dict:
        """
        Extract features from time series data using the selected autoencoder model.
        
        Data input priority:
        1. data_path / seq_len_path (if specified and exists)
        2. context['data']['X'] (if available)
        3. Raise error (if neither is available)
        
        Args:
            context (dict): Shared context containing data.
            
        Returns:
            dict: Updated context with extracted features.
            
        Raises:
            ValueError: If no data source is available
        """
        log_dir = self.get_log_dir(context)
        
        print("="*70)
        print("[FeatureExtract] Starting feature extraction step")
        print("="*70)
        
        # Print TensorFlow and GPU information
        print("[FeatureExtract] TensorFlow version:", tf.__version__)
        print("[FeatureExtract] GPU detected:", tf.config.list_physical_devices('GPU'))
        
        # Data input priority logic
        data = None
        data_source = None
        
        # Priority 1: data_path (with optional seq_len_path)
        if self.data_path:
            print(f"[FeatureExtract] Using specified input file: {self.data_path}")
            if self.seq_len_path:
                print(f"[FeatureExtract] Using specified lengths file: {self.seq_len_path}")
            try:
                data = self.load_data_from_file(self.data_path, self.seq_len_path)
                data_source = "file"
                print("[FeatureExtract] Data loaded successfully from file")
            except Exception as e:
                print(f"[FeatureExtract] Failed to load data from file: {e}")
                raise
        
        # Priority 2: context['data']['X']
        if data is None or not data:
            context_data = context.get('data', {})
            if isinstance(context_data, dict) and 'X' in context_data:
                data = {
                    'X': context_data['X'],
                    'lengths': context_data.get('lengths', None)
                }
                data_source = "context"
            else:
                data = {}
        
        # Priority 3: Raise error if neither is available
        if not data:
            raise ValueError("[FeatureExtract] Error: No data source available. "
                           "Please specify data_path or ensure context['data']['X'] contains a valid 3D tensor.")
        
        # Prepare model config
        model_config = {
            "latent_dim": self.latent_dim,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "patience": self.patience
        }
        
        if self.model_name in ['bilstm_ae_attention']:
            model_config["attention_size"] = self.attention_size
        
        if 'X' not in data:
            raise ValueError(
                "[FeatureExtract] Invalid context format: missing 'X'. "
                "Expected {'X': ndarray(num, len, dim), 'lengths': optional ndarray}."
            )

        np_data = data['X']
        lengths = data.get('lengths', None)

        if lengths is not None:
            model_config['lengths'] = lengths

        if not isinstance(np_data, np.ndarray) or np_data.ndim != 3:
            raise ValueError(
                f"[FeatureExtract] Invalid tensor shape: {getattr(np_data, 'shape', None)}. "
                "Expected ndarray with shape (num, len, dim)."
            )

        print(f"[FeatureExtract] Data loaded - Samples: {np_data.shape[0]}, "
              f"Timesteps: {np_data.shape[1]}, Features: {np_data.shape[2]}")

        # Select model and extract features
        try:
            if self.model_name == "lstm_ae":
                print(f"[FeatureExtract] Using LSTM Autoencoder - Output shape: (n_samples, {self.latent_dim})")
                extracted_features, training_history = lstm_ae(np_data, model_config)

            elif self.model_name == "bilstm_ae":
                print(f"[FeatureExtract] Using BiLSTM Autoencoder - Output shape: (n_samples, {self.latent_dim})")
                extracted_features, training_history = bilstm_ae(np_data, model_config)

            elif self.model_name == "cnn_ae":
                print(f"[FeatureExtract] Using CNN Autoencoder - Output shape: (n_samples, {self.latent_dim})")
                extracted_features, training_history = cnn_ae(np_data, model_config)

            elif self.model_name == "bilstm_ae_attention":
                print(f"[FeatureExtract] Using BiLSTM + Attention Autoencoder")
                print(f"[FeatureExtract] Output shape: (n_samples, {self.latent_dim})")
                extracted_features, training_history = bilstm_ae_attention(np_data, model_config)

            elif self.model_name == "detsec":
                print(f"[FeatureExtract] Using DETSEC Model")
                extracted_features, training_history = detsec_ae(np_data, model_config)

            else:
                print(f"[FeatureExtract] Unknown model: {self.model_name}")
                return context

            print(f"[FeatureExtract] Extracted features shape: {extracted_features.shape}")

            # Print training history summary
            if training_history:
                print(f"[FeatureExtract] Epochs trained: {training_history['epochs_trained']}")
                if 'loss' in training_history and len(training_history['loss']) > 0:
                    print(f"[FeatureExtract] Final training loss: {training_history['loss'][-1]:.6f}")
                    print(f"[FeatureExtract] Final validation loss: {training_history['val_loss'][-1]:.6f}")

            source_tag = "file" if data_source == "file" else "context"

            # Visualize training history
            self.visualize_training_history(training_history, self.model_name, log_dir, source_tag)

            # Save features to file if enabled
            if self.save_to_file:
                feature_save_dir = os.path.join(log_dir, 'extracted_features')
                os.makedirs(feature_save_dir, exist_ok=True)

                feature_file_path = os.path.join(
                    feature_save_dir,
                    f'{self.model_name}_{source_tag}_{self.latent_dim}dim.npy'
                )
                np.save(feature_file_path, extracted_features)
                print(f"[FeatureExtract] Features saved to: {feature_file_path}")

        except Exception as e:
            print(f"[FeatureExtract] Error during feature extraction: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Store features in context
        context['features'] = extracted_features
        context['feature_extract_config'] = {
            'model_name': self.model_name,
            'latent_dim': self.latent_dim,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'data_source': data_source,
            'data_source_detail': {
                'data_path': self.data_path if data_source == 'file' else None,
                'seq_len_path': self.seq_len_path if data_source == 'file' else None,
                'context_data_key': 'data.X' if data_source == 'context' else None
            },
            'input_tensor_shape': tuple(np_data.shape),
            'output_feature_shape': tuple(extracted_features.shape),
            'training_history': training_history
        }
        print("\n[FeatureExtract] Successfully extracted features")
        
        # Sliding context release: Step 3 (FeatureExtract) releases Step 1 (ExtractActiveData) data
        if 'data' in context and 'extract_active_data' in context['data']:
            print("[FeatureExtract] Releasing Step 1 (ExtractActiveData) context data")
            del context['data']['extract_active_data']

        print("\n" + "="*70)
        print("[FeatureExtract] Step completed")
        print("="*70)
        
        return context
