import os
import sys
import time
from datetime import datetime

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.spatial.distance import cdist
from scipy.signal import medfilt

from src.framework.step import Step

sys.stdout.flush()
sys.stderr.flush()

from src.utlis import clustering_utils


class TimeClusteringStep(Step):
    """
    Step for time series clustering.
    Supports DBSCAN and KMeans.
    """
    def __init__(
        self,
        name="TimeClustering",
        data_path=None,
        feature_path=None,
        seq_len_path=None,
        save_dir=None,
        col_index=2,
        eps=1.25,
        min_eps=0.5,
        max_eps=2.0,
        eps_gap=0.1,
        min_pts=20,
        metric='euclidean',
        normalization_method='zscore',
        cluster_method='dbscan',
        kmeans_n_clusters=8,
        min_cluster=2,
        max_cluster=10,
        kmeans_random_state=42,
        kmeans_n_init=10,
        kmeans_max_iter=300,
        enable_visualization=True,
        visualize_noise=True,
        visualization_language='zh',
        cluster_stack_count=50,
        few_shot_enabled=False,
        few_shot_method='avg_percent',
        few_shot_n_percent=50.0,
        few_shot_threshold=5,
        appliance_name=""
    ):
        super().__init__(name)
        self.data_path = data_path
        self.feature_path = feature_path
        self.seq_len_path = seq_len_path
        self.save_dir = save_dir or os.path.join('cluster_result', 'dbscan_result')
        self.col_index = col_index
        self.eps = eps
        self.min_eps = min_eps
        self.max_eps = max_eps
        self.eps_gap = eps_gap
        self.min_pts = min_pts
        self.metric = metric
        self.normalization_method = normalization_method
        self.cluster_method = str(cluster_method).lower()
        self.kmeans_n_clusters = kmeans_n_clusters
        self.min_cluster = min_cluster
        self.max_cluster = max_cluster
        self.kmeans_random_state = kmeans_random_state
        self.kmeans_n_init = kmeans_n_init
        self.kmeans_max_iter = kmeans_max_iter
        self.enable_visualization = enable_visualization
        self.visualize_noise = bool(visualize_noise)
        self.visualization_language = str(visualization_language).lower()
        self.cluster_stack_count = max(1, int(cluster_stack_count))
        self.few_shot_enabled = bool(few_shot_enabled)
        self.few_shot_method = str(few_shot_method).lower()
        self.few_shot_n_percent = float(few_shot_n_percent)
        self.few_shot_threshold = int(few_shot_threshold)
        self.selected_scan_eps = None
        self.appliance_name = appliance_name

    def load_data(self, context: dict) -> tuple:
        """Load feature data and sequence length from context or files."""
        data_np = None
        feature_matrix = None
        seq_len = None

        if self.data_path and os.path.exists(self.data_path):
            data_np = np.load(self.data_path)
        elif 'data' in context and context['data'] is not None:
            context_data = context['data']
            if isinstance(context_data, dict):
                # Preferred contract from WaveletSeparation: context['data']['X']
                data_np = context_data.get('X', None)
            elif isinstance(context_data, np.ndarray):
                # Backward compatibility for direct ndarray payload
                data_np = context_data

        if self.feature_path and os.path.exists(self.feature_path):
            feature_matrix = np.load(self.feature_path)
        elif 'features' in context and context['features'] is not None:
            context_features = context['features']
            if isinstance(context_features, np.ndarray):
                # New contract from FeatureExtractStep
                feature_matrix = context_features
            else:
                raise ValueError(
                    f"[TimeClustering] Invalid features type: {type(context_features).__name__}. "
                    "Expected context['features'] as numpy.ndarray."
                )

        if self.seq_len_path and os.path.exists(self.seq_len_path):
            seq_len = np.load(self.seq_len_path)
        elif 'seq_len' in context and context['seq_len'] is not None:
            seq_len = context['seq_len']
        elif 'data' in context and isinstance(context['data'], dict):
            # Preferred contract from WaveletSeparation: context['data']['lengths']
            seq_len = context['data'].get('lengths', None)

        # Normalize types
        if data_np is not None and not isinstance(data_np, np.ndarray):
            data_np = np.asarray(data_np)
        if feature_matrix is not None and not isinstance(feature_matrix, np.ndarray):
            feature_matrix = np.asarray(feature_matrix)
        if seq_len is not None and not isinstance(seq_len, np.ndarray):
            seq_len = np.asarray(seq_len)

        if data_np is not None:
            print(f"[TimeClustering] Loaded data shape: {data_np.shape}")
        if feature_matrix is not None and feature_matrix.size > 0:
            print(f"[TimeClustering] Loaded feature matrix shape: {feature_matrix.shape}")
        if seq_len is not None:
            print(f"[TimeClustering] Loaded seq_len shape: {seq_len.shape}")

        return data_np, feature_matrix, seq_len

    def _get_seq_length(self, seq_len: np.ndarray, idx: int, fallback_len: int) -> int:
        """Safely read sequence length value from seq_len with support for (n,) and (n,1)."""
        if seq_len is None or idx >= len(seq_len):
            return fallback_len

        raw_len = seq_len[idx]
        try:
            actual_len = int(np.asarray(raw_len).reshape(-1)[0])
        except Exception:
            return fallback_len

        if actual_len <= 0:
            return fallback_len
        return min(actual_len, fallback_len)

    def _validate_col_index(self, data_np: np.ndarray):
        """Validate col_index against raw data feature dimension before clustering/visualization."""
        if data_np is None:
            return

        if not isinstance(data_np, np.ndarray):
            raise ValueError(
                f"[TimeClustering] Invalid data type for col_index check: {type(data_np).__name__}. "
                "Expected numpy.ndarray."
            )

        if data_np.ndim < 3:
            raise ValueError(
                f"[TimeClustering] Invalid data shape for col_index check: {data_np.shape}. "
                "Expected at least 3D array with shape (n_samples, seq_len, n_features)."
            )

        feature_dim = int(data_np.shape[2])
        if feature_dim <= 0:
            raise ValueError(
                f"[TimeClustering] Invalid feature dimension: {feature_dim}. "
                "n_features must be positive."
            )

        if not (0 <= int(self.col_index) < feature_dim):
            raise ValueError(
                f"[TimeClustering] Invalid col_index={self.col_index}. "
                f"Valid range is 0..{feature_dim - 1} for input shape {data_np.shape}."
            )

    def normalize_features(self, feature_matrix: np.ndarray) -> list:
        """Normalize feature matrix using specified method."""
        if feature_matrix is None or feature_matrix.size == 0:
            print("[TimeClustering] Warning: Empty feature matrix, skipping normalization")
            return []

        data = feature_matrix

        if self.normalization_method == 'minmax':
            scaler = MinMaxScaler()
            print("[TimeClustering] Using Min-Max normalization")
        elif self.normalization_method == 'zscore':
            scaler = StandardScaler()
            print("[TimeClustering] Using Z-Score normalization")
        else:
            raise ValueError(f"Unsupported normalization method: {self.normalization_method}")

        if data.ndim == 2:
            normalized_features = scaler.fit_transform(data)
            normalized_feature_list = [normalized_features[i] for i in range(len(normalized_features))]
        elif data.ndim == 1:
            normalized_features = scaler.fit_transform(data.reshape(-1, 1)).flatten()
            normalized_feature_list = [[normalized_features[i]] for i in range(len(normalized_features))]
        else:
            raise ValueError(f"Unsupported data dimension: {data.ndim}")

        print(f"[TimeClustering] Normalization complete, valid samples: {len(normalized_feature_list)}")
        return normalized_feature_list

    def compute_distance_matrix(self, ts_list: list, metric: str = 'euclidean') -> np.ndarray:
        """Compute distance matrix for time series."""
        print(f"[TimeClustering] Computing {metric} distance matrix...")

        if metric == 'dtw' or metric == 'fastdtw':
            return self._compute_dtw_matrix(ts_list)
        else:
            distance_matrix = cdist(ts_list, ts_list, metric=metric)
            print(f"[TimeClustering] Distance matrix shape: {distance_matrix.shape}")
            return distance_matrix

    def _compute_dtw_matrix(self, ts_list: list) -> np.ndarray:
        """Compute DTW distance matrix using tslearn."""
        from tslearn.utils import to_time_series_dataset
        from tslearn.metrics import cdist_dtw

        start_time = time.time()
        X = to_time_series_dataset(ts_list)
        dist_matrix = cdist_dtw(X, n_jobs=-1)

        elapsed_time = time.time() - start_time
        print(f"[TimeClustering] DTW matrix computed in {elapsed_time:.2f}s, shape: {dist_matrix.shape}")
        return dist_matrix

    def get_distance_matrix(self, ts_list: list, metric: str = 'euclidean', cache_dir: str = None) -> np.ndarray:
        """Get distance matrix with caching support for DTW."""
        if metric in ['dtw', 'fastdtw'] and cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            n_samples = len(ts_list)
            seq_len = len(ts_list[0]) if ts_list else 0
            cache_filename = f"dtw_dist_matrix_{metric}_{n_samples}_{seq_len}.npy"
            cache_path = os.path.join(cache_dir, cache_filename)

            if os.path.exists(cache_path):
                print(f"[TimeClustering] Loading cached distance matrix from {cache_path}")
                return np.load(cache_path)

        dist_matrix = self.compute_distance_matrix(ts_list, metric)

        if metric in ['dtw', 'fastdtw'] and cache_dir:
            np.save(cache_path, dist_matrix)
            print(f"[TimeClustering] Saved distance matrix to cache: {cache_path}")

        return dist_matrix

    def run_dbscan(self, dist_matrix: np.ndarray, eps: float, min_pts: int) -> np.ndarray:
        """Execute DBSCAN clustering."""
        print(f"[TimeClustering] Running DBSCAN with eps={eps}, min_samples={min_pts}")

        dbscan_model = DBSCAN(
            eps=eps,
            min_samples=min_pts,
            metric="precomputed"
        )
        labels = dbscan_model.fit_predict(dist_matrix)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)

        print(f"[TimeClustering] Clustering complete: {n_clusters} clusters, {n_noise} noise points")

        unique_labels, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            label_name = "noise" if label == -1 else f"cluster_{label}"
            print(f"  {label_name}: {count} samples")

        return labels

    def run_kmeans(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Execute KMeans clustering on normalized feature vectors."""
        print(
            "[TimeClustering] Running KMeans with "
            f"n_clusters={self.kmeans_n_clusters}, random_state={self.kmeans_random_state}, "
            f"n_init={self.kmeans_n_init}, max_iter={self.kmeans_max_iter}"
        )

        kmeans_model = KMeans(
            n_clusters=self.kmeans_n_clusters,
            random_state=self.kmeans_random_state,
            n_init=self.kmeans_n_init,
            max_iter=self.kmeans_max_iter,
        )
        labels = kmeans_model.fit_predict(feature_matrix)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)
        print(f"[TimeClustering] Clustering complete: {n_clusters} clusters, {n_noise} noise points")

        unique_labels, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            label_name = "noise" if label == -1 else f"cluster_{label}"
            print(f"  {label_name}: {count} samples")

        return labels

    def run_kmeans_scan(self, feature_matrix: np.ndarray, save_dir: str) -> tuple:
        """Scan n_clusters in [min_cluster, max_cluster], save metrics JSON/plot, and return best labels."""
        min_k = int(self.min_cluster)
        max_k = int(self.max_cluster)
        n_samples = int(feature_matrix.shape[0])

        if min_k < 2:
            raise ValueError(f"[TimeClustering] min_cluster must be >= 2, got {min_k}")
        if max_k < min_k:
            raise ValueError(f"[TimeClustering] max_cluster must be >= min_cluster, got {max_k} < {min_k}")
        if min_k >= n_samples:
            raise ValueError(
                f"[TimeClustering] min_cluster={min_k} is invalid for n_samples={n_samples}. "
                "min_cluster must be less than number of samples."
            )

        scan_records = []
        best_score = -np.inf
        best_k = None
        best_labels = None
        dist_matrix = cdist(feature_matrix, feature_matrix, metric='euclidean')

        print(f"[TimeClustering] Running KMeans scan from k={min_k} to k={max_k}")

        for k in range(min_k, max_k + 1):
            if k >= n_samples:
                print(f"[TimeClustering] Skip k={k}: requires k < n_samples ({n_samples})")
                continue

            kmeans_model = KMeans(
                n_clusters=k,
                random_state=self.kmeans_random_state,
                n_init=self.kmeans_n_init,
                max_iter=self.kmeans_max_iter,
            )
            labels = kmeans_model.fit_predict(feature_matrix)

            sci = silhouette_score(feature_matrix, labels, metric='euclidean')
            dbi = davies_bouldin_score(feature_matrix, labels)
            chi = calinski_harabasz_score(feature_matrix, labels)
            dbcv = clustering_utils.calculate_dbcv_score(dist_matrix=dist_matrix, cluster_labels=labels)

            record = {
                'n_clusters': int(k),
                'sci': float(sci),
                'dbi': float(dbi),
                'chi': float(chi),
                'dbcv': (None if dbcv is None else float(dbcv)),
            }
            scan_records.append(record)

            print(
                f"[TimeClustering] k={k} -> "
                f"SCI={record['sci']:.6f}, DBI={record['dbi']:.6f}, CHI={record['chi']:.6f}"
                f", DBCV={record['dbcv']}"
            )

            # Choose best k by SCI (higher is better)
            if sci > best_score:
                best_score = float(sci)
                best_k = int(k)
                best_labels = labels

        if best_labels is None:
            raise ValueError("[TimeClustering] KMeans scan failed: no valid k in scan range.")

        scan_payload = clustering_utils.save_kmeans_scan_artifacts(
            scan_records,
            best_k,
            save_dir,
            data_path=self.data_path,
            appliance_name=self.appliance_name,
        )
        return best_labels, best_k, scan_records, scan_payload

    def run_dbscan_scan(self, dist_matrix: np.ndarray, feature_matrix: np.ndarray, save_dir: str) -> tuple:
        """Scan eps in [min_eps, max_eps] with step eps_gap for DBSCAN while keeping min_pts fixed."""
        min_eps = float(self.min_eps)
        max_eps = float(self.max_eps)
        eps_gap = float(self.eps_gap)
        min_pts = int(self.min_pts)
        n_samples = int(feature_matrix.shape[0])

        if min_pts < 1:
            raise ValueError(f"[TimeClustering] min_pts must be >= 1, got {min_pts}")
        if max_eps < min_eps:
            raise ValueError(f"[TimeClustering] max_eps must be >= min_eps, got {max_eps} < {min_eps}")
        if eps_gap <= 0:
            raise ValueError(f"[TimeClustering] eps_gap must be > 0, got {eps_gap}")

        scan_records = []
        best_score = -np.inf
        best_labels = None
        best_eps = None

        print(
            f"[TimeClustering] Running DBSCAN scan with min_pts={min_pts}, "
            f"eps={min_eps}..{max_eps}, gap={eps_gap}, metric={self.metric}"
        )

        eps_values = np.arange(min_eps, max_eps + (eps_gap * 0.5), eps_gap)
        for eps in eps_values:
            eps_val = float(round(float(eps), 10))
            labels = self.run_dbscan(dist_matrix, eps_val, min_pts)
            n_noise = int(np.sum(labels == -1))
            n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))

            sci = None
            dbi = None
            chi = None
            dbcv = None
            if n_clusters >= 2:
                valid_idx = labels != -1
                valid_labels = labels[valid_idx]
                valid_feature = feature_matrix[valid_idx]
                valid_dist = dist_matrix[valid_idx][:, valid_idx]
                if len(valid_labels) > 1 and len(np.unique(valid_labels)) >= 2:
                    sci = float(silhouette_score(valid_dist, valid_labels, metric='precomputed'))
                    dbi = float(davies_bouldin_score(valid_feature, valid_labels))
                    chi = float(calinski_harabasz_score(valid_feature, valid_labels))
                    dbcv = clustering_utils.calculate_dbcv_score(dist_matrix=dist_matrix, cluster_labels=labels)

            record = {
                'eps': eps_val,
                'min_pts': min_pts,
                'sci': sci,
                'dbi': dbi,
                'chi': chi,
                'dbcv': dbcv,
                'n_noise': n_noise,
                'n_clusters': n_clusters,
            }
            scan_records.append(record)
            print(
                f"[TimeClustering] eps={eps_val} -> "
                f"SCI={record['sci']}, DBI={record['dbi']}, CHI={record['chi']}, "
                f"DBCV={record['dbcv']}, n_noise={record['n_noise']}, n_clusters={record['n_clusters']}"
            )

            current_score = float(record['sci']) if record['sci'] is not None else -np.inf
            if current_score > best_score:
                best_score = current_score
                best_labels = labels
                best_eps = eps_val

        if best_labels is None:
            # Fallback: if all SCI invalid, pick the first run result.
            first = scan_records[0]
            best_eps = float(first['eps'])
            best_labels = self.run_dbscan(dist_matrix, best_eps, min_pts)

        scan_payload = clustering_utils.save_dbscan_scan_artifacts(
            scan_records,
            best_eps,
            save_dir,
            data_path=self.data_path,
            appliance_name=self.appliance_name,
        )
        return best_labels, best_eps, scan_records, scan_payload

    def evaluate_clustering(self, labels: np.ndarray, dist_matrix: np.ndarray,
                           org_data: np.ndarray, feature_matrix: np.ndarray,
                           seq_len: np.ndarray, save_dir: str,
                           enable_visualization: bool = False) -> dict:
        """Thin wrapper: forward to clustering_utils and assemble a lightweight metrics dict."""
        metrics = {}
        metrics_payload = None

        try:
            quant_dist_method = 'euclidean'
            if self.cluster_method == 'dbscan' and self.metric in ('dtw', 'fastdtw'):
                quant_dist_method = 'dtw'

            if self.cluster_method == 'dbscan':
                clustering_hyperparams = {
                    'eps': float(self.eps),
                    'min_pts': int(self.min_pts),
                    'metric': str(self.metric),
                }
            elif self.cluster_method in ('dbscan-scan', 'dbscan_scan'):
                clustering_hyperparams = {
                    'min_eps': float(self.min_eps),
                    'max_eps': float(self.max_eps),
                    'eps_gap': float(self.eps_gap),
                    'min_pts': int(self.min_pts),
                    'selected_eps': float(self.selected_scan_eps) if self.selected_scan_eps is not None else float(self.eps),
                    'metric': str(self.metric),
                }
            elif self.cluster_method == 'kmeans':
                clustering_hyperparams = {
                    'n_clusters': int(self.kmeans_n_clusters),
                    'random_state': int(self.kmeans_random_state),
                    'n_init': int(self.kmeans_n_init),
                    'max_iter': int(self.kmeans_max_iter),
                }
            elif self.cluster_method == 'kmeans-scan':
                clustering_hyperparams = {
                    'min_cluster': int(self.min_cluster),
                    'max_cluster': int(self.max_cluster),
                    'selected_n_clusters': int(len(np.unique(labels))),
                    'random_state': int(self.kmeans_random_state),
                    'n_init': int(self.kmeans_n_init),
                    'max_iter': int(self.kmeans_max_iter),
                }
            else:
                clustering_hyperparams = {}

            sil_score, db_score, ch_score, dbcv_score, metrics_payload = clustering_utils.cluster_result_quantification(
                cluster_labels=labels,
                dist_matrix=dist_matrix,
                org_data=org_data,
                feature_matrix=feature_matrix,
                save_dir=save_dir,
                seq_length=seq_len,
                dist_method=quant_dist_method,
                cluster_method=self.cluster_method,
                cluster_hyperparams=clustering_hyperparams,
                language=self.visualization_language,
                col_index=self.col_index,
                visualize=enable_visualization,
                visualize_noise=self.visualize_noise,
                cluster_stack_count=self.cluster_stack_count,
                sampling_threshold=200,
                data_path=self.data_path,
                appliance_name=self.appliance_name,
                few_shot_enabled=self.few_shot_enabled,
                few_shot_method=self.few_shot_method,
                few_shot_n_percent=self.few_shot_n_percent,
                few_shot_threshold=self.few_shot_threshold,
                return_metrics_payload=True,
            )

            if sil_score is not None:
                metrics['silhouette_score'] = float(sil_score)
                print(f"[TimeClustering] Silhouette Score: {sil_score:.4f}")
            if db_score is not None:
                metrics['davies_bouldin_score'] = float(db_score)
                print(f"[TimeClustering] Davies-Bouldin Score: {db_score:.4f}")
            if ch_score is not None:
                metrics['calinski_harabasz_score'] = float(ch_score)
                print(f"[TimeClustering] Calinski-Harabasz Score: {ch_score:.4f}")
            if dbcv_score is not None:
                metrics['dbcv_score'] = float(dbcv_score)
                print(f"[TimeClustering] DBCV Score: {dbcv_score:.4f}")
        except Exception as e:
            print(f"[TimeClustering] Quantification warning: {e}")

        if isinstance(metrics_payload, dict):
            metrics = metrics_payload
        else:
            unique_labels, counts = np.unique(labels, return_counts=True)
            metrics['cluster_distribution'] = {
                "noise" if label == -1 else f"cluster_{label}": int(count)
                for label, count in zip(unique_labels, counts)
            }
            metrics['n_clusters'] = len(set(labels)) - (1 if -1 in labels else 0)
            metrics['n_noise'] = int(np.sum(labels == -1))
        return metrics

    def save_clustering_results(self, data_np: np.ndarray, seq_len: np.ndarray,
                                 labels: np.ndarray, save_dir: str) -> str:
        """Save clustering results including labels and visualizations using clustering_utils."""
        labels_save_path = os.path.join(save_dir, 'cluster_labels.npy')
        np.save(labels_save_path, labels)
        print(f"[TimeClustering] Labels saved to {labels_save_path}")

        # Save cluster arrays
        self._save_cluster_arrays(data_np, seq_len, labels, save_dir)

        print(f"[TimeClustering] Results saved to {save_dir}")
        return save_dir

    def _save_cluster_arrays(self, data_np: np.ndarray, seq_len: np.ndarray,
                             labels: np.ndarray, save_dir: str):
        """Save cluster data as numpy arrays."""
        cluster_groups = {}
        for i in range(len(labels)):
            cluster_id = labels[i]
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            base_data = data_np[i]
            actual_len = self._get_seq_length(seq_len, i, len(base_data))
            data = base_data[:actual_len]
            cluster_groups[cluster_id].append(data)

        for cluster_id, cluster_data in cluster_groups.items():
            max_length = max(len(seq) for seq in cluster_data)
            sample_data = cluster_data[0]

            if len(sample_data.shape) == 1:
                feature_dim = 1
                cluster_data = [seq.reshape(-1, 1) if len(seq.shape) == 1 else seq for seq in cluster_data]
            else:
                feature_dim = sample_data.shape[1]

            n_samples = len(cluster_data)
            padded_data = np.zeros((n_samples, max_length, feature_dim))

            for i, seq in enumerate(cluster_data):
                actual_len = len(seq)
                padded_data[i, :actual_len, :] = seq

            cluster_file_path = os.path.join(save_dir, f'Cluster_{cluster_id}.npy')
            np.save(cluster_file_path, padded_data)
            print(f"[TimeClustering] Cluster_{cluster_id}.npy saved with shape {padded_data.shape}")

    def run(self, context: dict) -> dict:
        """Main execution logic for time series clustering."""
        log_dir = self.get_log_dir(context)
        save_dir = log_dir
        os.makedirs(save_dir, exist_ok=True)

        print(f"[TimeClustering] Starting clustering workflow")
        print(
            f"[TimeClustering] Parameters: method={self.cluster_method}, "
            f"eps={self.eps}, min_pts={self.min_pts}, metric={self.metric}"
        )

        data_np, feature_matrix, seq_len = self.load_data(context)

        # Validate visualization/index selection early to fail fast with clear diagnostics.
        self._validate_col_index(data_np)

        if data_np is None or feature_matrix is None:
            print("[TimeClustering] Error: Required data not found. Please provide feature data.")
            return context

        if feature_matrix.size == 0:
            print("[TimeClustering] Error: Feature matrix is empty.")
            return context

        normalized_feature_list = self.normalize_features(feature_matrix)

        if not normalized_feature_list:
            print("[TimeClustering] Error: No valid features after normalization.")
            return context

        normalized_feature_matrix = np.asarray(normalized_feature_list)

        if self.cluster_method == 'dbscan':
            dist_matrix = self.get_distance_matrix(
                normalized_feature_list,
                metric=self.metric,
                cache_dir=save_dir
            )
            labels = self.run_dbscan(dist_matrix, self.eps, self.min_pts)
        elif self.cluster_method in ('dbscan-scan', 'dbscan_scan'):
            dist_matrix = self.get_distance_matrix(
                normalized_feature_list,
                metric=self.metric,
                cache_dir=save_dir
            )
            _labels, best_eps, scan_records, scan_payload = self.run_dbscan_scan(
                dist_matrix=dist_matrix,
                feature_matrix=normalized_feature_matrix,
                save_dir=save_dir,
            )
            self.selected_scan_eps = float(best_eps)
            self.eps = float(best_eps)

            # In scan mode, only export scan artifacts (JSON + line chart),
            # and do not persist a selected-best clustering result.
            context['cluster_save_dir'] = save_dir
            context['dbscan_scan_best_eps'] = float(best_eps)
            context['dbscan_scan_records'] = scan_records
            if isinstance(scan_payload, dict):
                context['dbscan_scan_metrics'] = scan_payload
            print("[TimeClustering] DBSCAN scan mode completed (scan artifacts only)")
            return context
        elif self.cluster_method == 'kmeans':
            labels = self.run_kmeans(normalized_feature_matrix)
            # Keep quantification path unified by building a precomputed distance matrix.
            dist_matrix = cdist(normalized_feature_matrix, normalized_feature_matrix, metric='euclidean')
        elif self.cluster_method in ('kmeans-scan', 'kmeans_scan'):
            _labels, best_k, scan_records, scan_payload = self.run_kmeans_scan(normalized_feature_matrix, save_dir)
            self.kmeans_n_clusters = int(best_k)

            # In scan mode, only export scan artifacts (JSON + line chart),
            # and do not persist a selected-best clustering result.
            context['cluster_save_dir'] = save_dir
            context['kmeans_scan_best_k'] = int(best_k)
            context['kmeans_scan_records'] = scan_records
            if isinstance(scan_payload, dict):
                context['kmeans_scan_metrics'] = scan_payload
            print("[TimeClustering] KMeans scan mode completed (scan artifacts only)")
            return context
        else:
            raise ValueError(
                f"Unsupported cluster_method: {self.cluster_method}. "
                "Supported methods: dbscan, dbscan-scan, kmeans, kmeans-scan"
            )

        quant_metrics = self.evaluate_clustering(
            labels,
            dist_matrix,
            data_np,
            feature_matrix,
            seq_len,
            save_dir,
            enable_visualization=self.enable_visualization
        )

        self.save_clustering_results(data_np, seq_len, labels, save_dir)

        context['cluster_labels'] = labels
        context['cluster_save_dir'] = save_dir
        context['n_clusters'] = len(set(labels)) - (1 if -1 in labels else 0)
        context['n_noise'] = int(np.sum(labels == -1))
        context['evaluation_metrics'] = quant_metrics
        context['clustering_metrics'] = quant_metrics

        print(f"[TimeClustering] Clustering workflow complete")
        return context