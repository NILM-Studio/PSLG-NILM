import os
import sys

# Set threading and environment variables before any other imports
# to prevent segmentation faults caused by library conflicts (e.g., stumpy/numba and tensorflow)
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Prevent TensorFlow from pre-allocating all GPU memory
os.environ['NUMBA_THREADING_LAYER'] = 'workqueue' # Use the most stable threading layer for Numba

import yaml
import argparse
import numpy as np

# Add project root and models to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(project_root, "models")
time_seg_dir = os.path.join(models_dir, "time_segmentation")
feature_ext_dir = os.path.join(models_dir, "feature_extract")

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if models_dir not in sys.path:
    sys.path.insert(0, models_dir)
if time_seg_dir not in sys.path:
    sys.path.insert(0, time_seg_dir)
if feature_ext_dir not in sys.path:
    sys.path.insert(0, feature_ext_dir)

def run_workflow(config_path: str, resume: bool = False, sequence_id: str | None = None):
    """
    Main function to run the workflow based on config file.
    """
    # Load configuration from YAML file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Lazy import Workflow
    from src.framework.workflow import Workflow

    # Initialize workflow with name from config
    workflow_name = config['workflow'].get('name', 'ML_Workflow')
    appliance_name = config['workflow'].get('appliance_name', '')
    resume_cfg = bool(config.get('workflow', {}).get('resume', False))
    sequence_id_cfg = config.get('workflow', {}).get('sequence_id', None)
    save_interval_cfg = config.get('workflow', {}).get('save_interval', 0)
    wf = Workflow(
        workflow_name,
        appliance_name=appliance_name,
        sequence_id=sequence_id or sequence_id_cfg,
        resume=resume or resume_cfg,
        save_interval=save_interval_cfg
    )

    extract_active_cfg = config["steps"].get("extract_active_data", {})
    if extract_active_cfg.get("enabled", False):
        from src.steps.extract_active_data_step import ExtractActiveDataStep
        # 提取核心参数，其余通过 **kwargs 传递给模型
        step_params = extract_active_cfg.copy()
        step_params.pop("enabled", None)
        method = step_params.pop("method", "simple")
        input_file = step_params.pop("input_file", "")
        
        wf.add_step(
            ExtractActiveDataStep(
                name="ExtractActiveData",
                method=method,
                appliance_name=appliance_name,
                input_file=input_file,
                **step_params
            )
        )
    
    time_segmentation_cfg = config["steps"].get("time_segmentation", {})
    if time_segmentation_cfg.get("enabled", False):
        from src.steps.time_segmentation import TimeSegmentationStep
        wf.add_step(
            TimeSegmentationStep(
                name="TimeSegmentation",
                segment_method=time_segmentation_cfg.get("segment_method", "clasp"),
                appliance_name=appliance_name,
                window_size=time_segmentation_cfg.get("window_size", 100),
                n_regimes=time_segmentation_cfg.get("n_regimes", 3),
                excl_factor=time_segmentation_cfg.get("excl_factor", 5),
            )
        )
    
    if config['steps'].get('feature_extract', {}).get('enabled', True):
        from src.steps.feature_extract_step import FeatureExtractStep
        extract_config = config['steps']['feature_extract']
        wf.add_step(FeatureExtractStep(
            name="FeatureExtract",
            model_name=extract_config.get('model_name', 'detsec'),
            latent_dim=extract_config.get('latent_dim', 64),
            epochs=extract_config.get('epochs', 50),
            batch_size=extract_config.get('batch_size', 32),
            learning_rate=extract_config.get('learning_rate', 0.001),
            patience=extract_config.get('patience', 5),
            attention_size=extract_config.get('attention_size', 32),
            data_path=extract_config.get('data_path', ''),
            seq_len_path=extract_config.get('seq_len_path', ''),
            appliance_name=appliance_name
        ))
    
    if config['steps'].get('time_clustering', {}).get('enabled', True):
        from src.steps.time_clustering_step import TimeClusteringStep
        cluster_config = config['steps']['time_clustering']
        cluster_method = str(cluster_config.get('cluster_method', 'dbscan')).lower()
        method_specific = cluster_config.get('method_specific', {}) or {}
        method_config = method_specific.get(cluster_method, {}) if isinstance(method_specific, dict) else {}
        visualization_specific = cluster_config.get('visualization_specific', {}) or {}

        # Backward compatibility: keep supporting old flat keys if method_specific is absent.
        dbscan_eps = method_config.get('eps', cluster_config.get('eps', 1.25))
        dbscan_min_eps = method_config.get('min_eps', cluster_config.get('min_eps'))
        dbscan_max_eps = method_config.get('max_eps', cluster_config.get('max_eps'))
        dbscan_eps_gap = method_config.get('eps_gap', cluster_config.get('eps_gap', 0.1))
        dbscan_min_pts = method_config.get('min_pts', cluster_config.get('min_pts', 20))
        metric = method_config.get('metric', cluster_config.get('metric', 'euclidean'))
        kmeans_n_clusters = method_config.get('n_clusters', cluster_config.get('kmeans_n_clusters', 8))
        min_cluster = method_config.get('min_cluster', cluster_config.get('min_cluster', 2))
        max_cluster = method_config.get('max_cluster', cluster_config.get('max_cluster', 10))
        kmeans_random_state = method_config.get('random_state', cluster_config.get('kmeans_random_state', 42))
        kmeans_n_init = method_config.get('n_init', cluster_config.get('kmeans_n_init', 10))
        kmeans_max_iter = method_config.get('max_iter', cluster_config.get('kmeans_max_iter', 300))
        hdbscan_min_cluster_size = method_config.get(
            'min_cluster_size',
            cluster_config.get('hdbscan_min_cluster_size', cluster_config.get('min_cluster_size', 20))
        )
        hdbscan_min_samples = method_config.get(
            'min_samples',
            cluster_config.get('hdbscan_min_samples', cluster_config.get('min_samples'))
        )
        hdbscan_cluster_selection_method = method_config.get(
            'cluster_selection_method',
            cluster_config.get('hdbscan_cluster_selection_method', 'eom')
        )
        hdbscan_cluster_selection_epsilon = method_config.get(
            'cluster_selection_epsilon',
            cluster_config.get('hdbscan_cluster_selection_epsilon', 0.0)
        )

        enable_visualization = visualization_specific.get(
            'enabled',
            cluster_config.get('enable_visualization', cluster_config.get('enable_heatmap', True))
        )
        visualize_noise = visualization_specific.get(
            'visualize_noise',
            cluster_config.get('visualize_noise', cluster_config.get('visualize_noise_minus1', True))
        )
        visualization_language = visualization_specific.get(
            'language',
            cluster_config.get('visualization_language', 'en')
        )
        cluster_stack_count = visualization_specific.get('cluster_stack_count', 50)

        few_shot_config = cluster_config.get('few_shot_detection', {}) or {}
        few_shot_enabled = bool(few_shot_config.get('enabled', False))
        few_shot_method = str(few_shot_config.get('method', 'avg_percent')).lower()
        few_shot_n_percent = float(few_shot_config.get('n_percent', few_shot_config.get('n', 50.0)))
        few_shot_threshold = int(few_shot_config.get('threshold', 5))

        if cluster_method in ('dbscan-scan', 'dbscan_scan'):
            if dbscan_min_eps is None or dbscan_max_eps is None:
                raise ValueError(
                    "[Config] For cluster_method='dbscan-scan', both min_eps and max_eps must be provided."
                )

        wf.add_step(TimeClusteringStep(
            name="TimeClustering",
            data_path=cluster_config.get('data_path'),
            feature_path=cluster_config.get('feature_path'),
            seq_len_path=cluster_config.get('seq_len_path'),
            cluster_method=cluster_method,
            eps=dbscan_eps,
            min_eps=dbscan_min_eps,
            max_eps=dbscan_max_eps,
            eps_gap=dbscan_eps_gap,
            min_pts=dbscan_min_pts,
            kmeans_n_clusters=kmeans_n_clusters,
            min_cluster=min_cluster,
            max_cluster=max_cluster,
            kmeans_random_state=kmeans_random_state,
            kmeans_n_init=kmeans_n_init,
            kmeans_max_iter=kmeans_max_iter,
            hdbscan_min_cluster_size=hdbscan_min_cluster_size,
            hdbscan_min_samples=hdbscan_min_samples,
            hdbscan_cluster_selection_method=hdbscan_cluster_selection_method,
            hdbscan_cluster_selection_epsilon=hdbscan_cluster_selection_epsilon,
            metric=metric,
            normalization_method=cluster_config.get('normalization_method', 'zscore'),
            col_index=cluster_config.get('col_index', 2),
            enable_visualization=enable_visualization,
            visualize_noise=visualize_noise,
            visualization_language=visualization_language,
            cluster_stack_count=cluster_stack_count,
            few_shot_enabled=few_shot_enabled,
            few_shot_method=few_shot_method,
            few_shot_n_percent=few_shot_n_percent,
            few_shot_threshold=few_shot_threshold,
            appliance_name=appliance_name
        ))

    if config['steps'].get('primitive_activity_mapping', {}).get('enabled', False):
        from src.steps.primitive_activity_mapping_step import PrimitiveActivityMappingStep
        mapping_config = config['steps']['primitive_activity_mapping']
        wf.add_step(PrimitiveActivityMappingStep(
            name="PrimitiveActivityMapping",
            activity_sequence_dir=mapping_config.get('activity_sequence_dir'),
            primitive_sequence_dir=mapping_config.get('primitive_sequence_dir'),
            enable_tolerant_match=bool(mapping_config.get('enable_tolerant_match', False)),
            timestamp_tolerance=float(mapping_config.get('timestamp_tolerance', 0.0)),
            appliance_name=appliance_name,
        ))

    if config['steps'].get('dataset_split', {}).get('enabled', False):
        from src.steps.dataset_split_step import DatasetSplitStep
        split_config = config['steps']['dataset_split']
        wf.add_step(DatasetSplitStep(
            name="DatasetSplit",
            raw_series_path=split_config.get('raw_series_path'),
            mains_series_path=split_config.get('mains_series_path'),
            few_shot_tensor_path=split_config.get('few_shot_tensor_path'),
            non_few_shot_tensor_path=split_config.get('non_few_shot_tensor_path'),
            few_shot_activity_json_path=split_config.get('few_shot_activity_json_path'),
            non_few_shot_activity_json_path=split_config.get('non_few_shot_activity_json_path'),
            few_train_ratio=float(split_config.get('few_train_ratio', 0.5)),
            non_few_train_ratio=float(split_config.get('non_few_train_ratio', 0.8)),
            random_seed=int(split_config.get('random_seed', 42)),
            timestamp_tolerance_seconds=float(split_config.get('timestamp_tolerance_seconds', 0.0)),
            clip_negative_mains_to_zero=bool(split_config.get('clip_negative_mains_to_zero', True)),
            appliance_name=appliance_name,
        ))

    # Run the workflow
    wf.run()

def create_sample_data():
    """Helper to create some sample data in the input folder."""
    input_dir = 'input'
    os.makedirs(input_dir, exist_ok=True)
    
    # Create sample .csv file for wavelet separation
    import pandas as pd
    sample_csv_data = {
        'timestamp': range(100),
        'power': np.random.randn(100).cumsum() # Random walk to simulate power signal
    }
    pd.DataFrame(sample_csv_data).to_csv(os.path.join(input_dir, 'Kettle_sample.csv'), index=False)
    
    print(f"Sample data created in {input_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the ML Workflow framework.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to configuration file.")
    parser.add_argument("--sample", action="store_true", help="Create sample data in input folder.")
    parser.add_argument("--resume", action="store_true", help="Resume from cached step outputs if available.")
    parser.add_argument("--sequence-id", type=str, default=None, help="Specify existing sequence_id for resume.")
    args = parser.parse_args()

    # Create sample data if requested
    if args.sample:
        create_sample_data()

    # Run workflow
    run_workflow(args.config, resume=args.resume, sequence_id=args.sequence_id)
