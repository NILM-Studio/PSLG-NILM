import yaml
import os
import argparse
import sys
import numpy as np

# Add project root and models to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(project_root, "models")
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if models_dir not in sys.path:
    sys.path.insert(0, models_dir)

from src.framework.workflow import Workflow
from src.steps.data_loader import DataLoaderStep
from src.steps.data_processor import DataProcessorStep
from src.steps.model_step import ModelStep
from src.steps.wavelet_separation import WaveletSeparationStep

def run_workflow(config_path: str):
    """
    Main function to run the workflow based on config file.
    """
    # Load configuration from YAML file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize workflow with name from config
    workflow_name = config['workflow'].get('name', 'ML_Workflow')
    wf = Workflow(workflow_name)

    # Add steps to the workflow sequentially based on enabled flag in config
    if config['steps']['data_loader'].get('enabled', True):
        wf.add_step(DataLoaderStep("DataLoader"))

    if config['steps']['data_processor'].get('enabled', True):
        wf.add_step(DataProcessorStep("DataProcessor"))

    if config['steps'].get('wavelet_separation', {}).get('enabled', True):
        is_shape_dtw = config['steps']['wavelet_separation'].get('is_shape_dtw', False)
        wf.add_step(WaveletSeparationStep("WaveletSeparation", is_shape_dtw=is_shape_dtw))

    if config['steps']['model_training'].get('enabled', True):
        wf.add_step(ModelStep("ModelStep"))

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
    args = parser.parse_args()

    # Create sample data if requested
    if args.sample:
        create_sample_data()

    # Run workflow
    run_workflow(args.config)
