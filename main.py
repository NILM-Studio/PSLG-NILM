import yaml
import os
import argparse
import numpy as np
from src.framework.workflow import Workflow
from src.steps.data_loader import DataLoaderStep
from src.steps.data_processor import DataProcessorStep
from src.steps.model_step import ModelStep

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

    if config['steps']['model_training'].get('enabled', True):
        wf.add_step(ModelStep("ModelTraining", model_type="train"))

    if config['steps']['model_prediction'].get('enabled', True):
        wf.add_step(ModelStep("ModelPrediction", model_type="predict"))

    # Run the workflow
    wf.run()

def create_sample_data():
    """Helper to create some sample data in the input folder."""
    input_dir = 'input'
    os.makedirs(input_dir, exist_ok=True)
    
    # Create sample .npy file
    sample_npy = np.random.randn(100)
    np.save(os.path.join(input_dir, 'sample_data.npy'), sample_npy)
    
    # Create sample .txt file
    with open(os.path.join(input_dir, 'sample_data.txt'), 'w') as f:
        f.write("This is some sample text data for the workflow.")
    
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
