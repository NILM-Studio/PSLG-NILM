from models.base_model import BaseModel
import numpy as np
import os
import sys

class EspressoModel(BaseModel):
    """
    ESPRESSO algorithm for time series segmentation.
    Integrated from the local time-series-segmentation-benchmark (TSSB) repository.
    """
    def __init__(self, name="espresso", config=None):
        super().__init__(name, config)
        self.model = None
        self.change_points = []
        
        # Add local tssb_repo to sys.path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        repo_path = os.path.join(current_dir, "tssb_repo")
        if repo_path not in sys.path:
            sys.path.append(repo_path)
            
        # Lazy import to avoid errors if not found
        try:
            from tssb.search.espresso import ESPRESSO
            self.ESPRESSO_CLASS = ESPRESSO
        except ImportError as e:
            self.ESPRESSO_CLASS = None
            print(f"Warning: tssb library not found in {repo_path}. Error: {e}")

    def train(self, data):
        """
        Runs ESPRESSO segmentation on the provided time series data.
        """
        if self.ESPRESSO_CLASS is None:
            raise ImportError("tssb library is required for EspressoModel. "
                            "Please ensure the repository is cloned into models/time_segmentation/tssb_repo")

        # Ensure data is numpy array
        if isinstance(data, list):
            data = np.array(data)
        
        # Default parameters for ESPRESSO
        params = {
            "window_size": 50,
        }
        
        # Override with config if provided
        if self.config:
            params.update(self.config)

        self.model = self.ESPRESSO_CLASS(**params)
        
        # According to user snippet: espresso.fit_predict(X)
        self.change_points = self.model.fit_predict(data)
        
        print(f"[{self.name}] Detected {len(self.change_points)} change points.")
        return self.change_points

    def save(self, path: str):
        """Saves the detected change points."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, np.array(self.change_points))

    def load(self, path: str):
        """Loads the detected change points."""
        if os.path.exists(path):
            self.change_points = np.load(path).tolist()
        else:
            self.change_points = []

if __name__ == "__main__":
    # Test script based on user example
    try:
        # Simulate some data
        X = np.concatenate([
            np.random.randn(200, 1),
            np.random.randn(200, 1) + 5,
            np.random.randn(200, 1)
        ])
        
        # Initialize and run model
        config = {"window_size": 20}
        model = EspressoModel(config=config)
        cps = model.train(X)
        
        print("Detected change points:", cps)
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
