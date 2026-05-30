from models.base_model import BaseModel
from models.time_segmentation.claspy.segmentation import BinaryClaSPSegmentation
import numpy as np
import os

class ClaspOriginModel(BaseModel):
    """
    ClaSP Origin Model: Pure ClaSP segmentation without wavelet transform 
    or additional preprocessing as per project guidelines.
    """
    def __init__(self, name="clasp-origin", config=None):
        super().__init__(name, config)
        self.model = None
        self.change_points = []

    def train(self, data):
        """
        Runs ClaSP segmentation on the provided time series data.
        In this context, 'train' means performing the segmentation.
        """
        # Ensure data is 1D or 2D array
        if isinstance(data, list):
            data = np.array(data)
            
        # Default parameters for ClaSP
        params = {
            "n_segments": "learn",
            "window_size": "suss",
            "validation": "score_threshold",
            "threshold": 0.001,
            "distance": "znormed_euclidean_distance",
            "n_jobs": 1,
        }
        
        # Override with config if provided
        if self.config:
            params.update(self.config)

        self.model = BinaryClaSPSegmentation(**params)
        self.model.fit(data)
        self.change_points = self.model.change_points
        return self.change_points

    def save(self, path: str):
        """
        Saves the detected change points.
        """
        np.save(path, np.array(self.change_points))

    def load(self, path: str):
        """
        Loads the detected change points.
        """
        if os.path.exists(path):
            self.change_points = np.load(path).tolist()
        else:
            self.change_points = []
