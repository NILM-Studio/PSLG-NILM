import numpy as np
import pickle
import os
from .base_model import BaseModel

class DummyModel(BaseModel):
    """
    A simple dummy model for demonstration purposes.
    """
    def __init__(self, name="DummyModel", config=None):
        super().__init__(name, config)
        self.weights = None

    def train(self, data):
        """Simple 'training': calculate mean."""
        print(f"[{self.name}] Training on data...")
        self.weights = np.mean(data)
        return self.weights

    def save(self, path: str):
        """Saves model weights using pickle."""
        print(f"[{self.name}] Saving model to {path}")
        with open(path, 'wb') as f:
            pickle.dump(self.weights, f)

    def load(self, path: str):
        """Loads model weights using pickle."""
        print(f"[{self.name}] Loading model from {path}")
        with open(path, 'rb') as f:
            self.weights = pickle.load(f)
