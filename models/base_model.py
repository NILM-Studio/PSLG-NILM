from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    Abstract base class for all machine learning models in the framework.
    """
    def __init__(self, name: str, config: dict = None):
        self.name = name
        self.config = config or {}

    @abstractmethod
    def train(self, data):
        """Trains the model."""
        pass

    @abstractmethod
    def save(self, path: str):
        """Saves the model state to disk."""
        pass

    @abstractmethod
    def load(self, path: str):
        """Loads the model state from disk."""
        pass
