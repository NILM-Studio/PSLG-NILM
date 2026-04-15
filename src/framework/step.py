from abc import ABC, abstractmethod
import os

class Step(ABC):
    """
    Abstract base class for all workflow steps.
    """
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def run(self, context: dict) -> dict:
        """
        Execute the step logic.
        
        Args:
            context (dict): Shared dictionary containing sequence_id, paths, and intermediate data.
            
        Returns:
            dict: Updated context.
        """
        pass

    def get_log_dir(self, context: dict) -> str:
        """Helper to get the specific log directory for this step."""
        log_dir = os.path.join(context['log_root'], self.name)
        os.makedirs(log_dir, exist_ok=True)
        return log_dir
