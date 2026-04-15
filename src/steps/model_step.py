import os
import numpy as np
from src.framework.step import Step
from models.dummy_model import DummyModel

class ModelStep(Step):
    """
    Workflow step that wraps a model for training or prediction.
    """
    def __init__(self, name="ModelStep", model_type="train"):
        super().__init__(name)
        self.model_type = model_type
        # In a real framework, model selection could be dynamic from config
        self.model = DummyModel()

    def run(self, context: dict) -> dict:
        """
        Executes model training or prediction.
        """
        log_dir = self.get_log_dir(context)
        output_dir = os.path.join(context['output_root'], 'output')
        
        # Check if there's any data loaded into the context
        if not context['data']:
            print(f"Warning: No data found in context for {self.name}.")
            return context

        for file_name, data in list(context['data'].items()):
            if isinstance(data, np.ndarray):
                if self.model_type == "train":
                    print(f"[{self.name}] Training on {file_name}")
                    self.model.train(data)
                    
                    # Save trained model to cache
                    model_save_path = os.path.join(log_dir, f"{self.model.name}_weights.pkl")
                    self.model.save(model_save_path)
                    
                    # Store weight in context for next steps
                    context['data'][f'weights_{file_name}'] = self.model.weights
                
                elif self.model_type == "predict":
                    print(f"[{self.name}] Predicting on {file_name}")
                    
                    # Try to load weights from context if available
                    weight_key = f'weights_{file_name}'
                    if weight_key in context['data']:
                        self.model.weights = context['data'][weight_key]
                    
                    predictions = self.model.predict(data)
                    
                    # Save predictions to output
                    pred_file = os.path.join(output_dir, f"predictions_{file_name}")
                    np.save(pred_file, predictions)
                    
                    # Cache predictions
                    cache_file = os.path.join(log_dir, f"predictions_{file_name}")
                    np.save(cache_file, predictions)
                    
                    print(f"[{self.name}] Predictions saved to: {pred_file}")

        return context
