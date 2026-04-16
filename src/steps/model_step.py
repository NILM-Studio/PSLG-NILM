import os
import numpy as np
from src.framework.step import Step
from models.dummy_model import DummyModel

class ModelStep(Step):
    """
    Workflow step that wraps a model for training or prediction.
    """
    def __init__(self, name="ModelStep"):
        super().__init__(name)
        # In a real framework, model selection could be dynamic from config
        self.model = DummyModel()

    def run(self, context: dict) -> dict:
        """
        Executes model training and outputs results.
        """
        log_dir = self.get_log_dir(context)
        output_dir = os.path.join(context['output_root'], 'output')
        
        # Check if there's any data loaded into the context
        if not context['data']:
            print(f"Warning: No data found in context for {self.name}.")
            return context

        for file_name, data in list(context['data'].items()):
            if isinstance(data, np.ndarray):
                print(f"[{self.name}] Training and generating results for {file_name}")
                results = self.model.train(data)
                
                # Save trained model state to cache
                model_save_path = os.path.join(log_dir, f"{self.model.name}_weights.pkl")
                self.model.save(model_save_path)
                
                # Save results to output (since training directly yields results)
                result_file = os.path.join(output_dir, f"training_result_{file_name}")
                if isinstance(results, np.ndarray):
                    np.save(result_file, results)
                else:
                    # For simple types like float/int, save as text or pkl
                    with open(f"{result_file}.txt", 'w') as f:
                        f.write(str(results))
                
                # Store weight in context for next steps if any
                context['data'][f'weights_{file_name}'] = self.model.weights
                
                print(f"[{self.name}] Results saved to: {output_dir}")

        return context
