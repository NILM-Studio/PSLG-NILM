import datetime
import os
import shutil
import traceback
from typing import List, Type
from .step import Step
from .logger import setup_logger

class Workflow:
    """
    Manages the sequential execution of ML workflow steps.
    """
    def __init__(self, name: str):
        self.name = name
        self.steps: List[Step] = []
        self.sequence_id = self._generate_sequence_id()
        self.context = {
            'sequence_id': self.sequence_id,
            'input_root': 'input',
            'log_root': os.path.join('log', self.sequence_id),
            'output_root': os.path.join('output', self.sequence_id),
            'data': {} # Stores intermediate results
        }
        
        # Ensure directories exist for the specific run
        self._init_dirs()
        self.logger = setup_logger(self.name, self.context['log_root'])
        self.logger.info(f"Initialized Workflow: {self.name} with ID: {self.sequence_id}")

    def _generate_sequence_id(self) -> str:
        """Generates a unique ID based on the current timestamp."""
        return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    def _init_dirs(self):
        """Creates output subdirectories according to specification."""
        # Log directory (cache)
        os.makedirs(self.context['log_root'], exist_ok=True)
        
        # Output subdirectories
        os.makedirs(os.path.join(self.context['output_root'], 'output'), exist_ok=True)
        os.makedirs(os.path.join(self.context['output_root'], 'figure'), exist_ok=True)

    def add_step(self, step: Step):
        """Adds a step to the workflow."""
        self.steps.append(step)
        self.logger.info(f"Added step: {step.name}")

    def run(self):
        """Runs all steps in the order they were added."""
        self.logger.info("Starting workflow execution...")
        try:
            for i, step in enumerate(self.steps, 1):
                self.logger.info(f"--- Step {i}: {step.name} ---")
                
                # Update context with current step log dir
                step_log_dir = step.get_log_dir(self.context)
                
                # Execute step
                self.context = step.run(self.context)
                
                self.logger.info(f"Step {step.name} completed successfully.")
            
            self.logger.info("Workflow execution finished successfully.")
            
        except Exception as e:
            self.logger.error(f"Workflow failed at step {step.name if 'step' in locals() else 'initialization'}: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
        finally:
            self.logger.info(f"Execution logs and artifacts stored in {self.context['log_root']} and {self.context['output_root']}")
