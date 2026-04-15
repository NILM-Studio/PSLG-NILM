import os
import numpy as np
import matplotlib.pyplot as plt
from src.framework.step import Step

class DataProcessorStep(Step):
    """
    Example step that processes loaded data and saves intermediate/final results.
    """
    def __init__(self, name="DataProcessor"):
        super().__init__(name)

    def run(self, context: dict) -> dict:
        """
        Process data and save to log/cache and output subdirectories.
        """
        log_dir = self.get_log_dir(context)
        output_dir = os.path.join(context['output_root'], 'output')
        figure_dir = os.path.join(context['output_root'], 'figure')
        
        # Ensure directories exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(figure_dir, exist_ok=True)
        
        # Process each loaded data file
        for file_name, data in context['data'].items():
            if isinstance(data, np.ndarray):
                # Simple processing: normalization
                processed_data = (data - np.mean(data)) / (np.std(data) + 1e-6)
                
                # Save processed data to cache (log folder)
                cache_file = os.path.join(log_dir, f"processed_{file_name}")
                np.save(cache_file, processed_data)
                
                # Save processed data to final output
                output_file = os.path.join(output_dir, f"result_{file_name}")
                np.save(output_file, processed_data)
                
                # Generate figure
                plt.figure()
                plt.plot(processed_data[:100] if len(processed_data) > 100 else processed_data)
                plt.title(f"Processed Data: {file_name}")
                plt.xlabel("Sample Index")
                plt.ylabel("Normalized Value")
                
                # Save figure to figure subfolder
                fig_file = os.path.join(figure_dir, f"plot_{os.path.splitext(file_name)[0]}.png")
                plt.savefig(fig_file)
                plt.close()
                
                print(f"Processed and saved: {file_name}")
            
            elif isinstance(data, str):
                # Text processing: word count
                word_count = len(data.split())
                
                # Save word count to output
                output_file = os.path.join(output_dir, f"word_count_{file_name}")
                with open(output_file, 'w') as f:
                    f.write(f"Word count for {file_name}: {word_count}")
                
                print(f"Counted words in: {file_name}")
                
        return context
