import os
import numpy as np
import shutil
from src.framework.step import Step

class DataLoaderStep(Step):
    """
    Step to read data from input folder and copy to log cache.
    """
    def __init__(self, name="DataLoader"):
        super().__init__(name)

    def run(self, context: dict) -> dict:
        """
        Reads files from input folder and stores them in the context and cache.
        """
        input_dir = context['input_root']
        log_dir = self.get_log_dir(context)
        
        # Check if input directory exists
        if not os.path.exists(input_dir):
            os.makedirs(input_dir)
            print(f"Warning: Input directory '{input_dir}' was missing and has been created. Please place data there.")
            return context

        # Support .npy and .txt formats
        supported_extensions = ['.npy', '.txt']
        files_found = []
        
        for file in os.listdir(input_dir):
            ext = os.path.splitext(file)[1].lower()
            if ext in supported_extensions:
                source_path = os.path.join(input_dir, file)
                dest_path = os.path.join(log_dir, file)
                
                # Copy file to cache
                shutil.copy2(source_path, dest_path)
                
                # Load data into context
                if ext == '.npy':
                    data = np.load(source_path)
                elif ext == '.txt':
                    with open(source_path, 'r') as f:
                        data = f.read()
                
                context['data'][file] = data
                files_found.append(file)
                print(f"Loaded: {file}")

        if not files_found:
            print(f"No valid data files found in {input_dir}")
        
        return context
