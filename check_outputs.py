import numpy as np
import os

def check_outputs(log_dir):
    x_path = os.path.join(log_dir, 'X.npy')
    l_path = os.path.join(log_dir, 'lengths.npy')

    if not os.path.exists(x_path) or not os.path.exists(l_path):
        print(f"Error: Files not found in {log_dir}")
        return

    # Load data
    X = np.load(x_path)
    L = np.load(l_path)

    print("-" * 30)
    print("Output Verification Result")
    print("-" * 30)

    # 1. Check X.npy (n_samples, timestamp, feature)
    print(f"X.npy shape: {X.shape} (n_samples, timestamp, features)")
    print(f"X.npy dtype: {X.dtype}")
    
    # 2. Check lengths.npy (n_samples, 1)
    print(f"lengths.npy shape: {L.shape} (n_samples, 1)")
    print(f"lengths.npy dtype: {L.dtype}")

    # 3. Print first 3 samples' metadata and data preview
    num_to_show = min(3, len(X))
    print(f"\nShowing first {num_to_show} samples:")
    for i in range(num_to_show):
        actual_len = L[i, 0]
        print(f"\nSample {i}:")
        print(f"  Actual Length: {actual_len}")
        # Show a slice of the first 5 timestamps of the sample
        preview_len = min(5, actual_len)
        print(f"  First {preview_len} timestamps (Channel 0, Channel 1):")
        print(X[i, :preview_len, :])
        
        # Verify padding (check the last timestamp if max_len > actual_len)
        if X.shape[1] > actual_len:
            print(f"  Padding check at index {X.shape[1]-1}: {X[i, -1, :]}")

    print("-" * 30)

if __name__ == "__main__":
    target_log_dir = r'f:\B__ProfessionProject\PSLG-NILM\log\20260421_134643\WaveletSeparation'
    check_outputs(target_log_dir)
