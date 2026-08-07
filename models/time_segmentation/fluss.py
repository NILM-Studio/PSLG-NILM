import numpy as np
import matplotlib
import os

# Configure matplotlib for headless environment
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def fluss(ts, window_size, n_regimes=3, excl_factor=1, visualize=False):
    """
    FLUSS (Fast Low-cost Unsupervised Time Series Segmentation) implementation using stumpy.
    """
    # Lazy import stumpy to avoid conflicts and reduce startup time
    import stumpy
    
    # Ensure input is 1D float64 array
    ts = np.asarray(ts, dtype=np.float64).flatten()
    
    # Pre-check: Minimum length requirement for stumpy.stump
    if len(ts) < window_size * 2:
        print(f"[FLUSS] Signal length ({len(ts)}) too short for window_size ({window_size})")
        return ts, []

    # Pre-check: Constant signal check (stumpy.stump can have issues with zero variance)
    if np.allclose(ts, ts[0], atol=1e-9):
        print("[FLUSS] Signal is constant, skipping segmentation.")
        return ts, []

    # 1. Compute Matrix Profile
    try:
        # Using stump for Matrix Profile
        mp_res = stumpy.stump(ts, window_size)
        
        # Extract Matrix Profile (MP) and Matrix Profile Index (MPI)
        # We explicitly cast to avoid "ufunc 'isinf' not supported for the input types" error
        # which can happen if mp_res is an object array or structured array.
        mp = mp_res[:, 0].astype(np.float64)
        mpi = mp_res[:, 1].astype(np.int64)
        
        # Check if any valid distances were found
        if np.all(np.isinf(mp)):
            print("[FLUSS] No valid matches found in Matrix Profile (all distances are inf)")
            return ts, []
        
    except Exception as e:
        print(f"[FLUSS] stumpy.stump error: {e}")
        import traceback
        traceback.print_exc()
        return ts, []

    # 2. Compute FLUSS
    num_subseq = len(mpi)
    
    # Robustly adjust excl_factor if it's too large for the current signal
    effective_excl_factor = excl_factor
    if effective_excl_factor * window_size >= num_subseq:
        print(f"[FLUSS] excl_factor * window_size ({effective_excl_factor * window_size}) >= subsegments ({num_subseq}). Reducing.")
        effective_excl_factor = max(1, (num_subseq - 1) // window_size)
        if effective_excl_factor * window_size >= num_subseq:
            print("[FLUSS] Signal still too short for FLUSS logic after reduction. Skipping.")
            return ts, []

    try:
        # stumpy.fluss returns Corrected Arc Curve (CAC) and regime locations
        cac, regime_locations = stumpy.fluss(
            mpi, 
            L=window_size, 
            n_regimes=n_regimes, 
            excl_factor=effective_excl_factor
        )
        
        # Ensure regime_locations is a numpy array and filter out 0/N boundaries if they appear
        regime_locations = np.asarray(regime_locations)
        print(f"[FLUSS] Detected regime locations: {regime_locations}")
        
        if visualize:
            fluss_visualize(ts, mp, mpi, None, cac, regime_locations, window_size)
            
        return ts, regime_locations
        
    except Exception as e:
        print(f"[FLUSS] stumpy.fluss error: {e}")
        import traceback
        traceback.print_exc()
        return ts, []

def fluss_visualize(ts, mp=None, mpi=None, ac=None, cac=None, segments=None, window_size=None):
    """
    Visualize FLUSS results and save to project output directory.
    """
    # Use standard sans-serif font
    plt.rcParams['axes.unicode_minus'] = False

    subplot_count = 1
    if mp is not None: subplot_count += 1
    if mpi is not None: subplot_count += 1
    if ac is not None: subplot_count += 1
    if cac is not None: subplot_count += 1

    plt.figure(figsize=(12, 3 * subplot_count))
    plt.suptitle('FLUSS Time Series Segmentation', fontsize=16)

    curr = 1
    
    # Original Series
    plt.subplot(subplot_count, 1, curr)
    plt.plot(ts, label='Time Series')
    if segments is not None:
        for seg in segments:
            plt.axvline(x=seg, color='r', linestyle='--', alpha=0.8)
    plt.title('Segmentation Result')
    plt.legend()
    curr += 1

    # Matrix Profile
    if mp is not None:
        plt.subplot(subplot_count, 1, curr)
        plt.plot(mp, color='blue')
        plt.title('Matrix Profile (Distance)')
        curr += 1

    # MPI
    if mpi is not None:
        plt.subplot(subplot_count, 1, curr)
        plt.plot(mpi, color='green')
        plt.title('Matrix Profile Index')
        curr += 1

    # CAC
    if cac is not None:
        plt.subplot(subplot_count, 1, curr)
        plt.plot(cac, color='purple')
        plt.title('Corrected Arc Curve (CAC)')
        curr += 1

    plt.tight_layout()
    
    # Save to a sensible project directory (output/figures)
    # We don't have direct access to workflow context here, so we save to a generic path
    # or skip if directory creation fails.
    try:
        save_dir = 'output/figures/fluss'
        os.makedirs(save_dir, exist_ok=True)
        from datetime import datetime
        ts_str = datetime.now().strftime('%H%M%S')
        plt.savefig(f'{save_dir}/fluss_{ts_str}.png', dpi=150)
    except Exception as e:
        print(f"[FLUSS] Visualization save failed: {e}")
    
    plt.close()

if __name__ == "__main__":
    # Test code
    test_ts = np.concatenate([
        np.random.normal(0, 1, 100),
        np.random.normal(5, 1, 100),
        np.random.normal(0, 1, 100)
    ])
    fluss(test_ts, window_size=20, n_regimes=3)
