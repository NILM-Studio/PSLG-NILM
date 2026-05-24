import pandas as pd
import numpy as np
import pywt
import matplotlib
matplotlib.use('Agg') # 强制使用非交互式后端
import matplotlib.pyplot as plt
import os
import gc
import sys
from scipy.signal import medfilt
from claspy.segmentation import BinaryClaSPSegmentation
from src.framework.step import Step
from src.steps.dataset_split_step import DatasetSplitStep

class TimeSegmentationStep(Step):
    """
    Step for time series segmentation using wavelet separation and various algorithms (Clasp, FLUSS).
    Inherits from Step base class.
    """
    def __init__(
        self, 
        name="TimeSegmentation", 
        segment_method="clasp",
        appliance_name="",
        window_size=100,
        n_regimes=3,
        excl_factor=5
    ):
        super().__init__(name)
        self.segment_method = segment_method
        self.appliance_name = appliance_name
        # FLUSS specific params
        self.window_size = window_size
        self.n_regimes = n_regimes
        self.excl_factor = excl_factor

    def medfilt_outlier_removal(self, series):
        """Perform outlier removal using median filter."""
        ts = np.asarray(series)
        cleaned_series = medfilt(ts, kernel_size=5)
        outlier_mask = np.zeros_like(ts, dtype=bool)
        return cleaned_series, outlier_mask

    def get_segmentation_points(self, time_series, distance="znormed_euclidean_distance"):
        """Segmentation logic using BinaryClaSPSegmentation or FLUSS."""
        if self.segment_method == "fluss":
            from fluss import fluss
            try:
                # Ensure input is 1D for FLUSS
                ts_1d = time_series.flatten()
                
                # Check if length is sufficient for FLUSS
                if len(ts_1d) < self.window_size + self.n_regimes:
                    print(f"[TimeSegmentation] Skipping FLUSS: signal length ({len(ts_1d)}) too short for window_size ({self.window_size})")
                    return []

                _, change_points = fluss(
                    ts_1d, 
                    window_size=self.window_size, 
                    n_regimes=self.n_regimes, 
                    excl_factor=self.excl_factor,
                    visualize=False
                )
                return change_points
            except Exception as e:
                print(f"[TimeSegmentation] FLUSS error: {e}")
                return []
        elif self.segment_method == "espresso":
            from espresso import espresso_minimize
            try:
                # ESPRESSO is a logic minimizer, its integration here is experimental
                # and currently returns empty change points as a placeholder.
                print(f"[TimeSegmentation] ESPRESSO logic minimizer called (Experimental)")
                # Placeholder: ESPRESSO doesn't directly provide change points for time series
                return []
            except Exception as e:
                print(f"[TimeSegmentation] ESPRESSO error: {e}")
                return []
        else:  # Default: clasp
            try:
                clasp = BinaryClaSPSegmentation(
                    n_segments="learn",
                    window_size="suss",
                    validation="score_threshold",
                    threshold=0.001,
                    distance=distance,
                    n_jobs=1,
                )
                clasp.fit_predict(time_series)
                return clasp.change_points
            except Exception as e:
                print(f"[TimeSegmentation] Clasp error: {e}")
                return []

    def synthesize_changepoints(self, orig_cp, low_cp, high_cp):
        """Synthesizes changepoints from low and high frequency components."""
        if len(low_cp) == 0 and len(high_cp) == 0:
            return [], "None"

        if len(low_cp) >= len(high_cp):
            ref_cp = np.sort(low_cp)
            others = [np.sort(high_cp)]
            ref_name = "Low-Freq"
        else:
            ref_cp = np.sort(high_cp)
            others = [np.sort(low_cp)]
            ref_name = "High-Freq"

        if len(ref_cp) == 0:
            return [], "None"

        groups = {i: [ref_val] for i, ref_val in enumerate(ref_cp)}
        for other_list in others:
            for p in other_list:
                closest_idx = np.argmin(np.abs(ref_cp - p))
                groups[closest_idx].append(p)
                
        synthesized_cp = []
        for i in sorted(groups.keys()):
            group_mean = np.mean(groups[i])
            synthesized_cp.append(group_mean)
            
        return sorted(synthesized_cp), ref_name

    def run_wavelet_analysis(self, signal, wavelet, orig_cp):
        """Performs wavelet separation and segmentation."""
        level = 2
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        cA2, cD2, cD1 = coeffs
        
        zeros_cD2 = np.zeros_like(cD2)
        zeros_cD1 = np.zeros_like(cD1)
        zeros_cA2 = np.zeros_like(cA2)
        
        low_freq_signal = pywt.waverec([cA2, zeros_cD2, zeros_cD1], wavelet)
        high_freq_combined = pywt.waverec([zeros_cA2, cD2, cD1], wavelet)
        
        low_freq_signal = low_freq_signal[:len(signal)]
        high_freq_combined = high_freq_combined[:len(signal)]
        
        low_cp = self.get_segmentation_points(low_freq_signal, distance="znormed_euclidean_distance")
        high_cp = self.get_segmentation_points(high_freq_combined, distance="znormed_euclidean_distance")
        
        synthesized_cp, ref_name = self.synthesize_changepoints(orig_cp, low_cp, high_cp)
        
        return {
            'wavelet': wavelet,
            'low_freq_signal': low_freq_signal,
            'high_freq_combined': high_freq_combined,
            'low_cp': low_cp,
            'high_cp': high_cp,
            'synthesized_cp': synthesized_cp,
            'ref_name': ref_name,
            'num_low_cp': len(low_cp),
            'num_high_cp': len(high_cp),
            'cleaned_signal': signal
        }

    def restore(self, context: dict) -> dict:
        log_dir = self.get_log_dir(context)
        x_path = os.path.join(log_dir, "X.npy")
        lengths_path = os.path.join(log_dir, "lengths.npy")
        indices_path = os.path.join(log_dir, "indices.npy")
        if not (os.path.exists(x_path) and os.path.exists(lengths_path) and os.path.exists(indices_path)):
            return context

        X = np.load(x_path)
        L = np.load(lengths_path)
        I = np.load(indices_path)

        if "data" not in context:
            context["data"] = {}
        context["data"]["X"] = X
        context["data"]["lengths"] = L
        context["data"]["indices"] = I
        return context

    def run(self, context: dict) -> dict:
        """Main execution logic for TimeSegmentationStep."""
        # Memory management: Clear previous steps data if necessary (Sliding context release)
        # As per GUIDELINES.md: Step 3 (TimeSegmentation) should release Step 1 (ExtractActiveData)
        if 'data' in context and 'extract_active_data' in context['data']:
            print("[TimeSegmentation] Releasing Step 1 (ExtractActiveData) context data")
            del context['data']['extract_active_data']
            gc.collect()

        input_dir = context.get('input_root', os.path.join(context['log_root'], 'DataLoader'))
        log_dir = self.get_log_dir(context)
        
        if not os.path.exists(input_dir):
            print(f"Error: Input directory {input_dir} not found.")
            return context

        target_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.csv')])
        print(f"Found {len(target_files)} CSV files for time segmentation.")

        all_samples = []
        all_lengths = []
        all_indices = []

        for i, file_name in enumerate(target_files):
            print(f"Processing ({i+1}/{len(target_files)}): {file_name} using {self.segment_method}", flush=True)
            
            # Load
            csv_path = os.path.join(input_dir, file_name)
            df = pd.read_csv(csv_path)
            if 'power' not in df.columns:
                print(f"Skipping {file_name}: 'power' column not found.")
                continue
            
            signal = df['power'].values
            
            # 1. Outlier removal
            signal_cleaned, _ = self.medfilt_outlier_removal(signal)
            
            # 2. Initial segmentation
            orig_cp = self.get_segmentation_points(signal_cleaned, distance="znormed_euclidean_distance")
            
            # 3. Wavelet analysis
            res = self.run_wavelet_analysis(signal_cleaned, 'db4', orig_cp)
            
            # 4. Extract segments for tensor output
            low_freq = res['low_freq_signal']
            high_freq = res['high_freq_combined']
            cps = sorted(list(set([int(round(cp)) for cp in res['synthesized_cp']])))
            
            # Define segment boundaries
            boundaries = [0] + cps + [len(signal_cleaned)]
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                if end <= start:
                    continue
                
                # Slice and stack: (length, 4)
                # 0: Original, 1: Cleaned, 2: Low-freq, 3: High-freq
                s_orig = signal[start:end]
                s_cleaned = signal_cleaned[start:end]
                s_low = low_freq[start:end]
                s_high = high_freq[start:end]
                
                sample = np.stack([s_orig, s_cleaned, s_low, s_high], axis=1)
                
                all_samples.append(sample)
                all_lengths.append(len(sample))
                all_indices.append([i, start])

            del df, signal, signal_cleaned, res
            gc.collect()

        # Final tensorization and persistence
        if all_samples:
            max_len = max(all_lengths)
            n_samples = len(all_samples)
            X = np.zeros((n_samples, max_len, 4), dtype=np.float32)
            L = np.array(all_lengths, dtype=np.int32).reshape(-1, 1)
            I = np.array(all_indices, dtype=np.int32) # (n_samples, 2)
            
            for idx, sample in enumerate(all_samples):
                length = all_lengths[idx]
                X[idx, :length, :] = sample
            
            # Persistence (Path priority principle: save and deliver paths or ensure small footprint)
            x_path = os.path.join(log_dir, 'X.npy')
            l_path = os.path.join(log_dir, 'lengths.npy')
            i_path = os.path.join(log_dir, 'indices.npy')
            np.save(x_path, X)
            np.save(l_path, L)
            np.save(i_path, I)
            
            # Context delivery
            if 'data' not in context:
                context['data'] = {}
            context['data']['X'] = X
            context['data']['lengths'] = L
            context['data']['indices'] = I
            
            print(f"Tensorization complete: {n_samples} samples, max_length={max_len}")
            print(f"Saved to {log_dir} and updated context['data']")

        # Memory cleanup
        del all_samples, all_lengths, all_indices
        gc.collect()

        return context
