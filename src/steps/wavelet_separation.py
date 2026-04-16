import pandas as pd
import numpy as np
import pywt
import matplotlib.pyplot as plt
import os
import gc
import shutil
from scipy.signal import medfilt
from models.claspy.segmentation import BinaryClaSPSegmentation
from src.framework.step import Step

class WaveletSeparationStep(Step):
    """
    Step for wavelet separation and segmentation of power signals.
    Inherits from Step base class.
    """
    def __init__(self, name="WaveletSeparation", is_shape_dtw=False):
        super().__init__(name)
        self.is_shape_dtw = is_shape_dtw

    def medfilt_outlier_removal(self, series):
        """Perform outlier removal using median filter."""
        ts = np.asarray(series)
        cleaned_series = medfilt(ts, kernel_size=5)
        outlier_mask = np.zeros_like(ts, dtype=bool)
        return cleaned_series, outlier_mask

    def get_segmentation_points(self, time_series, distance="znormed_euclidean_distance"):
        """Segmentation logic using BinaryClaSPSegmentation."""
        try:
            clasp = BinaryClaSPSegmentation(
                n_segments="learn",
                window_size="suss",
                validation="score_threshold",
                threshold=0.001,
                distance=distance,
            )
            clasp.fit_predict(time_series)
            return clasp.change_points
        except Exception as e:
            print(f"Segmentation error: {e}")
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
        
        if self.is_shape_dtw:
            low_cp = self.get_segmentation_points(low_freq_signal, distance="shape_dtw")
        else:
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

    def plot_results(self, signal, signal_cleaned, orig_cp, results, output_dir, file_name):
        """Generates the 4-panel plot and heatmap."""
        wavelet = results['wavelet']
        low_freq_signal = results['low_freq_signal']
        high_freq_combined = results['high_freq_combined']
        low_cp = results['low_cp']
        high_cp = results['high_cp']
        synth_cp = results['synthesized_cp']
        ref_name = results['ref_name']
        
        plt.figure(figsize=(15, 12))
        blue = (74/255, 75/255, 157/255)
        red = (200/255, 22/255, 29/255)
        green = (90/255, 164/255, 174/255)
        yellow = (250/255, 192/255, 61/255)
        synth_color = (166/255, 85/255, 157/255)
        cleaned_color = (204/255, 93/255, 32/255)
        
        # 1. Original vs Cleaned
        plt.subplot(4, 1, 1)
        plt.plot(signal, label='Original Signal', color='gray', alpha=0.6)
        plt.plot(signal_cleaned, label='Cleaned Signal', color=cleaned_color, alpha=0.8)
        for cp in orig_cp:
            plt.axvline(x=cp, color=red, linestyle='--', alpha=0.8)
        plt.title(f'Signal Comparison - {file_name}')
        plt.legend(loc='upper right')
        plt.grid(True)
        
        # 2. Low Frequency
        plt.subplot(4, 1, 2)
        plt.plot(low_freq_signal, label=f'Low Freq ({wavelet})', color=blue)
        for cp in low_cp:
            plt.axvline(x=cp, color=red, linestyle='--', alpha=0.8)
        plt.legend(loc='upper right')
        plt.grid(True)
        
        # 3. High Frequency
        plt.subplot(4, 1, 3)
        plt.plot(high_freq_combined, label=f'High Freq ({wavelet})', color=green, alpha=0.8)
        for cp in high_cp:
            plt.axvline(x=cp, color=red, linestyle='--', alpha=0.8)
        plt.legend(loc='upper right')
        plt.grid(True)
        
        # 4. Comparison
        plt.subplot(4, 1, 4)
        plt.plot(signal, label='Original', color=blue, alpha=0.8)
        for i, cp in enumerate(orig_cp):
            plt.axvline(x=cp, color=green, linestyle='--', linewidth=1, alpha=1.0, label='Orig CP' if i == 0 else "")
        for i, cp in enumerate(synth_cp):
            plt.axvline(x=cp, color=synth_color, linestyle='-', linewidth=2, alpha=1.0, label=f'Synth CP ({ref_name})' if i == 0 else "")
        plt.legend(loc='upper right', fontsize='small', ncol=2)
        plt.grid(True)
        
        plt.tight_layout()
        filename_base = file_name.split('.')[0]
        plot_path = os.path.join(output_dir, f'wavelet_separation_{filename_base}_{wavelet}.png')
        plt.savefig(plot_path)
        plt.close()

        # Heatmap
        scales = np.arange(1, 128)
        cwt_wavelet = 'cmor1.5-1.0'
        try:
            cwtmatr, freqs = pywt.cwt(low_freq_signal, scales, cwt_wavelet)
        except:
            cwt_wavelet = 'mexh'
            cwtmatr, freqs = pywt.cwt(low_freq_signal, scales, cwt_wavelet)
        
        plt.figure(figsize=(15, 8))
        plt.imshow(np.abs(cwtmatr), extent=[0, len(low_freq_signal), scales[0], scales[-1]], 
                   cmap='jet', aspect='auto', interpolation='nearest', origin='lower')
        plt.colorbar(label='Energy')
        plt.title(f'Scalogram - {file_name}')
        heatmap_path = os.path.join(output_dir, f'wavelet_heatmap_{filename_base}.png')
        plt.savefig(heatmap_path)
        plt.close()

    def export_data(self, df, results, output_dir, file_name):
        """Exports processed signals and changepoints."""
        filename_base = file_name.split('.')[0]
        data_dir = os.path.join(output_dir, 'data')
        label_dir = os.path.join(output_dir, 'label')
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

        # Signal Export
        signal_data = {
            "timestamp": df["timestamp"] if "timestamp" in df.columns else df.index,
            "power": df["power"],
            "cleaned_power": results['cleaned_signal'],
            "high_freq": results['high_freq_combined'],
            "low_freq": results['low_freq_signal']
        }
        pd.DataFrame(signal_data).to_csv(os.path.join(data_dir, f"{filename_base}.csv"), index=False)

        # Label Export
        cp_export = []
        for cp_list, label_type in [(results['synthesized_cp'], 0), (results['low_cp'], 1), (results['high_cp'], 2)]:
            for cp in cp_list:
                idx = int(round(cp))
                if 0 <= idx < len(df):
                    cp_export.append({
                        "timestamp": df.iloc[idx]["timestamp"] if "timestamp" in df.columns else idx,
                        "power": df.iloc[idx]["power"],
                        "changepoint_index": idx,
                        "label_type": label_type
                    })
        if cp_export:
            pd.DataFrame(cp_export).to_csv(os.path.join(label_dir, f"Changepoints_{filename_base}.csv"), index=False)

    def run(self, context: dict) -> dict:
        """Main execution logic for WaveletSeparationStep."""
        input_dir = os.path.join(context['log_root'], 'DataLoader')
        log_dir = self.get_log_dir(context) # This is log/{seq_id}/WaveletSeparation
        
        if not os.path.exists(input_dir):
            print(f"Error: Input directory {input_dir} not found.")
            return context

        target_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.csv')]
        print(f"Found {len(target_files)} CSV files for wavelet analysis.")

        for i, file_name in enumerate(target_files):
            print(f"Processing ({i+1}/{len(target_files)}): {file_name}")
            
            # Load
            csv_path = os.path.join(input_dir, file_name)
            df = pd.read_csv(csv_path)
            if 'power' not in df.columns:
                print(f"Skipping {file_name}: 'power' column not found.")
                continue
            
            signal = df['power'].values
            
            # 1. Outlier removal (apply_diff=False logic: signal_cleaned = signal)
            signal_cleaned, _ = self.medfilt_outlier_removal(signal)
            
            # 2. Initial segmentation
            orig_cp = self.get_segmentation_points(signal_cleaned, distance="shape_dtw" if self.is_shape_dtw else "znormed_euclidean_distance")
            
            # 3. Wavelet analysis (test db4 as best choice as per original logic)
            res = self.run_wavelet_analysis(signal_cleaned, 'db4', orig_cp)
            
            # 4. Export
            self.export_data(df, res, log_dir, file_name)
            
            # 5. Plot (first 10 files)
            if i < 10:
                self.plot_results(signal, signal_cleaned, orig_cp, res, log_dir, file_name)
                
            del df, signal, signal_cleaned, res
            gc.collect()

        return context
