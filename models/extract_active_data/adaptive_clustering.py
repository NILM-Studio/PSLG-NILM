import numpy as np
import pandas as pd
import gc
from sklearn.cluster import KMeans
from typing import List, Dict, Any
from .base import BaseActiveDetector

class AdaptiveClusteringDetector(BaseActiveDetector):
    def __init__(self, name: str, config: dict = None):
        super().__init__(name, config)
        # 兼容 threshold 和 power_threshold 两种配置命名
        self.threshold = self.config.get("threshold") or self.config.get("power_threshold")
        self.t_drop = self.config.get("t_drop", None)
        self.t_min_work = self.config.get("t_min_work", None)

    def _to_unix_timestamps(self, timestamps: np.ndarray) -> np.ndarray:
        """统一转换各种格式的时间戳为 Unix 时间戳 (秒)"""
        if np.issubdtype(timestamps.dtype, np.number):
            return timestamps.astype(float)
        else:
            return pd.to_datetime(timestamps).view(np.int64) // 10**9

    def _moving_average(self, data, window_size=5):
        return np.convolve(data, np.ones(window_size)/window_size, mode='same')

    def fit_parameters_otsu(self, raw_power_series, num_bins=500):
        """基于 Otsu 算法自适应学习阈值 (幅度域)"""
        print(f"[{self.name}] 开始基于 Otsu 概率密度直方图的自适应阈值学习...")
        smoothed_power = self._moving_average(raw_power_series, window_size=5)
        min_p, max_p = np.min(smoothed_power), np.max(smoothed_power)
        counts, bin_edges = np.histogram(smoothed_power, bins=num_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        total_samples = len(smoothed_power)

        max_variance = -1
        best_threshold = (min_p + max_p) / 2
        saved_mean_0, saved_mean_1 = 0, 0

        for i in range(1, num_bins - 1):
            counts_0, bin_c_0 = counts[:i], bin_centers[:i]
            sum_0 = np.sum(counts_0)
            if sum_0 == 0: continue
            mean_0 = np.sum(counts_0 * bin_c_0) / sum_0

            counts_1, bin_c_1 = counts[i:], bin_centers[i:]
            sum_1 = np.sum(counts_1)
            if sum_1 == 0: continue
            mean_1 = np.sum(counts_1 * bin_c_1) / sum_1

            between_class_variance = (sum_0 / total_samples) * (sum_1 / total_samples) * ((mean_0 - mean_1) ** 2)
            if between_class_variance > max_variance:
                max_variance = between_class_variance
                best_threshold = bin_centers[i]
                saved_mean_0, saved_mean_1 = mean_0, mean_1

        sleep_data = smoothed_power[smoothed_power < best_threshold]
        safe_ceiling = np.mean(sleep_data) + 6 * np.std(sleep_data)
        self.threshold = float(min(best_threshold, safe_ceiling))

        print(f"  [幅值域] Otsu 密度切分完成。阈值设定为: {self.threshold:.2f} W")
        del smoothed_power, counts, bin_edges, bin_centers, sleep_data
        gc.collect()
        return self

    def find_time_parameters(self, power_series: np.ndarray, times: np.ndarray, threshold: float):
        """通过分析活跃点之间的时间间隔，自动寻找 T_drop 和 T_min_work"""
        
        # 1. 活跃点识别
        active_indices = np.where(power_series > threshold)[0]
        if len(active_indices) < 5:
            return 15.0, 30.0
        
        # 2. 计算活跃点之间的时间间隔 (Time Gaps)
        # 只有当索引连续但时间不连续，或者索引不连续时，才存在 gap
        # 这里直接计算所有活跃点之间的时间差
        time_gaps = np.diff(times[active_indices])
        
        # 3. 聚类分析时间间隔以确定 t_drop
        # 正常工作状态下的内部间隔 (短) vs 两次活动之间的间隔 (长)
        if len(time_gaps) >= 2:
            X_gap = np.log1p(time_gaps).reshape(-1, 1)
            kmeans_gap = KMeans(n_clusters=2, random_state=42, n_init='auto').fit(X_gap)
            short_cluster_idx = np.argmin(kmeans_gap.cluster_centers_.flatten())
            short_gaps = time_gaps[kmeans_gap.labels_ == short_cluster_idx]
            t_drop_learned = float(np.max(short_gaps) * 1.2) if len(short_gaps) > 0 else 15.0
        else:
            t_drop_learned = 15.0

        # 4. 基于初步的 t_drop 划分区间，分析区间长度以确定 t_min_work
        durations = []
        if len(active_indices) > 0:
            start_t = times[active_indices[0]]
            for i in range(1, len(active_indices)):
                if (times[active_indices[i]] - times[active_indices[i-1]]) > t_drop_learned:
                    durations.append(times[active_indices[i-1]] - start_t)
                    start_t = times[active_indices[i]]
            durations.append(times[active_indices[-1]] - start_t)

        if len(durations) >= 2:
            X_dur = np.log1p(durations).reshape(-1, 1)
            kmeans_dur = KMeans(n_clusters=2, random_state=42, n_init='auto').fit(X_dur)
            long_cluster_idx = np.argmax(kmeans_dur.cluster_centers_.flatten())
            # 最小工作时间应区分“噪声脉冲”与“真实活动”
            noise_durations = np.array(durations)[kmeans_dur.labels_ != long_cluster_idx]
            t_min_work_learned = float(np.max(noise_durations) + 1.0) if len(noise_durations) > 0 else 30.0
        else:
            t_min_work_learned = 30.0

        return t_drop_learned, t_min_work_learned

    def train(self, raw_power_series: np.ndarray, timestamps: np.ndarray = None):
        print(f"[{self.name}] 开始自适应参数学习...")
        times = self._to_unix_timestamps(timestamps)
        
        if self.threshold is None:
            self.fit_parameters_otsu(raw_power_series)
        
        learned_t_drop, learned_t_min_work = self.find_time_parameters(raw_power_series, times, self.threshold)
        
        if self.t_drop is None: self.t_drop = learned_t_drop
        if self.t_min_work is None: self.t_min_work = learned_t_min_work

        print(f"  学习结果: threshold={self.threshold:.2f}, t_drop={self.t_drop:.2f}, t_min_work={self.t_min_work:.2f}")
        gc.collect()

    def detect(self, raw_power_series: np.ndarray, timestamps: np.ndarray) -> List[Dict[str, Any]]:
        times = self._to_unix_timestamps(timestamps)
        if self.threshold is None:
            self.train(raw_power_series, timestamps)
            
        active_indices = np.where(raw_power_series >= self.threshold)[0]
        if len(active_indices) == 0: return []

        work_intervals = []
        current_group = [active_indices[0]]
        
        for i in range(1, len(active_indices)):
            curr_idx = active_indices[i]
            prev_idx = active_indices[i-1]
            if (times[curr_idx] - times[prev_idx]) <= self.t_drop:
                current_group.append(curr_idx)
            else:
                self._process_group(work_intervals, current_group, raw_power_series, times, timestamps)
                current_group = [curr_idx]
        
        if current_group:
            self._process_group(work_intervals, current_group, raw_power_series, times, timestamps)

        del times, active_indices
        gc.collect()
        return work_intervals

    def _process_group(self, work_intervals, group, powers, times, orig_times):
        start_idx, end_idx = group[0], group[-1]
        duration = times[end_idx] - times[start_idx]
        if duration >= self.t_min_work:
            context_pts = int(self.context_seconds * self.fs)
            c_start, c_end = max(0, start_idx - context_pts), min(len(powers) - 1, end_idx + context_pts)
            work_intervals.append({
                "start_time": int(times[start_idx]),
                "end_time": int(times[end_idx]),
                "duration_sec": float(duration),
                "data": pd.DataFrame({
                    "timestamp": orig_times[c_start:c_end + 1],
                    "power": powers[c_start:c_end + 1]
                })
            })
