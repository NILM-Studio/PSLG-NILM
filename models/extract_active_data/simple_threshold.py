import numpy as np
import pandas as pd
import gc
from typing import List, Dict, Any
from .base import BaseActiveDetector

class SimpleThresholdDetector(BaseActiveDetector):
    def __init__(self, name: str, config: dict = None):
        super().__init__(name, config)
        # 兼容 threshold 和 power_threshold 两种配置命名
        self.threshold = self.config.get("threshold")
        if self.threshold is None:
            self.threshold = self.config.get("power_threshold", 1.0)
            
        # 跌落容忍时间 (秒)
        self.t_drop = self.config.get("t_drop", 0)
        
        # 最小工作时长 (秒)
        self.t_min_work = self.config.get("t_min_work")
        if self.t_min_work is None:
            self.t_min_work = self.config.get("min_duration_seconds", 30)

    def _to_unix_timestamps(self, timestamps: np.ndarray) -> np.ndarray:
        """统一转换各种格式的时间戳为 Unix 时间戳 (秒)"""
        if np.issubdtype(timestamps.dtype, np.number):
            return timestamps.astype(float)
        else:
            return pd.to_datetime(timestamps).view(np.int64) // 10**9

    def train(self, raw_power_series: np.ndarray, timestamps: np.ndarray = None):
        """简单阈值模型无需训练"""
        print(f"[{self.name}] 使用固定参数: threshold={self.threshold} W, t_drop={self.t_drop} s, t_min_work={self.t_min_work} s")
        pass

    def detect(self, raw_power_series: np.ndarray, timestamps: np.ndarray) -> List[Dict[str, Any]]:
        """
        基于时间间隔的区间提取逻辑：
        1. 不进行补零重采样，直接在原始时间线上计算
        2. 如果两个活跃点之间的时间差 <= t_drop，则视为同一区间
        """
        if len(raw_power_series) == 0:
            return []

        # 1. 统一时间格式
        times = self._to_unix_timestamps(timestamps)
        
        # 2. 找到所有超过阈值的活跃点索引
        active_indices = np.where(raw_power_series >= self.threshold)[0]
        if len(active_indices) == 0:
            return []

        # 3. 基于 t_drop 进行聚类 (直接在时间戳上判断)
        work_intervals = []
        current_group = [active_indices[0]]
        
        for i in range(1, len(active_indices)):
            curr_idx = active_indices[i]
            prev_idx = active_indices[i-1]
            
            # 计算实际时间差
            time_diff = times[curr_idx] - times[prev_idx]
            
            if time_diff <= self.t_drop:
                current_group.append(curr_idx)
            else:
                # 结束当前组，检查是否满足最小工作时间
                self._process_group(work_intervals, current_group, raw_power_series, times, timestamps)
                current_group = [curr_idx]
        
        # 处理最后一组
        if current_group:
            self._process_group(work_intervals, current_group, raw_power_series, times, timestamps)

        # 内存管理
        del times, active_indices
        gc.collect()
        
        return work_intervals

    def _process_group(self, work_intervals, group, powers, times, orig_times):
        start_idx, end_idx = group[0], group[-1]
        duration = times[end_idx] - times[start_idx]
        
        if duration >= self.t_min_work:
            # 获取上下文范围
            context_pts = int(self.context_seconds * self.fs)
            c_start = max(0, start_idx - context_pts)
            c_end = min(len(powers) - 1, end_idx + context_pts)
            
            work_intervals.append({
                "start_time": int(times[start_idx]),
                "end_time": int(times[end_idx]),
                "duration_sec": float(duration),
                "data": pd.DataFrame({
                    "timestamp": orig_times[c_start:c_end + 1],
                    "power": powers[c_start:c_end + 1]
                })
            })

    def _create_interval_dict(self, *args, **kwargs):
        # 此方法已由 _process_group 整合
        pass
