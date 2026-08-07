from abc import abstractmethod
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from models.base_model import BaseModel

class BaseActiveDetector(BaseModel):
    def __init__(self, name: str, config: dict = None):
        super().__init__(name, config)
        self.appliance_name = self.config.get("appliance_name") or "appliance"
        self.fs = self.config.get("fs") if self.config.get("fs") is not None else 1
        self.context_seconds = self.config.get("context_seconds") if self.config.get("context_seconds") is not None else 60

    @abstractmethod
    def train(self, raw_power_series: np.ndarray, timestamps: np.ndarray = None):
        """学习提取参数"""
        pass

    @abstractmethod
    def detect(self, raw_power_series: np.ndarray, timestamps: np.ndarray) -> List[Dict[str, Any]]:
        """执行检测并返回区间数据"""
        pass

    def save(self, path: str):
        """默认不保存模型文件，如需保存子类实现"""
        pass

    def load(self, path: str):
        """默认不加载模型文件，如需加载子类实现"""
        pass
