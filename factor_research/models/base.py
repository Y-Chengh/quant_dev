"""方向预测模型及其工厂的抽象接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class DirectionModel(ABC):
    """滚动方向预测所需的最小模型接口。"""

    feature_importances_: np.ndarray

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "DirectionModel":
        """使用二维特征矩阵和 0/1 标签训练模型。"""

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """返回形状为（样本数, 2）的下跌、上涨概率。"""


class DirectionModelFactory(ABC):
    """为每个滚动预测日期创建一个全新模型。"""

    name: str

    @abstractmethod
    def create(self) -> DirectionModel:
        """创建尚未训练且不共享历史状态的模型实例。"""
