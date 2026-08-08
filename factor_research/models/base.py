"""方向预测模型及其工厂的抽象接口。"""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod

import numpy as np


class DirectionModel(ABC):
    """滚动方向预测所需的最小模型接口。"""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "DirectionModel":
        """使用二维特征矩阵和 0/1 标签训练模型。"""

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """返回形状为（样本数, 2）的下跌、上涨概率。"""


class DirectionModelFactory(ABC):
    """定义模型的 CLI 配置方式，并为每个预测日期创建全新模型。"""

    name: str

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """向主程序注册该模型专属的命令行参数。"""

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "DirectionModelFactory":
        """读取命令行参数；无专属配置的工厂默认使用无参构造。"""
        return cls()

    @abstractmethod
    def create(self) -> DirectionModel:
        """创建尚未训练且不共享历史状态的模型实例。"""
