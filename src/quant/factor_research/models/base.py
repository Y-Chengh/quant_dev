"""方向预测模型及其工厂的抽象接口。"""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

FitProgressCallback = Callable[[int, int], None]
"""训练进度回调：依次接收已完成轮数（从 1 起）与本次训练的总轮数。"""


class DirectionModel(ABC):
    """滚动方向预测所需的最小模型接口。

    ``set_fit_progress`` 是可选能力，用于在一次 ``fit`` 内部按迭代轮上报进度：

    .. code-block:: python

        def set_fit_progress(self, callback: FitProgressCallback | None) -> int | None

    实现该方法的模型应在训练过程中按轮调用 ``callback(已完成轮数, 总轮数)``，并
    返回本次训练预计的总轮数；传入 ``None`` 表示取消上报。无法预知轮数时返回
    ``None``。未实现该方法的模型不受影响，整段训练在调用方的进度显示中只占一步。
    该能力只用于进度显示，不得改变训练结果。
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> DirectionModel:
        """使用二维特征矩阵和 0/1 标签训练模型。"""

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """返回形状为（样本数, 2）的下跌、上涨概率。"""

    def predict(self, X: np.ndarray) -> np.ndarray:
        """返回连续预测值；仅回归模型需要实现。"""
        raise NotImplementedError(f"{type(self).__name__} 不支持连续值预测")


class DirectionModelFactory(ABC):
    """定义模型的 CLI 配置方式，并为每个预测日期创建全新模型。"""

    name: str
    supported_tasks: tuple[str, ...] = ("classification",)
    required_finite_feature_indices: tuple[int, ...] = ()

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """向主程序注册该模型专属的命令行参数。"""

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> DirectionModelFactory:
        """读取命令行参数；无专属配置的工厂默认使用无参构造。"""
        return cls()

    @abstractmethod
    def create(self) -> DirectionModel:
        """创建尚未训练且不共享历史状态的模型实例。"""
