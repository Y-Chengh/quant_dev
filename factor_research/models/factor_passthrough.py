"""原样输出最后一个因子值的无训练回归模型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from .base import DirectionModel, DirectionModelFactory
from .registry import register_model_factory


class FactorPassthroughRegressor(DirectionModel):
    """把特征矩阵最后一列直接作为连续预测值。"""

    def __init__(self) -> None:
        """创建尚未接收输入结构的直出模型。"""

        self.n_features_in_: int | None = None

    @staticmethod
    def _matrix(X: np.ndarray) -> np.ndarray:
        """校验并转换待训练或预测的二维有限特征矩阵。

        参数：
            X: 样本特征矩阵；最后一列必须是要原样输出的候选因子值。

        返回：
            浮点二维数组，不复制已经满足要求的输入。
        """

        matrix = np.asarray(X, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] == 0:
            raise ValueError("X 必须是至少包含一列特征的二维矩阵")
        if not np.isfinite(matrix).all():
            raise ValueError("X 不能包含 NaN 或无穷值")
        return matrix

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FactorPassthroughRegressor":
        """记录输入列数，不从目标值学习任何参数。

        参数：
            X: 训练样本特征矩阵；最后一列是后续要直出的因子。
            y: 与训练样本等长的目标收益；仅校验长度，不参与拟合。

        返回：
            当前模型实例。
        """

        matrix = self._matrix(X)
        target = np.asarray(y)
        if target.ndim != 1 or len(target) != len(matrix):
            raise ValueError("y 必须是与 X 样本数相同的一维数组")
        self.n_features_in_ = matrix.shape[1]
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """拒绝概率预测，因为该模型只支持连续因子回归。

        参数：
            X: 任意待预测特征矩阵；该参数不会被读取。

        返回：
            本方法不会返回结果，总是抛出 ``NotImplementedError``。
        """

        raise NotImplementedError("factor_passthrough 仅支持 regression 任务")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """原样返回预测矩阵的最后一列因子值。

        参数：
            X: 待预测二维特征矩阵；列数必须与拟合阶段一致。

        返回：
            与输入样本数相同的一维有限因子值数组。
        """

        if self.n_features_in_ is None:
            raise RuntimeError("模型尚未拟合")
        matrix = self._matrix(X)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError(
                "预测特征列数与拟合阶段不一致: "
                f"expected={self.n_features_in_} actual={matrix.shape[1]}"
            )
        return matrix[:, -1].copy()


@register_model_factory
@dataclass(frozen=True)
class FactorPassthroughModelFactory(DirectionModelFactory):
    """创建以最后一列为输出、无共享训练状态的直出模型。"""

    name: ClassVar[str] = "factor_passthrough"
    supported_tasks: ClassVar[tuple[str, ...]] = ("regression",)
    required_finite_feature_indices: ClassVar[tuple[int, ...]] = (-1,)

    def create(self) -> FactorPassthroughRegressor:
        """创建一个尚未拟合的因子直出模型。

        返回：
            无历史状态的 ``FactorPassthroughRegressor`` 实例。
        """

        return FactorPassthroughRegressor()
