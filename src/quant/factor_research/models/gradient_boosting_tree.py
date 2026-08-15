"""基于 scikit-learn 的梯度提升树二分类模型及其工厂。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from .base import DirectionModel, DirectionModelFactory
from .registry import register_model_factory


class GradientBoostingTreeClassifier(DirectionModel):
    """使用多棵回归树逐步拟合分类损失的方向预测模型。"""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_leaf: int = 20,
        subsample: float = 1.0,
        random_state: int | None = 42,
    ):
        self.estimator = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            subsample=subsample,
            random_state=random_state,
        )
        self.feature_importances_ = np.array([], dtype=float)
        self._constant_probability: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingTreeClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if X.ndim != 2 or len(X) != len(y) or len(X) == 0:
            raise ValueError("X和y的形状不合法")
        if not np.isin(y, [0, 1]).all():
            raise ValueError("y必须是0/1标签")

        classes = np.unique(y)
        if len(classes) == 1:
            # sklearn 梯度提升要求至少两个类别；早期滚动窗口用已观察类别回退。
            self._constant_probability = float(classes[0])
            self.feature_importances_ = np.zeros(X.shape[1], dtype=float)
            return self

        self._constant_probability = None
        self.estimator.fit(X, y)
        self.feature_importances_ = self.estimator.feature_importances_.copy()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self._constant_probability is not None:
            positive = np.full(len(X), self._constant_probability, dtype=float)
            return np.column_stack([1 - positive, positive])
        if not hasattr(self.estimator, "classes_"):
            raise RuntimeError("模型尚未训练")
        return self.estimator.predict_proba(X)


@register_model_factory
@dataclass(frozen=True)
class GradientBoostingTreeModelFactory(DirectionModelFactory):
    """从 CLI 配置创建相互独立的 scikit-learn 梯度提升树。"""

    n_estimators: int = 100
    learning_rate: float = 0.1
    max_depth: int = 3
    min_samples_leaf: int = 20
    subsample: float = 1.0
    random_state: int | None = 42
    name: ClassVar[str] = "gradient_boosting_tree"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--n-estimators",
            type=int,
            default=100,
            help="梯度提升迭代次数，即树的数量；默认 100",
        )
        parser.add_argument(
            "--learning-rate",
            type=float,
            default=0.1,
            help="每棵树的贡献缩减系数；默认 0.1",
        )
        parser.add_argument("--max-depth", type=int, default=3)
        parser.add_argument(
            "--min-samples-leaf",
            type=int,
            default=20,
            help="每棵基学习器叶节点的最少样本数；默认 20",
        )
        parser.add_argument(
            "--subsample",
            type=float,
            default=1.0,
            help="每轮训练使用的样本比例，范围 (0, 1]；默认 1.0",
        )
        parser.add_argument(
            "--random-state",
            type=int,
            default=42,
            help="随机种子；默认 42",
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "GradientBoostingTreeModelFactory":
        return cls(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            subsample=args.subsample,
            random_state=args.random_state,
        )

    def create(self) -> GradientBoostingTreeClassifier:
        return GradientBoostingTreeClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            subsample=self.subsample,
            random_state=self.random_state,
        )
