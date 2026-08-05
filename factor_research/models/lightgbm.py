"""基于 LightGBM 的方向二分类模型及其工厂。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from lightgbm import LGBMClassifier

from .base import DirectionModel, DirectionModelFactory
from .registry import register_model_factory


class LightGBMClassifier(DirectionModel):
    """使用支持多线程的直方图梯度提升树预测下一交易日方向。"""

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.03,
        num_leaves: int = 15,
        max_depth: int = 5,
        min_child_samples: int = 50,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        n_jobs: int = -1,
        random_state: int | None = 42,
    ):
        self.estimator = LGBMClassifier(
            objective="binary",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            subsample=subsample,
            # LightGBM 只有在 subsample_freq > 0 时才执行行采样。
            subsample_freq=1 if subsample < 1.0 else 0,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            n_jobs=n_jobs,
            random_state=random_state,
            importance_type="gain",
            verbosity=-1,
        )
        self.feature_importances_ = np.array([], dtype=float)
        self._constant_probability: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LightGBMClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if X.ndim != 2 or len(X) != len(y) or len(X) == 0:
            raise ValueError("X 和 y 的形状不合法")
        if not np.isin(y, [0, 1]).all():
            raise ValueError("y 必须是 0/1 标签")

        classes = np.unique(y)
        if len(classes) == 1:
            # 早期滚动窗口可能只有一个类别，直接返回已观察到的类别概率。
            self._constant_probability = float(classes[0])
            self.feature_importances_ = np.zeros(X.shape[1], dtype=float)
            return self

        self._constant_probability = None
        self.estimator.fit(X, y)
        self.feature_importances_ = np.asarray(
            self.estimator.feature_importances_, dtype=float
        ).copy()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self._constant_probability is not None:
            positive = np.full(len(X), self._constant_probability, dtype=float)
            return np.column_stack([1 - positive, positive])
        if not hasattr(self.estimator, "classes_"):
            raise RuntimeError("模型尚未训练")
        return np.asarray(self.estimator.predict_proba(X), dtype=float)


@register_model_factory
@dataclass(frozen=True)
class LightGBMModelFactory(DirectionModelFactory):
    """从 CLI 配置创建相互独立的 LightGBM 模型。"""

    n_estimators: int = 300
    learning_rate: float = 0.03
    num_leaves: int = 15
    max_depth: int = 5
    min_child_samples: int = 50
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    n_jobs: int = -1
    random_state: int | None = 42
    name: ClassVar[str] = "lightgbm"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--n-estimators", type=int, default=300)
        parser.add_argument("--learning-rate", type=float, default=0.03)
        parser.add_argument("--num-leaves", type=int, default=15)
        parser.add_argument("--max-depth", type=int, default=5)
        parser.add_argument("--min-child-samples", type=int, default=50)
        parser.add_argument("--subsample", type=float, default=0.8)
        parser.add_argument("--colsample-bytree", type=float, default=0.8)
        parser.add_argument("--reg-alpha", type=float, default=0.1)
        parser.add_argument("--reg-lambda", type=float, default=1.0)
        parser.add_argument(
            "--n-jobs",
            type=int,
            default=-1,
            help="LightGBM 训练线程数；-1 表示使用全部可用 CPU",
        )
        parser.add_argument("--random-state", type=int, default=42)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "LightGBMModelFactory":
        return cls(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            max_depth=args.max_depth,
            min_child_samples=args.min_child_samples,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            reg_alpha=args.reg_alpha,
            reg_lambda=args.reg_lambda,
            n_jobs=args.n_jobs,
            random_state=args.random_state,
        )

    def create(self) -> LightGBMClassifier:
        return LightGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
