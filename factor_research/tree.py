from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class _Node:
    probability: float
    samples: int
    feature: int | None = None
    threshold: float | None = None
    left: "_Node | None" = None
    right: "_Node | None" = None


class SimpleDecisionTreeClassifier:
    """仅支持数值特征的轻量二分类CART树，适合框架冒烟测试。"""

    def __init__(self, max_depth: int = 3, min_samples_leaf: int = 20, max_thresholds: int = 32):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_thresholds = max_thresholds
        self.root_: _Node | None = None
        self.feature_importances_: np.ndarray | None = None

    @staticmethod
    def _gini(y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        p = float(y.mean())
        return 2 * p * (1 - p)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleDecisionTreeClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if X.ndim != 2 or len(X) != len(y) or len(X) == 0:
            raise ValueError("X和y的形状不合法")
        if not np.isin(y, [0, 1]).all():
            raise ValueError("y必须是0/1标签")
        self.feature_importances_ = np.zeros(X.shape[1], dtype=float)
        self.root_ = self._grow(X, y, depth=0)
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total
        return self

    def _grow(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        node = _Node(probability=float(y.mean()), samples=len(y))
        if depth >= self.max_depth or len(y) < 2 * self.min_samples_leaf or len(np.unique(y)) == 1:
            return node
        parent_impurity = self._gini(y)
        best: tuple[float, int, float, np.ndarray] | None = None
        for feature in range(X.shape[1]):
            values = X[:, feature]
            unique = np.unique(values)
            if len(unique) < 2:
                continue
            if len(unique) > self.max_thresholds:
                quantiles = np.linspace(0, 1, self.max_thresholds + 2)[1:-1]
                thresholds = np.unique(np.quantile(values, quantiles))
            else:
                thresholds = (unique[:-1] + unique[1:]) / 2
            for threshold in thresholds:
                left = values <= threshold
                left_n = int(left.sum())
                right_n = len(y) - left_n
                if left_n < self.min_samples_leaf or right_n < self.min_samples_leaf:
                    continue
                impurity = (left_n * self._gini(y[left]) + right_n * self._gini(y[~left])) / len(y)
                gain = parent_impurity - impurity
                if best is None or gain > best[0]:
                    best = (gain, feature, float(threshold), left)
        if best is None or best[0] <= 0:
            return node
        gain, feature, threshold, left = best
        node.feature, node.threshold = feature, threshold
        self.feature_importances_[feature] += gain * len(y)
        node.left = self._grow(X[left], y[left], depth + 1)
        node.right = self._grow(X[~left], y[~left], depth + 1)
        return node

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.root_ is None:
            raise RuntimeError("模型尚未训练")
        X = np.asarray(X, dtype=float)
        positive = np.array([self._predict_row(row, self.root_) for row in X])
        return np.column_stack([1 - positive, positive])

    def _predict_row(self, row: np.ndarray, node: _Node) -> float:
        while node.feature is not None:
            node = node.left if row[node.feature] <= node.threshold else node.right  # type: ignore[assignment]
        return node.probability

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
