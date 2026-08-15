"""支持方向分类与连续收益率回归的标准化线性模型及其工厂。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from .base import DirectionModel, DirectionModelFactory
from .registry import register_model_factory


def _validate_finite_float(value: float, name: str, minimum: float, inclusive: bool) -> None:
    """校验浮点超参数的类型、有限性和下界。

    参数：
        value: 待校验的正则强度或收敛阈值。
        name: 报错信息中使用的超参数名称。
        minimum: 该超参数允许的数值下界。
        inclusive: ``True`` 表示允许等于下界，否则要求严格大于下界。

    返回：
        参数合法时不返回结果，否则抛出 ``ValueError``。
    """

    numeric_types = (int, float, np.integer, np.floating)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numeric_types):
        raise ValueError(f"{name} 必须是有限数值")
    invalid_bound = value < minimum if inclusive else value <= minimum
    if not np.isfinite(value) or invalid_bound:
        relation = "大于等于" if inclusive else "大于"
        raise ValueError(f"{name} 必须是{relation} {minimum:g} 的有限数值")


def _validate_positive_integer(value: int, name: str) -> None:
    """校验迭代次数等参数为排除布尔值的正整数。

    参数：
        value: 待校验的整数型超参数。
        name: 报错信息中使用的超参数名称。

    返回：
        参数合法时不返回结果，否则抛出 ``ValueError``。
    """

    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value <= 0
    ):
        raise ValueError(f"{name} 必须为正整数")


def _validate_random_state(value: int | None) -> None:
    """校验随机种子为 sklearn 接受的 32 位无符号整数或空值。

    参数：
        value: 逻辑回归求解器随机种子；``None`` 表示不固定种子。

    返回：
        参数合法时不返回结果，否则抛出 ``ValueError``。
    """

    if value is None:
        return
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 0
        or value > np.iinfo(np.uint32).max
    ):
        raise ValueError("random_state 必须是 32 位无符号整数或 None")


def _as_feature_matrix(X: np.ndarray, expected_features: int | None = None) -> np.ndarray:
    """校验并返回有限的二维浮点特征矩阵。

    参数：
        X: 样本位于行、因子位于列的模型输入矩阵。
        expected_features: 训练后要求的因子列数；未训练时为 ``None``。

    返回：
        转换为浮点类型且通过形状、有限值检查的二维数组。
    """

    matrix = np.asarray(X, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("X 必须是二维特征矩阵")
    if matrix.shape[1] == 0:
        raise ValueError("X 必须至少包含一个特征")
    if expected_features is not None and matrix.shape[1] != expected_features:
        raise ValueError(
            "X 的特征数与训练数据不一致: "
            f"expected={expected_features} actual={matrix.shape[1]}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("X 必须全部为有限数值")
    return matrix


class LogisticRegressionClassifier(DirectionModel):
    """标准化因子后预测下一交易日上涨概率的逻辑回归模型。"""

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-4,
        class_weight: str | None = None,
        random_state: int | None = 42,
    ) -> None:
        """初始化逻辑回归估计器，不读取或拟合任何行情数据。

        参数：
            C: L2 正则强度的倒数，必须大于零；数值越小正则越强。
            max_iter: 求解器的最大迭代次数，必须为正整数。
            tol: 求解器停止迭代的误差阈值，必须大于零。
            class_weight: 类别权重策略；``"balanced"`` 自动平衡涨跌样本，
                ``None`` 表示每条样本等权。
            random_state: sklearn 求解器随机种子；``None`` 表示不固定种子。
        """

        _validate_finite_float(C, "C", minimum=0.0, inclusive=False)
        _validate_positive_integer(max_iter, "max_iter")
        _validate_finite_float(tol, "tol", minimum=0.0, inclusive=False)
        if class_weight not in (None, "balanced"):
            raise ValueError("class_weight 只能是 None 或 'balanced'")
        _validate_random_state(random_state)
        self.scaler = StandardScaler()
        self.estimator = LogisticRegression(
            C=C,
            max_iter=max_iter,
            tol=tol,
            class_weight=class_weight,
            random_state=random_state,
            solver="lbfgs",
        )
        self.feature_importances_ = np.array([], dtype=float)
        self._constant_probability: float | None = None
        self._n_features: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionClassifier":
        """使用当前训练窗口的因子和 0/1 方向标签拟合模型。

        参数：
            X: 当前训练窗口的二维因子矩阵，所有元素必须为有限数值。
            y: 与 ``X`` 行一一对应的 0/1 下一交易日方向标签。

        返回：
            已拟合的当前模型；单类别窗口退化为该类别的常数概率。
        """

        matrix = _as_feature_matrix(X)
        target = np.asarray(y)
        if target.ndim != 1 or len(matrix) != len(target) or len(matrix) == 0:
            raise ValueError("X 和 y 的形状不合法")
        if not np.isin(target, [0, 1]).all():
            raise ValueError("y 必须是 0/1 标签")
        target = target.astype(int, copy=False)
        self._n_features = matrix.shape[1]

        classes = np.unique(target)
        if len(classes) == 1:
            # 滚动验证早期可能只有一个方向，使用当时已观察到的信息安全回退。
            self._constant_probability = float(classes[0])
            self.feature_importances_ = np.zeros(matrix.shape[1], dtype=float)
            return self

        self._constant_probability = None
        scaled = self.scaler.fit_transform(matrix)
        self.estimator.fit(scaled, target)
        # 换算回原始因子量纲，再取绝对系数作为线性模型的重要度。
        coefficient = self.estimator.coef_[0] / self.scaler.scale_
        self.feature_importances_ = np.abs(coefficient).astype(float, copy=True)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """返回每个样本的下跌、上涨概率。

        参数：
            X: 待预测的二维因子矩阵，列数必须与训练窗口一致。

        返回：
            形状为 ``(样本数, 2)`` 的有限概率数组，第二列为上涨概率。
        """

        if self._n_features is None:
            raise RuntimeError("模型尚未训练")
        matrix = _as_feature_matrix(X, self._n_features)
        if self._constant_probability is not None:
            positive = np.full(len(matrix), self._constant_probability, dtype=float)
            return np.column_stack([1.0 - positive, positive])
        scaled = self.scaler.transform(matrix)
        return np.asarray(self.estimator.predict_proba(scaled), dtype=float)


class RidgeRegressionModel(DirectionModel):
    """标准化因子后预测下一交易日连续收益率的岭回归模型。"""

    def __init__(self, alpha: float = 1.0, tol: float = 1e-4) -> None:
        """初始化岭回归估计器，不读取或拟合任何行情数据。

        参数：
            alpha: L2 正则强度，必须为非负数；零表示普通最小二乘口径。
            tol: 迭代型求解器可使用的停止误差阈值，必须大于零。
        """

        _validate_finite_float(alpha, "alpha", minimum=0.0, inclusive=True)
        _validate_finite_float(tol, "tol", minimum=0.0, inclusive=False)
        self.scaler = StandardScaler()
        self.estimator = Ridge(alpha=alpha, tol=tol)
        self.feature_importances_ = np.array([], dtype=float)
        self._constant_prediction: float | None = None
        self._n_features: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegressionModel":
        """使用当前训练窗口的因子和连续收益率拟合岭回归。

        参数：
            X: 当前训练窗口的二维因子矩阵，所有元素必须为有限数值。
            y: 与 ``X`` 行一一对应的下一交易日开盘至收盘收益率。

        返回：
            已拟合的当前模型；常数目标窗口退化为该收益率的常数预测。
        """

        matrix = _as_feature_matrix(X)
        target = np.asarray(y, dtype=float)
        if target.ndim != 1 or len(matrix) != len(target) or len(matrix) == 0:
            raise ValueError("X 和 y 的形状不合法")
        if not np.isfinite(target).all():
            raise ValueError("y 必须全部为有限数值")
        self._n_features = matrix.shape[1]

        if np.all(target == target[0]):
            self._constant_prediction = float(target[0])
            self.feature_importances_ = np.zeros(matrix.shape[1], dtype=float)
            return self

        self._constant_prediction = None
        scaled = self.scaler.fit_transform(matrix)
        self.estimator.fit(scaled, target)
        coefficient = np.asarray(self.estimator.coef_, dtype=float) / self.scaler.scale_
        self.feature_importances_ = np.abs(coefficient).astype(float, copy=True)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测下一交易日开盘至收盘的连续收益率。

        参数：
            X: 待预测的二维因子矩阵，列数必须与训练窗口一致。

        返回：
            与输入样本数相同的一维有限收益率数组。
        """

        if self._n_features is None:
            raise RuntimeError("模型尚未训练")
        matrix = _as_feature_matrix(X, self._n_features)
        if self._constant_prediction is not None:
            return np.full(len(matrix), self._constant_prediction, dtype=float)
        scaled = self.scaler.transform(matrix)
        return np.asarray(self.estimator.predict(scaled), dtype=float)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """拒绝在连续收益率回归模型上请求分类概率。

        参数：
            X: 未使用的特征矩阵；保留该参数以满足通用模型接口。

        返回：
            本方法不返回结果，总是抛出 ``NotImplementedError``。
        """

        raise NotImplementedError("RidgeRegressionModel 不提供分类概率")


@register_model_factory
@dataclass(frozen=True)
class LogisticRegressionModelFactory(DirectionModelFactory):
    """按任务创建相互独立的逻辑分类或岭回归线性模型。"""

    classification_c: float = 1.0
    regression_alpha: float = 1.0
    max_iter: int = 1000
    tol: float = 1e-4
    class_weight: str | None = None
    random_state: int | None = 42
    task: str = "classification"
    name: ClassVar[str] = "logistic_regression"
    supported_tasks: ClassVar[tuple[str, ...]] = ("classification", "regression")

    def __post_init__(self) -> None:
        """校验任务类型和正则、收敛参数，避免延迟到滚动训练时报错。"""

        if self.task not in self.supported_tasks:
            raise ValueError(f"logistic_regression 不支持任务 {self.task!r}")
        _validate_finite_float(
            self.classification_c,
            "classification_c",
            minimum=0.0,
            inclusive=False,
        )
        _validate_finite_float(
            self.regression_alpha,
            "regression_alpha",
            minimum=0.0,
            inclusive=True,
        )
        _validate_positive_integer(self.max_iter, "max_iter")
        _validate_finite_float(self.tol, "tol", minimum=0.0, inclusive=False)
        if self.class_weight not in (None, "balanced"):
            raise ValueError("class_weight 只能是 None 或 'balanced'")
        _validate_random_state(self.random_state)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """向第二阶段 CLI 解析器注册本模型独有的参数。

        参数：
            parser: 已包含通用实验参数、仅待加入所选模型参数的解析器。

        返回：
            本方法原地修改解析器，不返回结果。
        """

        parser.add_argument(
            "--classification-c",
            type=float,
            default=1.0,
            help="分类 L2 正则强度的倒数，必须大于零；默认 1.0",
        )
        parser.add_argument(
            "--regression-alpha",
            type=float,
            default=1.0,
            help="回归 L2 正则强度，必须为非负数；默认 1.0",
        )
        parser.add_argument(
            "--max-iter",
            type=int,
            default=1000,
            help="逻辑回归求解器最大迭代次数；默认 1000",
        )
        parser.add_argument(
            "--tol",
            type=float,
            default=1e-4,
            help="线性模型求解停止阈值；默认 1e-4",
        )
        parser.add_argument(
            "--class-weight",
            choices=["balanced"],
            default=None,
            help="分类时可选 balanced 自动平衡涨跌样本；默认等权",
        )
        parser.add_argument(
            "--random-state",
            type=int,
            default=42,
            help="逻辑回归随机种子；默认 42",
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "LogisticRegressionModelFactory":
        """从命令行或 YAML 已校验参数构建模型工厂。

        参数：
            args: 主程序两阶段解析后得到的实验与模型参数命名空间。

        返回：
            保存当前任务和超参数的不可变模型工厂。
        """

        return cls(
            classification_c=args.classification_c,
            regression_alpha=args.regression_alpha,
            max_iter=args.max_iter,
            tol=args.tol,
            class_weight=args.class_weight,
            random_state=args.random_state,
            task=getattr(args, "task", "classification"),
        )

    def create(self) -> DirectionModel:
        """按工厂任务创建一个没有历史训练状态的新模型。

        返回：
            分类任务对应逻辑回归模型，回归任务对应岭回归模型。
        """

        if self.task == "classification":
            return LogisticRegressionClassifier(
                C=self.classification_c,
                max_iter=self.max_iter,
                tol=self.tol,
                class_weight=self.class_weight,
                random_state=self.random_state,
            )
        return RidgeRegressionModel(alpha=self.regression_alpha, tol=self.tol)
