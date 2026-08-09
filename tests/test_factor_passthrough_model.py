"""因子值直出回归模型及其工厂注册测试。"""

from __future__ import annotations

import unittest

import numpy as np

from factor_research.models.factor_passthrough import (
    FactorPassthroughModelFactory,
    FactorPassthroughRegressor,
)
from factor_research.models.registry import available_models
from factor_research.models.registry import model_factory_from_args
from run_factor_demo import parse_args


class FactorPassthroughModelTests(unittest.TestCase):
    """验证直出语义、状态隔离、输入边界和注册结果。"""

    def test_predict_returns_last_feature_unchanged(self) -> None:
        """拟合不得改变末列因子，预测结果应与该列精确相等。"""

        X = np.array([[1.0, -0.2], [2.0, 0.4], [3.0, 1.5]])
        model = FactorPassthroughRegressor().fit(X, np.array([0.1, -0.1, 0.2]))

        np.testing.assert_array_equal(model.predict(X), X[:, -1])

    def test_model_rejects_invalid_shapes_and_unfitted_prediction(self) -> None:
        """空列、目标错位、预测列数变化和未拟合调用都应明确失败。"""

        with self.assertRaisesRegex(RuntimeError, "尚未拟合"):
            FactorPassthroughRegressor().predict(np.ones((1, 1)))
        with self.assertRaisesRegex(ValueError, "至少包含一列"):
            FactorPassthroughRegressor().fit(np.empty((2, 0)), np.zeros(2))
        with self.assertRaisesRegex(ValueError, "样本数相同"):
            FactorPassthroughRegressor().fit(np.ones((2, 1)), np.zeros(1))
        model = FactorPassthroughRegressor().fit(np.ones((2, 2)), np.zeros(2))
        with self.assertRaisesRegex(ValueError, "列数与拟合阶段不一致"):
            model.predict(np.ones((1, 1)))

    def test_factory_is_registered_for_regression_and_creates_fresh_models(self) -> None:
        """工厂应自动注册、只声明回归能力，并且每次创建独立实例。"""

        factory = FactorPassthroughModelFactory()

        self.assertIn("factor_passthrough", available_models())
        self.assertEqual(factory.supported_tasks, ("regression",))
        first = factory.create()
        second = factory.create()
        self.assertIsNot(first, second)

    def test_main_cli_can_select_passthrough_regression(self) -> None:
        """主实验 CLI 应通过注册表选择直出模型，无需增加模型分支。"""

        args = parse_args(
            [
                "--model",
                "factor_passthrough",
                "--task",
                "regression",
                "--factors",
                "return_1d",
            ]
        )

        factory = model_factory_from_args(args)
        self.assertIsInstance(factory, FactorPassthroughModelFactory)


if __name__ == "__main__":
    unittest.main()
