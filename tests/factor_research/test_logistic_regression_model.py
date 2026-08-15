from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from quant.cli.factor_demo import parse_args
from quant.factor_research.experiment import DirectionExperiment
from quant.factor_research.models.logistic_regression import (
    LogisticRegressionClassifier,
    LogisticRegressionModelFactory,
    RidgeRegressionModel,
)
from quant.factor_research.models.registry import available_models, model_factory_from_args


class LogisticRegressionModelTest(unittest.TestCase):
    """验证标准化线性模型的数值、边界、注册及实验集成行为。"""

    def test_factory_is_registered_and_reads_classification_cli(self) -> None:
        """分类超参数应通过两阶段 CLI 解析并传递给新模型。"""

        args = parse_args(
            [
                "--model",
                "logistic_regression",
                "--classification-c",
                "0.25",
                "--max-iter",
                "321",
                "--tol",
                "0.00001",
                "--class-weight",
                "balanced",
                "--random-state",
                "7",
            ]
        )
        factory = model_factory_from_args(args)

        self.assertIn("logistic_regression", available_models())
        self.assertIsInstance(factory, LogisticRegressionModelFactory)
        self.assertEqual(factory.classification_c, 0.25)
        self.assertEqual(factory.max_iter, 321)
        self.assertEqual(factory.tol, 0.00001)
        self.assertEqual(factory.class_weight, "balanced")
        self.assertEqual(factory.random_state, 7)
        self.assertIsInstance(factory.create(), LogisticRegressionClassifier)

    def test_regression_cli_creates_independent_ridge_models(self) -> None:
        """回归任务应读取岭正则参数且每次创建无共享状态的实例。"""

        args = parse_args(
            [
                "--model",
                "logistic_regression",
                "--task",
                "regression",
                "--regression-alpha",
                "2.5",
            ]
        )
        factory = model_factory_from_args(args)
        first = factory.create()
        second = factory.create()

        self.assertEqual(factory.supported_tasks, ("classification", "regression"))
        self.assertIsInstance(first, RidgeRegressionModel)
        self.assertIsNot(first, second)
        self.assertIsNot(first.estimator, second.estimator)
        self.assertEqual(first.estimator.alpha, 2.5)

    def test_classifier_learns_probabilities_and_handles_single_class(self) -> None:
        """逻辑回归应学习方向概率，并安全处理最小单类别窗口。"""

        random = np.random.default_rng(42)
        X = random.normal(size=(200, 3))
        y = (2.0 * X[:, 0] - X[:, 1] > 0).astype(int)
        model = LogisticRegressionClassifier(C=10.0).fit(X, y)

        probabilities = model.predict_proba(X[:10])
        self.assertEqual(probabilities.shape, (10, 2))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        self.assertGreater(np.mean((probabilities[:, 1] >= 0.5) == y[:10]), 0.8)
        self.assertEqual(model.feature_importances_.shape, (3,))
        self.assertTrue(np.isfinite(model.feature_importances_).all())

        constant = LogisticRegressionClassifier().fit(X[:1], np.array([1]))
        np.testing.assert_array_equal(
            constant.predict_proba(X[:2]), np.array([[0.0, 1.0], [0.0, 1.0]])
        )
        np.testing.assert_array_equal(constant.feature_importances_, np.zeros(3))

    def test_regressor_recovers_linear_returns_and_constant_target(self) -> None:
        """岭回归应输出连续收益率，并安全处理常数目标窗口。"""

        random = np.random.default_rng(7)
        X = random.normal(size=(240, 3))
        y = 0.02 * X[:, 0] - 0.01 * X[:, 1] + 0.003
        model = RidgeRegressionModel(alpha=0.0).fit(X, y)

        prediction = model.predict(X[:12])
        np.testing.assert_allclose(prediction, y[:12], atol=1e-10)
        self.assertEqual(model.feature_importances_.shape, (3,))
        self.assertTrue(np.isfinite(model.feature_importances_).all())

        constant = RidgeRegressionModel().fit(X[:1], np.array([0.012]))
        np.testing.assert_allclose(constant.predict(X[:2]), [0.012, 0.012])
        np.testing.assert_array_equal(constant.feature_importances_, np.zeros(3))

    def test_invalid_targets_and_predict_before_fit_are_rejected(self) -> None:
        """模型应明确拒绝非法目标、非有限特征和未训练预测。"""

        with self.assertRaises(RuntimeError):
            LogisticRegressionClassifier().predict_proba(np.zeros((1, 2)))
        with self.assertRaises(RuntimeError):
            RidgeRegressionModel().predict(np.zeros((1, 2)))
        with self.assertRaisesRegex(ValueError, "0/1"):
            LogisticRegressionClassifier().fit(np.zeros((2, 1)), np.array([0, 2]))
        with self.assertRaisesRegex(ValueError, "有限"):
            RidgeRegressionModel().fit(np.array([[np.inf]]), np.array([0.1]))
        with self.assertRaisesRegex(ValueError, "至少包含一个特征"):
            LogisticRegressionClassifier().fit(np.empty((2, 0)), np.array([0, 1]))

    def test_factory_rejects_non_finite_and_non_integer_parameters(self) -> None:
        """CLI、YAML 和直接注入都应在训练前拒绝非法超参数。"""

        invalid_cli = parse_args(
            [
                "--model",
                "logistic_regression",
                "--classification-c",
                "nan",
            ]
        )
        with self.assertRaisesRegex(ValueError, "classification_c"):
            model_factory_from_args(invalid_cli)

        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "logistic.yaml"
            config_path.write_text(
                "model: logistic_regression\nregression_alpha: .inf\n",
                encoding="utf-8",
            )
            invalid_yaml = parse_args(["--config", str(config_path)])
        with self.assertRaisesRegex(ValueError, "regression_alpha"):
            model_factory_from_args(invalid_yaml)

        for invalid_max_iter in (0, 1.5, True):
            with self.subTest(max_iter=invalid_max_iter):
                with self.assertRaisesRegex(ValueError, "max_iter"):
                    LogisticRegressionModelFactory(max_iter=invalid_max_iter)
        with self.assertRaisesRegex(ValueError, "tol"):
            LogisticRegressionModelFactory(tol=np.inf)

    def test_public_models_reject_invalid_parameters_before_constant_fallback(self) -> None:
        """公开模型构造器不能因单目标回退而绕过超参数校验。"""

        invalid_classification = (
            {"C": 0.0},
            {"max_iter": 1.5},
            {"tol": np.nan},
            {"class_weight": "unknown"},
            {"random_state": True},
        )
        for parameters in invalid_classification:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                LogisticRegressionClassifier(**parameters)

        for parameters in ({"alpha": -1.0}, {"tol": np.inf}):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                RidgeRegressionModel(**parameters)

    def test_classification_runs_walk_forward_without_future_labels(self) -> None:
        """逻辑分类工厂注入实验后，训练截止日期必须严格早于预测日期。"""

        target_dates = pd.date_range("2024-02-01", periods=7, freq="D")
        rows = []
        for date_position, target_date in enumerate(target_dates):
            for code_position, code in enumerate(["000001.SZ", "600000.SH"]):
                label = (date_position + code_position) % 2
                rows.append(
                    {
                        "feature_date": target_date - pd.Timedelta(days=1),
                        "target_date": target_date,
                        "code": code,
                        "factor_a": float(date_position),
                        "factor_b": float(code_position),
                        "label": label,
                        "target_return": 0.01 if label else -0.01,
                    }
                )
        dataset = pd.DataFrame(rows)

        result = DirectionExperiment(
            validation_start=target_dates[3],
            feature_columns=["factor_a", "factor_b"],
            model_factory=LogisticRegressionModelFactory(),
            task="classification",
        ).run(dataset)

        self.assertEqual(result.model_name, "logistic_regression")
        self.assertTrue(
            (result.predictions["training_end_date"] < result.predictions["target_date"]).all()
        )
        self.assertTrue(np.isfinite(result.predictions["up_probability"]).all())

    def test_regression_runs_walk_forward_without_future_labels(self) -> None:
        """工厂注入回归实验后，训练截止日期必须严格早于预测日期。"""

        target_dates = pd.date_range("2024-01-02", periods=7, freq="D")
        rows = []
        for date_position, target_date in enumerate(target_dates):
            for code_position, code in enumerate(["000001.SZ", "600000.SH"]):
                target_return = 0.01 * date_position - 0.002 * code_position
                rows.append(
                    {
                        "feature_date": target_date - pd.Timedelta(days=1),
                        "target_date": target_date,
                        "code": code,
                        "factor_a": float(date_position),
                        "factor_b": float(code_position),
                        "label": int(target_return > 0),
                        "target_return": target_return,
                    }
                )
        dataset = pd.DataFrame(rows)
        factory = LogisticRegressionModelFactory(task="regression", regression_alpha=0.1)

        result = DirectionExperiment(
            validation_start=target_dates[3],
            feature_columns=["factor_a", "factor_b"],
            model_factory=factory,
            task="regression",
        ).run(dataset)

        self.assertEqual(result.model_name, "logistic_regression")
        self.assertTrue(
            (result.predictions["training_end_date"] < result.predictions["target_date"]).all()
        )
        self.assertTrue(np.isfinite(result.predictions["predicted_return"]).all())


if __name__ == "__main__":
    unittest.main()
