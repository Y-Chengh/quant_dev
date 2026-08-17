from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from quant.cli.factor_demo import parse_args
from quant.factor_research.experiment import DirectionExperiment
from quant.factor_research.models.lightgbm import (
    LightGBMClassifier,
    LightGBMModelFactory,
    LightGBMRegressor,
)
from quant.factor_research.models.registry import available_models, model_factory_from_args


class LightGBMModelTest(unittest.TestCase):
    def test_model_is_auto_registered_and_configured_from_args(self):
        args = parse_args(
            [
                "--model",
                "lightgbm",
                "--n-estimators",
                "12",
                "--learning-rate",
                "0.05",
                "--num-leaves",
                "7",
                "--max-depth",
                "3",
                "--min-child-samples",
                "4",
                "--subsample",
                "0.7",
                "--colsample-bytree",
                "0.6",
                "--reg-alpha",
                "0.2",
                "--reg-lambda",
                "1.5",
                "--n-jobs",
                "2",
                "--random-state",
                "7",
                "--objective",
                "cross_entropy",
                "--boosting-type",
                "gbdt",
            ]
        )
        factory = model_factory_from_args(args)

        self.assertIn("lightgbm", available_models())
        self.assertIsInstance(factory, LightGBMModelFactory)
        self.assertEqual(factory.n_estimators, 12)
        self.assertEqual(factory.learning_rate, 0.05)
        self.assertEqual(factory.num_leaves, 7)
        self.assertEqual(factory.max_depth, 3)
        self.assertEqual(factory.min_child_samples, 4)
        self.assertEqual(factory.subsample, 0.7)
        self.assertEqual(factory.colsample_bytree, 0.6)
        self.assertEqual(factory.reg_alpha, 0.2)
        self.assertEqual(factory.reg_lambda, 1.5)
        self.assertEqual(factory.n_jobs, 2)
        self.assertEqual(factory.random_state, 7)
        self.assertEqual(factory.objective, "cross_entropy")
        self.assertEqual(factory.boosting_type, "gbdt")

    def test_legacy_namespace_defaults_to_binary_classification(self):
        args = parse_args(["--model", "lightgbm"])
        delattr(args, "task")
        delattr(args, "objective")
        delattr(args, "objective_alpha")

        factory = model_factory_from_args(args)

        self.assertEqual(factory.task, "classification")
        self.assertEqual(factory.objective, "binary")
        self.assertIsInstance(factory.create(), LightGBMClassifier)

    def test_factory_creates_independent_models_with_expected_parameters(self):
        factory = LightGBMModelFactory(subsample=0.7, n_jobs=2)

        first = factory.create()
        second = factory.create()

        self.assertIsNot(first, second)
        self.assertIsNot(first.estimator, second.estimator)
        self.assertEqual(first.estimator.get_params()["n_jobs"], 2)
        self.assertEqual(first.estimator.get_params()["subsample_freq"], 1)
        self.assertEqual(first.estimator.get_params()["objective"], "binary")

    def test_boosting_type_defaults_to_dart_and_passes_through(self):
        """缺省保持 dart 与既有结果可比；显式配置应透传到底层估计器。"""

        default_model = LightGBMModelFactory(n_jobs=1).create()
        self.assertEqual(
            default_model.estimator.get_params()["boosting_type"], "dart"
        )

        gbdt_model = LightGBMModelFactory(boosting_type="gbdt", n_jobs=1).create()
        self.assertEqual(gbdt_model.estimator.get_params()["boosting_type"], "gbdt")

        goss_model = LightGBMModelFactory(
            boosting_type="goss", subsample=1.0, n_jobs=1
        ).create()
        self.assertEqual(goss_model.estimator.get_params()["boosting_type"], "goss")
        # subsample=1.0 时不启用每轮行采样，GOSS 才能通过 LightGBM 自身校验。
        self.assertEqual(goss_model.estimator.get_params()["subsample_freq"], 0)

    def test_boosting_type_is_validated_at_factory_construction(self):
        """非法取值与 GOSS+行采样组合都应在工厂构造时立刻报错。"""

        with self.assertRaisesRegex(ValueError, "boosting_type"):
            LightGBMModelFactory(boosting_type="rf")
        with self.assertRaisesRegex(ValueError, "GOSS"):
            LightGBMModelFactory(boosting_type="goss", subsample=0.8)

    def test_gbdt_and_goss_models_fit_and_predict(self):
        """gbdt 与 goss 两种提升算法应能完成训练并给出有限预测。"""

        random = np.random.default_rng(42)
        X = random.normal(size=(200, 3))
        y = 0.01 * X[:, 0] - 0.005 * X[:, 1]
        labels = (y > 0).astype(int)

        gbdt_classifier = LightGBMClassifier(
            n_estimators=10,
            min_child_samples=5,
            n_jobs=1,
            boosting_type="gbdt",
        ).fit(X, labels)
        probabilities = gbdt_classifier.predict_proba(X[:5])
        self.assertEqual(probabilities.shape, (5, 2))
        self.assertTrue(np.isfinite(probabilities).all())

        goss_regressor = LightGBMRegressor(
            n_estimators=10,
            min_child_samples=5,
            subsample=1.0,
            n_jobs=1,
            boosting_type="goss",
        ).fit(X, y)
        prediction = goss_regressor.predict(X[:5])
        self.assertEqual(prediction.shape, (5,))
        self.assertTrue(np.isfinite(prediction).all())

    def test_regression_factory_uses_configured_objective(self):
        factory = LightGBMModelFactory(
            task="regression",
            objective="huber",
            objective_alpha=0.02,
            n_estimators=5,
            n_jobs=1,
        )

        model = factory.create()

        self.assertIsInstance(model, LightGBMRegressor)
        self.assertEqual(model.estimator.get_params()["objective"], "huber")
        self.assertEqual(model.estimator.get_params()["alpha"], 0.02)

    def test_huber_alpha_changes_outlier_response(self):
        """较小 Huber 阈值应削弱极端收益标签并改变实际预测。"""

        random = np.random.default_rng(42)
        X = random.normal(size=(400, 3))
        y = 0.02 * X[:, 0] + random.normal(0.0, 0.01, len(X))
        y[:8] += 0.5
        common = {
            "n_estimators": 20,
            "learning_rate": 0.1,
            "num_leaves": 7,
            "max_depth": 3,
            "min_child_samples": 5,
            "reg_alpha": 0.0,
            "n_jobs": 1,
            "random_state": 42,
        }
        squared = LightGBMRegressor(objective="regression", **common).fit(X, y)
        robust = LightGBMRegressor(
            objective="huber",
            objective_alpha=0.02,
            **common,
        ).fit(X, y)

        difference = np.max(np.abs(squared.predict(X) - robust.predict(X)))
        self.assertGreater(float(difference), 1e-6)

    def test_objective_alpha_is_validated_for_huber_and_quantile(self):
        """目标函数 alpha 应拒绝非有限、非正及非法分位点。"""

        for invalid in (0.0, -0.1, np.inf, np.nan):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "objective_alpha"):
                    LightGBMModelFactory(
                        task="regression",
                        objective="huber",
                        objective_alpha=invalid,
                    )
        with self.assertRaisesRegex(ValueError, "小于 1"):
            LightGBMModelFactory(
                task="regression",
                objective="quantile",
                objective_alpha=1.0,
            )

    def test_objective_must_match_task(self):
        with self.assertRaisesRegex(ValueError, "不兼容"):
            LightGBMModelFactory(task="regression", objective="binary")
        with self.assertRaisesRegex(ValueError, "不兼容"):
            LightGBMModelFactory(task="classification", objective="regression")

    def test_experiment_task_must_match_factory_task(self):
        with self.assertRaisesRegex(ValueError, "不一致"):
            DirectionExperiment(
                validation_start="2024-01-02",
                feature_columns=["factor_a"],
                model_factory=LightGBMModelFactory(task="classification"),
                task="regression",
            )
        with self.assertRaisesRegex(ValueError, "不一致"):
            DirectionExperiment(
                validation_start="2024-01-02",
                feature_columns=["factor_a"],
                model_factory=LightGBMModelFactory(task="regression"),
                task="classification",
            )

    def test_classifier_produces_probabilities_and_feature_importance(self):
        random = np.random.default_rng(42)
        X = random.normal(size=(160, 3))
        y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
        model = LightGBMClassifier(
            n_estimators=20,
            learning_rate=0.1,
            num_leaves=7,
            max_depth=3,
            min_child_samples=3,
            n_jobs=1,
            random_state=42,
        ).fit(X, y)

        probabilities = model.predict_proba(X[:8])
        self.assertEqual(probabilities.shape, (8, 2))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertEqual(model.feature_importances_.shape, (3,))
        self.assertTrue(np.isfinite(model.feature_importances_).all())
        self.assertGreater(float(model.feature_importances_.sum()), 0.0)

    def test_single_class_training_window_uses_constant_probability(self):
        X = np.arange(18, dtype=float).reshape(6, 3)
        y = np.ones(6, dtype=int)
        model = LightGBMClassifier(n_estimators=5, n_jobs=1).fit(X, y)

        probabilities = model.predict_proba(X[:2])
        np.testing.assert_array_equal(probabilities[:, 0], np.zeros(2))
        np.testing.assert_array_equal(probabilities[:, 1], np.ones(2))
        np.testing.assert_array_equal(model.feature_importances_, np.zeros(3))

    def test_predict_before_fit_is_rejected(self):
        model = LightGBMClassifier(n_jobs=1)
        with self.assertRaises(RuntimeError):
            model.predict_proba(np.zeros((1, 2)))

    def test_regressor_predicts_continuous_returns_and_constant_window(self):
        random = np.random.default_rng(42)
        X = random.normal(size=(120, 3))
        y = 0.01 * X[:, 0] - 0.005 * X[:, 1]
        model = LightGBMRegressor(
            n_estimators=20,
            learning_rate=0.1,
            num_leaves=7,
            max_depth=3,
            min_child_samples=3,
            n_jobs=1,
        ).fit(X, y)

        prediction = model.predict(X[:8])
        self.assertEqual(prediction.shape, (8,))
        self.assertTrue(np.isfinite(prediction).all())
        self.assertEqual(model.feature_importances_.shape, (3,))

        constant = LightGBMRegressor(n_estimators=5, n_jobs=1).fit(
            X[:6], np.full(6, 0.012)
        )
        np.testing.assert_allclose(constant.predict(X[:2]), [0.012, 0.012])

    def test_factory_runs_in_walk_forward_experiment_without_future_labels(self):
        target_dates = pd.date_range("2024-01-02", periods=8, freq="D")
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
        factory = LightGBMModelFactory(
            n_estimators=5,
            min_child_samples=2,
            n_jobs=1,
        )

        result = DirectionExperiment(
            validation_start=target_dates[4],
            feature_columns=["factor_a", "factor_b"],
            model_factory=factory,
        ).run(dataset)

        self.assertEqual(result.model_name, "lightgbm")
        self.assertTrue(
            (result.predictions["training_end_date"] < result.predictions["target_date"]).all()
        )
        self.assertTrue(np.isfinite(result.predictions["up_probability"]).all())

    @staticmethod
    def _regression_sample(rows: int = 60) -> tuple[np.ndarray, np.ndarray]:
        """构造可复现的回归训练样本。

        参数：
            rows: 样本行数；取值需大于叶节点最小样本数，保证真的会分裂建树。

        返回：
            四列特征矩阵与由首列线性生成、带小幅噪声的连续目标。
        """

        generator = np.random.default_rng(20260818)
        features = generator.normal(size=(rows, 4))
        target = features[:, 0] * 0.5 + generator.normal(size=rows) * 0.05
        return features, target

    def test_fit_progress_reports_every_boosting_round(self):
        """注册回调后，每完成一轮提升都应上报一次递增的已完成轮数。"""

        features, target = self._regression_sample()
        model = LightGBMRegressor(
            n_estimators=9, min_child_samples=2, n_jobs=1, boosting_type="gbdt"
        )
        reported: list[tuple[int, int]] = []

        declared = model.set_fit_progress(lambda completed, total: reported.append((completed, total)))
        model.fit(features, target)

        self.assertEqual(declared, 9)
        self.assertEqual([completed for completed, _ in reported], list(range(1, 10)))
        self.assertEqual({total for _, total in reported}, {9})

    def test_fit_progress_can_be_uninstalled_without_changing_predictions(self):
        """进度回调只用于显示，装卸它都不得改变训练结果。

        逐个提升算法核对：``dart`` 是生产默认值，``gbdt`` 与 ``goss`` 为常用替代。
        ``dart``、``gbdt`` 保留生产默认的 ``subsample=0.8``，因为每轮行采样会消耗
        随机数，是"回调是否扰动训练"最敏感的探测点；``goss`` 与行采样互斥只能关掉，
        并把轮数提到预热期（约 ``1/learning_rate``）之后，否则它退化成 ``gbdt``。
        """

        features, target = self._regression_sample()
        settings = (
            ("dart", 9, 0.8),
            ("gbdt", 9, 0.8),
            ("goss", 40, 1.0),
        )
        for boosting_type, n_estimators, subsample in settings:
            with self.subTest(boosting_type=boosting_type):
                parameters = {
                    "n_estimators": n_estimators,
                    "min_child_samples": 2,
                    "n_jobs": 1,
                    "boosting_type": boosting_type,
                    "subsample": subsample,
                }
                baseline = LightGBMRegressor(**parameters).fit(features, target)

                observed = LightGBMRegressor(**parameters)
                calls: list[int] = []
                observed.set_fit_progress(
                    lambda completed, total: calls.append(completed)
                )
                observed.fit(features, target)

                uninstalled = LightGBMRegressor(**parameters)
                self.assertIsNone(uninstalled.set_fit_progress(None))
                uninstalled.set_fit_progress(lambda completed, total: calls.append(-1))
                uninstalled.set_fit_progress(None)
                uninstalled.fit(features, target)

                self.assertEqual(len(calls), n_estimators)
                self.assertNotIn(-1, calls)
                np.testing.assert_allclose(
                    observed.predict(features), baseline.predict(features)
                )
                np.testing.assert_allclose(
                    uninstalled.predict(features), baseline.predict(features)
                )

    def test_fit_progress_is_silent_when_classifier_skips_training(self):
        """训练集只有一个类别时不会真正建树，回调也不应被调用。"""

        features, _ = self._regression_sample(rows=20)
        labels = np.ones(len(features), dtype=int)
        model = LightGBMClassifier(n_estimators=9, min_child_samples=2, n_jobs=1)
        reported: list[int] = []

        declared = model.set_fit_progress(lambda completed, total: reported.append(completed))
        model.fit(features, labels)

        self.assertEqual(declared, 9)
        self.assertEqual(reported, [])
        np.testing.assert_allclose(model.predict_proba(features)[:, 1], 1.0)


if __name__ == "__main__":
    unittest.main()
