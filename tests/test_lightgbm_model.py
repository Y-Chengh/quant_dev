from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from factor_research.experiment import DirectionExperiment
from factor_research.models.lightgbm import LightGBMClassifier, LightGBMModelFactory
from factor_research.models.registry import available_models, model_factory_from_args
from run_factor_demo import parse_args


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

    def test_factory_creates_independent_models_with_expected_parameters(self):
        factory = LightGBMModelFactory(subsample=0.7, n_jobs=2)

        first = factory.create()
        second = factory.create()

        self.assertIsNot(first, second)
        self.assertIsNot(first.estimator, second.estimator)
        self.assertEqual(first.estimator.get_params()["n_jobs"], 2)
        self.assertEqual(first.estimator.get_params()["subsample_freq"], 1)

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


if __name__ == "__main__":
    unittest.main()
