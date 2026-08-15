from __future__ import annotations

import unittest

import numpy as np

from quant.factor_research.models.gradient_boosting_tree import (
    GradientBoostingTreeClassifier,
    GradientBoostingTreeModelFactory,
)
from quant.factor_research.models.registry import available_models, model_factory_from_args
from quant.cli.factor_demo import parse_args


class GradientBoostingTreeTest(unittest.TestCase):
    def test_model_is_auto_registered_and_configured_from_args(self):
        args = parse_args(
            [
                "--model",
                "gradient_boosting_tree",
                "--n-estimators",
                "12",
                "--learning-rate",
                "0.05",
                "--max-depth",
                "2",
                "--min-samples-leaf",
                "3",
                "--subsample",
                "0.8",
                "--random-state",
                "7",
            ]
        )
        factory = model_factory_from_args(args)

        self.assertIn("gradient_boosting_tree", available_models())
        self.assertIsInstance(factory, GradientBoostingTreeModelFactory)
        self.assertEqual(factory.n_estimators, 12)
        self.assertEqual(factory.learning_rate, 0.05)
        self.assertEqual(factory.max_depth, 2)
        self.assertEqual(factory.min_samples_leaf, 3)
        self.assertEqual(factory.subsample, 0.8)
        self.assertEqual(factory.random_state, 7)

    def test_classifier_produces_probabilities_and_feature_importance(self):
        random = np.random.default_rng(42)
        X = random.normal(size=(120, 3))
        y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
        model = GradientBoostingTreeClassifier(
            n_estimators=20,
            learning_rate=0.1,
            max_depth=2,
            min_samples_leaf=3,
            random_state=42,
        ).fit(X, y)

        probabilities = model.predict_proba(X[:8])
        self.assertEqual(probabilities.shape, (8, 2))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        self.assertEqual(model.feature_importances_.shape, (3,))
        self.assertAlmostEqual(float(model.feature_importances_.sum()), 1.0)

    def test_single_class_training_window_uses_constant_probability(self):
        X = np.arange(18, dtype=float).reshape(6, 3)
        y = np.ones(6, dtype=int)
        model = GradientBoostingTreeClassifier(n_estimators=5).fit(X, y)

        probabilities = model.predict_proba(X[:2])
        np.testing.assert_array_equal(probabilities[:, 0], np.zeros(2))
        np.testing.assert_array_equal(probabilities[:, 1], np.ones(2))
        np.testing.assert_array_equal(model.feature_importances_, np.zeros(3))

    def test_predict_before_fit_is_rejected(self):
        model = GradientBoostingTreeClassifier()
        with self.assertRaises(RuntimeError):
            model.predict_proba(np.zeros((1, 2)))


if __name__ == "__main__":
    unittest.main()
