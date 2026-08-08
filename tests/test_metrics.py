from __future__ import annotations

import unittest

import numpy as np

from factor_research.metrics import (
    classification_metrics,
    information_coefficient,
    regression_metrics,
)


class InformationCoefficientTest(unittest.TestCase):
    def test_information_coefficient_matches_pearson_correlation(self):
        probability = np.array([0.1, 0.4, 0.8, 0.6])
        target_return = np.array([-0.03, 0.02, 0.07, 0.01])

        expected = float(np.corrcoef(probability, target_return)[0, 1])

        self.assertAlmostEqual(
            information_coefficient(probability, target_return), expected
        )
        self.assertAlmostEqual(
            classification_metrics(
                np.array([0, 1, 1, 1]), probability, target_return
            )["ic"],
            expected,
        )

    def test_information_coefficient_ignores_non_finite_pairs(self):
        probability = np.array([0.1, 0.2, np.nan, 0.4])
        target_return = np.array([0.01, 0.02, 0.03, np.inf])

        self.assertAlmostEqual(
            information_coefficient(probability, target_return), 1.0
        )

    def test_information_coefficient_is_nan_without_variation(self):
        self.assertTrue(
            np.isnan(
                information_coefficient(
                    np.array([0.5, 0.5, 0.5]),
                    np.array([-0.01, 0.0, 0.01]),
                )
            )
        )

    def test_information_coefficient_is_nan_with_fewer_than_two_valid_pairs(self):
        self.assertTrue(
            np.isnan(
                information_coefficient(
                    np.array([0.2, np.nan]),
                    np.array([0.01, 0.02]),
                )
            )
        )

    def test_classification_metrics_remains_backward_compatible(self):
        metrics = classification_metrics(
            np.array([0, 1]),
            np.array([0.2, 0.8]),
        )

        self.assertNotIn("ic", metrics)

    def test_regression_metrics_use_continuous_returns_and_direction(self):
        actual = np.array([-0.02, 0.01, 0.03, -0.04])
        predicted = np.array([-0.01, -0.01, 0.02, -0.02])

        metrics = regression_metrics(actual, predicted)

        error = predicted - actual
        self.assertEqual(metrics["samples"], 4.0)
        self.assertAlmostEqual(metrics["mae"], float(np.mean(np.abs(error))))
        self.assertAlmostEqual(metrics["rmse"], float(np.sqrt(np.mean(error**2))))
        self.assertEqual(metrics["direction_accuracy"], 0.75)
        self.assertAlmostEqual(
            metrics["ic"], float(np.corrcoef(predicted, actual)[0, 1])
        )

    def test_information_coefficient_requires_matching_shapes(self):
        with self.assertRaisesRegex(ValueError, "same shape"):
            information_coefficient(np.array([0.1]), np.array([0.01, 0.02]))


if __name__ == "__main__":
    unittest.main()
