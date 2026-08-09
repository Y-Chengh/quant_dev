from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from factor_research.metrics import (
    classification_metrics,
    cross_sectional_ic_metrics,
    daily_cross_sectional_ic,
    information_coefficient,
    rank_information_coefficient,
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

    def test_daily_cross_sectional_ic_and_rank_ic_are_isolated_by_date(self):
        dates = pd.to_datetime(
            ["2025-01-02"] * 3 + ["2025-01-03"] * 3 + ["2025-01-06"] * 3
        )
        predictions = pd.DataFrame(
            {
                "target_date": dates,
                "score": [1.0, 2.0, 3.0, 1.0, 1.0, 3.0, 2.0, 2.0, 2.0],
                "target_return": [
                    0.01, 0.02, 0.03,
                    0.03, 0.01, 0.02,
                    -0.01, 0.00, 0.01,
                ],
            }
        )

        daily = daily_cross_sectional_ic(predictions, "score")
        summary = cross_sectional_ic_metrics(daily)

        self.assertEqual(
            list(daily.columns), ["target_date", "samples", "ic", "rank_ic"]
        )
        np.testing.assert_allclose(daily.loc[:1, "ic"], [1.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(
            daily.loc[:1, "rank_ic"], [1.0, 0.0], atol=1e-12
        )
        self.assertTrue(pd.isna(daily.loc[2, "ic"]))
        self.assertTrue(pd.isna(daily.loc[2, "rank_ic"]))
        self.assertAlmostEqual(summary["ic"], 0.5)
        self.assertAlmostEqual(summary["rank_ic"], 0.5)
        self.assertAlmostEqual(summary["icir"], 1 / np.sqrt(2))
        self.assertEqual(summary["ic_win_rate"], 0.5)
        self.assertEqual(summary["ic_dates"], 2.0)
        self.assertEqual(summary["rank_ic_dates"], 2.0)

    def test_icir_and_win_rate_are_nan_without_valid_variation(self):
        """有效日不足或日 IC 无波动时，ICIR 应缺失但胜率仍应可计算。

        返回：
            无；断言失败时由测试框架报告差异。
        """

        summary = cross_sectional_ic_metrics(
            pd.DataFrame({"ic": [0.2, 0.2, np.nan], "rank_ic": [0.1, 0.1, np.nan]})
        )

        self.assertTrue(np.isnan(summary["icir"]))
        self.assertEqual(summary["ic_win_rate"], 1.0)

    def test_rank_ic_uses_average_ranks_for_ties(self):
        score = np.array([0.1, 0.1, 0.9, 0.5])
        returns = np.array([0.0, 0.2, 0.3, 0.1])
        expected = float(
            np.corrcoef(
                pd.Series(score).rank(method="average"),
                pd.Series(returns).rank(method="average"),
            )[0, 1]
        )
        self.assertAlmostEqual(
            rank_information_coefficient(score, returns), expected
        )


if __name__ == "__main__":
    unittest.main()
