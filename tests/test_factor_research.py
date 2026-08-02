from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from factor_research.dataset import build_direction_dataset, split_by_date
from factor_research.experiment import DirectionExperiment
from factor_research.factor_factories import FACTOR_FACTORIES
from factor_research.factors import build_daily_features
from run_factor_demo import resolve_window


def synthetic_bars(days: int = 80, symbols: int = 3) -> pd.DataFrame:
    rows = []
    dates = pd.bdate_range("2024-01-02", periods=days)
    for symbol_index in range(symbols):
        price = 10.0 + symbol_index
        for day_index, date in enumerate(dates):
            direction = 1 if (day_index + symbol_index) % 4 in (0, 1, 2) else -1
            for minute, offset in zip((35, 40, 45, 50, 55, 60), range(6)):
                timestamp = date + pd.Timedelta(hours=9, minutes=minute)
                opening = price
                closing = opening * (1 + direction * 0.001 * (offset + 1))
                rows.append(
                    {
                        "code": f"S{symbol_index}",
                        "trade_time": timestamp,
                        "open": opening,
                        "high": max(opening, closing) * 1.001,
                        "low": min(opening, closing) * 0.999,
                        "close": closing,
                        "volume": 1000 + day_index * 10 + offset,
                        "amount": closing * 1000,
                    }
                )
                price = closing
    return pd.DataFrame(rows)


class FactorResearchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.daily = build_daily_features(synthetic_bars())
        cls.dataset = build_direction_dataset(cls.daily)

    def test_feature_and_target_dates_are_strictly_ordered(self):
        self.assertTrue((self.dataset["target_date"] > self.dataset["feature_date"]).all())
        self.assertIn("last_30m_return", self.dataset.columns)

    def test_selected_factors_are_cached_separately(self):
        bars = synthetic_bars(days=12, symbols=2)
        with TemporaryDirectory() as cache_dir:
            selected = ["return_1d", "intraday_return"]
            first = build_daily_features(bars, selected, cache_dir)
            second = build_daily_features(bars, selected, cache_dir)
            pd.testing.assert_frame_equal(first, second)
            cache_files = list(Path(cache_dir).rglob("*.parquet"))
            self.assertEqual({path.parent.name for path in cache_files}, set(selected))
            self.assertEqual(len(cache_files), len(selected))

    def test_all_factory_modules_are_auto_registered(self):
        self.assertEqual(
            set(FACTOR_FACTORIES),
            {
                "return_1d", "return_5d", "volatility_5d", "volume_ratio_5d",
                "amplitude", "close_position", "intraday_return", "realized_vol",
                "positive_bar_ratio", "last_30m_return", "last_30m_volume_ratio",
            },
        )

    def test_split_keeps_dates_isolated(self):
        cutoff = self.dataset["target_date"].drop_duplicates().sort_values().iloc[48]
        split = split_by_date(self.dataset, cutoff)
        train_dates = set(split.train["target_date"])
        validation_dates = set(split.validation["target_date"])
        self.assertFalse(train_dates & validation_dates)
        self.assertLess(max(train_dates), min(validation_dates))
        self.assertEqual(min(validation_dates), cutoff)

    def test_tree_experiment_runs_end_to_end(self):
        cutoff = self.dataset["target_date"].drop_duplicates().sort_values().iloc[48]
        result = DirectionExperiment(cutoff, max_depth=2, min_samples_leaf=5).run(self.dataset)
        self.assertGreater(len(result.predictions), 0)
        self.assertTrue(np.isfinite(result.predictions["up_probability"]).all())
        self.assertAlmostEqual(float(result.feature_importance.sum()), 1.0, places=6)
        self.assertIn("auc", result.metrics)
        self.assertTrue(np.isfinite(result.metrics["auc"]))
        self.assertEqual(
            list(result.daily_accuracy_trend.columns),
            ["target_date", "samples", "accuracy", "accuracy_change"],
        )
        self.assertEqual(
            len(result.daily_accuracy_trend),
            result.predictions["target_date"].nunique(),
        )
        self.assertTrue(result.daily_accuracy_trend["accuracy"].between(0.0, 1.0).all())
        self.assertTrue(pd.isna(result.daily_accuracy_trend["accuracy_change"].iloc[0]))

    def test_walk_forward_training_precedes_each_prediction_date(self):
        cutoff = self.dataset["target_date"].drop_duplicates().sort_values().iloc[48]
        result = DirectionExperiment(cutoff, max_depth=2, min_samples_leaf=5).run(self.dataset)
        self.assertTrue((result.predictions["training_end_date"] < result.predictions["target_date"]).all())
        samples_by_date = result.predictions.groupby("target_date")["training_samples"].first()
        self.assertTrue(samples_by_date.is_monotonic_increasing)

    def test_default_window_uses_last_three_years(self):
        start, end = resolve_window(
            {"first_time": "2018-01-01 09:35:00", "last_time": "2025-06-30 15:00:00"}
        )
        self.assertEqual(start.date().isoformat(), "2022-06-30")
        self.assertEqual(end.date().isoformat(), "2025-06-30")


if __name__ == "__main__":
    unittest.main()
