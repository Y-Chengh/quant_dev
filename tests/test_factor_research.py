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
from factor_research.models import DirectionModel, DirectionModelFactory
from factor_research.reporting import write_evaluation_report
from run_factor_demo import resolve_run_output_paths, resolve_window


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
                "amplitude", "candle_body_ratio", "close_position",
                "intraday_return", "realized_vol",
                "positive_bar_ratio", "last_30m_return", "last_30m_volume_ratio",
                "close_to_ma_5d", "ma_distance_change_5d", "ma_5d_slope",
                "ma_spread_5d_20d", "ma_spread_change_5d_20d", "return_10d",
                "return_20d", "momentum_acceleration_5d_20d",
                "up_days_ratio_5d", "breakout_strength_20d",
                "signed_volume_imbalance", "close_to_vwap", "overnight_gap",
                "intraday_path_efficiency", "downside_semivol",
                "channel_position_20d",
            },
        )

    def test_continuous_momentum_factor_values(self):
        dates = pd.bdate_range("2024-01-02", periods=25)
        daily = pd.DataFrame(
            {
                "code": ["UP"] * 25 + ["DOWN"] * 25,
                "trade_date": list(dates) * 2,
                "close": list(range(1, 26)) + list(range(100, 75, -1)),
            }
        )
        empty_bars = pd.DataFrame()
        values = {
            name: FACTOR_FACTORIES[name].compute(empty_bars, daily)
            for name in {
                "close_to_ma_5d", "ma_distance_change_5d", "ma_5d_slope",
                "ma_spread_5d_20d", "ma_spread_change_5d_20d", "return_10d",
                "return_20d", "momentum_acceleration_5d_20d",
                "up_days_ratio_5d", "breakout_strength_20d",
            }
        }

        up_last = 24
        self.assertAlmostEqual(values["close_to_ma_5d"].iloc[up_last], 25 / 23 - 1)
        self.assertAlmostEqual(
            values["ma_distance_change_5d"].iloc[up_last],
            (25 / 23 - 1) - (24 / 22 - 1),
        )
        self.assertAlmostEqual(values["ma_5d_slope"].iloc[up_last], 23 / 22 - 1)
        self.assertAlmostEqual(values["ma_spread_5d_20d"].iloc[up_last], 23 / 15.5 - 1)
        self.assertAlmostEqual(
            values["ma_spread_change_5d_20d"].iloc[up_last],
            (23 / 15.5 - 1) - (22 / 14.5 - 1),
        )
        self.assertAlmostEqual(values["return_10d"].iloc[up_last], 25 / 15 - 1)
        self.assertAlmostEqual(values["return_20d"].iloc[up_last], 25 / 5 - 1)
        self.assertAlmostEqual(
            values["momentum_acceleration_5d_20d"].iloc[up_last],
            np.log(25 / 20) / 5 - np.log(25 / 5) / 20,
        )
        self.assertEqual(values["up_days_ratio_5d"].iloc[up_last], 1.0)
        self.assertAlmostEqual(values["breakout_strength_20d"].iloc[up_last], 25 / 24 - 1)

        # 第二只证券独立计算，下降序列不会继承第一只证券的窗口状态。
        down_last = 49
        self.assertEqual(values["up_days_ratio_5d"].iloc[down_last], 0.0)
        self.assertAlmostEqual(
            values["breakout_strength_20d"].iloc[down_last], 76 / 96 - 1
        )
        self.assertTrue(pd.isna(values["close_to_ma_5d"].iloc[3]))
        self.assertTrue(pd.isna(values["return_20d"].iloc[19]))
        self.assertTrue(pd.isna(values["breakout_strength_20d"].iloc[19]))

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
        self.assertEqual(result.model_name, "simple_decision_tree")
        self.assertGreater(len(result.predictions), 0)
        self.assertTrue(np.isfinite(result.predictions["up_probability"]).all())
        self.assertAlmostEqual(float(result.feature_importance.sum()), 1.0, places=6)
        self.assertIn("auc", result.metrics)
        self.assertTrue(np.isfinite(result.metrics["auc"]))
        self.assertIn("ic", result.metrics)
        self.assertTrue(np.isfinite(result.metrics["ic"]))
        self.assertAlmostEqual(
            result.metrics["ic"],
            result.predictions["up_probability"].corr(
                result.predictions["target_return"]
            ),
        )
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
        with TemporaryDirectory() as report_dir:
            report_path = Path(report_dir) / "evaluation.md"
            chart_path = Path(report_dir) / "evaluation_accuracy.svg"
            write_evaluation_report(result, report_path, chart_path, "test-run", {"max_depth": 2})
            report = report_path.read_text(encoding="utf-8")
            self.assertIn("ROC AUC", report)
            self.assertIn("| IC |", report)
            self.assertIn(chart_path.name, report)
            self.assertGreater(report.index("## 每日预估汇总"), report.index("## 运行参数"))
            self.assertEqual(report.rfind("## "), report.index("## 每日预估汇总"))
            self.assertIn("<polyline", chart_path.read_text(encoding="utf-8"))

    def test_experiment_accepts_an_injected_model_factory(self):
        class ConstantProbabilityModel(DirectionModel):
            def __init__(self):
                self.feature_importances_ = np.array([], dtype=float)

            def fit(self, X: np.ndarray, y: np.ndarray):
                self.feature_importances_ = np.full(X.shape[1], 1 / X.shape[1])
                return self

            def predict_proba(self, X: np.ndarray) -> np.ndarray:
                positive = np.full(len(X), 0.75)
                return np.column_stack([1 - positive, positive])

        class CountingModelFactory(DirectionModelFactory):
            name = "constant_probability"

            def __init__(self):
                self.created = 0

            def create(self):
                self.created += 1
                return ConstantProbabilityModel()

        cutoff = self.dataset["target_date"].drop_duplicates().sort_values().iloc[48]
        factory = CountingModelFactory()
        result = DirectionExperiment(cutoff, model_factory=factory).run(self.dataset)

        self.assertEqual(result.model_name, "constant_probability")
        self.assertTrue((result.predictions["up_probability"] == 0.75).all())
        self.assertEqual(factory.created, result.predictions["target_date"].nunique())

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

    def test_run_artifacts_are_archived_by_start_date(self):
        log_path, report_path, chart_path = resolve_run_output_paths(
            Path("logs"),
            pd.Timestamp("2026-08-02 23:59:58").to_pydatetime(),
            "a1b2c3d4",
        )
        self.assertEqual(log_path.parent, Path("logs/2026-08-02"))
        self.assertEqual(report_path.parent, log_path.parent)
        self.assertEqual(chart_path.parent, log_path.parent)
        self.assertEqual(log_path.name, "factor_demo_20260802_235958_a1b2c3d4.log")


if __name__ == "__main__":
    unittest.main()
