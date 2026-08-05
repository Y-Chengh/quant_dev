from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from factor_research.factor_factories import FACTOR_FACTORIES
from factor_research.factors import build_daily_features


class AdditionalFactorsTest(unittest.TestCase):
    def test_intraday_factor_exact_values_and_zero_activity_boundaries(self):
        date = pd.Timestamp("2024-01-02")
        bars = pd.DataFrame(
            [
                {"code": "A", "trade_time": date + pd.Timedelta(hours=9, minutes=35),
                 "open": 10.0, "high": 11.0, "low": 10.0, "close": 11.0, "volume": 100.0},
                {"code": "A", "trade_time": date + pd.Timedelta(hours=9, minutes=40),
                 "open": 11.0, "high": 11.0, "low": 10.0, "close": 10.0, "volume": 300.0},
                {"code": "B", "trade_time": date + pd.Timedelta(hours=9, minutes=35),
                 "open": 20.0, "high": 20.0, "low": 20.0, "close": 20.0, "volume": 0.0},
            ]
        )
        names = [
            "signed_volume_imbalance", "close_to_vwap",
            "intraday_path_efficiency", "downside_semivol",
        ]
        result = build_daily_features(bars, names).set_index("code")

        self.assertAlmostEqual(result.at["A", "signed_volume_imbalance"], -0.5)
        self.assertAlmostEqual(result.at["A", "close_to_vwap"], 10.0 / 10.25 - 1)
        self.assertEqual(result.at["A", "intraday_path_efficiency"], 0.0)
        self.assertAlmostEqual(
            result.at["A", "downside_semivol"],
            np.sqrt(((0.0 ** 2) + ((10.0 / 11.0 - 1) ** 2)) / 2),
        )
        self.assertTrue(pd.isna(result.at["B", "signed_volume_imbalance"]))
        self.assertTrue(pd.isna(result.at["B", "close_to_vwap"]))
        self.assertTrue(pd.isna(result.at["B", "intraday_path_efficiency"]))
        self.assertEqual(result.at["B", "downside_semivol"], 0.0)

    def test_intraday_factors_do_not_mix_securities(self):
        date = pd.Timestamp("2024-01-02")
        bars = pd.DataFrame(
            [
                {"code": "UP", "trade_time": date + pd.Timedelta(hours=9, minutes=35),
                 "open": 10.0, "high": 11.0, "low": 10.0, "close": 11.0, "volume": 5.0},
                {"code": "DOWN", "trade_time": date + pd.Timedelta(hours=9, minutes=35),
                 "open": 20.0, "high": 20.0, "low": 19.0, "close": 19.0, "volume": 7.0},
            ]
        )
        result = build_daily_features(
            bars, ["signed_volume_imbalance", "intraday_path_efficiency"]
        ).set_index("code")

        self.assertEqual(result.at["UP", "signed_volume_imbalance"], 1.0)
        self.assertEqual(result.at["DOWN", "signed_volume_imbalance"], -1.0)
        self.assertEqual(result.at["UP", "intraday_path_efficiency"], 1.0)
        self.assertEqual(result.at["DOWN", "intraday_path_efficiency"], -1.0)

    def test_overnight_gap_is_grouped_by_security(self):
        dates = pd.bdate_range("2024-01-02", periods=2)
        daily = pd.DataFrame(
            {
                "code": ["A", "A", "B", "B"],
                "trade_date": [dates[0], dates[1], dates[0], dates[1]],
                "open": [9.0, 12.0, 19.0, 18.0],
                "close": [10.0, 12.5, 20.0, 17.5],
            }
        )
        values = FACTOR_FACTORIES["overnight_gap"].compute(pd.DataFrame(), daily)

        self.assertTrue(pd.isna(values.iloc[0]))
        self.assertAlmostEqual(values.iloc[1], 12.0 / 10.0 - 1)
        self.assertTrue(pd.isna(values.iloc[2]))
        self.assertAlmostEqual(values.iloc[3], 18.0 / 20.0 - 1)

    def test_channel_position_uses_only_previous_twenty_days(self):
        dates = pd.bdate_range("2024-01-02", periods=21)
        daily = pd.DataFrame(
            {
                "code": ["A"] * 21 + ["B"] * 21,
                "trade_date": list(dates) * 2,
                "high": list(np.arange(10.0, 30.0)) + [1000.0] + [5.0] * 21,
                "low": list(np.arange(0.0, 20.0)) + [-1000.0] + [5.0] * 21,
                "close": [5.0] * 20 + [24.0] + [5.0] * 21,
            }
        )
        values = FACTOR_FACTORIES["channel_position_20d"].compute(
            pd.DataFrame(), daily
        )

        self.assertTrue(pd.isna(values.iloc[19]))
        self.assertAlmostEqual(values.iloc[20], 24.0 / 29.0)
        self.assertTrue(pd.isna(values.iloc[41]))


if __name__ == "__main__":
    unittest.main()
