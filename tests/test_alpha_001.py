from __future__ import annotations

import unittest

import pandas as pd

from factor_research.factor_factories import FACTOR_FACTORIES


def _daily_frame(prices_by_code: dict[str, list[float]]) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=len(next(iter(prices_by_code.values()))))
    rows = [
        {"code": code, "trade_date": date, "close": close}
        for code, prices in prices_by_code.items()
        for date, close in zip(dates, prices)
    ]
    return pd.DataFrame(rows)


class Alpha001Test(unittest.TestCase):
    def test_is_auto_registered(self):
        self.assertIn("alpha_001", FACTOR_FACTORIES)

    def test_exact_cross_sectional_rank_and_original_index_order(self):
        prefix = [10.0] * 19 + [9.0]
        daily = _daily_frame(
            {
                "A": prefix + [10.0, 11.0, 12.0, 13.0, 14.0],
                "B": prefix + [10.0, 12.0, 11.0, 13.0, 12.0],
                "C": prefix + [20.0, 19.0, 18.0, 17.0, 16.0],
            }
        ).sample(frac=1.0, random_state=7)

        values = FACTOR_FACTORIES["alpha_001"].compute(pd.DataFrame(), daily)
        last_date = daily["trade_date"].max()
        result = daily.assign(alpha_001=values).loc[
            lambda frame: frame["trade_date"].eq(last_date)
        ].set_index("code")

        # 最近 5 日最大输入的位置分别为 A=5、B=4、C=1。
        self.assertAlmostEqual(result.at["A", "alpha_001"], 1.0)
        self.assertAlmostEqual(result.at["B", "alpha_001"], 2.0 / 3.0)
        self.assertAlmostEqual(result.at["C", "alpha_001"], 1.0 / 3.0)
        self.assertTrue(values.index.equals(daily.index))

    def test_negative_returns_require_full_volatility_and_argmax_windows(self):
        prices = [100.0 * (0.99 ** offset) for offset in range(25)]
        daily = _daily_frame({"ONLY": prices})

        values = FACTOR_FACTORIES["alpha_001"].compute(pd.DataFrame(), daily)

        self.assertTrue(values.iloc[:24].isna().all())
        self.assertEqual(values.iloc[24], 1.0)

    def test_argmax_and_cross_sectional_ties_use_documented_methods(self):
        prefix = [10.0] * 19 + [9.0]
        daily = _daily_frame(
            {
                # A 的最大输入在窗口第 1、2 日并列，应取最早的第 1 日。
                "A": prefix + [20.0, 20.0, 19.0, 18.0, 17.0],
                "B": prefix + [20.0, 19.0, 18.0, 17.0, 16.0],
                "C": prefix + [10.0, 11.0, 12.0, 13.0, 14.0],
            }
        )

        values = FACTOR_FACTORIES["alpha_001"].compute(pd.DataFrame(), daily)
        result = daily.assign(alpha_001=values).groupby("code", sort=False).tail(1)
        result = result.set_index("code")

        # A、B 的极值位置均为 1，横截面并列平均名次为 (1 + 2) / 2 / 3。
        self.assertEqual(result.at["A", "alpha_001"], 0.5)
        self.assertEqual(result.at["B", "alpha_001"], 0.5)
        self.assertEqual(result.at["C", "alpha_001"], 1.0)

    def test_securities_do_not_share_return_or_rolling_state(self):
        dates = pd.bdate_range("2024-01-02", periods=5)
        daily = pd.DataFrame(
            {
                "code": ["A", "B"] * 5,
                "trade_date": [date for date in dates for _ in range(2)],
                "close": [value for pair in zip(range(1, 6), range(105, 100, -1)) for value in pair],
            }
        )

        values = FACTOR_FACTORIES["alpha_001"].compute(pd.DataFrame(), daily)
        result = daily.assign(alpha_001=values).set_index(["trade_date", "code"])

        self.assertEqual(result.at[(dates[-1], "A"), "alpha_001"], 1.0)
        self.assertTrue(pd.isna(result.at[(dates[-1], "B"), "alpha_001"]))


if __name__ == "__main__":
    unittest.main()
