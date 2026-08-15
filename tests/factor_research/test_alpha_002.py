from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from quant.factor_research.factor_factories import FACTOR_FACTORIES


def _rank_pattern_frame(days: int = 8) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=days)
    volume_a_high = [True, False, True, False, True, False]
    return_a_high = [True, True, False, False, True, False]
    log_volumes = {"A": [10.0, 10.0], "B": [10.0, 10.0]}

    for offset in range(max(0, days - 2)):
        a_delta = 1.0 if volume_a_high[offset] else 0.0
        b_delta = 0.0 if volume_a_high[offset] else 1.0
        log_volumes["A"].append(log_volumes["A"][offset] + a_delta)
        log_volumes["B"].append(log_volumes["B"][offset] + b_delta)

    rows = []
    for code in ("A", "B"):
        for offset, date in enumerate(dates):
            if offset < 2:
                intraday_return = 0.01
            else:
                a_is_high = return_a_high[offset - 2]
                intraday_return = 0.02 if (code == "A") == a_is_high else 0.01
            rows.append(
                {
                    "code": code,
                    "trade_date": date,
                    "open": 100.0,
                    "close": 100.0 * (1.0 + intraday_return),
                    "volume": np.exp(log_volumes[code][offset]),
                }
            )
    return pd.DataFrame(rows)


class Alpha002Test(unittest.TestCase):
    def test_is_auto_registered(self):
        self.assertIn("alpha_002", FACTOR_FACTORIES)

    def test_exact_cross_sectional_ranks_rolling_correlation_and_index_order(self):
        daily = _rank_pattern_frame().sample(frac=1.0, random_state=11)

        values = FACTOR_FACTORIES["alpha_002"].compute(pd.DataFrame(), daily)
        result = daily.assign(alpha_002=values).set_index(["trade_date", "code"])
        last_date = daily["trade_date"].max()

        # A 的两组排名等价于 010101 与 001101，相关系数为 1/3，因子取负。
        self.assertAlmostEqual(result.at[(last_date, "A"), "alpha_002"], -1.0 / 3.0)
        self.assertAlmostEqual(result.at[(last_date, "B"), "alpha_002"], -1.0 / 3.0)
        self.assertTrue(values.index.equals(daily.index))

    def test_requires_two_day_delta_and_six_complete_rank_observations(self):
        daily = _rank_pattern_frame(days=7)

        values = FACTOR_FACTORIES["alpha_002"].compute(pd.DataFrame(), daily)

        self.assertTrue(values.isna().all())

    def test_nonpositive_volume_and_zero_open_produce_missing_windows(self):
        daily = _rank_pattern_frame()
        last_date = daily["trade_date"].max()
        daily.loc[
            daily["code"].eq("A") & daily["trade_date"].eq(last_date), "volume"
        ] = 0.0
        daily.loc[
            daily["code"].eq("B") & daily["trade_date"].eq(last_date), "open"
        ] = 0.0

        values = FACTOR_FACTORIES["alpha_002"].compute(pd.DataFrame(), daily)
        result = daily.assign(alpha_002=values).set_index(["trade_date", "code"])

        self.assertTrue(pd.isna(result.at[(last_date, "A"), "alpha_002"]))
        self.assertTrue(pd.isna(result.at[(last_date, "B"), "alpha_002"]))

    def test_missing_daily_input_produces_missing_window(self):
        last_date = _rank_pattern_frame()["trade_date"].max()

        for column in ("volume", "open", "close"):
            with self.subTest(column=column):
                daily = _rank_pattern_frame()
                daily.loc[
                    daily["code"].eq("A") & daily["trade_date"].eq(last_date),
                    column,
                ] = np.nan

                values = FACTOR_FACTORIES["alpha_002"].compute(
                    pd.DataFrame(), daily
                )
                result = daily.assign(alpha_002=values).set_index(
                    ["trade_date", "code"]
                )

                self.assertTrue(
                    pd.isna(result.at[(last_date, "A"), "alpha_002"])
                )

    def test_constant_rank_window_has_undefined_correlation(self):
        daily = _rank_pattern_frame()
        dates = daily["trade_date"].drop_duplicates().sort_values()

        # 最近 6 个可计算日中 A 的成交量差分始终高于 B，因此两者各自的
        # volume rank 都是常数，Pearson 相关系数因零方差而没有定义。
        for code, delta in (("A", 1.0), ("B", 0.0)):
            log_volumes = [10.0, 10.0]
            for offset in range(6):
                log_volumes.append(log_volumes[offset] + delta)
            for date, log_volume in zip(dates, log_volumes):
                daily.loc[
                    daily["code"].eq(code) & daily["trade_date"].eq(date),
                    "volume",
                ] = np.exp(log_volume)

        values = FACTOR_FACTORIES["alpha_002"].compute(pd.DataFrame(), daily)
        result = daily.assign(alpha_002=values).set_index(["trade_date", "code"])
        last_date = dates.iloc[-1]

        self.assertTrue(pd.isna(result.at[(last_date, "A"), "alpha_002"]))
        self.assertTrue(pd.isna(result.at[(last_date, "B"), "alpha_002"]))

    def test_securities_do_not_share_delta_or_rolling_state(self):
        daily = _rank_pattern_frame()
        extra = daily.loc[daily["code"].eq("A")].tail(1).copy()
        extra["code"] = "C"
        combined = pd.concat([extra, daily], ignore_index=True)

        values = FACTOR_FACTORIES["alpha_002"].compute(pd.DataFrame(), combined)

        self.assertTrue(pd.isna(values.iloc[0]))
        self.assertTrue(values.iloc[1:].notna().any())


if __name__ == "__main__":
    unittest.main()
