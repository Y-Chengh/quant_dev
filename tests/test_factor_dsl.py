from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from factor_research.factor_dsl import DailyFactorFrame, ExpressionNode
from factor_research.factor_factories import FACTOR_FACTORIES


def _daily_frame(days: int = 8, codes: tuple[str, ...] = ("A", "B")) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=days)
    rows = []
    for code_index, code in enumerate(codes):
        for day_index, date in enumerate(dates):
            close = 10.0 + code_index * 10 + day_index
            rows.append(
                {
                    "code": code,
                    "trade_date": date,
                    "open": close - 0.5,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 100.0 + code_index * 10 + day_index,
                }
            )
    return pd.DataFrame(rows)


def _alpha_002_frame(days: int = 8) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=days)
    volume_a_high = [True, False, True, False, True, False]
    return_a_high = [True, True, False, False, True, False]
    log_volumes = {"A": [10.0, 10.0], "B": [10.0, 10.0]}
    for offset in range(days - 2):
        log_volumes["A"].append(
            log_volumes["A"][offset] + (1.0 if volume_a_high[offset] else 0.0)
        )
        log_volumes["B"].append(
            log_volumes["B"][offset] + (0.0 if volume_a_high[offset] else 1.0)
        )
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


class FactorDslTest(unittest.TestCase):
    def test_normalized_daily_keys_must_remain_unique(self):
        base = _daily_frame(days=2, codes=("1",))

        # 同一自然日的不同时刻会在 normalize 后碰撞，必须在滚动计算前拒绝。
        time_collision = pd.concat(
            [
                base.iloc[[0]],
                base.iloc[[0]].assign(
                    trade_date=base.iloc[0]["trade_date"] + pd.Timedelta(hours=12)
                ),
            ],
            ignore_index=True,
        )
        with self.assertRaisesRegex(ValueError, "规范化后存在重复"):
            DailyFactorFrame(time_collision)

        # 数值代码 1 与字符串代码 "1" 也会规范化为相同证券代码。
        code_collision = pd.concat(
            [base.iloc[[0]].assign(code=1), base.iloc[[0]].assign(code="1")],
            ignore_index=True,
        )
        with self.assertRaisesRegex(ValueError, "规范化后存在重复"):
            DailyFactorFrame(code_collision)

    def test_elementwise_arithmetic_preserves_original_index_and_row_order(self):
        daily = _daily_frame().sample(frac=1.0, random_state=17)
        daily.index = [index // 2 for index in range(len(daily))]  # 重复索引也必须逐行对齐。
        frame = DailyFactorFrame(daily)

        values = ((frame.close() - frame.open()) / frame.open()).compute("ratio")
        expected = (daily["close"] - daily["open"]) / daily["open"]

        self.assertTrue(values.index.equals(daily.index))
        np.testing.assert_allclose(values.to_numpy(), expected.to_numpy())
        self.assertEqual(values.name, "ratio")

    def test_time_series_windows_are_isolated_by_code(self):
        daily = _daily_frame(days=4).sort_values("trade_date", kind="stable")
        frame = DailyFactorFrame(daily)

        mean = frame.close().mean(3).compute()
        delta = frame.close().delta(2).compute()
        result = daily.assign(mean=mean, delta=delta).sort_values(["code", "trade_date"])

        for _, group in result.groupby("code"):
            self.assertTrue(group["mean"].iloc[:2].isna().all())
            self.assertAlmostEqual(group["mean"].iloc[2], group["close"].iloc[:3].mean())
            self.assertTrue(group["delta"].iloc[:2].isna().all())
            self.assertEqual(group["delta"].iloc[2], 2.0)

    def test_remaining_rolling_operators_have_exact_documented_values(self):
        daily = _daily_frame(days=3, codes=("A",))
        daily["close"] = [1.0, 2.0, 3.0]
        daily["volume"] = [2.0, 4.0, 6.0]
        frame = DailyFactorFrame(daily)

        self.assertEqual(frame.close().sum(3).compute().iloc[-1], 6.0)
        self.assertEqual(frame.close().min(3).compute().iloc[-1], 1.0)
        self.assertEqual(frame.close().max(3).compute().iloc[-1], 3.0)
        self.assertEqual(frame.close().ts_rank(3).compute().iloc[-1], 1.0)
        self.assertEqual(frame.close().argmin(3).compute().iloc[-1], 1.0)
        self.assertEqual(frame.close().delay(1).compute().iloc[-1], 2.0)
        self.assertEqual(frame.close().returns(1).compute().iloc[-1], 0.5)
        self.assertAlmostEqual(
            frame.close().correlation(frame.volume(), 3).compute().iloc[-1], 1.0
        )
        self.assertAlmostEqual(
            frame.close().covariance(frame.volume(), 3).compute().iloc[-1], 2.0
        )

    def test_cross_sectional_normalizers_and_winsorize_are_exact(self):
        daily = _daily_frame(days=1, codes=("A", "B", "C"))
        daily["close"] = [1.0, 2.0, 3.0]
        frame = DailyFactorFrame(daily)

        demeaned = frame.close().demean().compute()
        zscore = frame.close().zscore().compute()
        scaled = frame.close().scale().compute()
        winsorized = frame.close().winsorize(0.25, 0.75).compute()

        np.testing.assert_allclose(demeaned, [-1.0, 0.0, 1.0])
        np.testing.assert_allclose(
            zscore, [-np.sqrt(1.5), 0.0, np.sqrt(1.5)]
        )
        np.testing.assert_allclose(scaled, [1 / 6, 2 / 6, 3 / 6])
        np.testing.assert_allclose(winsorized, [1.5, 2.0, 2.5])

    def test_missing_comparison_is_false_and_where_uses_false_branch(self):
        daily = _daily_frame(days=2, codes=("A",))
        daily.loc[0, "close"] = np.nan
        frame = DailyFactorFrame(daily)

        values = frame.where(frame.close() < 0, -1.0, 1.0).compute()

        np.testing.assert_allclose(values, [1.0, 1.0])

    def test_cross_sectional_rank_and_ties_are_isolated_by_date(self):
        daily = _daily_frame(days=2, codes=("A", "B", "C"))
        first_date = daily["trade_date"].min()
        daily.loc[daily["trade_date"].eq(first_date), "close"] = [1.0, 1.0, 3.0]

        ranked = DailyFactorFrame(daily).close().rank().compute()
        result = daily.assign(rank=ranked).set_index(["trade_date", "code"])

        self.assertEqual(result.at[(first_date, "A"), "rank"], 0.5)
        self.assertEqual(result.at[(first_date, "B"), "rank"], 0.5)
        self.assertEqual(result.at[(first_date, "C"), "rank"], 1.0)

    def test_argmax_uses_first_one_based_position_and_full_window(self):
        daily = _daily_frame(days=5, codes=("A",))
        daily["close"] = [5.0, 5.0, 4.0, 3.0, 2.0]

        values = DailyFactorFrame(daily).close().argmax(5).compute()

        self.assertTrue(values.iloc[:4].isna().all())
        self.assertEqual(values.iloc[4], 1.0)

    def test_argmax_and_argmin_ignore_nan_but_preserve_original_position(self):
        daily = _daily_frame(days=3, codes=("A",))
        daily["close"] = [1.0, np.nan, 2.0]
        frame = DailyFactorFrame(daily)

        argmax = frame.close().argmax(3, min_periods=2).compute()
        argmin = frame.close().argmin(3, min_periods=2).compute()

        self.assertEqual(argmax.iloc[-1], 3.0)
        self.assertEqual(argmin.iloc[-1], 1.0)

        leading_missing = daily.copy()
        leading_missing["close"] = [np.nan, 4.0, 3.0]
        leading_frame = DailyFactorFrame(leading_missing)
        self.assertEqual(
            leading_frame.close().argmax(3, min_periods=2).compute().iloc[-1],
            2.0,
        )

    def test_invalid_or_future_looking_windows_are_rejected(self):
        frame = DailyFactorFrame(_daily_frame())

        with self.assertRaisesRegex(ValueError, "禁止"):
            frame.close().delay(-1)
        with self.assertRaisesRegex(ValueError, "正整数"):
            frame.close().delta(0)
        with self.assertRaisesRegex(ValueError, "min_periods"):
            frame.close().stddev(5, min_periods=6)

    def test_missing_and_invalid_numeric_inputs_follow_documented_rules(self):
        daily = _daily_frame(days=3, codes=("A",))
        daily.loc[0, "volume"] = 0.0
        daily.loc[1, "open"] = 0.0
        frame = DailyFactorFrame(daily)

        logged = frame.volume().log().compute()
        divided = (frame.close() / frame.open()).compute()

        self.assertTrue(pd.isna(logged.iloc[0]))
        self.assertTrue(pd.isna(divided.iloc[1]))
        self.assertTrue(np.isfinite(logged.iloc[1:]).all())

        daily.loc[2, "close"] = np.inf
        finite_values = DailyFactorFrame(daily).close().compute()
        self.assertTrue(pd.isna(finite_values.iloc[2]))

    def test_expression_serialization_is_stable_and_round_trips(self):
        frame = DailyFactorFrame(_daily_frame())
        expression = frame.close().delta(2).stddev(3).rank()

        restored = ExpressionNode.from_dict(expression.to_dict())

        self.assertEqual(restored.canonical, expression.node.canonical)
        self.assertEqual(restored.factor_id, expression.node.factor_id)
        self.assertEqual(restored.lookback, 4)
        self.assertTrue(restored.causal)

        with self.assertRaisesRegex(ValueError, "不能包含 inputs"):
            ExpressionNode.from_dict(
                {
                    "operator": "column",
                    "inputs": [{"operator": "constant", "parameters": {"value": 1}}],
                    "parameters": {"name": "close"},
                }
            )

    def test_changing_future_data_cannot_change_historical_outputs(self):
        daily = _daily_frame(days=8, codes=("A", "B", "C"))
        expression = lambda frame: frame.close().returns(1).stddev(3).rank()
        original = expression(DailyFactorFrame(daily)).compute()
        cutoff = daily["trade_date"].sort_values().unique()[4]

        modified = daily.copy()
        modified.loc[modified["trade_date"] > cutoff, "close"] *= 1000
        changed = expression(DailyFactorFrame(modified)).compute()

        historical = daily["trade_date"] <= cutoff
        np.testing.assert_allclose(
            original[historical], changed[historical], equal_nan=True
        )

    def test_dsl_reproduces_existing_alpha_001_exactly(self):
        dates = pd.bdate_range("2024-01-02", periods=25)
        prefix = [10.0] * 19 + [9.0]
        prices = {
            "A": prefix + [10.0, 11.0, 12.0, 13.0, 14.0],
            "B": prefix + [10.0, 12.0, 11.0, 13.0, 12.0],
            "C": prefix + [20.0, 19.0, 18.0, 17.0, 16.0],
        }
        daily = pd.DataFrame(
            [
                {"code": code, "trade_date": date, "close": close}
                for code, series in prices.items()
                for date, close in zip(dates, series)
            ]
        ).sample(frac=1.0, random_state=7)
        frame = DailyFactorFrame(daily)
        returns = frame.close().returns(1)
        conditional = frame.where(
            returns < 0,
            returns.stddev(20),
            frame.close(),
        )
        dsl = conditional.signed_power(2).argmax(5).rank().compute()

        existing = FACTOR_FACTORIES["alpha_001"].compute(pd.DataFrame(), daily)

        np.testing.assert_allclose(dsl, existing, equal_nan=True)

    def test_dsl_reproduces_existing_alpha_002_exactly(self):
        daily = _alpha_002_frame().sample(frac=1.0, random_state=11)
        frame = DailyFactorFrame(daily)
        volume_side = frame.volume().log().delta(2).rank()
        return_side = ((frame.close() - frame.open()) / frame.open()).rank()
        dsl = -volume_side.correlation(return_side, 6).compute()

        existing = FACTOR_FACTORIES["alpha_002"].compute(pd.DataFrame(), daily)

        np.testing.assert_allclose(dsl, existing, equal_nan=True)


if __name__ == "__main__":
    unittest.main()
