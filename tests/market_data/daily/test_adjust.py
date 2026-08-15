"""验证复权系数的累乘口径、基准日与分组隔离。

口径依据：``adjustment_factor(t) == close(t-1) / pre_close(t)``，已在真实数据上
标定（2024 年 4524 个事件，比值中位数 1.0）。
"""

from __future__ import annotations

import unittest
from datetime import date

import pandas as pd

from quant.market_data.daily.adjust import (
    apply_adjustment,
    cumulative_factors,
    observed_event_ratio,
    theoretical_event_ratio,
)
from quant.market_data.daily.models import AdjustMode


def _bars() -> pd.DataFrame:
    """构造两只证券、各三个交易日的最小行情表。

    ``000001.SZ`` 在第二天除权，因子 1.05；``600000.SH`` 全程无除权。

    返回：
        含 ``code``、``trade_date`` 与价格成交列的日线表。
    """
    rows = [
        ("000001.SZ", "2024-01-02", 10.0, 10.0, 10.0, 10.0, 9.5, 100, 1000.0),
        ("000001.SZ", "2024-01-03", 11.0, 11.0, 11.0, 11.0, 10.0 / 1.05, 100, 1100.0),
        ("000001.SZ", "2024-01-04", 12.0, 12.0, 12.0, 12.0, 11.0, 100, 1200.0),
        ("600000.SH", "2024-01-02", 7.0, 7.0, 7.0, 7.0, 7.0, 100, 700.0),
        ("600000.SH", "2024-01-03", 7.2, 7.2, 7.2, 7.2, 7.0, 100, 720.0),
        ("600000.SH", "2024-01-04", 7.4, 7.4, 7.4, 7.4, 7.2, 100, 740.0),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "code", "trade_date", "open", "high", "low", "close",
            "pre_close", "volume", "amount",
        ],
    )
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    return frame


def _actions() -> pd.DataFrame:
    """构造只含一次现金分红的除权表。

    返回：
        含 ``code``、``ex_date`` 与各项除权字段的表。
    """
    return pd.DataFrame(
        [
            {
                "code": "000001.SZ",
                "ex_date": pd.Timestamp("2024-01-03"),
                "cash_dividend_per_share": 0.5,
                "bonus_share_per_share": 0.0,
                "capitalization_per_share": 0.0,
                "rights_issue_per_share": 0.0,
                "rights_issue_price": 0.0,
                "share_reform_flag": 0.0,
                "adjustment_factor": 1.05,
            }
        ]
    )


class CumulativeFactorTest(unittest.TestCase):
    """累乘系数的构造规则。"""

    def test_factor_is_cumulative_product_per_code(self) -> None:
        """多次除权应按时间顺序连乘，且各证券互不影响。"""
        actions = pd.DataFrame(
            [
                {"code": "A", "ex_date": pd.Timestamp("2024-01-03"), "adjustment_factor": 1.1},
                {"code": "A", "ex_date": pd.Timestamp("2024-02-03"), "adjustment_factor": 1.2},
                {"code": "B", "ex_date": pd.Timestamp("2024-01-03"), "adjustment_factor": 2.0},
            ]
        )
        result = cumulative_factors(actions).set_index(["code", "ex_date"])["hfq_factor"]
        self.assertAlmostEqual(result[("A", pd.Timestamp("2024-01-03"))], 1.1)
        self.assertAlmostEqual(result[("A", pd.Timestamp("2024-02-03"))], 1.1 * 1.2)
        self.assertAlmostEqual(result[("B", pd.Timestamp("2024-01-03"))], 2.0)

    def test_invalid_factor_is_treated_as_one(self) -> None:
        """非正或缺失的因子按 1.0 处理，避免把整条曲线算废。"""
        actions = pd.DataFrame(
            [
                {"code": "A", "ex_date": pd.Timestamp("2024-01-03"), "adjustment_factor": 0.0},
                {"code": "A", "ex_date": pd.Timestamp("2024-02-03"), "adjustment_factor": None},
            ]
        )
        result = cumulative_factors(actions)
        self.assertTrue((result["hfq_factor"] == 1.0).all())

    def test_empty_actions_return_empty_frame(self) -> None:
        """没有除权记录时返回具有标准列的空表。"""
        result = cumulative_factors(pd.DataFrame())
        self.assertTrue(result.empty)
        self.assertEqual(list(result.columns), ["code", "ex_date", "hfq_factor"])


class ApplyAdjustmentTest(unittest.TestCase):
    """复权后价格的精确取值。"""

    def test_none_mode_keeps_raw_prices(self) -> None:
        """不复权时价格必须与输入完全一致。"""
        result = apply_adjustment(_bars(), _actions(), mode=AdjustMode.NONE)
        self.assertTrue((result["adjust_factor"] == 1.0).all())
        self.assertAlmostEqual(
            result.loc[result["code"] == "000001.SZ", "close"].tolist()[1], 11.0
        )

    def test_backward_adjustment_restores_continuity(self) -> None:
        """后复权后，除权日的复权前收应当等于前一日的复权收盘。"""
        result = apply_adjustment(_bars(), _actions(), mode=AdjustMode.HFQ)
        subset = result[result["code"] == "000001.SZ"].reset_index(drop=True)
        self.assertAlmostEqual(subset.loc[0, "close"], 10.0)
        self.assertAlmostEqual(subset.loc[1, "close"], 11.0 * 1.05)
        self.assertAlmostEqual(subset.loc[2, "close"], 12.0 * 1.05)
        # 连续性：除权日的复权前收 == 前一日复权收盘。
        self.assertAlmostEqual(subset.loc[1, "pre_close"], subset.loc[0, "close"], places=9)

    def test_backward_adjustment_leaves_other_codes_untouched(self) -> None:
        """没有除权的证券在后复权下取值不变，证明分组之间没有串联。"""
        result = apply_adjustment(_bars(), _actions(), mode=AdjustMode.HFQ)
        subset = result[result["code"] == "600000.SH"]
        self.assertTrue((subset["adjust_factor"] == 1.0).all())
        self.assertAlmostEqual(subset["close"].tolist()[-1], 7.4)

    def test_forward_adjustment_pins_anchor_date(self) -> None:
        """前复权以基准日为 1，基准日之前的价格按比例缩小。"""
        result = apply_adjustment(
            _bars(), _actions(), mode=AdjustMode.QFQ, anchor=date(2024, 1, 4)
        )
        subset = result[result["code"] == "000001.SZ"].reset_index(drop=True)
        self.assertAlmostEqual(subset.loc[2, "close"], 12.0)
        self.assertAlmostEqual(subset.loc[1, "close"], 11.0)
        self.assertAlmostEqual(subset.loc[0, "close"], 10.0 / 1.05)

    def test_forward_adjustment_requires_anchor(self) -> None:
        """前复权缺少基准日时必须报错，不能静默使用会漂移的默认值。"""
        with self.assertRaises(ValueError):
            apply_adjustment(_bars(), _actions(), mode=AdjustMode.QFQ)

    def test_amount_is_never_adjusted(self) -> None:
        """成交额是真实成交金额，任何口径下都不调整。"""
        result = apply_adjustment(_bars(), _actions(), mode=AdjustMode.HFQ)
        subset = result[result["code"] == "000001.SZ"].reset_index(drop=True)
        self.assertAlmostEqual(subset.loc[1, "amount"], 1100.0)

    def test_volume_adjusted_only_on_request(self) -> None:
        """成交量默认保持原始股数，显式要求时才反向调整。"""
        plain = apply_adjustment(_bars(), _actions(), mode=AdjustMode.HFQ)
        scaled = apply_adjustment(
            _bars(), _actions(), mode=AdjustMode.HFQ, adjust_volume=True
        )
        plain_row = plain[(plain["code"] == "000001.SZ")].reset_index(drop=True).loc[1]
        scaled_row = scaled[(scaled["code"] == "000001.SZ")].reset_index(drop=True).loc[1]
        self.assertAlmostEqual(plain_row["volume"], 100)
        self.assertAlmostEqual(scaled_row["volume"], 100 / 1.05)


class QfqWindowIndependenceTest(unittest.TestCase):
    """前复权取值不得随查询窗口变化。"""

    def test_anchor_after_window_end_still_sees_the_event(self) -> None:
        """基准日晚于窗口终点时，必须仍然把基准日之前的除权算进来。

        回归用例：若除权记录只截到窗口终点，实际基准会退化成
        ``min(anchor, end)``，同一天的复权价就会随窗口变化。
        """
        bars = _bars()
        window = bars[bars["trade_date"] <= pd.Timestamp("2024-01-02")]
        full = apply_adjustment(
            _bars(), _actions(), mode=AdjustMode.QFQ, anchor=date(2024, 1, 4)
        )
        narrow = apply_adjustment(
            window, _actions(), mode=AdjustMode.QFQ, anchor=date(2024, 1, 4)
        )
        full_value = full[
            (full["code"] == "000001.SZ")
            & (full["trade_date"] == pd.Timestamp("2024-01-02"))
        ]["close"].iloc[0]
        narrow_value = narrow[narrow["code"] == "000001.SZ"]["close"].iloc[0]
        self.assertAlmostEqual(full_value, narrow_value, places=9)
        self.assertAlmostEqual(narrow_value, 10.0 / 1.05, places=9)


class ObservedRatioTest(unittest.TestCase):
    """由行情反推的除权比例，用于校验除权表。"""

    def test_observed_ratio_matches_declared_factor(self) -> None:
        """观测比例应当与声明的 adjustment_factor 一致。"""
        ratios = observed_event_ratio(_bars())
        row = ratios[
            (ratios["code"] == "000001.SZ")
            & (ratios["trade_date"] == pd.Timestamp("2024-01-03"))
        ]
        self.assertAlmostEqual(float(row["observed_ratio"].iloc[0]), 1.05, places=9)

    def test_first_row_of_each_code_has_no_ratio(self) -> None:
        """每只证券的首行没有前一日收盘，比例应为缺失。"""
        ratios = observed_event_ratio(_bars())
        first = ratios.groupby("code", sort=False).head(1)
        self.assertTrue(first["observed_ratio"].isna().all())

    def test_theoretical_ratio_matches_cash_dividend(self) -> None:
        """纯现金分红的理论比例应等于 前收/(前收-每股分红)。"""
        actions = _actions()
        ratio = theoretical_event_ratio(actions, pd.Series([10.0]))
        self.assertAlmostEqual(float(ratio.iloc[0]), 10.0 / 9.5, places=9)


if __name__ == "__main__":
    unittest.main()
