"""校验 ``adjust_factor`` 作为基础列贯通到因子层的契约。

覆盖四件事：日频路径把系数原样传给因子工厂、分钟路径补 1.0 使两条路径的列宇宙
一致、DSL 表达式可以引用它、以及它进入缓存输入指纹从而不会跨口径串味。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from quant.factor_research.data import validate_daily_bars
from quant.factor_research.factors import (
    BASE_DAILY_COLUMNS,
    _daily_input_fingerprint,
    _prepare_daily_bars,
    aggregate_daily_bars,
    build_daily_features,
    build_daily_features_from_daily,
)


def _daily_bars(with_factor: bool = True) -> pd.DataFrame:
    """构造两只证券、二十五个交易日的日频行情。

    参数：
        with_factor: 是否附带 ``adjust_factor`` 列。为 ``True`` 时第 10 个交易日
            起跳到 2.0，模拟一次除权；为 ``False`` 时整列缺席，用于验证补 1.0
            的缺省行为。

    返回：
        含标准日频列的行情表。
    """

    rows = []
    for code, base in (("000001.SZ", 10.0), ("600000.SH", 20.0)):
        for index in range(25):
            close = base + index * 0.1
            row = {
                "code": code,
                "trade_date": pd.Timestamp("2024-01-02") + pd.Timedelta(days=index),
                "open": close - 0.05,
                "high": close + 0.08,
                "low": close - 0.09,
                "close": close,
                "volume": 1000 + index,
                "amount": (1000 + index) * close,
            }
            if with_factor:
                row["adjust_factor"] = 1.0 if index < 10 else 2.0
            rows.append(row)
    return pd.DataFrame(rows)


def _minute_bars() -> pd.DataFrame:
    """构造两只证券、三个交易日、每天四根的分钟行情。

    返回：
        含标准分钟列的行情表。
    """

    rows = []
    for code, base in (("000001.SZ", 10.0), ("600000.SH", 20.0)):
        for day in range(3):
            for bar in range(4):
                price = base + day * 0.2 + bar * 0.01
                rows.append(
                    {
                        "code": code,
                        "trade_time": pd.Timestamp("2024-01-02")
                        + pd.Timedelta(days=day, hours=9, minutes=35 + 5 * bar),
                        "open": price,
                        "high": price + 0.02,
                        "low": price - 0.02,
                        "close": price + 0.01,
                        "volume": 500 + bar,
                    }
                )
    return pd.DataFrame(rows)


class PrepareDailyBarsTest(unittest.TestCase):
    """``_prepare_daily_bars`` 的列契约。"""

    def test_adjust_factor_is_preserved(self) -> None:
        """输入带系数时必须原样保留，且只保留基础列。"""
        prepared = _prepare_daily_bars(validate_daily_bars(_daily_bars()))
        self.assertEqual(list(prepared.columns), list(BASE_DAILY_COLUMNS))
        # 按 (code, trade_date) 对齐比较，不依赖两边的行序。
        expected = (
            _daily_bars()
            .set_index(["code", "trade_date"])["adjust_factor"]
            .astype(float)
        )
        actual = prepared.set_index(["code", "trade_date"])["adjust_factor"]
        pd.testing.assert_series_equal(actual, expected.reindex(actual.index))

    def test_missing_column_defaults_to_one(self) -> None:
        """输入没有系数列时补 1.0，含义是这份行情未经复权。"""
        prepared = _prepare_daily_bars(validate_daily_bars(_daily_bars(with_factor=False)))
        self.assertEqual(list(prepared.columns), list(BASE_DAILY_COLUMNS))
        self.assertTrue((prepared["adjust_factor"] == 1.0).all())

    def test_extra_columns_are_still_dropped(self) -> None:
        """``amount`` 这类额外列必须继续被丢掉，列宇宙不能扩散。"""
        prepared = _prepare_daily_bars(validate_daily_bars(_daily_bars()))
        self.assertNotIn("amount", prepared.columns)

    def test_dtype_is_normalized_to_float64(self) -> None:
        """整数与可空扩展 dtype 都要落到 numpy float64，否则指纹会随 backend 漂移。"""
        for dtype in ("int64", "Float64"):
            with self.subTest(dtype=dtype):
                bars = _daily_bars()
                bars["adjust_factor"] = bars["adjust_factor"].astype(dtype)
                prepared = _prepare_daily_bars(validate_daily_bars(bars))
                self.assertEqual(prepared["adjust_factor"].dtype, np.dtype("float64"))


class ValidateAdjustFactorTest(unittest.TestCase):
    """``validate_daily_bars`` 对系数列的校验。"""

    def test_non_positive_factor_is_rejected(self) -> None:
        """非正系数会让还原原始价除出无穷，必须报错而不是替换成 1.0。"""
        bars = _daily_bars()
        bars.loc[3, "adjust_factor"] = 0.0
        with self.assertRaises(ValueError) as context:
            validate_daily_bars(bars)
        self.assertIn("非法复权系数", str(context.exception))

    def test_missing_factor_value_is_rejected(self) -> None:
        """列内缺失与整列缺席是两回事，前者必须报错。"""
        bars = _daily_bars()
        bars.loc[3, "adjust_factor"] = np.nan
        with self.assertRaises(ValueError):
            validate_daily_bars(bars)

    def test_nullable_dtype_missing_value_is_rejected(self) -> None:
        """可空扩展 dtype 下的 ``pd.NA`` 同样要报错。

        ``np.isfinite`` 作用在 ``Float64`` 上返回带 ``pd.NA`` 的 BooleanArray，
        而 ``any()`` 默认 skipna 会把缺失位当 False 跳过——不先归一成 numpy
        ``float64``，这一行就会被静默放行，一路变成因子层的 NaN。
        """
        bars = _daily_bars()
        bars["adjust_factor"] = bars["adjust_factor"].astype("Float64")
        bars.loc[3, "adjust_factor"] = pd.NA
        with self.assertRaises(ValueError) as context:
            validate_daily_bars(bars)
        self.assertIn("非法复权系数", str(context.exception))

    def test_infinite_factor_is_rejected(self) -> None:
        """无穷系数会让还原出的原始价变成 0，同样必须报错。"""
        bars = _daily_bars()
        bars.loc[3, "adjust_factor"] = np.inf
        with self.assertRaises(ValueError):
            validate_daily_bars(bars)

    def test_error_message_locates_first_bad_row(self) -> None:
        """报错要能一眼定位到证券和交易日，否则全库排查无从下手。"""
        bars = _daily_bars()
        bars.loc[3, "adjust_factor"] = -1.0
        with self.assertRaises(ValueError) as context:
            validate_daily_bars(bars)
        self.assertIn("000001.SZ", str(context.exception))

    def test_absent_column_stays_optional(self) -> None:
        """没有这一列的既有调用方不受影响。"""
        result = validate_daily_bars(_daily_bars(with_factor=False))
        self.assertNotIn("adjust_factor", result.columns)


class DailyPathTest(unittest.TestCase):
    """日频路径的端到端行为。"""

    def test_output_carries_adjust_factor(self) -> None:
        """因子表里必须能拿到系数，供还原原始价使用。"""
        features = build_daily_features_from_daily(
            _daily_bars(), feature_columns=["return_1d"], source_name="qmt_daily"
        )
        self.assertIn("adjust_factor", features.columns)
        # 跳档两侧都要断言：只查除权前那一半的话，「无条件填 1.0」的回退能蒙混过关。
        boundary = pd.Timestamp("2024-01-12")
        early = features.loc[features["trade_date"] < boundary]
        late = features.loc[features["trade_date"] >= boundary]
        self.assertTrue((early["adjust_factor"] == 1.0).all())
        self.assertTrue((late["adjust_factor"] == 2.0).all())

    def test_expression_can_reference_adjust_factor(self) -> None:
        """DSL 表达式引用该列不再被基础列守卫拒绝，且取值正确。"""
        bars = _daily_bars()
        features = build_daily_features_from_daily(
            bars,
            feature_columns=[],
            factor_expressions=["divide(column(close),column(adjust_factor))"],
            source_name="qmt_daily",
        )
        expression_column = [
            name for name in features.columns if name not in BASE_DAILY_COLUMNS
        ]
        self.assertEqual(len(expression_column), 1)
        recovered = features[expression_column[0]].to_numpy(dtype=float)
        expected = (features["close"] / features["adjust_factor"]).to_numpy(dtype=float)
        np.testing.assert_allclose(recovered, expected, rtol=1e-12)

    def test_unknown_column_is_still_rejected(self) -> None:
        """基础列守卫只多放行了这一列，其它列必须继续报错。"""
        with self.assertRaises(ValueError) as context:
            build_daily_features_from_daily(
                _daily_bars(),
                feature_columns=[],
                factor_expressions=["column(amount)"],
                source_name="qmt_daily",
            )
        self.assertIn("未知列", str(context.exception))


class IntradayPathTest(unittest.TestCase):
    """分钟路径补 1.0 后两条路径的列宇宙一致。"""

    def test_minute_path_fills_one(self) -> None:
        """分钟库没有复权概念，系数恒为 1.0。"""
        features = build_daily_features(_minute_bars(), feature_columns=["return_1d"])
        self.assertIn("adjust_factor", features.columns)
        self.assertTrue((features["adjust_factor"] == 1.0).all())

    def test_aggregate_daily_bars_fills_one(self) -> None:
        """搜索用的聚合入口同样补齐，避免同一表达式换入口就报错。"""
        daily = aggregate_daily_bars(_minute_bars())
        self.assertEqual(list(daily.columns), list(BASE_DAILY_COLUMNS))
        self.assertTrue((daily["adjust_factor"] == 1.0).all())

    def test_both_paths_expose_the_same_base_columns(self) -> None:
        """同一个读系数的因子换数据源不应 KeyError。"""
        minute_daily = aggregate_daily_bars(_minute_bars())
        daily_daily = _prepare_daily_bars(validate_daily_bars(_daily_bars()))
        self.assertEqual(set(minute_daily.columns), set(daily_daily.columns))


class InputFingerprintTest(unittest.TestCase):
    """系数参与缓存输入指纹。"""

    def test_factor_change_changes_fingerprint(self) -> None:
        """价格相同、系数不同时指纹必须不同，否则缓存会串味。"""
        base = _prepare_daily_bars(validate_daily_bars(_daily_bars()))
        shifted = base.copy()
        shifted["adjust_factor"] = shifted["adjust_factor"] * 3.0
        self.assertNotEqual(
            _daily_input_fingerprint(base, "qmt_daily/hfq"),
            _daily_input_fingerprint(shifted, "qmt_daily/hfq"),
        )

    def test_identical_input_keeps_fingerprint(self) -> None:
        """输入逐值相同时指纹必须稳定，否则缓存永远命中不了。"""
        first = _prepare_daily_bars(validate_daily_bars(_daily_bars()))
        second = _prepare_daily_bars(validate_daily_bars(_daily_bars()))
        self.assertEqual(
            _daily_input_fingerprint(first, "qmt_daily/hfq"),
            _daily_input_fingerprint(second, "qmt_daily/hfq"),
        )

    def test_expression_cache_round_trips(self) -> None:
        """读系数的表达式因子走一次缓存后取值不变。"""
        expression = "divide(column(close),column(adjust_factor))"
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            first = build_daily_features_from_daily(
                _daily_bars(),
                feature_columns=[],
                cache_dir=cache_dir,
                factor_expressions=[expression],
                cache_namespace="qmt_daily/hfq",
                source_name="qmt_daily",
            )
            second = build_daily_features_from_daily(
                _daily_bars(),
                feature_columns=[],
                cache_dir=cache_dir,
                factor_expressions=[expression],
                cache_namespace="qmt_daily/hfq",
                source_name="qmt_daily",
            )
        pd.testing.assert_frame_equal(first, second)


if __name__ == "__main__":
    unittest.main()
