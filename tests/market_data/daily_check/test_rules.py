"""逐条验证日线库审计规则的正例与反例。"""

from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from quant.market_data.daily.schema import apply_schema
from quant.market_data.daily_check import DailyCheckConfig, run_daily_market_check
from quant.market_data.daily_check.limits import resolve_limit_ratio


def _write_store(root: Path, bars, instruments=None, calendar=None, actions=None) -> Path:
    """在临时目录里手工搭一个最小日线库。

    参数：
        root: 日线库数据根目录。
        bars: 日线记录字典列表。
        instruments: 证券静态信息字典列表；``None`` 时按 ``bars`` 里出现的代码
            生成一份 2024-01-01 上市、未退市的默认信息。
        calendar: 交易日列表；``None`` 时取 ``bars`` 里出现过的交易日。
        actions: 除权记录字典列表；``None`` 表示没有除权。

    返回：
        目录库文件路径。
    """
    frame = pd.DataFrame(bars)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    codes = sorted(frame["code"].unique())
    if instruments is None:
        instruments = [
            {"code": code, "instrument_name": "测试", "open_date": date(2024, 1, 1),
             "expire_date": None, "board": "main", "is_st": False}
            for code in codes
        ]
    if calendar is None:
        calendar = sorted(frame["trade_date"].dt.date.unique())

    shard = root / "bars_1d" / "year=2024" / "month=01"
    shard.mkdir(parents=True, exist_ok=True)
    database = root / "qmt_daily.duckdb"
    connection = duckdb.connect(str(database))
    try:
        connection.register("bars_frame", frame)
        # 与生产的 COPY 一致，把 trade_date 落成 DATE 而不是 TIMESTAMP。
        connection.execute(
            "COPY (SELECT code, CAST(trade_date AS DATE) AS trade_date, open, high, low, "
            "close, pre_close, CAST(volume AS BIGINT) AS volume, amount, "
            "CAST(suspend_flag AS TINYINT) AS suspend_flag FROM bars_frame) "
            "TO '{0}' (FORMAT PARQUET)".format(
                str(shard / "bars.parquet").replace("\\", "/")
            )
        )
        apply_schema(connection, root, create_view=True)
        connection.register("inst_frame", pd.DataFrame(instruments))
        connection.execute(
            "INSERT INTO instruments SELECT code, instrument_name, CAST(open_date AS DATE), "
            "CAST(expire_date AS DATE), NULL, '0', board, is_st FROM inst_frame"
        )
        connection.register(
            "cal_frame", pd.DataFrame({"trade_date": pd.to_datetime(pd.Series(calendar))})
        )
        connection.execute(
            "INSERT INTO trading_calendar SELECT CAST(trade_date AS DATE), '000001.SH' "
            "FROM cal_frame"
        )
        if actions:
            connection.register("act_frame", pd.DataFrame(actions))
            connection.execute(
                "INSERT INTO corporate_actions SELECT code, CAST(ex_date AS DATE), "
                "cash_dividend_per_share, 0.0, 0.0, 0.0, 0.0, 0.0, adjustment_factor "
                "FROM act_frame"
            )
        connection.execute(
            "INSERT INTO monthly_inventory SELECT 2024, 1, count(*), count(DISTINCT code), "
            "count(DISTINCT trade_date), min(trade_date), max(trade_date) FROM bars_1d"
        )
        connection.execute(
            "INSERT INTO symbols SELECT code, min(trade_date), max(trade_date), count(*), "
            "sum(CASE WHEN suspend_flag <> 0 THEN 1 ELSE 0 END) FROM bars_1d GROUP BY code"
        )
    finally:
        connection.close()
    return database


def _bar(code, day, close, pre_close, *, volume=1000, amount=None, suspend=0):
    """构造一行日线记录。

    参数：
        code: 证券代码。
        day: ``YYYY-MM-DD`` 交易日文本。
        close: 收盘价，同时用作开高低价。
        pre_close: 前收盘价。
        volume: 成交量。
        amount: 成交额；``None`` 时按 ``收盘价 × 成交量 × 100`` 计算，
            与 A 股「手」的口径一致。
        suspend: 停牌标记。

    返回：
        与日线列对齐的字典。
    """
    return {
        "code": code, "trade_date": day, "open": close, "high": close,
        "low": close, "close": close, "pre_close": pre_close,
        "volume": volume,
        "amount": close * volume * 100 if amount is None else amount,
        "suspend_flag": suspend,
    }


def _codes(result, issue_code):
    """取出某个问题编号对应的全部记录。

    参数：
        result: ``run_daily_market_check`` 的返回值。
        issue_code: 目标问题编号。

    返回：
        该编号对应的问题子表。
    """
    return result.issues[result.issues["issue_code"] == issue_code]


class DailyCheckRuleTest(unittest.TestCase):
    """各条审计规则的正例与反例。"""

    def _run(self, bars, **kwargs):
        """在临时目录里建库并跑一次审计。

        参数：
            bars: 日线记录字典列表。
            **kwargs: 透传给 ``_write_store`` 的 ``instruments``、``calendar``、
                ``actions``，以及透传给 ``DailyCheckConfig`` 的其余键。

        返回：
            审计结果。
        """
        store_keys = {"instruments", "calendar", "actions"}
        store_kwargs = {k: v for k, v in kwargs.items() if k in store_keys}
        config_kwargs = {k: v for k, v in kwargs.items() if k not in store_keys}
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        database = _write_store(root, bars, **store_kwargs)
        return run_daily_market_check(
            DailyCheckConfig(
                database=database,
                report_dir=root / "report",
                **config_kwargs,
            )
        )

    def test_clean_store_passes(self) -> None:
        """完全干净的库不应报出任何 ERROR。"""
        bars = [
            _bar("000001.SZ", "2024-01-02", 10.0, 9.9),
            _bar("000001.SZ", "2024-01-03", 10.1, 10.0),
        ]
        result = self._run(bars)
        errors = result.issues[result.issues["level"] == "ERROR"]
        self.assertEqual(list(errors["issue_code"]), [], msg=errors.to_string())
        self.assertEqual(result.exit_code, 0)

    def test_missing_span_is_reported(self) -> None:
        """存续期内缺整段交易日应报连续缺失区间。"""
        bars = [
            _bar("000001.SZ", "2024-01-02", 10.0, 9.9),
            _bar("000001.SZ", "2024-01-05", 10.3, 10.2),
        ]
        calendar = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)]
        result = self._run(bars, calendar=calendar)
        spans = _codes(result, "DAILY_MISSING_SPAN")
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans.iloc[0]["start_date"], "2024-01-03")
        self.assertEqual(spans.iloc[0]["end_date"], "2024-01-04")
        self.assertEqual(result.exit_code, 1)

    def test_active_zero_volume_is_error(self) -> None:
        """非停牌日成交量为零必须报错。"""
        bars = [
            _bar("000001.SZ", "2024-01-02", 10.0, 9.9),
            _bar("000001.SZ", "2024-01-03", 10.1, 10.0, volume=0, amount=0.0),
        ]
        result = self._run(bars)
        self.assertEqual(len(_codes(result, "DAILY_ACTIVE_ZERO_VOLUME")), 1)

    def test_suspended_zero_volume_is_not_reported(self) -> None:
        """停牌日成交量为零是正常的，不应报错。"""
        bars = [
            _bar("000001.SZ", "2024-01-02", 10.0, 9.9),
            _bar("000001.SZ", "2024-01-03", 10.0, 10.0, volume=0, amount=0.0, suspend=1),
        ]
        result = self._run(bars)
        self.assertEqual(len(_codes(result, "DAILY_ACTIVE_ZERO_VOLUME")), 0)

    def test_suspended_with_turnover_is_error(self) -> None:
        """停牌日却有成交必须报错。"""
        bars = [
            _bar("000001.SZ", "2024-01-02", 10.0, 9.9),
            _bar("000001.SZ", "2024-01-03", 10.0, 10.0, suspend=1),
        ]
        result = self._run(bars)
        self.assertEqual(len(_codes(result, "DAILY_SUSPENDED_WITH_TURNOVER")), 1)

    def test_ohlc_violation_is_error(self) -> None:
        """最高价低于收盘价必须报错。"""
        bars = [_bar("000001.SZ", "2024-01-02", 10.0, 9.9)]
        bars[0]["high"] = 9.0
        result = self._run(bars)
        self.assertEqual(len(_codes(result, "DAILY_OHLC_VIOLATION")), 1)

    def test_pre_close_gap_without_action_is_error(self) -> None:
        """前收断点且当日无除权记录必须报错。"""
        bars = [
            _bar("000001.SZ", "2024-01-02", 10.0, 9.9),
            _bar("000001.SZ", "2024-01-03", 9.0, 9.0),
        ]
        result = self._run(bars)
        self.assertEqual(len(_codes(result, "DAILY_PRE_CLOSE_MISMATCH")), 1)

    def test_pre_close_gap_with_matching_action_is_clean(self) -> None:
        """有匹配的除权记录时前收断点不应报错。"""
        bars = [
            _bar("000001.SZ", "2024-01-02", 10.0, 9.9),
            _bar("000001.SZ", "2024-01-03", 9.6, 9.5),
        ]
        actions = [{
            "code": "000001.SZ", "ex_date": date(2024, 1, 3),
            "cash_dividend_per_share": 0.5,
            "adjustment_factor": 10.0 / 9.5,
        }]
        result = self._run(bars, actions=actions)
        self.assertEqual(len(_codes(result, "DAILY_PRE_CLOSE_MISMATCH")), 0)
        self.assertEqual(len(_codes(result, "DAILY_ADJUST_FACTOR_MISMATCH")), 0)

    def test_adjust_factor_inconsistent_with_bars_is_error(self) -> None:
        """除权因子与行情对不上时必须报错。"""
        bars = [
            _bar("000001.SZ", "2024-01-02", 10.0, 9.9),
            _bar("000001.SZ", "2024-01-03", 9.6, 9.5),
        ]
        actions = [{
            "code": "000001.SZ", "ex_date": date(2024, 1, 3),
            "cash_dividend_per_share": 0.5, "adjustment_factor": 2.0,
        }]
        result = self._run(bars, actions=actions)
        self.assertEqual(len(_codes(result, "DAILY_ADJUST_FACTOR_MISMATCH")), 1)

    def test_rounding_within_one_tick_is_tolerated(self) -> None:
        """前收只差一分钱时属于四舍五入，不应报错。"""
        bars = [
            _bar("000001.SZ", "2024-01-02", 30.84, 30.0),
            _bar("000001.SZ", "2024-01-03", 30.70, 30.40),
        ]
        actions = [{
            "code": "000001.SZ", "ex_date": date(2024, 1, 3),
            "cash_dividend_per_share": 0.446, "adjustment_factor": 1.014807,
        }]
        result = self._run(bars, actions=actions)
        self.assertEqual(len(_codes(result, "DAILY_ADJUST_FACTOR_MISMATCH")), 0)

    def test_data_outside_lifecycle_is_error(self) -> None:
        """行情早于上市日必须报错。"""
        bars = [_bar("000001.SZ", "2024-01-02", 10.0, 9.9)]
        instruments = [{
            "code": "000001.SZ", "instrument_name": "测试", "open_date": date(2024, 1, 10),
            "expire_date": None, "board": "main", "is_st": False,
        }]
        result = self._run(bars, instruments=instruments)
        self.assertEqual(len(_codes(result, "DAILY_DATA_BEFORE_LISTING")), 1)

    def test_no_delisted_symbols_is_warned(self) -> None:
        """全部证券都没有退市日时应提示幸存者偏差。"""
        result = self._run([_bar("000001.SZ", "2024-01-02", 10.0, 9.9)])
        self.assertEqual(len(_codes(result, "DAILY_NO_DELISTED_SYMBOLS")), 1)

    def test_date_outside_calendar_is_error(self) -> None:
        """库里出现交易日历之外的日期必须报错。"""
        bars = [
            _bar("000001.SZ", "2024-01-02", 10.0, 9.9),
            _bar("000001.SZ", "2024-01-03", 10.1, 10.0),
        ]
        result = self._run(bars, calendar=[date(2024, 1, 2)])
        self.assertEqual(len(_codes(result, "DAILY_DATE_NOT_IN_CALENDAR")), 1)

    def test_report_files_are_written(self) -> None:
        """报告目录必须产出全套文件，且问题表列顺序与 self_check 一致。"""
        from quant.qmt_downloader.self_check import ISSUE_COLUMNS

        result = self._run([_bar("000001.SZ", "2024-01-02", 10.0, 9.9)])
        for name in (
            "summary.json", "summary.md", "issues.csv", "missing_spans.csv",
            "coverage_by_date.csv", "coverage_by_symbol.csv",
            "adjust_factor_audit.csv", "price_limit_violations.csv",
        ):
            with self.subTest(name=name):
                self.assertTrue((result.report_dir / name).is_file())
        self.assertEqual(list(result.issues.columns), ISSUE_COLUMNS)


class PriceLimitTest(unittest.TestCase):
    """涨跌停口径的推导。"""

    def test_new_listing_window_counts_trading_sessions(self) -> None:
        """新股豁免必须按交易日计数：第 5 个交易日仍在豁免期内。"""
        self.assertIsNone(
            resolve_limit_ratio("chinext", False, date(2024, 10, 9), 4)
        )
        self.assertEqual(
            resolve_limit_ratio("chinext", False, date(2024, 10, 10), 5), 0.20
        )

    def test_st_is_not_tightened_by_default(self) -> None:
        """缺省不按当前 ST 状态收紧，避免历史状态不可知造成的大量误报。"""
        self.assertEqual(resolve_limit_ratio("main", True, date(2024, 5, 6), 100), 0.10)
        self.assertEqual(
            resolve_limit_ratio("main", True, date(2024, 5, 6), 100, apply_st_limit=True),
            0.05,
        )

    def test_board_specific_ratios(self) -> None:
        """各板块与各生效日期应套用正确的幅度。"""
        self.assertEqual(resolve_limit_ratio("main", False, date(2024, 5, 6), 100), 0.10)
        self.assertEqual(resolve_limit_ratio("star", False, date(2024, 5, 6), 100), 0.20)
        self.assertEqual(resolve_limit_ratio("bse", False, date(2024, 5, 6), 100), 0.30)
        self.assertIsNone(resolve_limit_ratio("star", False, date(2018, 5, 6), 100))
        self.assertEqual(
            resolve_limit_ratio("chinext", False, date(2019, 5, 6), 100), 0.10
        )
        self.assertIsNone(resolve_limit_ratio("unknown", False, date(2024, 5, 6), 100))

    def test_listing_day_is_exempt(self) -> None:
        """上市首日不设涨跌停。"""
        self.assertIsNone(resolve_limit_ratio("main", False, date(2024, 5, 6), 0))


if __name__ == "__main__":
    unittest.main()
