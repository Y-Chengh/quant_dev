from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import duckdb
import pandas as pd

import market_service.app as market_app
from market_service import KlinePeriod, KlineQuery, MarketDataClient, RawBarQuery
from market_service.app import records
from market_service.codes import normalize_security_code
from market_service.database import MarketDatabase


class MarketServiceKlineTest(unittest.TestCase):
    """验证网页K线所需的标准涨跌幅字段及REST缺失值转换。"""

    def _create_database(self, path: Path) -> MarketDatabase:
        """创建包含跳空和连续两个聚合周期的最小行情数据库。

        参数：
            path: 临时DuckDB文件路径，仅在当前测试目录中创建。

        返回：
            指向已关闭写连接、可按只读模式查询的行情仓储。
        """
        con = duckdb.connect(str(path))
        try:
            con.execute(
                """
                CREATE TABLE bars_5m(
                    code VARCHAR, trade_time TIMESTAMP, open DOUBLE, high DOUBLE,
                    low DOUBLE, close DOUBLE, volume BIGINT, amount DOUBLE,
                    pre_close DOUBLE, change DOUBLE, pct_change DOUBLE,
                    trade_date DATE GENERATED ALWAYS AS (CAST(trade_time AS DATE)) VIRTUAL
                )
                """
            )
            con.executemany(
                """
                INSERT INTO bars_5m(
                    code, trade_time, open, high, low, close, volume, amount
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    ("000001.SZ", "2026-08-09 09:55:00", 10.0, 10.0, 10.0, 10.0, 100, 1_000.0),
                    ("000001.SZ", "2026-08-09 10:00:00", 11.0, 11.2, 10.8, 11.0, 110, 1_210.0),
                    ("000001.SZ", "2026-08-09 10:05:00", 11.0, 12.2, 10.9, 12.0, 120, 1_440.0),
                    ("000001.SZ", "2026-08-09 10:10:00", 12.0, 12.1, 10.8, 11.0, 130, 1_430.0),
                    ("000001.SZ", "2026-08-09 10:15:00", 9.0, 10.0, 8.9, 9.9, 140, 1_386.0),
                    ("000001.SZ", "2026-08-09 10:20:00", 9.9, 10.0, 9.8, None, 150, 1_485.0),
                    ("000001.SZ", "2026-08-09 10:25:00", 9.9, 10.0, 9.8, float("inf"), 160, 1_584.0),
                    ("000001.SZ", "2026-08-09 10:30:00", 9.9, 10.0, 0.0, 0.0, 170, 1_683.0),
                    ("000001.SZ", "2026-08-09 10:35:00", 12.0, 12.1, 11.9, 12.0, 180, 2_160.0),
                    ("000002.SZ", "2026-08-09 09:55:00", 10.0, 10.0, 10.0, 10.0, 100, 1_000.0),
                    ("000002.SZ", "2026-08-09 10:00:00", None, 11.2, 10.8, 11.0, 110, 1_210.0),
                    ("000002.SZ", "2026-08-09 10:05:00", 0.0, 9.2, 8.8, 9.0, 120, 1_080.0),
                    ("000002.SZ", "2026-08-09 10:10:00", float("inf"), 12.2, 11.8, 12.0, 130, 1_560.0),
                    ("000002.SZ", "2026-08-09 10:15:00", float("-inf"), 8.2, 7.8, 8.0, 140, 1_120.0),
                    ("000002.SZ", "2026-08-09 10:20:00", 10.0, 11.2, 9.9, 11.0, 150, 1_650.0),
                    ("600000.SH", "2026-08-09 10:00:00", 20.0, 20.2, 19.8, 20.1, 200, 4_020.0),
                ],
            )
        finally:
            con.close()
        return MarketDatabase(path)

    def test_five_minute_change_uses_previous_close_and_captures_gap(self) -> None:
        """5分钟K线应使用上一根收盘价，首根跳空不能误报为零涨幅。"""
        with tempfile.TemporaryDirectory() as directory:
            database = self._create_database(Path(directory) / "market.duckdb")
            frame = database.kline(
                "000001.sz",
                datetime(2026, 8, 9, 10, 0),
                datetime(2026, 8, 9, 10, 5),
                "5m",
            )

        self.assertEqual(frame["pre_close"].tolist(), [10.0, 11.0])
        self.assertEqual(frame["change"].tolist(), [1.0, 1.0])
        self.assertAlmostEqual(frame.loc[0, "pct_change"], 10.0)
        self.assertAlmostEqual(frame.loc[1, "pct_change"], 100.0 / 11.0)
        self.assertAlmostEqual(frame.loc[0, "intraday_pct_change"], 0.0)
        self.assertAlmostEqual(frame.loc[1, "intraday_pct_change"], 100.0 / 11.0)

    def test_aggregated_change_uses_previous_aggregated_close(self) -> None:
        """聚合K线应逐周期衔接收盘价，且证券代码规范化不影响结果。"""
        with tempfile.TemporaryDirectory() as directory:
            database = self._create_database(Path(directory) / "market.duckdb")
            frame = database.kline(
                " 000001.sz ",
                datetime(2026, 8, 9, 10, 0),
                datetime(2026, 8, 9, 10, 15),
                "15m",
            )

        self.assertEqual(frame["close"].tolist(), [11.0, 9.9])
        self.assertEqual(frame["pre_close"].tolist(), [10.0, 11.0])
        self.assertAlmostEqual(frame.loc[0, "pct_change"], 10.0)
        self.assertAlmostEqual(frame.loc[1, "pct_change"], -10.0)
        self.assertAlmostEqual(frame.loc[0, "intraday_pct_change"], 0.0)
        self.assertAlmostEqual(frame.loc[1, "intraday_pct_change"], 10.0)

    def test_invalid_close_does_not_break_previous_valid_close_chain(self) -> None:
        """区间内缺失、无穷或零收盘无效，后续K线仍应衔接最近有效收盘。"""
        with tempfile.TemporaryDirectory() as directory:
            database = self._create_database(Path(directory) / "market.duckdb")
            frame = database.kline(
                "000001.SZ",
                datetime(2026, 8, 9, 10, 20),
                datetime(2026, 8, 9, 10, 35),
                "5m",
            )

        self.assertEqual(frame["pre_close"].tolist(), [9.9, 9.9, 9.9, 9.9])
        self.assertTrue(frame.loc[:2, "change"].isna().all())
        self.assertTrue(frame.loc[:2, "pct_change"].isna().all())
        self.assertTrue(frame.loc[:2, "intraday_pct_change"].isna().all())
        self.assertAlmostEqual(frame.loc[3, "pct_change"], (12.0 / 9.9 - 1.0) * 100)
        self.assertAlmostEqual(frame.loc[3, "intraday_pct_change"], 0.0)

    def test_aggregated_change_is_independent_of_query_start(self) -> None:
        """聚合K线首根应查找上一根同周期有效收盘，不得随查询起点改变口径。"""
        with tempfile.TemporaryDirectory() as directory:
            database = self._create_database(Path(directory) / "market.duckdb")
            full = database.kline(
                "000001.SZ",
                datetime(2026, 8, 9, 10, 0),
                datetime(2026, 8, 9, 10, 35),
                "15m",
            )
            direct = database.kline(
                "000001.SZ",
                datetime(2026, 8, 9, 10, 35),
                datetime(2026, 8, 9, 10, 35),
                "15m",
            )

        for column in ["time", "open", "high", "low", "close", "volume", "amount"]:
            self.assertEqual(full.loc[2, column], direct.loc[0, column])
        self.assertEqual(full.loc[2, "pre_close"], 11.0)
        self.assertEqual(direct.loc[0, "pre_close"], 11.0)
        self.assertAlmostEqual(full.loc[2, "pct_change"], direct.loc[0, "pct_change"])

    def test_missing_previous_close_serializes_as_json_null(self) -> None:
        """没有更早行情时前收相关字段应为JSON空值，日内涨跌幅仍可计算。"""
        with tempfile.TemporaryDirectory() as directory:
            database = self._create_database(Path(directory) / "market.duckdb")
            frame = database.kline(
                "000001.SZ",
                datetime(2026, 8, 9, 9, 55),
                datetime(2026, 8, 9, 9, 55),
                "5m",
            )

        item = records(frame)[0]
        self.assertIsNone(item["pre_close"])
        self.assertIsNone(item["change"])
        self.assertIsNone(item["pct_change"])
        self.assertEqual(item["intraday_pct_change"], 0.0)

    def test_invalid_open_produces_missing_intraday_change(self) -> None:
        """开盘价缺失、为零或非有限时日内涨跌幅应缺失，有限开盘价精确计算。"""
        with tempfile.TemporaryDirectory() as directory:
            database = self._create_database(Path(directory) / "market.duckdb")
            frame = database.kline(
                "000002.SZ",
                datetime(2026, 8, 9, 10, 0),
                datetime(2026, 8, 9, 10, 20),
                "5m",
            )

        self.assertTrue(frame.loc[:3, "intraday_pct_change"].isna().all())
        self.assertAlmostEqual(frame.loc[4, "intraday_pct_change"], 10.0)
        items = records(frame)
        self.assertEqual([item["intraday_pct_change"] for item in items[:4]], [None] * 4)
        self.assertAlmostEqual(items[4]["intraday_pct_change"], 10.0)

    def test_records_converts_all_non_finite_numbers_to_json_null(self) -> None:
        """REST记录必须把NaN及正负无穷统一转为JSON空值，有限值保持不变。"""
        items = records(pd.DataFrame({
            "nan": [float("nan")],
            "positive_inf": [float("inf")],
            "negative_inf": [float("-inf")],
            "finite": [12.5],
        }))

        self.assertEqual(items, [{
            "nan": None,
            "positive_inf": None,
            "negative_inf": None,
            "finite": 12.5,
        }])

    def test_bare_stock_codes_are_supported_by_all_database_queries(self) -> None:
        """单只、批量、快照和原始行情查询均应自动补全六位沪深股票代码。"""
        with tempfile.TemporaryDirectory() as directory:
            database = self._create_database(Path(directory) / "market.duckdb")
            start = datetime(2026, 8, 9, 10, 0)
            end = datetime(2026, 8, 9, 10, 0)

            shanghai = database.kline(" 600000 ", start, end, "5m")
            batch = database.klines_5m(["000001", "600000"], start, end)
            snapshot = database.snapshot(start, ["600000"])
            raw, total = database.raw("000001", start.date())

        self.assertEqual(shanghai["close"].tolist(), [20.1])
        self.assertEqual(batch["code"].tolist(), ["000001.SZ", "600000.SH"])
        self.assertEqual(snapshot["code"].tolist(), ["600000.SH"])
        self.assertEqual(total, 9)
        self.assertTrue(raw["code"].eq("000001.SZ").all())

    def test_security_code_normalization_preserves_explicit_and_unknown_codes(self) -> None:
        """代码规范化应覆盖沪深裸代码，并保持显式后缀及未知市场代码兼容。"""
        expected_codes = {
            "000001": "000001.SZ",
            "100001": "100001.SZ",
            "200001": "200001.SZ",
            "300001": "300001.SZ",
            "500001": "500001.SH",
            "600000": "600000.SH",
            "900001": "900001.SH",
        }
        for bare_code, expected in expected_codes.items():
            with self.subTest(code=bare_code):
                self.assertEqual(normalize_security_code(bare_code), expected)
        self.assertEqual(normalize_security_code(" 000001.sz "), "000001.SZ")
        self.assertEqual(normalize_security_code("430001"), "430001")
        self.assertEqual(normalize_security_code("AAPL"), "AAPL")

    def test_public_client_accepts_bare_codes_for_each_query_type(self) -> None:
        """Python公共客户端的四类行情查询均应接受不带后缀的沪深代码。"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "market.duckdb"
            self._create_database(path)
            client = MarketDataClient(path)
            start = datetime(2026, 8, 9, 10, 0)

            kline = client.get_kline(KlineQuery(
                "600000", start, start, KlinePeriod.MIN_5
            ))
            batch = client.get_klines_5m(["000001", "600000"], start, start)
            snapshot = client.get_snapshot(start, ["600000"])
            raw = client.get_raw_bars(RawBarQuery("000001", start.date()))

        self.assertEqual(kline["close"].tolist(), [20.1])
        self.assertEqual(batch["code"].tolist(), ["000001.SZ", "600000.SH"])
        self.assertEqual(snapshot["code"].tolist(), ["600000.SH"])
        self.assertEqual(raw.total, 9)
        self.assertTrue(raw.data["code"].eq("000001.SZ").all())

    def test_rest_responses_use_normalized_code(self) -> None:
        """REST K线响应及CSV文件名应展示补全交易所后缀后的规范代码。"""
        at = datetime(2026, 8, 9, 10, 0)
        with patch.object(market_app.client, "get_kline", return_value=pd.DataFrame()):
            result = market_app.kline("600000", at, at)
        with patch.object(
            market_app,
            "raw_query",
            return_value=(pd.DataFrame({"code": ["600000.SH"]}), 1),
        ):
            response = market_app.raw_csv("600000", at.date())

        self.assertEqual(result["code"], "600000.SH")
        self.assertEqual(
            response.headers["content-disposition"],
            'attachment; filename="600000_SH_2026-08-09.csv"',
        )


if __name__ == "__main__":
    unittest.main()
