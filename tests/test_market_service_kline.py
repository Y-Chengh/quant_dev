from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

from market_service.app import records
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
                    low DOUBLE, close DOUBLE, volume BIGINT, amount DOUBLE
                )
                """
            )
            con.executemany(
                "INSERT INTO bars_5m VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
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
        self.assertAlmostEqual(frame.loc[3, "pct_change"], (12.0 / 9.9 - 1.0) * 100)

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
        """没有更早行情时首根派生字段应缺失，并转换为JSON空值而非NaN。"""
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


if __name__ == "__main__":
    unittest.main()
