"""验证日线库的列契约、生命周期哨兵归一化与板块推导。"""

from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

import duckdb

from quant.market_data.daily.schema import (
    BAR_COLUMN_TYPES,
    BAR_COLUMNS,
    apply_schema,
    classify_board,
    is_risk_warned,
    normalize_lifecycle_date,
)


class LifecycleSentinelTest(unittest.TestCase):
    """上市退市日期的哨兵值必须一律归一化为空。"""

    def test_qmt_sentinels_become_none(self) -> None:
        """QMT 用来表达「没有这个日期」的各种取值都应返回 None。"""
        for value in (
            "",
            None,
            "0",
            "99999999",
            "19700101",
            "19700427",
            "19700428",
            "nan",
            "NaT",
        ):
            with self.subTest(value=value):
                self.assertIsNone(normalize_lifecycle_date(value))

    def test_real_dates_are_parsed(self) -> None:
        """真实的八位日期应当被正确解析。"""
        self.assertEqual(normalize_lifecycle_date("19910403"), date(1991, 4, 3))
        self.assertEqual(normalize_lifecycle_date(20240102), date(2024, 1, 2))

    def test_unparseable_text_returns_none(self) -> None:
        """无法解析的文本按无日期处理，不应抛异常。"""
        self.assertIsNone(normalize_lifecycle_date("abc"))
        self.assertIsNone(normalize_lifecycle_date("20241350"))


class BoardClassificationTest(unittest.TestCase):
    """板块与风险警示标记的推导规则。"""

    def test_board_by_code_prefix(self) -> None:
        """各交易所与板块的代码前缀应归入正确板块。"""
        expected = {
            "600000.SH": "main",
            "601398.SH": "main",
            "000001.SZ": "main",
            "002415.SZ": "main",
            "300750.SZ": "chinext",
            "301001.SZ": "chinext",
            "688981.SH": "star",
            "689009.SH": "star",
            "430047.BJ": "bse",
            "920002.BJ": "bse",
            "999999.XX": "unknown",
        }
        for code, board in expected.items():
            with self.subTest(code=code):
                self.assertEqual(classify_board(code), board)

    def test_risk_warning_detected_from_name(self) -> None:
        """名称含 ST 或退字时应识别为风险警示。"""
        self.assertTrue(is_risk_warned("*ST 中安"))
        self.assertTrue(is_risk_warned("ST 沪科"))
        self.assertTrue(is_risk_warned("退市海润"))
        self.assertFalse(is_risk_warned("平安银行"))
        self.assertFalse(is_risk_warned(""))
        self.assertFalse(is_risk_warned(None))


class SchemaDdlTest(unittest.TestCase):
    """建表语句必须幂等，并与列契约保持一致。"""

    def test_apply_schema_is_idempotent(self) -> None:
        """重复执行建表语句不应报错，也不应改变表数量。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            connection = duckdb.connect(str(root / "qmt_daily.duckdb"))
            try:
                apply_schema(connection, root, create_view=False)
                first = connection.execute(
                    "SELECT count(*) FROM duckdb_tables()"
                ).fetchone()[0]
                apply_schema(connection, root, create_view=False)
                second = connection.execute(
                    "SELECT count(*) FROM duckdb_tables()"
                ).fetchone()[0]
            finally:
                connection.close()
        self.assertEqual(first, second)
        self.assertGreaterEqual(first, 8)

    def test_bar_columns_and_types_align(self) -> None:
        """列顺序常量与类型映射必须一一对应。"""
        self.assertEqual(set(BAR_COLUMNS), set(BAR_COLUMN_TYPES))
        self.assertEqual(BAR_COLUMNS[0], "code")
        self.assertEqual(BAR_COLUMNS[1], "trade_date")


if __name__ == "__main__":
    unittest.main()
