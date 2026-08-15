"""源数据勘误表装载、应用与转宽表逻辑测试。"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from quant.qmt_downloader import errata


class ErrataLoadingTests(unittest.TestCase):
    """load_errata_overrides 的路径缺失、字段校验与聚合行为。"""

    def test_missing_path_returns_empty_dict(self):
        self.assertEqual(errata.load_errata_overrides(None), {})
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(
                errata.load_errata_overrides(Path(directory) / "not_exists.csv"), {}
            )

    def test_loads_and_groups_by_symbol_and_date(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "errata.csv"
            path.write_text(
                "symbol,trade_date,field,value,issue_type,description,found_date,source_report\n"
                "002062.sz,20210524,suspend_flag,1,missing_suspension,test,20260815,test\n",
                encoding="utf-8",
            )
            overrides = errata.load_errata_overrides(path)
            self.assertEqual(overrides, {("002062.SZ", "20210524"): {"suspend_flag": "1"}})

    def test_multiple_fields_for_same_key_are_merged(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "errata.csv"
            path.write_text(
                "symbol,trade_date,field,value,issue_type,description,found_date,source_report\n"
                "000001.SZ,20240102,suspend_flag,1,x,x,x,x\n"
                "000001.SZ,20240102,volume,0,x,x,x,x\n",
                encoding="utf-8",
            )
            overrides = errata.load_errata_overrides(path)
            self.assertEqual(
                overrides,
                {("000001.SZ", "20240102"): {"suspend_flag": "1", "volume": "0"}},
            )

    def test_rejects_field_outside_overridable_set(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "errata.csv"
            path.write_text(
                "symbol,trade_date,field,value,issue_type,description,found_date,source_report\n"
                "000001.SZ,20240102,code,999999,x,x,x,x\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                errata.load_errata_overrides(path)

    def test_rejects_missing_required_column(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "errata.csv"
            path.write_text("symbol,trade_date,field\n000001.SZ,20240102,volume\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                errata.load_errata_overrides(path)


class ApplyErrataOverridesTests(unittest.TestCase):
    """apply_errata_overrides 对内存 DataFrame 的按行覆盖行为。"""

    def _frame(self):
        return pd.DataFrame(
            [
                {"code": "600000.SH", "trade_date": "20240103", "volume": 0.0, "amount": 0.0,
                 "suspend_flag": 0.0, "close": 20.0},
                {"code": "000001.SZ", "trade_date": "20240103", "volume": 500.0, "amount": 5000.0,
                 "suspend_flag": 0.0, "close": 10.0},
            ]
        )

    def test_empty_overrides_is_a_no_op(self):
        frame = self._frame()
        result, applied = errata.apply_errata_overrides(frame, {})
        self.assertEqual(applied, 0)
        pd.testing.assert_frame_equal(result, frame)

    def test_overrides_only_matching_row_and_field(self):
        frame = self._frame()
        overrides = {("600000.SH", "20240103"): {"suspend_flag": "1"}}
        result, applied = errata.apply_errata_overrides(frame, overrides)
        self.assertEqual(applied, 1)
        self.assertEqual(result.loc[0, "suspend_flag"], 1.0)
        # 其它字段与另一只证券的行必须保持原值不变。
        self.assertEqual(result.loc[0, "volume"], 0.0)
        self.assertEqual(result.loc[0, "close"], 20.0)
        self.assertEqual(result.loc[1, "suspend_flag"], 0.0)

    def test_unknown_key_does_not_affect_frame(self):
        frame = self._frame()
        overrides = {("999999.SZ", "20240103"): {"suspend_flag": "1"}}
        result, applied = errata.apply_errata_overrides(frame, overrides)
        self.assertEqual(applied, 0)
        pd.testing.assert_frame_equal(result, frame)


class ErrataPivotFrameTests(unittest.TestCase):
    """errata_pivot_frame 生成的宽表供市场数据入库 SQL 使用。"""

    def test_empty_overrides_returns_typed_empty_frame(self):
        frame = errata.errata_pivot_frame({})
        self.assertEqual(len(frame), 0)
        self.assertIn("code", frame.columns)
        for field in errata.ERRATA_OVERRIDABLE_FIELDS:
            self.assertIn(field, frame.columns)
            self.assertEqual(frame[field].dtype, "float64")

    def test_pivots_overrides_into_wide_row_with_nulls_for_unset_fields(self):
        overrides = {("600000.SH", "20240103"): {"suspend_flag": "1"}}
        frame = errata.errata_pivot_frame(overrides)
        self.assertEqual(len(frame), 1)
        row = frame.iloc[0]
        self.assertEqual(row["code"], "600000.SH")
        self.assertEqual(row["trade_date"], "20240103")
        self.assertEqual(row["suspend_flag"], 1.0)
        self.assertTrue(pd.isna(row["volume"]))
        self.assertTrue(pd.isna(row["close"]))


if __name__ == "__main__":
    unittest.main()
