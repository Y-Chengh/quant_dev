# -*- coding: utf-8 -*-
"""QMT 全样本日线数据内部质量自检测试。"""

from __future__ import annotations

import tempfile
import unittest
import hashlib
import json
from pathlib import Path

import pandas as pd

from quant.qmt_downloader.self_check import SelfCheckConfig, run_full_sample_self_check
from quant.qmt_downloader.storage import DailyPartitionStore


SYMBOLS = ["000001.SZ", "600000.SH"]
KLINE_COLUMNS = [
    "code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "volume",
    "amount",
    "suspend_flag",
]
INSTRUMENT_COLUMNS = ["code", "open_date", "expire_date"]


def _write_calendar(root: Path, dates: list[str]) -> Path:
    """写出测试使用的权威 QMT 交易日历。

    参数：
        root: 临时 QMT 数据根目录。
        dates: 需要参与审计的升序八位交易日期。

    返回：
        已写出的单列交易日历 CSV 路径。
    """

    path = root / "calendar.csv"
    pd.DataFrame({"trade_date": dates}).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _write_instruments(
    store: DailyPartitionStore,
    rows: list[dict[str, object]],
    symbols: list[str],
) -> None:
    """通过正式分区存储器写出测试证券生命周期快照。

    参数：
        store: 指向临时 QMT 根目录的日分区存储器。
        rows: 每只证券的代码、上市日期和可选退市日期记录。
        symbols: 完成标记 ``partition_scope`` 使用的完整证券池。

    返回：
        无返回值。
    """

    store.write_partition(
        "instrument_info",
        "snapshot",
        "latest",
        pd.DataFrame(rows),
        INSTRUMENT_COLUMNS,
        ["code"],
        ["code"],
        {"partition_scope": {"symbols": symbols}},
    )


def _write_kline(
    store: DailyPartitionStore,
    date_value: str,
    rows: list[dict[str, object]],
    symbols: list[str],
) -> None:
    """通过正式分区存储器写出一个测试日线分区。

    参数：
        store: 指向临时 QMT 根目录的日分区存储器。
        date_value: 当前分区的八位交易日期。
        rows: 当前日期各证券的标准 OHLCV 与停牌记录。
        symbols: 完成标记 ``partition_scope`` 使用的完整证券池。

    返回：
        无返回值。
    """

    store.write_partition(
        "kline_1d",
        "date",
        date_value,
        pd.DataFrame(rows),
        KLINE_COLUMNS,
        ["code", "trade_date"],
        ["code"],
        {"partition_scope": {"symbols": symbols}},
    )


def _bar(
    code: str,
    date_value: str,
    close: float,
    pre_close: float,
    volume: float = 1000.0,
    amount: float = 10000.0,
    suspend_flag: int = 0,
) -> dict[str, object]:
    """构造价格关系合法的测试日线记录。

    参数：
        code: 含市场后缀的证券代码。
        date_value: 当前记录的八位交易日期。
        close: 当前收盘价；同时作为默认开盘价。
        pre_close: QMT 返回的昨收价。
        volume: 当前成交量，缺省为一千。
        amount: 当前成交额，缺省为一万。
        suspend_flag: QMT 停牌标志，0 为未停牌、1 为停牌。

    返回：
        包含标准日线字段的字典。
    """

    return {
        "code": code,
        "trade_date": date_value,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "pre_close": pre_close,
        "volume": volume,
        "amount": amount,
        "suspend_flag": suspend_flag,
    }


def _write_staging_kline(root: Path, date_value: str, rows: list[dict[str, object]]) -> None:
    """写入一个与 QMT runner 相同的 staging 日批次及行数元数据。

    参数：
        root: staging 作业目录。
        date_value: 八位交易日期。
        rows: 需要写入该日期的标准日线行。
    返回：
        无返回值。
    """

    directory = root / ("kline_daily_" + date_value)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "batch_00000.csv"
    pd.DataFrame(rows, columns=KLINE_COLUMNS).to_csv(
        path, index=False, encoding="utf-8-sig", lineterminator="\n"
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (directory / "batch_00000.meta.json").write_text(
        json.dumps({"rows": len(rows), "sha256": digest}), encoding="utf-8"
    )


class QmtDataSelfCheckTests(unittest.TestCase):
    """验证全样本覆盖率、生命周期、成交量和分区完整性检查。"""

    def test_complete_sample_with_filled_suspension_passes(self) -> None:
        """完整证券日矩阵及零成交停牌补齐行不应产生硬错误。"""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = DailyPartitionStore(root)
            calendar = _write_calendar(root, ["20240102", "20240103", "20240104"])
            _write_instruments(
                store,
                [
                    {"code": "000001.SZ", "open_date": "19910403", "expire_date": ""},
                    {"code": "600000.SH", "open_date": "19991110", "expire_date": ""},
                ],
                SYMBOLS,
            )
            _write_kline(
                store,
                "20240102",
                [_bar("000001.SZ", "20240102", 10.0, 9.8), _bar("600000.SH", "20240102", 20.0, 19.8)],
                SYMBOLS,
            )
            _write_kline(
                store,
                "20240103",
                [
                    _bar("000001.SZ", "20240103", 10.1, 10.0),
                    _bar("600000.SH", "20240103", 20.0, 20.0, volume=0, amount=0, suspend_flag=1),
                ],
                SYMBOLS,
            )
            _write_kline(
                store,
                "20240104",
                [_bar("000001.SZ", "20240104", 10.2, 10.1), _bar("600000.SH", "20240104", 20.2, 20.0)],
                SYMBOLS,
            )

            result = run_full_sample_self_check(
                SelfCheckConfig(
                    output_root=root,
                    calendar_csv=calendar,
                    report_dir=root / "audit",
                )
            )

            self.assertEqual(result.exit_code, 0)
            self.assertEqual(result.summary["expected_rows"], 6)
            self.assertEqual(result.summary["missing_rows"], 0)
            self.assertTrue((result.report_dir / "issues.csv").is_file())
            self.assertTrue((result.report_dir / "coverage_by_symbol.csv").is_file())

    def test_missing_lifecycle_and_volume_errors_include_evidence(self) -> None:
        """缺失区间、上市退市越界和停牌成交量矛盾应含完整定位与建议。"""

        symbols = ["000001.SZ", "600000.SH", "300001.SZ"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = DailyPartitionStore(root)
            calendar = _write_calendar(root, ["20240102", "20240103", "20240104"])
            _write_instruments(
                store,
                [
                    {"code": "000001.SZ", "open_date": "20240102", "expire_date": ""},
                    {"code": "600000.SH", "open_date": "20240103", "expire_date": "20240103"},
                    {"code": "300001.SZ", "open_date": "20240102", "expire_date": ""},
                ],
                symbols,
            )
            _write_kline(
                store,
                "20240102",
                [
                    _bar("000001.SZ", "20240102", 10.0, 9.8),
                    _bar("600000.SH", "20240102", 20.0, 19.8),
                ],
                symbols,
            )
            _write_kline(
                store,
                "20240103",
                [_bar("000001.SZ", "20240103", 10.0, 10.0, volume=0, amount=0)],
                symbols,
            )
            _write_kline(
                store,
                "20240104",
                [
                    _bar("000001.SZ", "20240104", 10.0, 10.0, volume=100, amount=1000, suspend_flag=1),
                    _bar("600000.SH", "20240104", 20.0, 20.0),
                ],
                symbols,
            )

            result = run_full_sample_self_check(
                SelfCheckConfig(
                    output_root=root,
                    calendar_csv=calendar,
                    report_dir=root / "audit",
                )
            )

            codes = set(result.issues["issue_code"])
            self.assertIn("DATA_BEFORE_LISTING", codes)
            self.assertIn("DATA_AFTER_DELISTING", codes)
            self.assertIn("ACTIVE_ZERO_VOLUME", codes)
            self.assertIn("SUSPENDED_WITH_TURNOVER", codes)
            self.assertIn("SYMBOL_ALL_DATA_MISSING", codes)
            missing = result.missing_spans.set_index("code")
            self.assertEqual(missing.loc["300001.SZ", "start_date"], "20240102")
            self.assertEqual(missing.loc["300001.SZ", "end_date"], "20240104")
            symbol_coverage = result.coverage_by_symbol.set_index("code")
            self.assertEqual(symbol_coverage.loc["600000.SH", "expected_rows"], 1)
            self.assertEqual(symbol_coverage.loc["600000.SH", "actual_rows"], 0)
            zero_volume = result.issues.loc[
                result.issues["issue_code"] == "ACTIVE_ZERO_VOLUME"
            ].iloc[0]
            self.assertIn("suspend_flag=0", zero_volume["actual"])
            self.assertIn("重新下载", zero_volume["suggested_action"])
            self.assertTrue(zero_volume["source_file"].endswith("data.csv"))
            self.assertEqual(int(zero_volume["source_row"]), 2)

    def test_detects_missing_partition_and_hash_mismatch(self) -> None:
        """权威日历中的整日缺失及被修改的完成分区必须阻止通过。"""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = DailyPartitionStore(root)
            calendar = _write_calendar(root, ["20240102", "20240103"])
            _write_instruments(
                store,
                [{"code": "000001.SZ", "open_date": "20240102", "expire_date": ""}],
                ["000001.SZ"],
            )
            _write_kline(
                store,
                "20240102",
                [_bar("000001.SZ", "20240102", 10.0, 9.8)],
                ["000001.SZ"],
            )
            data_path = root / "kline_1d" / "date=20240102" / "data.csv"
            with data_path.open("a", encoding="utf-8") as handle:
                handle.write("\n")

            result = run_full_sample_self_check(
                SelfCheckConfig(
                    output_root=root,
                    calendar_csv=calendar,
                    report_dir=root / "audit",
                )
            )

            codes = set(result.issues["issue_code"])
            self.assertIn("PARTITION_HASH_MISMATCH", codes)
            self.assertIn("PARTITION_ROW_COUNT_MISMATCH", codes)
            self.assertIn("MISSING_DATE_PARTITION", codes)
            missing_partition = result.issues.loc[
                result.issues["issue_code"] == "MISSING_DATE_PARTITION"
            ].iloc[0]
            self.assertEqual(missing_partition["date"], "20240103")
            self.assertIn("repair", missing_partition["suggested_action"])
            self.assertEqual(result.exit_code, 1)

    def test_requires_authoritative_calendar_and_audits_extra_partition(self) -> None:
        """没有权威日历不能宣称全样本通过，日历外分区也必须被报告。"""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = DailyPartitionStore(root)
            _write_instruments(
                store,
                [{"code": "000001.SZ", "open_date": "20240102", "expire_date": ""}],
                ["000001.SZ"],
            )
            _write_kline(
                store,
                "20240102",
                [_bar("000001.SZ", "20240102", 10.0, 9.8)],
                ["000001.SZ"],
            )
            _write_kline(
                store,
                "20240103",
                [_bar("000001.SZ", "20240103", 10.1, 10.0)],
                ["000001.SZ"],
            )

            inferred = run_full_sample_self_check(
                SelfCheckConfig(output_root=root, report_dir=root / "inferred")
            )
            self.assertEqual(inferred.exit_code, 1)
            self.assertIn(
                "CALENDAR_INFERRED_FROM_PARTITIONS",
                set(inferred.issues["issue_code"]),
            )

            calendar = _write_calendar(root, ["20240102"])
            result = run_full_sample_self_check(
                SelfCheckConfig(
                    output_root=root,
                    calendar_csv=calendar,
                    report_dir=root / "authoritative",
                )
            )
            self.assertIn("DATA_ON_NON_TRADING_DATE", set(result.issues["issue_code"]))
            self.assertEqual(result.exit_code, 1)

    def test_handles_qmt_timestamp_calendar_invalid_expire_and_bad_marker(self) -> None:
        """支持 QMT 秒/毫秒时间戳，并把非法退市日期和畸形标记写入报告。"""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = DailyPartitionStore(root)
            calendar = _write_calendar(root, ["1704153600000", "1704240000000"])
            _write_instruments(
                store,
                [{"code": "000001.SZ", "open_date": "20240102", "expire_date": "bad-date"}],
                ["000001.SZ"],
            )
            _write_kline(
                store,
                "20240102",
                [_bar("000001.SZ", "20240102", 10.0, 9.8)],
                ["000001.SZ"],
            )
            _write_kline(
                store,
                "20240103",
                [_bar("000001.SZ", "20240103", 10.1, 10.0)],
                ["000001.SZ"],
            )
            marker = root / "instrument_info" / "snapshot=latest" / "_SUCCESS.json"
            marker.write_text("[]", encoding="utf-8")

            result = run_full_sample_self_check(
                SelfCheckConfig(
                    output_root=root,
                    calendar_csv=calendar,
                    report_dir=root / "audit",
                )
            )
            codes = set(result.issues["issue_code"])
            self.assertIn("AUXILIARY_MARKER_SCHEMA_INVALID", codes)
            self.assertIn("EXPIRE_DATE_INVALID", codes)
            self.assertEqual(result.coverage_by_date["trade_date"].tolist(), ["20240102", "20240103"])

            errors = result.issues.loc[result.issues["level"] == "ERROR"]
            for column in ("expected", "actual", "evidence", "possible_causes", "suggested_action", "source_file"):
                self.assertTrue(errors[column].map(bool).all(), column)

    def test_date_mismatch_does_not_count_as_present(self) -> None:
        """分区内 trade_date 错位行不能掩盖同日理论缺失。"""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = DailyPartitionStore(root)
            calendar = _write_calendar(root, ["20240102", "20240103"])
            _write_instruments(
                store,
                [{"code": "000001.SZ", "open_date": "20240102", "expire_date": ""}],
                ["000001.SZ"],
            )
            _write_kline(
                store,
                "20240102",
                [_bar("000001.SZ", "20240103", 10.0, 9.8)],
                ["000001.SZ"],
            )
            _write_kline(
                store,
                "20240103",
                [_bar("000001.SZ", "20240103", 10.1, 10.0)],
                ["000001.SZ"],
            )

            result = run_full_sample_self_check(
                SelfCheckConfig(
                    output_root=root,
                    calendar_csv=calendar,
                    report_dir=root / "audit",
                )
            )
            self.assertIn("KLINE_DATE_PARTITION_MISMATCH", set(result.issues["issue_code"]))
            coverage = result.coverage_by_date.set_index("trade_date")
            self.assertEqual(int(coverage.loc["20240102", "missing_symbols"]), 1)

    def test_reads_staging_daily_directories(self) -> None:
        """staging 日目录不含最终分区标记时也应执行完整行级审计。"""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = DailyPartitionStore(root)
            staging = root / "staging" / "qmt_job"
            calendar = _write_calendar(root, ["20240102", "20240103"])
            _write_instruments(
                store,
                [{"code": "000001.SZ", "open_date": "20240102", "expire_date": ""}],
                ["000001.SZ"],
            )
            _write_staging_kline(staging, "20240102", [_bar("000001.SZ", "20240102", 10.0, 9.8)])
            _write_staging_kline(staging, "20240103", [_bar("000001.SZ", "20240103", 10.1, 10.0)])

            result = run_full_sample_self_check(
                SelfCheckConfig(
                    output_root=root,
                    staging_root=root / "staging",
                    calendar_csv=calendar,
                    report_dir=root / "audit",
                )
            )

            self.assertEqual(result.exit_code, 0)
            self.assertEqual(result.summary["source_mode"], "staging")
            self.assertEqual(result.summary["expected_rows"], 2)


if __name__ == "__main__":
    unittest.main()
