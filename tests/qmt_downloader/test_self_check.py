"""QMT 全样本日线数据内部质量自检测试。"""

from __future__ import annotations

import hashlib
import json
import logging
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from quant.qmt_downloader.self_check import (
    LOGGER_NAME,
    SelfCheckConfig,
    run_full_sample_self_check,
)
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


def _write_corporate_action(
    store: DailyPartitionStore, date_value: str, code: str
) -> None:
    """写出一个只含单只证券的除权事件分区。

    参数：
        store: 指向临时 QMT 根目录的日分区存储器。
        date_value: 除权日期，即分区键 ``ex_date``。
        code: 发生除权的证券代码。

    返回：
        无返回值。
    """

    store.write_partition(
        "corporate_actions",
        "ex_date",
        date_value,
        pd.DataFrame([{"code": code, "ex_date": date_value}]),
        ["code", "ex_date"],
        ["code"],
        ["code"],
        {"partition_scope": {"symbols": [code]}},
    )


def _write_staging_kline(
    root: Path,
    date_value: str,
    rows: list[dict[str, object]],
    batch_id: int = 0,
) -> None:
    """写入一个与 QMT runner 相同的 staging 日批次及行数元数据。

    参数：
        root: staging 作业目录。
        date_value: 八位交易日期。
        rows: 需要写入该日期的标准日线行。
        batch_id: 批次编号，决定文件名 ``batch_%05d.csv``；同一日期写多个批次时
            用它区分，缺省 0 表示单批次场景。
    返回：
        无返回值。
    """

    directory = root / ("kline_daily_" + date_value)
    directory.mkdir(parents=True, exist_ok=True)
    name = f"batch_{batch_id:05d}"
    path = directory / (name + ".csv")
    pd.DataFrame(rows, columns=KLINE_COLUMNS).to_csv(
        path, index=False, encoding="utf-8-sig", lineterminator="\n"
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (directory / (name + ".meta.json")).write_text(
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
            # 快照按 code 排序落盘，非法退市日期的证券位于 CSV 第 3 行，用来核对
            # 生命周期问题的行号取自该证券首次出现的位置，而不是固定的第一行。
            _write_instruments(
                store,
                [
                    {"code": "000001.SZ", "open_date": "20240102", "expire_date": ""},
                    {"code": "600000.SH", "open_date": "20240102", "expire_date": "bad-date"},
                ],
                ["000001.SZ", "600000.SH"],
            )
            _write_kline(
                store,
                "20240102",
                [
                    _bar("000001.SZ", "20240102", 10.0, 9.8),
                    _bar("600000.SH", "20240102", 20.0, 19.8),
                ],
                ["000001.SZ", "600000.SH"],
            )
            _write_kline(
                store,
                "20240103",
                [
                    _bar("000001.SZ", "20240103", 10.1, 10.0),
                    _bar("600000.SH", "20240103", 20.1, 20.0),
                ],
                ["000001.SZ", "600000.SH"],
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
            expire = result.issues.loc[result.issues["issue_code"] == "EXPIRE_DATE_INVALID"]
            self.assertEqual(len(expire), 1)
            self.assertEqual(expire.iloc[0]["code"], "600000.SH")
            self.assertEqual(int(expire.iloc[0]["source_row"]), 3)
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

    def test_progress_logs_cover_phases_and_daily_scan(self) -> None:
        """自检应按阶段和交易日输出进度日志，便于观察长时间运行。"""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = DailyPartitionStore(root)
            dates = ["20240102", "20240103", "20240104"]
            calendar = _write_calendar(root, dates)
            _write_instruments(
                store,
                [{"code": "000001.SZ", "open_date": "20240102", "expire_date": ""}],
                ["000001.SZ"],
            )
            previous = 9.8
            for date_value in dates:
                _write_kline(
                    store,
                    date_value,
                    [_bar("000001.SZ", date_value, previous + 0.1, previous)],
                    ["000001.SZ"],
                )
                previous = previous + 0.1

            with self.assertLogs(LOGGER_NAME, level="INFO") as captured:
                result = run_full_sample_self_check(
                    SelfCheckConfig(
                        output_root=root,
                        calendar_csv=calendar,
                        report_dir=root / "audit",
                        # 步长为 1 时每个交易日都必须留下一条 INFO 进度。
                        progress_every=1,
                    )
                )

            self.assertEqual(result.exit_code, 0)
            messages = captured.output
            for phase in (
                "发现日线分区",
                "装载证券信息与证券池",
                "装载交易日历",
                "解析证券生命周期与除权事件",
                "逐日扫描行情",
                "汇总并写出报告",
            ):
                self.assertTrue(
                    any("阶段开始 " + phase in line for line in messages),
                    f"缺少阶段开始日志: {phase}",
                )
                self.assertTrue(
                    any("阶段完成 " + phase in line for line in messages),
                    f"缺少阶段完成日志: {phase}",
                )
            scan_lines = [line for line in messages if "逐日扫描 " in line]
            self.assertEqual(len(scan_lines), len(dates))
            for index, date_value in enumerate(dates):
                self.assertIn(
                    f"{index + 1}/{len(dates)}", scan_lines[index]
                )
                self.assertIn("date=" + date_value, scan_lines[index])
            self.assertTrue(
                any("分区校验 1/3" in line for line in messages),
                "缺少分区校验进度日志",
            )
            self.assertTrue(
                any("[自检] 结束" in line and "总耗时" in line for line in messages),
                "缺少总耗时结束日志",
            )

    def test_progress_every_default_limits_scan_log_volume(self) -> None:
        """缺省步长下逐日进度按总量的 5% 输出，不应逐日刷屏。"""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = DailyPartitionStore(root)
            dates = [f"202401{day:02d}" for day in range(1, 29)] + [
                f"202402{day:02d}" for day in range(1, 13)
            ]
            calendar = _write_calendar(root, dates)
            _write_instruments(
                store,
                [{"code": "000001.SZ", "open_date": "20240101", "expire_date": ""}],
                ["000001.SZ"],
            )
            previous = 9.8
            for date_value in dates:
                _write_kline(
                    store,
                    date_value,
                    [_bar("000001.SZ", date_value, previous + 0.1, previous)],
                    ["000001.SZ"],
                )
                previous = previous + 0.1

            with self.assertLogs(LOGGER_NAME, level="INFO") as captured:
                run_full_sample_self_check(
                    SelfCheckConfig(
                        output_root=root,
                        calendar_csv=calendar,
                        report_dir=root / "audit",
                    )
                )

            scan_lines = [line for line in captured.output if "逐日扫描 " in line]
            # 40 个交易日、步长 2：第 2、4、…、40 日各一条共 20 条，加上首日 1 条。
            self.assertEqual(len(scan_lines), 21)
            self.assertIn("逐日扫描 1/40", scan_lines[0])
            self.assertIn("逐日扫描 40/40", scan_lines[-1])

    def test_progress_every_above_total_keeps_first_and_last_only(self) -> None:
        """步长大于总量时仍应保留首尾两条进度，不能一条都不输出。"""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = DailyPartitionStore(root)
            dates = ["20240102", "20240103", "20240104"]
            calendar = _write_calendar(root, dates)
            _write_instruments(
                store,
                [{"code": "000001.SZ", "open_date": "20240102", "expire_date": ""}],
                ["000001.SZ"],
            )
            previous = 9.8
            for date_value in dates:
                _write_kline(
                    store,
                    date_value,
                    [_bar("000001.SZ", date_value, previous + 0.1, previous)],
                    ["000001.SZ"],
                )
                previous = previous + 0.1

            with self.assertLogs(LOGGER_NAME, level="INFO") as captured:
                run_full_sample_self_check(
                    SelfCheckConfig(
                        output_root=root,
                        calendar_csv=calendar,
                        report_dir=root / "audit",
                        progress_every=1000,
                    )
                )

            scan_lines = [line for line in captured.output if "逐日扫描 " in line]
            self.assertEqual(len(scan_lines), 2)
            self.assertIn("逐日扫描 1/3", scan_lines[0])
            self.assertIn("逐日扫描 3/3", scan_lines[1])

    def test_debug_level_logs_every_trading_day(self) -> None:
        """DEBUG 级别应逐日输出，用于定位卡在哪个交易日。"""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = DailyPartitionStore(root)
            dates = ["20240102", "20240103", "20240104"]
            calendar = _write_calendar(root, dates)
            _write_instruments(
                store,
                [{"code": "000001.SZ", "open_date": "20240102", "expire_date": ""}],
                ["000001.SZ"],
            )
            previous = 9.8
            for date_value in dates:
                _write_kline(
                    store,
                    date_value,
                    [_bar("000001.SZ", date_value, previous + 0.1, previous)],
                    ["000001.SZ"],
                )
                previous = previous + 0.1

            with self.assertLogs(LOGGER_NAME, level="DEBUG") as captured:
                run_full_sample_self_check(
                    SelfCheckConfig(
                        output_root=root,
                        calendar_csv=calendar,
                        report_dir=root / "audit",
                        progress_every=1000,
                    )
                )

            scan_lines = [line for line in captured.output if "逐日扫描 " in line]
            self.assertEqual(len(scan_lines), len(dates))
            self.assertTrue(
                any(line.startswith("DEBUG") for line in scan_lines),
                "非里程碑进度必须降级为 DEBUG",
            )

    def test_library_run_installs_no_log_handler(self) -> None:
        """库层只写日志器，不得安装处理器或改动级别，避免污染宿主日志配置。"""

        logger = logging.getLogger(LOGGER_NAME)
        self.assertEqual(logger.handlers, [])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = DailyPartitionStore(root)
            calendar = _write_calendar(root, ["20240102"])
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

            run_full_sample_self_check(
                SelfCheckConfig(
                    output_root=root,
                    calendar_csv=calendar,
                    report_dir=root / "audit",
                )
            )

        self.assertEqual(logger.handlers, [])
        self.assertEqual(logger.level, logging.NOTSET)

    def test_negative_progress_every_is_rejected(self) -> None:
        """负步长会让进度日志无法输出，必须在配置校验阶段拒绝。"""

        with self.assertRaises(ValueError):
            SelfCheckConfig(output_root=Path("."), progress_every=-1)

    def test_row_issues_keep_per_row_code_and_csv_line(self) -> None:
        """多行分区中每条问题必须落到正确的证券和 CSV 行号，不能整体错位。"""

        symbols = ["000001.SZ", "000002.SZ", "600000.SH"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = DailyPartitionStore(root)
            calendar = _write_calendar(root, ["20240102"])
            _write_instruments(
                store,
                [
                    {"code": code, "open_date": "20240102", "expire_date": ""}
                    for code in symbols
                ],
                symbols,
            )
            # 分区按 code 排序落盘，第二只证券（CSV 第 3 行）的最高价低于收盘价。
            bad = _bar("000002.SZ", "20240102", 20.0, 19.8)
            bad["high"] = 19.0
            bad["low"] = 18.5
            # 第三只证券（CSV 第 4 行）的 trade_date 与分区日期不符：逐行日期规范化
            # 带缓存，必须保证同一分区内不同日期各自解析，不会复用第一行的结果。
            wrong_date = _bar("600000.SH", "20240103", 30.0, 29.8)
            _write_kline(
                store,
                "20240102",
                [
                    _bar("000001.SZ", "20240102", 10.0, 9.8),
                    bad,
                    wrong_date,
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

            ohlc = result.issues.loc[
                result.issues["issue_code"] == "INVALID_OHLC_RELATION"
            ]
            self.assertEqual(len(ohlc), 1)
            self.assertEqual(ohlc.iloc[0]["code"], "000002.SZ")
            self.assertEqual(int(ohlc.iloc[0]["source_row"]), 3)
            self.assertIn("high=19.0", ohlc.iloc[0]["actual"])
            mismatch = result.issues.loc[
                result.issues["issue_code"] == "KLINE_DATE_PARTITION_MISMATCH"
            ]
            self.assertEqual(len(mismatch), 1)
            self.assertEqual(mismatch.iloc[0]["code"], "600000.SH")
            self.assertEqual(int(mismatch.iloc[0]["source_row"]), 4)
            self.assertEqual(mismatch.iloc[0]["actual"], "20240103")

    def test_staging_row_issues_use_batch_file_and_row(self) -> None:
        """staging 模式的问题必须指向批次文件及其内部行号，而非合并后的位置。"""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = DailyPartitionStore(root)
            staging = root / "staging" / "qmt_job"
            calendar = _write_calendar(root, ["20240102"])
            _write_instruments(
                store,
                [
                    {"code": "000001.SZ", "open_date": "20240102", "expire_date": ""},
                    {"code": "600000.SH", "open_date": "20240102", "expire_date": ""},
                    {"code": "300001.SZ", "open_date": "20240102", "expire_date": ""},
                ],
                ["000001.SZ", "600000.SH", "300001.SZ"],
            )
            bad = _bar("600000.SH", "20240102", 20.0, 19.8, volume=0.0, amount=0.0)
            # 分成两个批次：问题必须落到自己所在批次文件，而不是合并帧中的位置。
            _write_staging_kline(
                staging,
                "20240102",
                [_bar("000001.SZ", "20240102", 10.0, 9.8), bad],
                batch_id=0,
            )
            _write_staging_kline(
                staging,
                "20240102",
                [_bar("300001.SZ", "20240102", 30.0, 29.8, volume=0.0, amount=0.0)],
                batch_id=1,
            )

            result = run_full_sample_self_check(
                SelfCheckConfig(
                    output_root=root,
                    staging_root=root / "staging",
                    calendar_csv=calendar,
                    report_dir=root / "audit",
                )
            )

            zero_volume = result.issues.loc[
                result.issues["issue_code"] == "ACTIVE_ZERO_VOLUME"
            ].set_index("code")
            self.assertEqual(len(zero_volume), 2)
            self.assertEqual(int(zero_volume.loc["600000.SH", "source_row"]), 3)
            self.assertTrue(
                str(zero_volume.loc["600000.SH", "source_file"]).endswith(
                    "batch_00000.csv"
                )
            )
            self.assertEqual(int(zero_volume.loc["300001.SZ", "source_row"]), 2)
            self.assertTrue(
                str(zero_volume.loc["300001.SZ", "source_file"]).endswith(
                    "batch_00001.csv"
                )
            )

    def test_partitions_outside_audit_range_are_not_read(self) -> None:
        """限定区间时不得读取区间外分区，其损坏也不应进入本次报告。"""

        dates = ["20240102", "20240103", "20240104"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = DailyPartitionStore(root)
            calendar = _write_calendar(root, dates)
            _write_instruments(
                store,
                [{"code": "000001.SZ", "open_date": "20240102", "expire_date": ""}],
                ["000001.SZ"],
            )
            previous = 9.8
            for date_value in dates:
                _write_kline(
                    store,
                    date_value,
                    [_bar("000001.SZ", date_value, previous + 0.1, previous)],
                    ["000001.SZ"],
                )
                previous = previous + 0.1
            # 破坏区间外分区：内容与完成标记的 SHA-256 不再一致。
            outside = root / "kline_1d" / "date=20240102" / "data.csv"
            outside.write_text(
                outside.read_text(encoding="utf-8-sig") + "\n", encoding="utf-8-sig"
            )

            with self.assertLogs(LOGGER_NAME, level="INFO") as captured:
                result = run_full_sample_self_check(
                    SelfCheckConfig(
                        output_root=root,
                        calendar_csv=calendar,
                        report_dir=root / "audit",
                        start_date="20240103",
                        end_date="20240104",
                    )
                )

            self.assertEqual(result.exit_code, 0)
            self.assertNotIn("PARTITION_HASH_MISMATCH", set(result.issues["issue_code"]))
            self.assertTrue(
                any("待校验日线分区 2 个" in line for line in captured.output),
                "区间外分区不应进入校验循环",
            )

    def test_corrupted_partition_inside_audit_range_is_still_reported(self) -> None:
        """区间内分区仍必须做完整哈希校验，范围裁剪不能放过真正的损坏。"""

        dates = ["20240102", "20240103", "20240104"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = DailyPartitionStore(root)
            calendar = _write_calendar(root, dates)
            _write_instruments(
                store,
                [{"code": "000001.SZ", "open_date": "20240102", "expire_date": ""}],
                ["000001.SZ"],
            )
            previous = 9.8
            for date_value in dates:
                _write_kline(
                    store,
                    date_value,
                    [_bar("000001.SZ", date_value, previous + 0.1, previous)],
                    ["000001.SZ"],
                )
                previous = previous + 0.1
            inside = root / "kline_1d" / "date=20240103" / "data.csv"
            inside.write_text(
                inside.read_text(encoding="utf-8-sig") + "\n", encoding="utf-8-sig"
            )

            result = run_full_sample_self_check(
                SelfCheckConfig(
                    output_root=root,
                    calendar_csv=calendar,
                    report_dir=root / "audit",
                    start_date="20240103",
                    end_date="20240104",
                )
            )

            self.assertIn("PARTITION_HASH_MISMATCH", set(result.issues["issue_code"]))

    def test_corporate_actions_are_limited_to_audit_range(self) -> None:
        """除权分区只按审计区间读取，区间内事件仍能解释昨收断层。"""

        dates = ["20240102", "20240103", "20240104"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = DailyPartitionStore(root)
            calendar = _write_calendar(root, dates)
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
            # 20240104 的昨收相对前一日收盘跳空，需由当日除权事件解释。
            _write_kline(
                store,
                "20240104",
                [_bar("000001.SZ", "20240104", 9.2, 9.1)],
                ["000001.SZ"],
            )
            for date_value in dates:
                _write_corporate_action(store, date_value, "000001.SZ")

            with self.assertLogs(LOGGER_NAME, level="INFO") as captured:
                result = run_full_sample_self_check(
                    SelfCheckConfig(
                        output_root=root,
                        calendar_csv=calendar,
                        report_dir=root / "audit",
                        start_date="20240103",
                        end_date="20240104",
                    )
                )

            issue_codes = set(result.issues["issue_code"])
            self.assertIn("PRE_CLOSE_DISCONTINUITY_EXPLAINED", issue_codes)
            self.assertNotIn("PRE_CLOSE_DISCONTINUITY", issue_codes)
            self.assertTrue(
                any("待读取除权事件分区 2 个" in line for line in captured.output),
                "区间外除权分区不应被读取",
            )


if __name__ == "__main__":
    unittest.main()
