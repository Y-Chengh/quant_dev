"""端到端验证日线库的增量入库：过滤、幂等、按月重写与锁降级。

夹具一律通过下载器真实的 ``DailyPartitionStore`` 写出，因此 ``_SUCCESS.json``
里的行数与 SHA-256 都是真的，增量检查走的是与生产完全一致的路径。
"""

from __future__ import annotations

import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import duckdb
import pandas as pd

from quant.market_data.daily import AdjustMode, DailyMarketClient
from quant.market_data.daily.ingest import DailySyncConfig, aux_tables, sync_daily_store
from quant.market_data.daily.ingest.locking import DailyStoreLockedError, sync_lock
from quant.market_data.daily.ingest.shards import rewrite_month_shard, shard_directory
from quant.market_data.daily.schema import apply_schema
from quant.qmt_downloader.storage import DailyPartitionStore

KLINE_COLUMNS = [
    "code", "trade_date", "open", "high", "low", "close",
    "pre_close", "volume", "amount", "suspend_flag",
]
INSTRUMENT_COLUMNS = [
    "code", "instrument_name", "open_date", "expire_date",
    "is_trading", "instrument_status",
]
ACTION_COLUMNS = [
    "code", "ex_date", "cash_dividend_per_share", "bonus_share_per_share",
    "capitalization_per_share", "rights_issue_per_share", "rights_issue_price",
    "share_reform_flag", "adjustment_factor",
]
_SCOPE = {"partition_scope": {"symbols": ["000001.SZ", "600000.SH", "300001.SZ"]}}


def _bar(code, day, close, pre_close, suspend=0.0, volume=1000.0):
    """构造一行日线记录。

    参数：
        code: 证券代码。
        day: 八位交易日字符串。
        close: 收盘价，同时用作开高低价，便于断言。
        pre_close: 前收盘价。
        suspend: 停牌标记，1.0 表示停牌。
        volume: 成交量。

    返回：
        与 ``KLINE_COLUMNS`` 对齐的字典。
    """
    return {
        "code": code, "trade_date": day, "open": close, "high": close,
        "low": close, "close": close, "pre_close": pre_close,
        "volume": volume, "amount": close * volume, "suspend_flag": suspend,
    }


def _build_source(root: Path, days=("20240102", "20240103", "20240104")) -> None:
    """写出一份最小但结构完整的 QMT 落盘目录。

    刻意复刻真实数据的两个特征：每个交易日都返回全部证券（含尚未上市的填充行），
    以及退市日用 ``19700427`` 这类哨兵表达「无退市日」。

    参数：
        root: 落盘根目录。
        days: 要写出的交易日序列。

    返回：
        无返回值。
    """
    store = DailyPartitionStore(root)
    instruments = pd.DataFrame([
        {"code": "000001.SZ", "instrument_name": "平安银行", "open_date": "20240101",
         "expire_date": "", "is_trading": "", "instrument_status": "0"},
        {"code": "600000.SH", "instrument_name": "浦发银行", "open_date": "20240103",
         "expire_date": "19700427", "is_trading": "", "instrument_status": "0"},
        {"code": "300001.SZ", "instrument_name": "*ST特锐", "open_date": "20240101",
         "expire_date": "20240103", "is_trading": "", "instrument_status": "0"},
    ])
    store.write_partition("instrument_info", "snapshot", "latest", instruments,
                          INSTRUMENT_COLUMNS, ["code"], ["code"], _SCOPE, overwrite=True)
    calendar = pd.DataFrame({"trade_date": list(days),
                             "calendar_symbol": ["000001.SH"] * len(days)})
    store.write_partition("trading_calendar", "snapshot", "latest", calendar,
                          ["trade_date", "calendar_symbol"], ["trade_date"],
                          ["trade_date"], _SCOPE, overwrite=True)

    prices = {
        "20240102": {"000001.SZ": (10.0, 9.5), "600000.SH": (7.0, 7.0), "300001.SZ": (5.0, 5.0)},
        "20240103": {"000001.SZ": (11.0, 10.0 / 1.05), "600000.SH": (7.2, 7.0),
                     "300001.SZ": (5.1, 5.0)},
        "20240104": {"000001.SZ": (12.0, 11.0), "600000.SH": (7.4, 7.2),
                     "300001.SZ": (5.2, 5.1)},
    }
    for day in days:
        rows = [
            _bar(code, day, close, pre_close,
                 suspend=1.0 if (code == "600000.SH" and day == "20240102") else 0.0)
            for code, (close, pre_close) in prices[day].items()
        ]
        store.write_partition("kline_1d", "date", day, pd.DataFrame(rows),
                              KLINE_COLUMNS, ["code", "trade_date"], ["code"],
                              _SCOPE, overwrite=True)
        actions = []
        if day == "20240103":
            actions.append({
                "code": "000001.SZ", "ex_date": day, "cash_dividend_per_share": 0.5,
                "bonus_share_per_share": 0.0, "capitalization_per_share": 0.0,
                "rights_issue_per_share": 0.0, "rights_issue_price": 0.0,
                "share_reform_flag": 0.0, "adjustment_factor": 1.05,
            })
        store.write_partition("corporate_actions", "ex_date", day,
                              pd.DataFrame(actions, columns=ACTION_COLUMNS),
                              ACTION_COLUMNS, ["code", "ex_date"], ["code"],
                              _SCOPE, overwrite=True)
        store.write_run_date_complete(
            day,
            [f"kline_1d/date={day}", f"corporate_actions/ex_date={day}"],
            {"watermark_scope": {"datasets": ["kline_1d"]}},
        )


class DailySyncTest(unittest.TestCase):
    """增量入库的主要行为。"""

    def _config(self, base: Path, mode: str = "full") -> DailySyncConfig:
        """构造指向临时目录的同步配置。

        参数：
            base: 临时根目录，其下建立 ``qmt`` 源目录与 ``daily`` 库目录。
            mode: 同步模式。

        返回：
            可直接交给 ``sync_daily_store`` 的配置。
        """
        return DailySyncConfig(
            database=base / "daily" / "qmt_daily.duckdb",
            source_root=base / "qmt",
            mode=mode,
        )

    def test_first_sync_filters_unlisted_and_delisted_rows(self) -> None:
        """未上市与退市之后的填充行必须被生命周期过滤掉。"""
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            _build_source(base / "qmt")
            report = sync_daily_store(self._config(base))
            self.assertEqual(report.status, "synced")
            # 源侧 3 天 × 3 只 = 9 行；过滤后应为 7 行：
            # 600000.SH 上市前的 0102 与 300001.SZ 退市后的 0104 各去掉一行。
            self.assertEqual(report.rows_after, 7)

            client = DailyMarketClient(base / "daily" / "qmt_daily.duckdb")
            bars = client.get_klines_1d([], date(2024, 1, 1), date(2024, 1, 31),
                                        include_suspended=True)
            pairs = set(zip(bars["code"], bars["trade_date"].astype(str)))
            self.assertNotIn(("600000.SH", "2024-01-02"), pairs)
            self.assertNotIn(("300001.SZ", "2024-01-04"), pairs)
            self.assertIn(("000001.SZ", "2024-01-02"), pairs)

    def test_sync_report_lists_filter_reasons_and_counts(self) -> None:
        """同步报告应按原因汇总本次实际丢弃的行数，供排查用。"""
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            _build_source(base / "qmt")
            report = sync_daily_store(self._config(base))
            self.assertEqual(report.status, "synced")
            totals = dict(report.filtered_totals)
            # 600000.SH 上市前的 0102 一行、300001.SZ 退市后的 0104 一行。
            self.assertEqual(totals.get("before_listing"), 1)
            self.assertEqual(totals.get("after_delisting"), 1)
            self.assertNotIn("invalid_trade_date", totals)
            self.assertNotIn("unknown_code", totals)

    def test_shard_result_reports_reason_breakdown(self) -> None:
        """单个分片的过滤明细应能独立取到，不必跑完整个同步。"""
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "qmt"
            daily_root = base / "daily"
            _build_source(source)
            lifecycle = aux_tables.load_lifecycle_frame(source)
            connection = duckdb.connect()
            try:
                result = rewrite_month_shard(connection, source, daily_root, 2024, 1, lifecycle)
            finally:
                connection.close()
            breakdown = dict(result.filtered)
            self.assertEqual(breakdown.get("before_listing"), 1)
            self.assertEqual(breakdown.get("after_delisting"), 1)
            self.assertEqual(result.rows, 7)

    def test_shard_result_classifies_invalid_date_and_unknown_code(self) -> None:
        """非法 trade_date 与不在生命周期表里的代码要各自归到对应的过滤原因。"""
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "qmt"
            daily_root = base / "daily"
            partition = source / "kline_1d" / "date=20240102"
            partition.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([
                _bar("000001.SZ", "20240102", 10.0, 9.5),
                _bar("000001.SZ", "abcdefgh", 10.0, 9.5),
                _bar("999999.SZ", "20240102", 1.0, 1.0),
            ]).to_csv(partition / "data.csv", index=False, columns=KLINE_COLUMNS)
            lifecycle = pd.DataFrame([
                {"code": "000001.SZ", "open_date": pd.Timestamp("2024-01-01"),
                 "expire_date": pd.NaT},
            ])
            connection = duckdb.connect()
            try:
                result = rewrite_month_shard(connection, source, daily_root, 2024, 1, lifecycle)
            finally:
                connection.close()
            breakdown = dict(result.filtered)
            self.assertEqual(breakdown.get("invalid_trade_date"), 1)
            self.assertEqual(breakdown.get("unknown_code"), 1)
            self.assertEqual(result.rows, 1)

    def test_missing_open_date_does_not_drop_the_symbol(self) -> None:
        """open_date 缺失时不应把该证券的全部行情都过滤掉，口径与 self_check 一致。"""
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "qmt"
            _build_source(source)
            store = DailyPartitionStore(source)
            instruments = pd.DataFrame([
                {"code": "000001.SZ", "instrument_name": "平安银行", "open_date": "",
                 "expire_date": "", "is_trading": "", "instrument_status": "0"},
                {"code": "600000.SH", "instrument_name": "浦发银行", "open_date": "20240103",
                 "expire_date": "19700427", "is_trading": "", "instrument_status": "0"},
                {"code": "300001.SZ", "instrument_name": "*ST特锐", "open_date": "20240101",
                 "expire_date": "20240103", "is_trading": "", "instrument_status": "0"},
            ])
            store.write_partition("instrument_info", "snapshot", "latest", instruments,
                                  INSTRUMENT_COLUMNS, ["code"], ["code"], _SCOPE, overwrite=True)
            report = sync_daily_store(self._config(base))
            self.assertEqual(report.status, "synced")
            client = DailyMarketClient(base / "daily" / "qmt_daily.duckdb")
            bars = client.get_klines_1d(["000001.SZ"], date(2024, 1, 1), date(2024, 1, 31),
                                        include_suspended=True)
            self.assertEqual(len(bars), 3)

    def test_missing_open_date_still_filters_by_expire_date(self) -> None:
        """open_date 缺失只放开下界，退市日过滤上界必须照常生效。"""
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "qmt"
            _build_source(source)
            store = DailyPartitionStore(source)
            instruments = pd.DataFrame([
                {"code": "000001.SZ", "instrument_name": "平安银行", "open_date": "",
                 "expire_date": "", "is_trading": "", "instrument_status": "0"},
                {"code": "600000.SH", "instrument_name": "浦发银行", "open_date": "20240103",
                 "expire_date": "19700427", "is_trading": "", "instrument_status": "0"},
                {"code": "300001.SZ", "instrument_name": "*ST特锐", "open_date": "",
                 "expire_date": "20240103", "is_trading": "", "instrument_status": "0"},
            ])
            store.write_partition("instrument_info", "snapshot", "latest", instruments,
                                  INSTRUMENT_COLUMNS, ["code"], ["code"], _SCOPE, overwrite=True)
            report = sync_daily_store(self._config(base))
            self.assertEqual(report.status, "synced")
            client = DailyMarketClient(base / "daily" / "qmt_daily.duckdb")
            bars = client.get_klines_1d(["300001.SZ"], date(2024, 1, 1), date(2024, 1, 31),
                                        include_suspended=True)
            pairs = set(bars["trade_date"].astype(str))
            # open_date 缺失放开了下界，但 expire_date=20240103 之后的 0104 仍应被过滤。
            self.assertEqual(pairs, {"2024-01-02", "2024-01-03"})

    def test_second_sync_is_a_no_op(self) -> None:
        """源目录没有变化时应报告无增量，且不重写任何分片。"""
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            _build_source(base / "qmt")
            config = self._config(base)
            sync_daily_store(config)
            shard = shard_directory(base / "daily", 2024, 1) / "bars.parquet"
            before = shard.stat().st_mtime_ns

            report = sync_daily_store(config)
            self.assertEqual(report.status, "up_to_date")
            self.assertEqual(report.dirty, 0)
            self.assertEqual(report.rewritten_shards, 0)
            self.assertEqual(shard.stat().st_mtime_ns, before)

    def test_detection_never_parses_csv_when_nothing_changed(self) -> None:
        """无增量时不允许打开任何 data.csv：源目录有上万个分区，解析不起。"""
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            _build_source(base / "qmt")
            config = self._config(base)
            sync_daily_store(config)

            opened: list[str] = []
            original = Path.open

            def tracking_open(self, *args, **kwargs):
                """记录被打开的数据文件路径后照常打开。"""
                if self.name == "data.csv":
                    opened.append(str(self))
                return original(self, *args, **kwargs)

            with patch.object(Path, "open", tracking_open):
                report = sync_daily_store(config)
            self.assertEqual(report.status, "up_to_date")
            self.assertEqual(opened, [])

    def test_changed_day_rewrites_only_its_month(self) -> None:
        """改动一个交易日只应重写它所在的那个月度分片。"""
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "qmt"
            _build_source(source, days=("20240102", "20240103", "20240104"))
            config = self._config(base)
            sync_daily_store(config)

            # 追加一个二月的交易日，一月分片不应被改写。
            january = shard_directory(base / "daily", 2024, 1) / "bars.parquet"
            before = january.stat().st_mtime_ns
            store = DailyPartitionStore(source)
            store.write_partition(
                "kline_1d", "date", "20240205",
                pd.DataFrame([_bar("000001.SZ", "20240205", 13.0, 12.0)]),
                KLINE_COLUMNS, ["code", "trade_date"], ["code"], _SCOPE, overwrite=True,
            )
            report = sync_daily_store(config)

            self.assertEqual(report.status, "synced")
            self.assertEqual(report.rewritten_shards, 1)
            self.assertEqual(report.touched_months, ((2024, 2),))
            self.assertEqual(january.stat().st_mtime_ns, before)
            self.assertTrue(
                (shard_directory(base / "daily", 2024, 2) / "bars.parquet").is_file()
            )

    def test_sync_then_read_in_same_process(self) -> None:
        """同一进程内先同步再读取必须成功，不能被自己的写锁挡住。"""
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            _build_source(base / "qmt")
            sync_daily_store(self._config(base))
            client = DailyMarketClient(base / "daily" / "qmt_daily.duckdb")
            self.assertEqual(client.get_metadata()["rows"], 7)
            self.assertEqual(
                client.get_trading_calendar(),
                [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)],
            )

    def test_adjustment_survives_round_trip(self) -> None:
        """入库存原始价，读取时后复权应当还原出连续序列。"""
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            _build_source(base / "qmt")
            sync_daily_store(self._config(base))
            client = DailyMarketClient(base / "daily" / "qmt_daily.duckdb")
            raw = client.get_klines_1d(["000001.SZ"], date(2024, 1, 1), date(2024, 1, 31))
            hfq = client.get_klines_1d(["000001.SZ"], date(2024, 1, 1), date(2024, 1, 31),
                                       adjust=AdjustMode.HFQ)
            self.assertAlmostEqual(raw["close"].tolist()[1], 11.0)
            self.assertAlmostEqual(hfq["close"].tolist()[1], 11.0 * 1.05)

    def test_suspended_rows_are_dropped_by_default(self) -> None:
        """缺省剔除停牌行，避免平价零量造出假零收益。"""
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "qmt"
            _build_source(source)
            # 让 000001.SZ 在 0104 停牌。
            store = DailyPartitionStore(source)
            store.write_partition(
                "kline_1d", "date", "20240104",
                pd.DataFrame([
                    _bar("000001.SZ", "20240104", 11.0, 11.0, suspend=1.0, volume=0.0),
                    _bar("600000.SH", "20240104", 7.4, 7.2),
                    _bar("300001.SZ", "20240104", 5.2, 5.1),
                ]),
                KLINE_COLUMNS, ["code", "trade_date"], ["code"], _SCOPE, overwrite=True,
            )
            sync_daily_store(self._config(base))
            client = DailyMarketClient(base / "daily" / "qmt_daily.duckdb")
            without = client.get_klines_1d(["000001.SZ"], date(2024, 1, 1), date(2024, 1, 31))
            with_susp = client.get_klines_1d(
                ["000001.SZ"], date(2024, 1, 1), date(2024, 1, 31), include_suspended=True
            )
            self.assertEqual(len(without), 2)
            self.assertEqual(len(with_susp), 3)

    def test_errata_overrides_suspend_flag_in_dumped_data(self) -> None:
        """勘误表覆盖的字段应在落盘（dump）到日线库的数据中生效，其余字段不变。"""
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "qmt"
            _build_source(source)
            errata_csv = base / "errata.csv"
            errata_csv.write_text(
                "symbol,trade_date,field,value,issue_type,description,found_date,source_report\n"
                "000001.SZ,20240102,suspend_flag,1,missing_suspension,test,20240102,test\n",
                encoding="utf-8",
            )
            config = self._config(base)
            config = DailySyncConfig(
                database=config.database,
                source_root=config.source_root,
                mode=config.mode,
                errata_csv=errata_csv,
            )
            sync_daily_store(config)
            client = DailyMarketClient(base / "daily" / "qmt_daily.duckdb")
            without = client.get_klines_1d(["000001.SZ"], date(2024, 1, 1), date(2024, 1, 31))
            with_susp = client.get_klines_1d(
                ["000001.SZ"], date(2024, 1, 1), date(2024, 1, 31), include_suspended=True
            )
            # 0102 被勘误覆盖为停牌，缺省查询应剔除它，只剩 0103、0104。
            self.assertEqual(len(without), 2)
            self.assertEqual(len(with_susp), 3)
            corrected = with_susp[with_susp["trade_date"] == pd.Timestamp("2024-01-02")].iloc[0]
            # 只覆盖了 suspend_flag，价格字段应保持源数据原值不变。
            self.assertAlmostEqual(corrected["close"], 10.0)
            self.assertAlmostEqual(corrected["pre_close"], 9.5)

    def test_qfq_price_is_independent_of_query_window_end(self) -> None:
        """前复权取值必须只由基准日决定，不随查询窗口终点变化。

        回归用例：除权记录若只截到窗口终点，基准日晚于终点时实际基准会退化成
        ``min(anchor, end)``，同一天的复权价就会随窗口漂移。本用例走真实的
        ``DailyMarketClient.get_klines_1d``，覆盖 ``client.py`` 里截断上界那一行。
        """
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            _build_source(base / "qmt")
            sync_daily_store(self._config(base))
            client = DailyMarketClient(base / "daily" / "qmt_daily.duckdb")

            anchor = date(2024, 1, 4)
            values = []
            for end in (date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)):
                frame = client.get_klines_1d(
                    ["000001.SZ"], date(2024, 1, 1), end,
                    adjust=AdjustMode.QFQ, adjust_anchor=anchor,
                )
                row = frame[frame["trade_date"] == pd.Timestamp("2024-01-02")]
                values.append(round(float(row["close"].iloc[0]), 9))

            # 三个窗口下 2024-01-02 的前复权收盘必须完全一致，且等于 10/1.05。
            self.assertEqual(len(set(values)), 1, msg=f"随窗口漂移: {values}")
            self.assertAlmostEqual(values[0], 10.0 / 1.05, places=9)

    def test_removed_corporate_action_partition_clears_its_rows(self) -> None:
        """源侧除权分区消失时，库里的对应记录必须一并清掉。

        回归用例：只删指纹不删数据的话，这些记录会永远留在表里，
        之后每一次复权都被它们带偏。
        """
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "qmt"
            _build_source(source)
            config = self._config(base)
            sync_daily_store(config)

            database = base / "daily" / "qmt_daily.duckdb"
            connection = duckdb.connect(str(database), read_only=True)
            try:
                before = connection.execute("SELECT count(*) FROM corporate_actions").fetchone()[0]
            finally:
                connection.close()
            self.assertEqual(before, 1)

            # 直接删掉 20240103 这个除权分区目录，模拟源侧回滚。
            import shutil

            shutil.rmtree(source / "corporate_actions" / "ex_date=20240103")
            sync_daily_store(config)

            connection = duckdb.connect(str(database), read_only=True)
            try:
                after = connection.execute("SELECT count(*) FROM corporate_actions").fetchone()[0]
            finally:
                connection.close()
            self.assertEqual(after, 0)


class CorporateActionDeletionTest(unittest.TestCase):
    """``delete_corporate_actions`` 的删除范围。"""

    @staticmethod
    def _seeded_connection(ex_dates):
        """建一个已建表并灌入给定除权日记录的内存目录库。

        参数：
            ex_dates: 需要预先写入的八位除权日序列，每个日期一条记录。

        返回：
            已插入 ``len(ex_dates)`` 行 ``corporate_actions`` 的 DuckDB 连接。
        """
        connection = duckdb.connect()
        apply_schema(connection, Path("."), create_view=False)
        frame = pd.DataFrame(
            [
                {
                    "code": "000001.SZ",
                    "ex_date": datetime.strptime(value, "%Y%m%d").date(),
                    "cash_dividend_per_share": 0.1,
                    "bonus_share_per_share": 0.0,
                    "capitalization_per_share": 0.0,
                    "rights_issue_per_share": 0.0,
                    "rights_issue_price": 0.0,
                    "share_reform_flag": 0.0,
                    "adjustment_factor": 1.01,
                }
                for value in ex_dates
            ]
        )
        connection.register("seed_actions", frame)
        connection.execute("INSERT INTO corporate_actions SELECT * FROM seed_actions")
        connection.unregister("seed_actions")
        return connection

    def test_large_deletion_only_removes_the_named_dates(self) -> None:
        """删除数超过分批阈值时，未点名的除权日必须原样保留。

        回归用例：之前这种规模直接 ``DELETE FROM corporate_actions`` 整表清空，
        并指望随后的 refresh 重新灌回来；但本次没有脏除权分区时 refresh 会直接
        返回，除权表就此为空，之后所有复权系数都退化成 1.0。
        """
        # 夹具规模跟着真实阈值走：写死 200 的话，阈值一旦调大，用例会静默退化成
        # 单批路径，看上去仍然通过，却不再覆盖分批。
        threshold = aux_tables._GLOB_THRESHOLD
        removed_count = threshold + 100
        all_dates = [
            (date(2024, 1, 1) + timedelta(days=offset)).strftime("%Y%m%d")
            for offset in range(removed_count + 100)
        ]
        removed, kept = all_dates[:removed_count], all_dates[removed_count:]
        connection = self._seeded_connection(all_dates)
        try:
            self.assertGreater(len(removed), threshold)
            deleted = aux_tables.delete_corporate_actions(connection, removed)
            self.assertEqual(deleted, len(removed))
            remaining = connection.execute(
                "SELECT count(*) FROM corporate_actions"
            ).fetchone()[0]
            self.assertEqual(remaining, len(kept))
            earliest = connection.execute(
                "SELECT min(ex_date) FROM corporate_actions"
            ).fetchone()[0]
            self.assertEqual(earliest, datetime.strptime(kept[0], "%Y%m%d").date())
        finally:
            connection.close()

    def test_small_deletion_still_removes_exactly_the_named_dates(self) -> None:
        """低于阈值时的删除范围与分批路径保持一致。"""
        all_dates = ["20240102", "20240103", "20240104"]
        connection = self._seeded_connection(all_dates)
        try:
            self.assertEqual(
                aux_tables.delete_corporate_actions(connection, ["20240103"]), 1
            )
            rows = connection.execute(
                "SELECT ex_date FROM corporate_actions ORDER BY ex_date"
            ).fetchall()
            self.assertEqual(
                [item[0] for item in rows],
                [date(2024, 1, 2), date(2024, 1, 4)],
            )
        finally:
            connection.close()


class SyncLockTest(unittest.TestCase):
    """同步锁与写锁降级。"""

    def test_lock_is_exclusive(self) -> None:
        """已经被占用的同步锁不允许再次获取。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with sync_lock(root):
                with self.assertRaises(DailyStoreLockedError):
                    with sync_lock(root):
                        pass

    def test_stale_lock_is_reclaimed(self) -> None:
        """超时的残留锁应当被夺回，避免异常退出后永久卡死。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".sync.lock").write_text("{}", encoding="utf-8")
            with sync_lock(root, timeout_seconds=0):
                pass
            self.assertFalse((root / ".sync.lock").exists())

    def test_auto_sync_degrades_when_store_is_locked(self) -> None:
        """库被其它进程写锁占用时应降级跳过，绝不让调用方崩溃。"""
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            _build_source(base / "qmt")
            database = base / "daily" / "qmt_daily.duckdb"
            database.parent.mkdir(parents=True, exist_ok=True)
            holder = duckdb.connect(str(database))
            try:
                report = sync_daily_store(
                    DailySyncConfig(
                        database=database, source_root=base / "qmt", mode="full"
                    )
                )
            finally:
                holder.close()
            self.assertEqual(report.status, "skipped_locked")


if __name__ == "__main__":
    unittest.main()
