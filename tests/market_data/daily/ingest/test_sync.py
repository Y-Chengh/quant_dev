"""端到端验证日线库的增量入库：过滤、幂等、按月重写与锁降级。

夹具一律通过下载器真实的 ``DailyPartitionStore`` 写出，因此 ``_SUCCESS.json``
里的行数与 SHA-256 都是真的，增量检查走的是与生产完全一致的路径。
"""

from __future__ import annotations

import random
import shutil
import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import duckdb
import pandas as pd

from quant.market_data.daily import AdjustMode, DailyMarketClient
from quant.market_data.daily.ingest import DailySyncConfig, aux_tables, sync_daily_store
from quant.market_data.daily.ingest.filter_reasons import (
    FILTER_REASON_INFO,
    FILTER_REASONS,
    FILTERED_SAMPLE_LIMIT,
    FilteredSample,
    describe_reason,
    format_filter_summary,
    sort_by_reason,
)
from quant.market_data.daily.ingest.locking import DailyStoreLockedError, sync_lock
from quant.market_data.daily.ingest.shards import (
    ShardResult,
    rewrite_month_shard,
    shard_directory,
)
from quant.market_data.daily.ingest.sync import _pick_filtered_samples
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


class FilterReasonOrderingTest(unittest.TestCase):
    """过滤原因的排序与跨月样例合并这两段纯逻辑。"""

    def _sample(self, reason: str, code: str) -> FilteredSample:
        """构造一条只填了必要字段的样例。

        参数：
            reason: 该样例命中的过滤原因。
            code: 证券代码，同时用作排序断言的抓手。

        返回：
            其余字段一律为空的 ``FilteredSample``。
        """
        return FilteredSample(
            reason=reason, code=code, trade_date="20240102", open_date=None,
            expire_date=None, suspend_flag=None, volume=None, close=None,
        )

    def test_sort_by_reason_follows_case_branch_order(self) -> None:
        """排序必须跟 SQL 判定顺序一致，未知原因兜底排在最后。"""
        ordered = sort_by_reason([
            ("zzz_unknown", 1),
            ("after_delisting", 2),
            ("aaa_unknown", 3),
            ("trade_date_null", 4),
        ])
        self.assertEqual(
            [reason for reason, _ in ordered],
            ["trade_date_null", "after_delisting", "aaa_unknown", "zzz_unknown"],
        )
        # 全量原因按原顺序进出，保证排序本身不会打乱既定顺序。
        self.assertEqual(
            [reason for reason, _ in sort_by_reason(
                (reason, 0) for reason in reversed(FILTER_REASONS))],
            list(FILTER_REASONS),
        )

    def test_cross_month_samples_are_capped_and_sorted(self) -> None:
        """跨月合并后条数要封顶，且只保留本原因的样例并按代码排序。"""
        pool = {
            "before_listing_padding": [
                self._sample("before_listing_padding", f"{600000 + index}.SH")
                for index in range(FILTERED_SAMPLE_LIMIT * 3)
            ],
            "unknown_code": [self._sample("unknown_code", "999999.SZ")],
            "after_delisting": [],
        }
        picked = dict(_pick_filtered_samples(pool))
        # 空列表的原因不应出现，否则摘要会打印一个没有内容的小标题。
        self.assertEqual(set(picked), {"before_listing_padding", "unknown_code"})
        before = picked["before_listing_padding"]
        self.assertEqual(len(before), FILTERED_SAMPLE_LIMIT)
        self.assertEqual(len(set(sample.code for sample in before)), FILTERED_SAMPLE_LIMIT)
        self.assertTrue(all(sample.reason == "before_listing_padding" for sample in before))
        self.assertEqual([sample.code for sample in before],
                         sorted(sample.code for sample in before))
        self.assertEqual(len(picked["unknown_code"]), 1)

    def test_empty_pool_yields_no_samples(self) -> None:
        """一行都没过滤时不应产出任何样例。"""
        self.assertEqual(_pick_filtered_samples({}), ())

    def test_sampling_does_not_disturb_global_random_state(self) -> None:
        """抽样例必须走自带随机源，不能消耗全局 RNG。

        同一进程里可能正在跑按固定种子复现的遗传搜索，入库顺手抽几条样例就把
        全局随机流推进一格的话，复现结果会莫名其妙地变。
        """
        pool = {
            "before_listing_padding": [
                self._sample("before_listing_padding", f"{600000 + index}.SH")
                for index in range(FILTERED_SAMPLE_LIMIT * 3)
            ]
        }
        # 固定种子只是让本用例自身可复现，跑完必须还原，免得把后续用例的全局
        # 随机流也钉死。
        original = random.getstate()
        self.addCleanup(random.setstate, original)
        random.seed(20240102)
        state = random.getstate()
        _pick_filtered_samples(pool)
        self.assertEqual(random.getstate(), state)


class FilterSummaryFormatTest(unittest.TestCase):
    """过滤汇总的渲染：说明是否齐全、结论是否正确。"""

    def _sample(self, reason: str, code: str) -> FilteredSample:
        """构造一条只填了必要字段的样例。

        参数：
            reason: 该样例命中的过滤原因。
            code: 证券代码。

        返回：
            其余字段一律为空的 ``FilteredSample``。
        """
        return FilteredSample(
            reason=reason, code=code, trade_date="20240102", open_date=None,
            expire_date=None, suspend_flag=None, volume=None, close=None,
        )

    def test_every_reason_has_registered_explanation(self) -> None:
        """七个原因都必须登记说明，不能靠兜底文案糊弄过去。"""
        self.assertEqual(set(FILTER_REASON_INFO), set(FILTER_REASONS))
        for reason in FILTER_REASONS:
            info = describe_reason(reason)
            self.assertTrue(info.condition.strip())
            self.assertTrue(info.meaning.strip())
            self.assertTrue(info.action.strip())
        # 只有占位填充行是「非 0 也正常」的，其余六个正常都应为 0。
        self.assertEqual(
            {reason for reason in FILTER_REASONS if not describe_reason(reason).needs_review},
            {"before_listing_padding"},
        )

    def test_unknown_reason_falls_back_instead_of_raising(self) -> None:
        """漏登记说明时要走兜底，不能让整段日志输出崩掉。"""
        info = describe_reason("brand_new_reason")
        self.assertTrue(info.needs_review)
        self.assertIn("没有登记判定说明", info.condition)
        lines = format_filter_summary((("brand_new_reason", 3),))
        self.assertIn("  brand_new_reason: 3 行", lines)

    def test_summary_carries_condition_meaning_and_action(self) -> None:
        """每个原因下面都要带判定、含义、处理三行，0 行的也不例外。"""
        totals = tuple((reason, 0) for reason in FILTER_REASONS)
        lines = format_filter_summary(totals)
        for reason in FILTER_REASONS:
            info = describe_reason(reason)
            self.assertIn(f"  {reason}: 0 行", lines)
            self.assertIn(f"    判定: {info.condition}", lines)
            self.assertIn(f"    含义: {info.meaning}", lines)
            self.assertIn(f"    处理: {info.action}", lines)

    def test_verdict_says_expected_when_only_padding_hit(self) -> None:
        """只命中占位填充时结论应是「无需人工核对」。"""
        totals = tuple(
            (reason, 17252623 if reason == "before_listing_padding" else 0)
            for reason in FILTER_REASONS
        )
        self.assertEqual(
            format_filter_summary(totals)[-1], "本次过滤全部落在预期原因内，无需人工核对。"
        )

    def test_verdict_names_reasons_needing_review(self) -> None:
        """命中需核对的原因时，结论要点名是哪几个、各多少行。"""
        totals = tuple(
            (reason, {"before_listing_padding": 100, "unknown_code": 7,
                      "before_listing_with_data": 3}.get(reason, 0))
            for reason in FILTER_REASONS
        )
        verdict = format_filter_summary(totals)[-1]
        self.assertEqual(verdict, "需人工核对: unknown_code(7 行)、before_listing_with_data(3 行)")
        # 占位填充行数再多也不该被点名。
        self.assertNotIn("before_listing_padding", verdict)

    def test_samples_are_nested_under_their_reason(self) -> None:
        """样例要挂在对应原因的说明之后，而不是全堆在末尾。"""
        totals = (("unknown_code", 2), ("after_delisting", 0))
        samples = (("unknown_code", (self._sample("unknown_code", "999999.SZ"),)),)
        lines = format_filter_summary(totals, samples)
        self.assertLess(lines.index("  unknown_code: 2 行"), lines.index("    随机样例 1 条:"))
        self.assertLess(lines.index("    随机样例 1 条:"), lines.index("  after_delisting: 0 行"))

    def test_empty_totals_render_nothing(self) -> None:
        """没跑过过滤时整节都不输出，免得空标题误导。"""
        self.assertEqual(format_filter_summary(()), ())


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
            # 600000.SH 上市前的 0102 一行（suspend_flag=1，占位）、
            # 300001.SZ 退市后的 0104 一行。
            self.assertEqual(totals.get("before_listing_padding"), 1)
            self.assertEqual(totals.get("after_delisting"), 1)
            # 没发生的原因也要显式报 0，否则读日志的人分不清「查过了」和「没查」。
            self.assertEqual(totals.get("trade_date_null"), 0)
            self.assertEqual(totals.get("trade_date_bad_length"), 0)
            self.assertEqual(totals.get("trade_date_unparsable"), 0)
            self.assertEqual(totals.get("unknown_code"), 0)
            self.assertEqual(totals.get("before_listing_with_data"), 0)
            # 原因顺序必须跟着 FILTER_REASONS 走，而不是字母序。
            self.assertEqual(
                [reason for reason, _ in report.filtered_totals], list(FILTER_REASONS)
            )

    def test_up_to_date_sync_reports_no_filter_totals(self) -> None:
        """一个分片都没重写时不能报「全 0」，那会谎称查过。"""
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            _build_source(base / "qmt")
            sync_daily_store(self._config(base))
            again = sync_daily_store(self._config(base))
            self.assertEqual(again.rewritten_shards, 0)
            self.assertEqual(again.filtered_totals, ())
            self.assertEqual(again.filtered_samples, ())

    def test_non_kline_delta_reports_no_filter_totals(self) -> None:
        """只有非日线数据集变脏时走完整入库但不重写分片，同样不能报「全 0」。

        这条路径会真正进到 ``_apply_delta``，与 ``up_to_date`` 的提前返回不同；
        守住 gate 用的是哪个计数则由 ``test_removed_month_is_not_reported_as_all_zero``
        负责。
        """
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "qmt"
            _build_source(source)
            sync_daily_store(self._config(base))
            # 只动除权分区：touched_months() 只看 kline_1d，因此不会重写任何分片。
            store = DailyPartitionStore(source)
            store.write_partition(
                "corporate_actions", "ex_date", "20240104",
                pd.DataFrame([{
                    "code": "000001.SZ", "ex_date": "20240104",
                    "cash_dividend_per_share": 0.25, "bonus_share_per_share": 0.0,
                    "capitalization_per_share": 0.0, "rights_issue_per_share": 0.0,
                    "rights_issue_price": 0.0, "share_reform_flag": 0.0,
                    "adjustment_factor": 1.01,
                }], columns=ACTION_COLUMNS),
                ACTION_COLUMNS, ["code", "ex_date"], ["code"], _SCOPE, overwrite=True,
            )
            report = sync_daily_store(self._config(base))
            self.assertEqual(report.status, "synced")
            self.assertEqual(report.rewritten_shards, 0)
            self.assertEqual(report.filtered_totals, ())
            self.assertEqual(report.filtered_samples, ())

    def test_removed_month_is_not_reported_as_all_zero(self) -> None:
        """整月源分区被删光只会删分片，没读过一行源数据，不能报「全 0」。"""
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "qmt"
            _build_source(source)
            sync_daily_store(self._config(base))
            for partition in sorted((source / "kline_1d").glob("date=2024*")):
                shutil.rmtree(partition)
            report = sync_daily_store(self._config(base, mode="full"))
            self.assertEqual(report.status, "synced")
            self.assertEqual(report.rewritten_shards, 1)
            # 分片确实被删了，但一行源数据都没读过，所以没有可汇报的过滤结果。
            self.assertEqual(report.filtered_totals, ())
            self.assertEqual(report.filtered_samples, ())

    def test_sync_report_carries_concrete_filtered_samples(self) -> None:
        """同步报告要带回每个原因的具体样例行，字段足以判断该不该滤掉。"""
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            _build_source(base / "qmt")
            report = sync_daily_store(self._config(base))
            samples = dict(report.filtered_samples)
            # 计数为 0 的原因不该带样例，否则摘要会打印空的样例小标题。
            self.assertEqual(set(samples), {"before_listing_padding", "after_delisting"})

            before = samples["before_listing_padding"]
            self.assertEqual(len(before), 1)
            self.assertEqual(before[0].code, "600000.SH")
            self.assertEqual(before[0].trade_date, "20240102")
            self.assertEqual(before[0].open_date, "2024-01-03")
            # 19700427 是 QMT 的「无退市日」哨兵，样例里应显示为缺失。
            self.assertIsNone(before[0].expire_date)
            self.assertEqual(
                before[0].describe(),
                "600000.SH 20240102 open=2024-01-03 expire=- suspend=1 volume=1000 close=7.0000",
            )

            after = samples["after_delisting"]
            self.assertEqual(len(after), 1)
            self.assertEqual(after[0].code, "300001.SZ")
            self.assertEqual(after[0].trade_date, "20240104")
            self.assertEqual(after[0].expire_date, "2024-01-03")

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
            self.assertEqual(breakdown.get("before_listing_padding"), 1)
            self.assertEqual(breakdown.get("after_delisting"), 1)
            self.assertEqual(result.rows, 7)
            # 分片级明细刻意保持稀疏：没发生的原因不出现，免得 320 行日志被刷屏。
            self.assertNotIn("unknown_code", breakdown)

    def test_before_listing_splits_padding_from_real_quotes(self) -> None:
        """上市前的行要按 suspend_flag 分成占位填充与真实行情两类。

        口径与 ``quant.qmt_downloader.self_check`` 的 ``DATA_BEFORE_LISTING``
        一致：``suspend_flag=1`` 是 QMT 的占位填充，无害；不等于 1 说明上市日
        或行情归属有问题，必须能单独看到。
        """
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "qmt"
            partition = source / "kline_1d" / "date=20240102"
            partition.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([
                # 上市前的停牌占位行：全零、suspend_flag=1。
                _bar("000001.SZ", "20240102", 0.0, 0.0, suspend=1.0, volume=0.0),
                # 上市前却有真实成交，suspend_flag=0——这才是要报出来的那类。
                _bar("600000.SH", "20240102", 7.0, 6.9, suspend=0.0, volume=1000.0),
                # suspend_flag 缺失按 0 处理，同样算真实行情，不能悄悄归进占位。
                dict(_bar("300001.SZ", "20240102", 5.0, 5.0), suspend_flag=None),
            ]).to_csv(partition / "data.csv", index=False, columns=KLINE_COLUMNS)
            lifecycle = pd.DataFrame([
                {"code": code, "open_date": pd.Timestamp("2024-06-01"), "expire_date": pd.NaT}
                for code in ("000001.SZ", "600000.SH", "300001.SZ")
            ])
            connection = duckdb.connect()
            try:
                result = rewrite_month_shard(connection, source, base / "daily", 2024, 1,
                                             lifecycle)
            finally:
                connection.close()
            breakdown = dict(result.filtered)
            self.assertEqual(breakdown.get("before_listing_padding"), 1)
            self.assertEqual(breakdown.get("before_listing_with_data"), 2)
            samples = dict(result.filtered_samples)
            self.assertEqual(samples["before_listing_padding"][0].code, "000001.SZ")
            self.assertEqual(
                [sample.code for sample in samples["before_listing_with_data"]],
                ["300001.SZ", "600000.SH"],
            )

    def _run_bad_trade_date_shard(self, base: Path) -> ShardResult:
        """写出一份含三种坏 trade_date 与一个未知代码的源分区并重建分片。

        参数：
            base: 临时根目录，其下建立 ``qmt`` 源目录与 ``daily`` 库目录。

        返回：
            ``rewrite_month_shard`` 的 ``ShardResult``。
        """
        source = base / "qmt"
        daily_root = base / "daily"
        partition = source / "kline_1d" / "date=20240102"
        partition.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([
            _bar("000001.SZ", "20240102", 10.0, 9.5),
            # 空字段被 read_csv 读成 NULL，对应 trade_date_null。
            _bar("000001.SZ", "", 10.0, 9.5),
            # 七位数字，长度不足八位，对应 trade_date_bad_length。
            _bar("000001.SZ", "2024010", 10.0, 9.5),
            # 八位但不是合法日期，对应 trade_date_unparsable。
            _bar("000001.SZ", "abcdefgh", 10.0, 9.5),
            _bar("999999.SZ", "20240102", 1.0, 1.0),
        ]).to_csv(partition / "data.csv", index=False, columns=KLINE_COLUMNS)
        lifecycle = pd.DataFrame([
            {"code": "000001.SZ", "open_date": pd.Timestamp("2024-01-01"),
             "expire_date": pd.NaT},
        ])
        connection = duckdb.connect()
        try:
            return rewrite_month_shard(connection, source, daily_root, 2024, 1, lifecycle)
        finally:
            connection.close()

    def test_shard_result_splits_trade_date_failures_by_kind(self) -> None:
        """trade_date 的三种坏法必须各自成为独立原因，不再混成一个总类。"""
        with tempfile.TemporaryDirectory() as directory:
            result = self._run_bad_trade_date_shard(Path(directory))
            breakdown = dict(result.filtered)
            self.assertEqual(breakdown.get("trade_date_null"), 1)
            self.assertEqual(breakdown.get("trade_date_bad_length"), 1)
            self.assertEqual(breakdown.get("trade_date_unparsable"), 1)
            self.assertEqual(breakdown.get("unknown_code"), 1)
            self.assertNotIn("invalid_trade_date", breakdown)
            self.assertEqual(result.rows, 1)
            # 判定顺序即展示顺序：坏日期三类在前，代码与生命周期在后。
            self.assertEqual(
                [reason for reason, _ in result.filtered],
                [
                    "trade_date_null",
                    "trade_date_bad_length",
                    "trade_date_unparsable",
                    "unknown_code",
                ],
            )

    def test_shard_samples_keep_raw_trade_date_text(self) -> None:
        """坏 trade_date 的样例要保留源文本，否则无从判断到底坏在哪。"""
        with tempfile.TemporaryDirectory() as directory:
            result = self._run_bad_trade_date_shard(Path(directory))
            samples = dict(result.filtered_samples)
            self.assertIsNone(samples["trade_date_null"][0].trade_date)
            self.assertEqual(samples["trade_date_bad_length"][0].trade_date, "2024010")
            self.assertEqual(samples["trade_date_unparsable"][0].trade_date, "abcdefgh")
            # 生命周期表里没有这个代码，上市与退市日都应为空。
            unknown = samples["unknown_code"][0]
            self.assertEqual(unknown.code, "999999.SZ")
            self.assertIsNone(unknown.open_date)
            self.assertIsNone(unknown.expire_date)
            self.assertEqual(
                unknown.describe(),
                "999999.SZ 20240102 open=- expire=- suspend=0 volume=1000 close=1.0000",
            )

    def test_shard_samples_are_capped_per_reason(self) -> None:
        """同一原因丢弃行很多时，样例条数必须封顶，且都来自该原因。"""
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "qmt"
            partition = source / "kline_1d" / "date=20240102"
            partition.mkdir(parents=True, exist_ok=True)
            # 刻意避开 000001.SZ，让这批代码全部落进 unknown_code。
            codes = [f"{900000 + index:06d}.SZ" for index in range(FILTERED_SAMPLE_LIMIT + 5)]
            pd.DataFrame(
                [_bar("000001.SZ", "20240102", 10.0, 9.5)]
                + [_bar(code, "20240102", 1.0, 1.0) for code in codes]
            ).to_csv(partition / "data.csv", index=False, columns=KLINE_COLUMNS)
            lifecycle = pd.DataFrame([
                {"code": "000001.SZ", "open_date": pd.Timestamp("2024-01-01"),
                 "expire_date": pd.NaT},
            ])
            connection = duckdb.connect()
            try:
                result = rewrite_month_shard(connection, source, base / "daily", 2024, 1,
                                             lifecycle)
            finally:
                connection.close()
            samples = dict(result.filtered_samples)
            self.assertEqual(dict(result.filtered)["unknown_code"], len(codes))
            self.assertEqual(len(samples["unknown_code"]), FILTERED_SAMPLE_LIMIT)
            self.assertTrue(
                {sample.code for sample in samples["unknown_code"]} <= set(codes)
            )
            self.assertTrue(
                all(sample.reason == "unknown_code" for sample in samples["unknown_code"])
            )

    def test_fully_filtered_month_still_reports_reasons(self) -> None:
        """整月被过滤光时分片会被删除，但过滤明细必须保留下来。"""
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "qmt"
            partition = source / "kline_1d" / "date=20240102"
            partition.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([
                _bar("000001.SZ", "20240102", 10.0, 9.5),
            ]).to_csv(partition / "data.csv", index=False, columns=KLINE_COLUMNS)
            lifecycle = pd.DataFrame([
                {"code": "000001.SZ", "open_date": pd.Timestamp("2024-06-01"),
                 "expire_date": pd.NaT},
            ])
            connection = duckdb.connect()
            try:
                result = rewrite_month_shard(connection, source, base / "daily", 2024, 1,
                                             lifecycle)
            finally:
                connection.close()
            self.assertTrue(result.removed)
            self.assertEqual(result.rows, 0)
            self.assertTrue(result.filter_applied)
            self.assertEqual(dict(result.filtered), {"before_listing_with_data": 1})
            self.assertEqual(
                dict(result.filtered_samples)["before_listing_with_data"][0].open_date,
                "2024-06-01",
            )

    def test_month_without_source_reports_no_filtering(self) -> None:
        """源目录里压根没有该月分区时不算「过滤了 0 行」，应返回空明细。"""
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            lifecycle = pd.DataFrame([
                {"code": "000001.SZ", "open_date": pd.Timestamp("2024-01-01"),
                 "expire_date": pd.NaT},
            ])
            connection = duckdb.connect()
            try:
                result = rewrite_month_shard(connection, base / "qmt", base / "daily",
                                             2024, 1, lifecycle)
            finally:
                connection.close()
            self.assertTrue(result.removed)
            self.assertEqual(result.filtered, ())
            self.assertEqual(result.filtered_samples, ())
            # 空元组只说明「没丢弃」，filter_applied 才区分「没查」和「查过是 0」。
            self.assertFalse(result.filter_applied)

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
