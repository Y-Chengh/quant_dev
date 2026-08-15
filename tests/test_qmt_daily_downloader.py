# -*- coding: utf-8 -*-
"""大 QMT 下载器的接口替身、日分区和断点续传测试。"""

import json
import io
import logging
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from qmt_daily_downloader.config import DownloaderConfig, _resolve_incremental_lag_days
from qmt_daily_downloader.finance import materialize_finance_daily
from qmt_daily_downloader.gateway import FINANCE_FIELDS, QmtGateway
from qmt_daily_downloader.runner import QmtDailyDownloader, _make_job_key
from qmt_daily_downloader.state import CheckpointStore
from qmt_daily_downloader.storage import DailyPartitionStore
from qmt_daily_downloader.validation import (
    correct_kline_prices,
    find_missing_kline,
    validate_kline,
)


class FakeContext(object):
    """模拟大 QMT 内置 ContextInfo 的最小测试接口。

    子类通过覆盖 ``trading_dates`` 或 ``_kline_dates`` 配置缺失场景，
    共用同一份造帧逻辑、交易日 count 断言和请求参数记录。
    """

    trading_dates = ["20240102", "20240103"]

    def _kline_dates(self, code, start_date, end_date):
        """返回单只证券在请求区间内应返回的交易日；子类按场景覆盖。"""
        return [
            value for value in self.trading_dates if start_date <= value <= end_date
        ]

    def get_market_data_ex(self, fields, symbols, **kwargs):
        """按 ``_kline_dates`` 生成请求区间内的未复权日线。

        参数：
            fields: 请求的行情字段列表；测试替身不据此删列。
            symbols: 当前证券批次。
            **kwargs: 大 QMT 的周期、日期、复权和填充参数。

        返回：
            以证券代码为键的日线 ``DataFrame`` 字典。
        """
        self.last_market_kwargs = dict(kwargs)
        start_date = kwargs.get("start_time")
        end_date = kwargs.get("end_time")
        output = {}
        for offset, code in enumerate(symbols):
            dates = self._kline_dates(code, start_date, end_date)
            steps = [self.trading_dates.index(value) for value in dates]
            output[code] = pd.DataFrame(
                {
                    "open": [10.0 + offset + 0.5 * step for step in steps],
                    "high": [11.0 + offset + 0.5 * step for step in steps],
                    "low": [9.5 + offset + 0.5 * step for step in steps],
                    "close": [10.5 + offset + 0.5 * step for step in steps],
                    "preClose": [9.8 + offset + 0.7 * step for step in steps],
                    "volume": [1000.0 + 200.0 * step for step in steps],
                    "amount": [10200.0 + 2800.0 * step for step in steps],
                    "suspendFlag": [0.0] * len(dates),
                },
                index=dates,
            )
        return output

    def get_stock_list_in_sector(self, sector):
        """返回测试用沪深 A 股证券池。"""
        return ["000001.SZ", "600000.SH"]

    def get_instrument_detail(self, code):
        """返回测试股票的上市、退市和交易状态信息。"""
        return {
            "InstrumentName": code,
            "OpenDate": 20240101,
            "ExpireDate": 99999999,
            "IsTrading": True,
            "InstrumentStatus": 0,
        }

    def get_trading_dates(self, stockcode, start_date, end_date, count, period="1d"):
        """返回请求区间内的预期交易日。

        参数：
            stockcode: 交易日历基准代码。
            start_date: 查询起点。
            end_date: 查询终点。
            count: 最大返回数量，负数表示不限。
            period: 日历周期，测试要求为日线。

        返回：
            ``trading_dates`` 中位于请求区间内的八位交易日字符串。
        """
        if count <= 0:
            raise AssertionError("大 QMT 交易日 count 必须大于 0")
        return [
            value for value in self.trading_dates if start_date <= value <= end_date
        ]

    def get_raw_financial_data(
        self, fields, symbols, start_date, end_date, report_type="report_time"
    ):
        """返回一条在首个交易日公告的财务记录。

        参数：
            fields: 当前单张财务表的完整字段列表。
            symbols: 当前证券批次。
            start_date: 报告期查询起点。
            end_date: 报告期查询终点。
            report_type: 财务键口径，测试要求为 ``report_time``。

        返回：
            与大 QMT 原始财务接口相同的嵌套字典。
        """
        output = {}
        for code in symbols:
            output[code] = {}
            for position, field in enumerate(fields):
                if field.endswith(".m_timetag"):
                    value = 1703980800000
                elif field.endswith(".m_anntime"):
                    value = 1704153600000
                else:
                    value = 100.0 + position
                output[code][field] = {1703980800000: value}
        return output

    def get_divid_factors(self, code):
        """返回一条每股现金和送转测试记录。

        参数：
            code: 请求的证券代码。

        返回：
            以毫秒时间戳为键的七元素除权记录字典。
        """
        return {1704240000000: [0.1, 0.2, 0.3, 0.0, 0.0, 0, 1.6]}


class GapRepairContext(FakeContext):
    """模拟一只股票中间交易日缺失且可选择是否补回的行情接口。

    仅通过 ``_kline_dates`` 配置缺失场景并记录行情调用，造帧、交易日
    count 断言和 ``last_market_kwargs`` 记录全部复用基类。
    """

    trading_dates = ["20240102", "20240103", "20240104"]

    def __init__(self, persistent=False):
        """初始化调用记录和永久缺失开关。

        参数：
            persistent: 为 ``True`` 时补下载仍不返回缺失日，用于验证重试耗尽路径。
        """
        self.persistent = bool(persistent)
        self.market_calls = []

    def get_market_data_ex(self, fields, symbols, **kwargs):
        """记录每次行情请求后复用基类造帧逻辑。

        参数：
            fields: 请求的日线字段序列。
            symbols: 当前请求的证券代码序列。
            **kwargs: 大 QMT 的日期、周期、复权和填充参数。

        返回：
            按证券代码映射的日线 ``DataFrame`` 字典。
        """
        self.market_calls.append(
            (tuple(symbols), kwargs.get("start_time"), kwargs.get("end_time"))
        )
        return super(GapRepairContext, self).get_market_data_ex(
            fields, symbols, **kwargs
        )

    def _kline_dates(self, code, start_date, end_date):
        """首轮漏掉 000001.SZ 的中间日，定向补下载按开关决定是否补回。"""
        dates = super(GapRepairContext, self)._kline_dates(code, start_date, end_date)
        if code == "000001.SZ":
            if start_date == "20240102" and end_date == "20240104":
                return [value for value in dates if value != "20240103"]
            if self.persistent and start_date <= "20240103" <= end_date:
                return [value for value in dates if value != "20240103"]
        return dates


class RaisingContext(object):
    """确保断点重跑不会再次调用远端接口的失败替身。"""

    def get_market_data_ex(self, fields, symbols, **kwargs):
        """若断点失效则抛出异常。

        参数：
            fields: 请求字段。
            symbols: 请求证券。
            **kwargs: 其他大 QMT 查询参数。

        返回：
            本方法始终抛出 ``AssertionError``。
        """
        raise AssertionError("断点续传不应重新读取行情")

    def get_trading_dates(self, stockcode, start_date, end_date, count, period="1d"):
        """断点重跑仍允许读取交易日历。

        参数：
            stockcode: 交易日历基准代码。
            start_date: 查询起点。
            end_date: 查询终点。
            count: 最大返回数量。
            period: 日历周期。

        返回：
            与首次运行相同的两个交易日。
        """
        return ["20240102", "20240103"]

    def get_raw_financial_data(self, fields, symbols, start_date, end_date, report_type="report_time"):
        """若断点失效则抛出财务接口异常。

        参数：
            fields: 请求字段。
            symbols: 请求证券。
            start_date: 查询起点。
            end_date: 查询终点。
            report_type: 财务键口径。

        返回：
            本方法始终抛出 ``AssertionError``。
        """
        raise AssertionError("断点续传不应重新读取财务")

    def get_divid_factors(self, code):
        """若断点失效则抛出除权接口异常。

        参数：
            code: 请求证券代码。

        返回：
            本方法始终抛出 ``AssertionError``。
        """
        raise AssertionError("断点续传不应重新读取除权")


class EmptyFinanceContext(object):
    """模拟大 QMT 财务缓存尚未下载的空结果。"""

    def get_raw_financial_data(self, fields, symbols, start_date, end_date, report_type="report_time"):
        """对所有财务表返回空字典。

        参数：
            fields: 当前财务表请求字段。
            symbols: 当前批次证券代码。
            start_date: 报告期查询起点。
            end_date: 报告期查询终点。
            report_type: 财务键口径。

        返回：
            空字典，用于模拟客户端财务缓存缺失。
        """
        return {}

    def get_trading_dates(self, stockcode, start_date, end_date, count, period="1d"):
        """为仅财务测试提供预期交易日。

        参数：
            stockcode: 交易日历基准代码。
            start_date: 查询起点。
            end_date: 查询终点。
            count: 最大返回数量。
            period: 日历周期。

        返回：
            测试区间内的两个交易日。
        """
        return ["20240102", "20240103"]


class MissingMarketDayContext(FakeContext):
    """模拟整个股票池缺少一个预期交易日的行情缓存。"""

    def get_market_data_ex(self, fields, symbols, **kwargs):
        """只返回首个预期交易日。

        参数：
            fields: 请求行情字段。
            symbols: 当前证券批次。
            **kwargs: 其他大 QMT 查询参数。

        返回：
            删除第二个交易日后的行情字典。
        """
        output = super(MissingMarketDayContext, self).get_market_data_ex(
            fields, symbols, **kwargs
        )
        return {code: frame.iloc[:1].copy() for code, frame in output.items()}


class PartialFinanceContext(FakeContext):
    """模拟同一批次中只有第一只证券存在财务缓存。"""

    def get_raw_financial_data(
        self, fields, symbols, start_date, end_date, report_type="report_time"
    ):
        """仅为批次第一只证券返回财务记录。

        参数：
            fields: 当前财务表请求字段。
            symbols: 当前批次证券代码。
            start_date: 报告期查询起点。
            end_date: 报告期查询终点。
            report_type: 财务键口径。

        返回：
            缺少后续证券键的部分财务嵌套字典。
        """
        return super(PartialFinanceContext, self).get_raw_financial_data(
            fields, list(symbols)[:1], start_date, end_date, report_type
        )


class InvalidKlineContext(FakeContext):
    """模拟最高价低于收盘价的无效行情缓存。"""

    def get_market_data_ex(self, fields, symbols, **kwargs):
        """返回价格关系不成立的日线。

        参数：
            fields: 请求行情字段。
            symbols: 当前证券批次。
            **kwargs: 其他大 QMT 查询参数。

        返回：
            将最高价改为零后的行情字典。
        """
        output = super(InvalidKlineContext, self).get_market_data_ex(
            fields, symbols, **kwargs
        )
        for frame in output.values():
            frame.loc[:, "high"] = 0.0
        return output


class AllNoneFinanceContext(FakeContext):
    """模拟报告日期存在但所有财务业务字段均为空。"""

    def get_raw_financial_data(
        self, fields, symbols, start_date, end_date, report_type="report_time"
    ):
        """保留报告期和公告日并清空全部业务字段。

        参数：
            fields: 当前财务表请求字段。
            symbols: 当前批次证券代码。
            start_date: 报告期查询起点。
            end_date: 报告期查询终点。
            report_type: 财务键口径。

        返回：
            与真实接口同结构但业务值全为 ``None`` 的嵌套字典。
        """
        output = super(AllNoneFinanceContext, self).get_raw_financial_data(
            fields, symbols, start_date, end_date, report_type
        )
        for values_by_field in output.values():
            for field in fields[2:]:
                values_by_field[field] = {1703980800000: None}
        return output


class ChangedFinanceContext(FakeContext):
    """模拟上游财务缓存修订后的新业务数值。"""

    def get_raw_financial_data(
        self, fields, symbols, start_date, end_date, report_type="report_time"
    ):
        """把全部有效财务业务字段改为固定修订值。

        参数：
            fields: 当前财务表请求字段。
            symbols: 当前批次证券代码。
            start_date: 报告期查询起点。
            end_date: 报告期查询终点。
            report_type: 财务键口径。

        返回：
            与真实接口同结构且业务字段值为 ``999`` 的嵌套字典。
        """
        output = super(ChangedFinanceContext, self).get_raw_financial_data(
            fields, symbols, start_date, end_date, report_type
        )
        for values_by_field in output.values():
            for field in fields[2:]:
                values_by_field[field] = {1703980800000: 999.0}
        return output


class DownloaderTests(unittest.TestCase):
    """验证完整小区间保存流程和关键边界条件。"""

    def test_small_backfill_writes_daily_partitions_and_resumes(self):
        """两证券两交易日应落为独立分区且再次运行命中断点。"""
        with tempfile.TemporaryDirectory() as directory:
            config = _config(directory)
            download_calls = []
            first = _build_runner(
                config,
                FakeContext(),
                lambda code, period, start, end: download_calls.append(code),
            )
            summary = first.run()
            self.assertEqual(summary["symbols"], 2)
            self.assertEqual(summary["trade_dates"], 2)
            self.assertEqual(download_calls, ["000001.SZ", "600000.SH"])

            root = Path(directory)
            kline_path = root / "kline_1d" / "date=20240102" / "data.csv"
            finance_path = root / "finance_daily" / "date=20240103" / "data.csv"
            action_path = root / "corporate_actions" / "ex_date=20240103" / "data.csv"
            instrument_path = root / "instrument_info" / "snapshot=latest" / "data.csv"
            self.assertTrue(kline_path.is_file())
            self.assertTrue((kline_path.parent / "_SUCCESS.json").is_file())
            self.assertTrue(instrument_path.is_file())
            instrument_info = pd.read_csv(str(instrument_path), dtype={"code": str})
            self.assertEqual(instrument_info["code"].tolist(), ["000001.SZ", "600000.SH"])
            self.assertTrue(
                (root / "run_complete" / "date=20240103" / "_SUCCESS.json").is_file()
            )
            marker = json.loads(
                (root / "run_complete" / "date=20240103" / "_SUCCESS.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertIn(
                "finance_raw/table=income/announce_date=20240103",
                marker["required_partitions"],
            )
            self.assertEqual(len(pd.read_csv(str(kline_path))), 2)
            daily_finance = pd.read_csv(str(finance_path), dtype={"code": str})
            self.assertTrue((daily_finance["has_finance"] == 1).all())
            actions = pd.read_csv(str(action_path), dtype={"code": str})
            self.assertEqual(len(actions), 2)
            self.assertAlmostEqual(actions.iloc[0]["cash_dividend_per_share"], 0.1)

            second = _build_runner(
                config,
                RaisingContext(),
                lambda code, period, start, end: (_ for _ in ()).throw(
                    AssertionError("断点续传不应重新下载行情")
                ),
            )
            second_summary = second.run()
            self.assertEqual(second_summary["trade_dates"], 2)

    def test_finance_snapshot_excludes_same_day_announcement(self):
        """公告当天财报不得进入当天收盘快照，只能从下一交易日可见。"""
        raw = pd.DataFrame(
            [
                {
                    "code": "000001.SZ",
                    "report_date": "20231231",
                    "announce_date": "20240102",
                    "revenue": 123.0,
                }
            ]
        )
        result = materialize_finance_daily(
            {"income": raw}, ["20240102", "20240103"], ["000001.SZ"]
        )
        same_day = result[result["trade_date"] == "20240102"].iloc[0]
        next_day = result[result["trade_date"] == "20240103"].iloc[0]
        self.assertEqual(same_day["has_finance"], 0)
        self.assertEqual(next_day["has_finance"], 1)
        self.assertEqual(next_day["income_revenue"], 123.0)

    def test_config_rejects_reverse_date_range(self):
        """结束日早于开始日时应拒绝启动，避免写错分区。"""
        values = _config_values("D:\\unused")
        values["start_date"] = "20240103"
        values["end_date"] = "20240102"
        with self.assertRaises(ValueError):
            DownloaderConfig(values)

    def test_incremental_auto_lag_uses_16_clock_boundary(self):
        """增量自动滞后应在边界前后分别使用 1 天和 0 天，且边界可配置。"""
        from datetime import datetime

        self.assertEqual(
            _resolve_incremental_lag_days("auto", datetime(2026, 8, 15, 15, 59)), 1
        )
        self.assertEqual(
            _resolve_incremental_lag_days("auto", datetime(2026, 8, 15, 16, 0)), 0
        )
        self.assertEqual(
            _resolve_incremental_lag_days(
                "auto", datetime(2026, 8, 15, 16, 59), "17:30"
            ),
            1,
        )
        self.assertEqual(
            _resolve_incremental_lag_days(
                "auto", datetime(2026, 8, 15, 17, 30), "17:30"
            ),
            0,
        )
        self.assertEqual(_resolve_incremental_lag_days(1), 1)
        with self.assertRaises(ValueError):
            _resolve_incremental_lag_days("auto", datetime(2026, 8, 15, 16, 0), "25:00")

    def test_config_rejects_empty_datasets(self):
        """空数据集配置不得生成没有任何业务文件的虚假成功任务。"""
        values = _config_values("D:\\unused")
        values["datasets"] = []
        with self.assertRaises(ValueError):
            DownloaderConfig(values)

    def test_config_rejects_invalid_kline_gap_retry_count(self):
        """缺口补下载次数必须为有限正整数，避免静默关闭或无限循环。"""
        with tempfile.TemporaryDirectory() as directory:
            values = _config_values(directory)
            values["kline_gap_retry_count"] = 0
            with self.assertRaises(ValueError):
                DownloaderConfig(values)

    def test_checkpoint_key_changes_with_batch_and_finance_policy(self):
        """批大小或财务完整性策略变化后不得复用旧批次 staging。"""
        first_values = _config_values("D:\\unused")
        second_values = dict(first_values)
        second_values["batch_size"] = 2
        third_values = dict(first_values)
        third_values["allow_partial_finance"] = True
        symbols = ["000001.SZ", "600000.SH"]
        keys = {
            _make_job_key(DownloaderConfig(first_values), symbols),
            _make_job_key(DownloaderConfig(second_values), symbols),
            _make_job_key(DownloaderConfig(third_values), symbols),
        }
        self.assertEqual(len(keys), 3)

    def test_missing_kline_is_reported_per_symbol_without_cross_talk(self):
        """某证券缺日线时只应提示该证券，不得串联到其他证券状态。"""
        frame = pd.DataFrame(
            [
                {"code": "000001.SZ", "trade_date": "20240102"},
                {"code": "600000.SH", "trade_date": "20240102"},
                {"code": "600000.SH", "trade_date": "20240103"},
            ]
        )
        issues = find_missing_kline(
            frame, ["000001.SZ", "600000.SH"], ["20240102", "20240103"]
        )
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["code"], "000001.SZ")
        self.assertEqual(issues[0]["date"], "20240103")

    def test_kline_requests_fill_data_and_suspension_is_not_missing(self):
        """日线请求应启用填充，停牌标记行不应生成缺口警告。"""
        context = FakeContext()
        logger = logging.getLogger("qmt_fill_data_test_{0}".format(id(context)))
        logger.handlers = [logging.NullHandler()]
        logger.propagate = False
        gateway = QmtGateway(context, lambda *args: None, logger, retry_count=1)
        frame, issues = gateway.fetch_kline(
            ["000001.SZ"], "20240102", "20240103", download_first=False
        )
        self.assertEqual(issues, [])
        self.assertTrue(context.last_market_kwargs["fill_data"])
        suspended = frame.copy()
        suspended.loc[suspended["trade_date"] == "20240103", "suspend_flag"] = 1
        missing = find_missing_kline(
            suspended, ["000001.SZ"], ["20240102", "20240103"]
        )
        self.assertEqual(missing, [])

    def test_instrument_info_filters_pre_listing_kline_warning(self):
        """上市前日期的 K 线缺失提示应被生命周期信息过滤。"""
        runner = object.__new__(QmtDailyDownloader)
        runner.logger = logging.getLogger("qmt_lifecycle_filter_test")
        runner.logger.handlers = [logging.NullHandler()]
        runner.logger.propagate = False
        runner.issues = type("Issues", (object,), {})()
        runner.issues.items = [
            {
                "level": "WARNING",
                "dataset": "kline_1d",
                "code": "000001.SZ",
                "date": "20231229",
                "message": "填充后仍无日线；可能为未上市、退市或本地缓存缺失",
            },
            {
                "level": "WARNING",
                "dataset": "kline_1d",
                "code": "000001.SZ",
                "date": "20240102",
                "message": "填充后仍无日线；可能为未上市、退市或本地缓存缺失",
            },
        ]
        info = pd.DataFrame(
            [
                {
                    "code": "000001.SZ",
                    "open_date": "20240101",
                    "expire_date": None,
                }
            ]
        )
        runner._filter_kline_issues(info)
        self.assertEqual(len(runner.issues.items), 1)
        self.assertEqual(runner.issues.items[0]["date"], "20240102")

    def test_instrument_info_symbols_include_previous_day_live_snapshot_codes(self):
        """instrument_info 查询应合并当前证券池与前一日仍存续的历史代码。"""
        with tempfile.TemporaryDirectory() as directory:
            values = _config_values(directory)
            values["end_date"] = "20240103"
            config = DownloaderConfig(values)
            runner = _build_runner(config, FakeContext(), lambda *args: None)
            runner.symbols = ["999999.SZ"]
            store = runner.store
            previous = pd.DataFrame(
                [
                    {"code": "000001.SZ", "expire_date": ""},
                    {"code": "002000.SZ", "expire_date": ""},
                    {"code": "600000.SH", "expire_date": "20240102"},
                    {"code": "300000.SZ", "expire_date": "20240101"},
                ]
            )
            store.write_partition(
                "instrument_info",
                "snapshot",
                "latest",
                previous,
                ["code", "expire_date"],
                ["code"],
                ["code"],
                {},
                overwrite=True,
            )
            self.assertEqual(
                runner._instrument_info_symbols(),
                ["000001.SZ", "002000.SZ", "600000.SH", "999999.SZ"],
            )

    def test_empty_finance_cache_keeps_batch_resumable(self):
        """财务缓存全空时批次不得标为完成，以便用户补数据后原配置续传。"""
        with tempfile.TemporaryDirectory() as directory:
            values = _config_values(directory)
            values["symbols"] = ["000001.SZ"]
            values["datasets"] = ["finance_raw"]
            config = DownloaderConfig(values)
            runner = _build_runner(config, EmptyFinanceContext(), lambda *args: None)
            summary = runner.run()
            state = CheckpointStore(
                Path(directory) / "state" / "downloader_state.sqlite"
            )
            self.assertGreater(summary["errors"], 0)
            self.assertFalse(state.is_completed(summary["job_key"], "finance_raw", 0))

    def test_incremental_auto_starts_after_latest_completed_partition(self):
        """自动增量应从最近完成日线的下一自然日继续且不修改历史分区。"""
        with tempfile.TemporaryDirectory() as directory:
            values = _config_values(directory)
            scope = {
                "datasets": sorted(values["datasets"]),
                "symbols": sorted(values["symbols"]),
                "sector": "",
            }
            store = DailyPartitionStore(directory)
            for dataset, partition_name in (
                ("kline_1d", "date"),
                ("finance_daily", "date"),
                ("corporate_actions", "ex_date"),
            ):
                store.write_partition(
                    dataset,
                    partition_name,
                    "20240102",
                    pd.DataFrame(columns=["code"]),
                    ["code"],
                    [],
                    ["code"],
                    {},
                )
            store.write_run_date_complete(
                "20240102",
                [
                    "kline_1d/date=20240102",
                    "finance_daily/date=20240102",
                    "corporate_actions/ex_date=20240102",
                ],
                {"watermark_scope": scope},
            )
            values["mode"] = "incremental"
            values["start_date"] = "auto"
            values["end_date"] = "20240105"
            values["incremental_initial_start_date"] = "20200101"
            config = DownloaderConfig(values)
            self.assertEqual(config.start_date, "20240103")
            self.assertEqual(config.end_date, "20240105")
            self.assertFalse(config.no_work)

    def test_whole_market_missing_expected_day_remains_resumable(self):
        """交易日历存在但全市场日线缺失时不得完成批次或推进分区。"""
        with tempfile.TemporaryDirectory() as directory:
            values = _config_values(directory)
            values["datasets"] = ["kline_1d"]
            config = DownloaderConfig(values)
            runner = _build_runner(
                config, MissingMarketDayContext(), lambda *args: None
            )
            summary = runner.run()
            state = CheckpointStore(
                Path(directory) / "state" / "downloader_state.sqlite"
            )
            self.assertGreater(summary["errors"], 0)
            self.assertFalse(state.is_completed(summary["job_key"], "kline_1d", 0))
            self.assertFalse((Path(directory) / "kline_1d" / "date=20240102").exists())

    def test_partial_finance_does_not_advance_global_watermark(self):
        """日线成功但批内财务部分缺失时整日水位必须保持未完成。"""
        with tempfile.TemporaryDirectory() as directory:
            values = _config_values(directory)
            values["batch_size"] = 2
            config = DownloaderConfig(values)
            runner = _build_runner(config, PartialFinanceContext(), lambda *args: None)
            summary = runner.run()
            state = CheckpointStore(
                Path(directory) / "state" / "downloader_state.sqlite"
            )
            self.assertGreater(summary["errors"], 0)
            self.assertTrue((Path(directory) / "kline_1d" / "date=20240102" / "_SUCCESS.json").is_file())
            self.assertFalse(state.is_completed(summary["job_key"], "finance_raw", 0))
            self.assertFalse((Path(directory) / "run_complete" / "date=20240102").exists())

    def test_fragment_dates_are_normalized_after_mixed_missing_values(self):
        """公告日有效值与空值混合时重读后仍应保持八位日期而非浮点文本。"""
        with tempfile.TemporaryDirectory() as directory:
            store = DailyPartitionStore(directory)
            frame = pd.DataFrame(
                [
                    {"code": "000001.SZ", "announce_date": "20240102"},
                    {"code": "600000.SH", "announce_date": None},
                ]
            )
            store.write_fragment("job", "finance_income", 0, frame)
            result = store.read_fragments("job", "finance_income", [0])
            self.assertEqual(result.iloc[0]["announce_date"], "20240102")
            self.assertTrue(pd.isna(result.iloc[1]["announce_date"]))

    def test_corrupt_partition_is_not_skipped(self):
        """数据文件被篡改后完成校验应失败，下一次写入必须主动修复。"""
        with tempfile.TemporaryDirectory() as directory:
            store = DailyPartitionStore(directory)
            frame = pd.DataFrame([{"code": "000001.SZ", "trade_date": "20240102"}])
            store.write_partition(
                "kline_1d", "date", "20240102", frame, list(frame.columns),
                ["code", "trade_date"], ["code"], {},
            )
            data_path = Path(directory) / "kline_1d" / "date=20240102" / "data.csv"
            data_path.write_text("broken", encoding="utf-8")
            self.assertFalse(store.is_partition_complete("kline_1d", "date", "20240102"))
            result = store.write_partition(
                "kline_1d", "date", "20240102", frame, list(frame.columns),
                ["code", "trade_date"], ["code"], {}, overwrite=False,
            )
            self.assertEqual(result["status"], "written")
            self.assertTrue(store.is_partition_complete("kline_1d", "date", "20240102"))

    def test_partition_scope_change_requires_explicit_repair(self):
        """相同日期扩大证券池时不得静默跳过旧分区或伪造新范围水位。"""
        with tempfile.TemporaryDirectory() as directory:
            store = DailyPartitionStore(directory)
            frame = pd.DataFrame([{"code": "000001.SZ", "trade_date": "20240102"}])
            common = (
                "kline_1d", "date", "20240102", frame, list(frame.columns),
                ["code", "trade_date"], ["code"],
            )
            store.write_partition(
                *common, metadata={"partition_scope": {"symbols": ["000001.SZ"]}}
            )
            with self.assertRaises(ValueError):
                store.write_partition(
                    *common,
                    metadata={
                        "partition_scope": {"symbols": ["000001.SZ", "600000.SH"]}
                    }
                )

    def test_same_scope_key_change_rewrites_completed_partition(self):
        """范围一致但主键集合变化时应原地重写；范围不同时必须仍然报错。"""
        with tempfile.TemporaryDirectory() as directory:
            store = DailyPartitionStore(directory)
            scope = {"symbols": ["000001.SZ", "600000.SH"]}
            first = pd.DataFrame([{"code": "000001.SZ", "trade_date": "20240102"}])
            store.write_partition(
                "kline_1d", "date", "20240102", first, list(first.columns),
                ["code", "trade_date"], ["code"], {"partition_scope": scope},
            )
            repaired = pd.DataFrame(
                [
                    {"code": "000001.SZ", "trade_date": "20240102"},
                    {"code": "600000.SH", "trade_date": "20240102"},
                ]
            )
            result = store.write_partition(
                "kline_1d", "date", "20240102", repaired, list(repaired.columns),
                ["code", "trade_date"], ["code"], {"partition_scope": scope},
            )
            self.assertEqual(result["status"], "written")
            result = store.write_partition(
                "kline_1d", "date", "20240102", repaired, list(repaired.columns),
                ["code", "trade_date"], ["code"], {"partition_scope": scope},
            )
            self.assertEqual(result["status"], "skipped")
            narrower = pd.DataFrame([{"code": "300001.SZ", "trade_date": "20240102"}])
            with self.assertRaises(ValueError):
                store.write_partition(
                    "kline_1d", "date", "20240102", narrower, list(narrower.columns),
                    ["code", "trade_date"], ["code"],
                    {"partition_scope": {"symbols": ["300001.SZ"]}},
                )

    def test_legacy_partition_without_identity_digest_still_compares(self):
        """旧版本写出的分区没有主键摘要时应回落到读取 CSV 比较，不误判为不一致。"""
        with tempfile.TemporaryDirectory() as directory:
            store = DailyPartitionStore(directory)
            frame = pd.DataFrame(
                [
                    {"code": "000001.SZ", "trade_date": "20240102"},
                    {"code": "600000.SH", "trade_date": "20240102"},
                ]
            )
            store.write_partition(
                "kline_1d", "date", "20240102", frame, list(frame.columns),
                ["code", "trade_date"], ["code"], {},
            )
            success_path = (
                Path(directory) / "kline_1d" / "date=20240102" / "_SUCCESS.json"
            )
            metadata = json.loads(success_path.read_text(encoding="utf-8"))
            self.assertIsNotNone(metadata.pop("identity_sha256"))
            success_path.write_text(
                json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
            )
            self.assertTrue(
                store.partition_identity_matches(
                    "kline_1d", "date", "20240102", frame, ["code", "trade_date"]
                )
            )
            changed = frame.replace({"600000.SH": "600001.SH"})
            self.assertFalse(
                store.partition_identity_matches(
                    "kline_1d", "date", "20240102", changed, ["code", "trade_date"]
                )
            )

    def test_failed_gap_repair_keeps_previous_final_partition_valid(self):
        """补下载失败不得使先前完整的最终日线分区失去完成标记。"""
        with tempfile.TemporaryDirectory() as directory:
            values = _config_values(directory)
            values["end_date"] = "20240104"
            values["batch_size"] = 2
            values["datasets"] = ["kline_1d"]
            config = DownloaderConfig(values)
            first = _build_runner(config, GapRepairContext(), lambda *args: None)
            self.assertEqual(first.run()["errors"], 0)
            store = first.store
            self.assertTrue(store.is_partition_complete("kline_1d", "date", "20240103"))
            # 删除 staging 断点，迫使第二次运行重新下载并遭遇永久缺口。
            shutil.rmtree(Path(directory) / "staging")
            summary = _build_runner(
                config, GapRepairContext(persistent=True), lambda *args: None
            ).run()
            self.assertGreater(summary["errors"], 0)
            self.assertTrue(store.is_partition_complete("kline_1d", "date", "20240103"))

    def test_corrupt_staging_fragment_is_not_resumable(self):
        """staging CSV 被截断后必须使断点失效并触发重新下载。"""
        with tempfile.TemporaryDirectory() as directory:
            store = DailyPartitionStore(directory)
            store.write_fragment(
                "job", "kline_1d", 0,
                pd.DataFrame([{"code": "000001.SZ", "trade_date": "20240102"}]),
            )
            path = Path(directory) / "staging" / "job" / "kline_1d" / "batch_00000.csv"
            path.write_text("code,trade_date\n", encoding="utf-8")
            self.assertFalse(store.fragment_exists("job", "kline_1d", 0))

    def test_read_fragments_supports_legacy_pandas_concat(self):
        """合并 staging 时不得向旧版 pandas.concat 传入 sort 参数。"""
        with tempfile.TemporaryDirectory() as directory:
            store = DailyPartitionStore(directory)
            for batch_id, code in enumerate(("000001.SZ", "600000.SH")):
                store.write_fragment(
                    "job",
                    "kline_1d_date_20240102",
                    batch_id,
                    pd.DataFrame([{"code": code, "trade_date": "20240102"}]),
                )

            real_concat = pd.concat

            def legacy_concat(frames, ignore_index=False, **kwargs):
                """模拟不接受 ``sort`` 参数的大 QMT 旧版 pandas。

                参数：
                    frames: 待拼接的批次数据表序列。
                    ignore_index: 是否重建连续行索引。
                    **kwargs: 调用方传入的其他关键字参数；出现 ``sort`` 即模拟旧版报错。

                返回：
                    由当前 pandas 生成的拼接结果。
                """
                if "sort" in kwargs:
                    raise TypeError("concat() got an unexpected keyword argument 'sort'")
                return real_concat(frames, ignore_index=ignore_index, **kwargs)

            with patch(
                "qmt_daily_downloader.storage.pd.concat",
                side_effect=legacy_concat,
            ):
                result = store.read_fragments(
                    "job", "kline_1d_date_20240102", range(2)
                )
            self.assertEqual(result["code"].tolist(), ["000001.SZ", "600000.SH"])

    def test_invalid_kline_high_corrected_and_recorded(self):
        """价格关系错误应被行级修正、批次照常完成并写入 line_correct 报告。"""
        with tempfile.TemporaryDirectory() as directory:
            values = _config_values(directory)
            values["symbols"] = ["000001.SZ"]
            values["datasets"] = ["kline_1d"]
            config = DownloaderConfig(values)
            runner = _build_runner(config, InvalidKlineContext(), lambda *args: None)
            summary = runner.run()
            state = CheckpointStore(Path(directory) / "state" / "downloader_state.sqlite")
            self.assertEqual(summary["errors"], 0)
            self.assertEqual(summary["corrections"], 2)
            self.assertTrue(state.is_completed(summary["job_key"], "kline_1d", 0))
            partition = pd.read_csv(
                Path(directory) / "kline_1d" / "date=20240102" / "data.csv"
            )
            self.assertAlmostEqual(float(partition["high"].iloc[0]), 10.5)
            report = pd.read_csv(summary["line_correct_report"])
            self.assertEqual(len(report), 2)
            self.assertEqual(set(report["field"]), {"high"})
            self.assertEqual(set(report["code"]), {"000001.SZ"})
            self.assertTrue(
                (Path(directory) / "run_complete" / "date=20240103" / "_SUCCESS.json").is_file()
            )

    def test_correct_kline_prices_fixes_rows_and_keeps_original(self):
        """修正函数应抬高低于开收低的最高价、压低高于开收高的最低价且不改原表。"""
        frame = pd.DataFrame(
            {
                "code": ["600690.SH", "600807.SH", "000001.SZ"],
                "trade_date": ["19940404", "19940404", "19940404"],
                "open": [10.0, 10.0, None],
                "high": [9.0, 12.0, 11.0],
                "low": [9.5, 11.5, 9.0],
                "close": [10.5, 11.0, 10.0],
            }
        )
        corrected, records, issues = correct_kline_prices(frame)
        self.assertAlmostEqual(float(corrected["high"].iloc[0]), 10.5)
        self.assertAlmostEqual(float(corrected["low"].iloc[1]), 10.0)
        self.assertAlmostEqual(float(corrected["high"].iloc[2]), 11.0)
        self.assertAlmostEqual(float(frame["high"].iloc[0]), 9.0)
        self.assertEqual(
            [(item["code"], item["field"]) for item in records],
            [("600690.SH", "high"), ("600807.SH", "low")],
        )
        self.assertEqual({item["level"] for item in issues}, {"WARNING"})
        remaining_errors = [
            item for item in validate_kline(corrected) if item["level"] == "ERROR"
        ]
        self.assertEqual(remaining_errors, [])

    def test_all_none_finance_fields_keep_checkpoint_failed(self):
        """财务记录存在但业务字段全空时应提示并保持批次可重试。"""
        with tempfile.TemporaryDirectory() as directory:
            values = _config_values(directory)
            values["symbols"] = ["000001.SZ"]
            values["datasets"] = ["finance_raw"]
            config = DownloaderConfig(values)
            runner = _build_runner(config, AllNoneFinanceContext(), lambda *args: None)
            summary = runner.run()
            state = CheckpointStore(Path(directory) / "state" / "downloader_state.sqlite")
            self.assertGreater(summary["errors"], 0)
            self.assertFalse(state.is_completed(summary["job_key"], "finance_raw", 0))

    def test_repair_overwrites_raw_and_daily_finance(self):
        """repair 模式应使用新缓存值覆盖原始财务与日财务快照。"""
        with tempfile.TemporaryDirectory() as directory:
            first_config = _config(directory)
            _build_runner(first_config, FakeContext(), lambda *args: None).run()

            values = _config_values(directory)
            values["mode"] = "repair"
            values["overwrite_completed_partition"] = True
            repair_config = DownloaderConfig(values)
            summary = _build_runner(
                repair_config, ChangedFinanceContext(), lambda *args: None
            ).run()
            self.assertEqual(summary["errors"], 0)

            root = Path(directory)
            raw = pd.read_csv(
                str(
                    root
                    / "finance_raw"
                    / "table=income"
                    / "announce_date=20240102"
                    / "data.csv"
                )
            )
            daily = pd.read_csv(
                str(root / "finance_daily" / "date=20240103" / "data.csv")
            )
            self.assertTrue((raw["revenue"] == 999.0).all())
            self.assertTrue((daily["income_revenue"] == 999.0).all())

    def test_qmt_entry_source_is_ascii_safe(self):
        """确保由大 QMT 编辑器直接载入的入口源码不受 GBK/UTF-8 转码影响。"""
        entry_path = Path(__file__).resolve().parents[1] / "qmt_run_downloader.py"
        source = entry_path.read_bytes()
        self.assertEqual(source.decode("ascii").encode("ascii"), source)

    def test_sales_gross_profit_is_not_requested(self):
        """确保不请求对银行股通常为空的销售毛利率字段。"""
        self.assertNotIn(
            "PERSHAREINDEX.sales_gross_profit",
            FINANCE_FIELDS["pershareindex"],
        )

    def test_progress_log_contains_batch_and_date_percentages(self):
        """进度日志应包含批次、日期序号及百分比，便于终端观察长任务。"""
        stream = io.StringIO()
        logger = logging.getLogger("qmt_progress_test_{0}".format(id(stream)))
        logger.handlers = []
        logger.propagate = False
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(stream)
        logger.addHandler(handler)
        try:
            runner = object.__new__(QmtDailyDownloader)
            runner.logger = logger
            runner.batches = [["000001.SZ"], ["600000.SH"], ["300001.SZ"]]
            runner.symbols = ["000001.SZ", "600000.SH", "300001.SZ"]
            runner.config = type(
                "Config",
                (object,),
                {"batch_size": 1, "start_date": "20260810", "end_date": "20260812"},
            )()
            runner._log_batch_progress("kline_1d", 1, runner.batches[1], "完成")
            runner._log_date_progress("finance_daily", 1, 3, "20260811", "完成")
            output = stream.getvalue()
        finally:
            logger.removeHandler(handler)
            handler.close()
        self.assertIn("批次 2/3 (66.7%)", output)
        self.assertIn("证券序号 2-2/3", output)
        self.assertIn("日期 2/3 (66.7%) date=20260811", output)

    def test_parallel_save_writes_all_kline_fragments(self):
        """启用多个保存线程时，所有日期 staging 仍应完整落盘。"""
        with tempfile.TemporaryDirectory() as directory:
            values = _config_values(directory)
            values["datasets"] = ["kline_1d", "corporate_actions"]
            values["save_workers"] = 2
            config = DownloaderConfig(values)
            summary = _build_runner(config, FakeContext(), lambda *args: None).run()
            self.assertEqual(summary["errors"], 0)
            root = Path(directory)
            for trade_date in ("20240102", "20240103"):
                self.assertTrue(
                    (root / "kline_1d" / ("date=" + trade_date) / "data.csv").is_file()
                )
                self.assertTrue(
                    (root / "corporate_actions" / ("ex_date=" + trade_date) / "data.csv").is_file()
                )

    def test_kline_error_log_contains_stage_and_request_context(self):
        """行情接口异常日志应包含阶段、证券、日期和参数定位信息。"""
        class ErrorContext(object):
            """模拟大 QMT 日线读取失败。"""

            def get_market_data_ex(self, fields, symbols, **kwargs):
                """抛出可定位的测试异常。"""
                raise RuntimeError("模拟行情服务断开")

        stream = io.StringIO()
        logger = logging.getLogger("qmt_error_detail_test_{0}".format(id(stream)))
        logger.handlers = []
        logger.propagate = False
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(stream)
        logger.addHandler(handler)
        try:
            gateway = QmtGateway(ErrorContext(), lambda *args: None, logger, retry_count=1)
            _, issues = gateway.fetch_kline(
                ["000001.SZ"], "20240102", "20240103", download_first=False
            )
            output = stream.getvalue()
        finally:
            logger.removeHandler(handler)
            handler.close()
        self.assertIn("stage=get_market_data_ex", issues[0]["message"])
        self.assertIn("symbols=000001.SZ", issues[0]["message"])
        self.assertIn("start=20240102", issues[0]["message"])
        self.assertIn("error_type=RuntimeError", output)
        self.assertIn("模拟行情服务断开", output)

    def test_parallel_save_error_log_contains_target_label(self):
        """本地文件写入失败时日志应直接包含目标日期或路径标签。"""
        stream = io.StringIO()
        logger = logging.getLogger("qmt_parallel_error_test_{0}".format(id(stream)))
        logger.handlers = []
        logger.propagate = False
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(stream)
        logger.addHandler(handler)

        class Config(object):
            """提供并行保存线程数的最小配置。"""

            save_workers = 2

        runner = object.__new__(QmtDailyDownloader)
        runner.config = Config()
        runner.logger = logger

        def failing_task():
            """模拟指定分区写入失败。"""
            raise OSError("磁盘写入失败")

        try:
            with self.assertRaises(RuntimeError):
                runner._parallel_save([failing_task], "kline_daily_final", ["20240103"])
            output = stream.getvalue()
        finally:
            logger.removeHandler(handler)
            handler.close()
        self.assertIn("label=20240103", output)
        self.assertIn("error_type=OSError", output)
        self.assertIn("磁盘写入失败", output)

    def test_internal_kline_gap_is_redownloaded_and_merged(self):
        """首尾行情之间的缺口应定向补下载并写回对应日分区。"""
        with tempfile.TemporaryDirectory() as directory:
            values = _config_values(directory)
            values["end_date"] = "20240104"
            values["batch_size"] = 2
            values["datasets"] = ["kline_1d"]
            values["kline_gap_retry_count"] = 2
            context = GapRepairContext()
            summary = _build_runner(
                DownloaderConfig(values), context, lambda *args: None
            ).run()
            self.assertEqual(summary["errors"], 0)
            self.assertIn(
                (("000001.SZ",), "20240103", "20240103"),
                context.market_calls,
            )
            repaired = pd.read_csv(
                str(
                    Path(directory)
                    / "kline_1d"
                    / "date=20240103"
                    / "data.csv"
                ),
                encoding="utf-8-sig",
                dtype={"code": str},
            )
            self.assertEqual(
                repaired["code"].tolist(), ["000001.SZ", "600000.SH"]
            )
            self.assertFalse(repaired.duplicated(["code", "trade_date"]).any())

    def test_internal_gap_ranges_are_grouped_without_edge_dates(self):
        """连续缺口应合并、分离缺口应拆分，首尾观测之外日期不得进入请求。"""
        dates = [
            "20240101",
            "20240102",
            "20240103",
            "20240104",
            "20240105",
            "20240106",
            "20240107",
            "20240108",
        ]
        frame = pd.DataFrame(
            {
                "code": ["000001.SZ"] * 4,
                "trade_date": ["20240102", "20240105", "20240107", "20240108"],
            }
        )
        gaps = QmtDailyDownloader._find_internal_kline_gaps(
            frame, ["000001.SZ"], dates
        )
        self.assertEqual(
            gaps["000001.SZ"],
            [
                ("20240103", "20240104", ("20240103", "20240104")),
                ("20240106", "20240106", ("20240106",)),
            ],
        )

    def test_completed_checkpoint_replaces_stale_final_kline_partition(self):
        """补齐 staging 后若旧最终分区仍完整标记，重启也必须按主键差异覆盖。"""
        with tempfile.TemporaryDirectory() as directory:
            values = _config_values(directory)
            values["end_date"] = "20240104"
            values["batch_size"] = 2
            values["datasets"] = ["kline_1d"]
            config = DownloaderConfig(values)
            first_runner = _build_runner(
                config, GapRepairContext(), lambda *args: None
            )
            first_summary = first_runner.run()
            self.assertEqual(first_summary["errors"], 0)
            data_path = (
                Path(directory) / "kline_1d" / "date=20240103" / "data.csv"
            )
            complete = pd.read_csv(
                str(data_path), encoding="utf-8-sig", dtype={"code": str}
            )
            stale = complete[complete["code"] == "600000.SH"]
            first_runner.store.write_partition(
                "kline_1d",
                "date",
                "20240103",
                stale,
                list(complete.columns),
                ["code", "trade_date"],
                ["code"],
                {
                    "job_key": first_runner.job_key,
                    "mode": config.mode,
                    "partition_scope": first_runner.partition_scope,
                },
                overwrite=True,
            )
            second_context = GapRepairContext()
            second_summary = _build_runner(
                config, second_context, lambda *args: None
            ).run()
            self.assertEqual(second_summary["errors"], 0)
            self.assertEqual(second_context.market_calls, [])
            restored = pd.read_csv(
                str(data_path), encoding="utf-8-sig", dtype={"code": str}
            )
            self.assertEqual(
                restored["code"].tolist(), ["000001.SZ", "600000.SH"]
            )

    def test_resolved_gap_keeps_transient_download_error_as_warning(self):
        """补下载首次报错后恢复时应成功完成，并在问题报告保留降级警告。"""
        with tempfile.TemporaryDirectory() as directory:
            values = _config_values(directory)
            values["end_date"] = "20240104"
            values["batch_size"] = 2
            values["datasets"] = ["kline_1d"]
            attempts = []

            def flaky_history_downloader(code, period, start_date, end_date):
                """仅让缺口日期的第一次历史缓存下载失败。

                参数：
                    code: 当前下载的证券代码。
                    period: 行情周期，测试中固定为日线。
                    start_date: 当前下载区间起始日。
                    end_date: 当前下载区间结束日。

                返回：
                    成功时无返回值；第一次定向请求抛出 ``RuntimeError``。
                """
                if code == "000001.SZ" and start_date == end_date == "20240103":
                    attempts.append((code, start_date))
                    if len(attempts) == 1:
                        raise RuntimeError("模拟缺口缓存下载失败")

            summary = _build_runner(
                DownloaderConfig(values),
                GapRepairContext(),
                flaky_history_downloader,
            ).run()
            self.assertEqual(summary["errors"], 0)
            self.assertGreater(summary["warnings"], 0)
            issues = pd.read_csv(summary["issue_report"], encoding="utf-8-sig")
            warnings = issues[
                issues["message"]
                .astype(str)
                .str.contains("接口曾报错但缺口最终已补齐")
            ]
            self.assertEqual(len(warnings), 1)
            self.assertEqual(warnings.iloc[0]["level"], "WARNING")

    def test_unresolved_internal_kline_gap_keeps_batch_failed(self):
        """补下载达到上限仍缺失时不得生成最终 K 线分区或推进完成状态。"""
        with tempfile.TemporaryDirectory() as directory:
            values = _config_values(directory)
            values["end_date"] = "20240104"
            values["batch_size"] = 2
            values["datasets"] = ["kline_1d"]
            values["kline_gap_retry_count"] = 2
            context = GapRepairContext(persistent=True)
            summary = _build_runner(
                DownloaderConfig(values), context, lambda *args: None
            ).run()
            self.assertGreater(summary["errors"], 0)
            targeted = [
                call
                for call in context.market_calls
                if call == (("000001.SZ",), "20240103", "20240103")
            ]
            self.assertEqual(len(targeted), 2)
            self.assertFalse(
                (
                    Path(directory)
                    / "kline_1d"
                    / "date=20240103"
                    / "data.csv"
                ).exists()
            )
            issues = pd.read_csv(
                summary["issue_report"],
                encoding="utf-8-sig",
                dtype={"code": str, "date": str},
            )
            unresolved = issues[
                issues["message"].astype(str).str.contains("补下载 2 次后仍缺失")
            ]
            self.assertEqual(len(unresolved), 1)
            self.assertEqual(str(unresolved.iloc[0]["code"]), "000001.SZ")
            self.assertEqual(str(unresolved.iloc[0]["date"]), "20240103")


def _config(directory):
    """构造使用临时输出目录的测试配置。

    参数：
        directory: ``TemporaryDirectory`` 提供的独立输出目录。

    返回：
        已通过校验的 ``DownloaderConfig``。
    """
    return DownloaderConfig(_config_values(directory))


def _config_values(directory):
    """生成小日期区间的配置字典。

    参数：
        directory: 测试输出根目录。

    返回：
        两只证券、两个交易日、每批一只的配置字典。
    """
    return {
        "output_root": directory,
        "mode": "backfill",
        "start_date": "20240102",
        "end_date": "20240103",
        "symbols": ["000001.SZ", "600000.SH"],
        "batch_size": 1,
        "retry_count": 1,
        "download_kline": True,
        "datasets": ["kline_1d", "finance_raw", "finance_daily", "corporate_actions"],
        "finance_lookback_start": "20230101",
    }


def _build_runner(config, context, history_downloader):
    """组装使用测试替身的下载器。

    参数：
        config: 测试下载配置。
        context: 大 QMT ``ContextInfo`` 替身。
        history_downloader: 历史行情下载函数替身。

    返回：
        可直接执行的 ``QmtDailyDownloader``。
    """
    logger = logging.getLogger("qmt_daily_downloader_test_{0}".format(id(context)))
    logger.handlers = [logging.NullHandler()]
    logger.propagate = False
    store = DailyPartitionStore(config.output_root)
    state = CheckpointStore(config.output_root / "state" / "downloader_state.sqlite")
    gateway = QmtGateway(context, history_downloader, logger, retry_count=1)
    return QmtDailyDownloader(config, gateway, store, state, logger)


if __name__ == "__main__":
    unittest.main()
