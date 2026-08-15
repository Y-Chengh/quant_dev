# -*- coding: utf-8 -*-
"""运行器的编排层：建立共享状态并按固定顺序驱动各职责 mixin。"""

import json
import time
from datetime import datetime

from ..gateway import FINANCE_FIELDS
from ..storage import IssueCollector
from .corporate_actions import _CorporateActionMixin
from .finance_steps import _FinanceCollectionMixin
from .helpers import (
    _format_elapsed,
    _iter_batches,
    _lifecycle_windows,
    _make_job_key,
)
from .instruments import _InstrumentInfoMixin
from .issues import _IssueHandlingMixin
from .kline import _KlineCollectionMixin
from .partitions import _PartitionWriteMixin
from .progress import _ProgressLoggingMixin
from .trading_calendar import _TradingCalendarMixin


class QmtDailyDownloader(
    _InstrumentInfoMixin,
    _IssueHandlingMixin,
    _ProgressLoggingMixin,
    _TradingCalendarMixin,
    _KlineCollectionMixin,
    _FinanceCollectionMixin,
    _CorporateActionMixin,
    _PartitionWriteMixin,
):
    """执行一次可恢复的大 QMT 日频数据保存任务。"""

    def __init__(self, config, gateway, store, checkpoints, logger):
        """注入任务所需的配置和基础设施。

        参数：
            config: 已完成校验的 ``DownloaderConfig``。
            gateway: 封装大 QMT 内置接口的 ``QmtGateway``。
            store: 管理日分区和 staging 文件的 ``DailyPartitionStore``。
            checkpoints: 仅保存批次状态的 ``CheckpointStore``。
            logger: 同时输出到终端和滚动文件的日志对象。
        """
        self.config = config
        self.gateway = gateway
        self.store = store
        self.checkpoints = checkpoints
        self.logger = logger
        self.issues = IssueCollector()
        self.line_corrections = []
        self.symbols = []
        self.batches = []
        self.job_key = ""
        self.partition_scope = {}
        self.started_at = None

    def _with_elapsed(self, summary):
        """把本次运行的总耗时写入摘要字典。

        耗时从 ``run`` 入口开始计时，覆盖证券池解析、下载、校验和分区落盘的全过程；
        无论任务成功、无业务下载还是已是最新，返回的摘要都带有耗时字段。

        参数：
            summary: 已填好业务字段的摘要字典；本方法原地补充耗时后返回同一对象。

        返回：
            追加了 ``elapsed_seconds``（浮点秒，保留三位小数）和 ``elapsed``
            （``H:MM:SS`` 文本）的摘要字典。
        """
        seconds = 0.0 if self.started_at is None else time.time() - self.started_at
        summary["elapsed_seconds"] = round(seconds, 3)
        summary["elapsed"] = _format_elapsed(seconds)
        return summary

    def run(self):
        """依次执行证券池解析、数据抽取、质量检查和日分区保存。

        返回：
            本次运行摘要字典，包含任务标识、证券数、交易日数、问题数、报告路径和
            总耗时。
        """
        # 计时起点放在最前，让证券池解析和交易日历请求也计入总耗时。
        self.started_at = time.time()
        self.symbols = self.gateway.resolve_symbols(
            self.config.symbols, self.config.sector, self.config.expired_sectors
        )
        self.partition_scope = {
            "schema_version": 2,
            # 只登记业务数据集：交易日历不改变任何业务分区的内容，若纳入范围，仅仅
            # 打开落表就会让已完成分区的范围核验失败并拒绝写入。
            "datasets": sorted(self.config.business_datasets),
            "symbols": list(self.symbols),
            "finance_lookback_start": self.config.finance_lookback_start,
            "finance_fields": {
                name: list(fields) for name, fields in sorted(FINANCE_FIELDS.items())
            },
        }
        self.batches = list(_iter_batches(self.symbols, self.config.batch_size))
        self.job_key = _make_job_key(self.config, self.symbols)
        run_id = "{0}_{1}".format(
            datetime.now().strftime("%Y%m%d_%H%M%S"), self.job_key[-10:]
        )
        self.logger.info(
            "任务开始 mode=%s dates=%s..%s symbols=%d batches=%d job=%s",
            self.config.mode,
            self.config.start_date,
            self.config.end_date,
            len(self.symbols),
            len(self.batches),
            self.job_key,
        )
        if self.config.mode == "incremental" and self.config.incremental_lag_days_auto:
            self.logger.info(
                "增量滞后天数自动判定 now=%s cutoff=%s lag_days=%d target_date=%s",
                self.config.incremental_lag_decision_time.strftime("%Y-%m-%d %H:%M:%S %Z"),
                self.config.incremental_lag_auto_cutoff,
                self.config.incremental_lag_days,
                self.config.end_date,
            )
        if self.config.no_work:
            report_path = self.store.write_issue_report([], run_id)
            summary = {
                "job_key": self.job_key,
                "symbols": len(self.symbols),
                "trade_dates": 0,
                "errors": 0,
                "warnings": 0,
                "issue_report": str(report_path),
                "status": "up_to_date",
            }
            self._with_elapsed(summary)
            self.logger.info(
                "增量数据已是最新，无需下载 用时=%s summary=%s",
                summary["elapsed"],
                json.dumps(summary, ensure_ascii=False),
            )
            return summary

        trade_dates, calendar_issues = self.gateway.fetch_trading_dates(
            self.config.calendar_symbol,
            self.config.start_date,
            self.config.end_date,
        )
        self.issues.extend(calendar_issues)
        if calendar_issues:
            return self._finish_without_download(run_id, "calendar_error")
        if not trade_dates:
            return self._finish_without_download(run_id, "no_trading_dates")

        all_datasets_succeeded = True
        if self.config.save_trading_calendar:
            # 交易日历只依赖上面已经取得的 trade_dates，放在业务数据之前落表：即使随后
            # 的日线或财务失败，自检也已经拿得到本次区间的权威日历。落表失败按所选数据
            # 集失败处理，不推进整日水位，重跑时业务分区会因范围一致而快速跳过。
            try:
                self._write_trading_calendar(trade_dates)
            except Exception as error:
                all_datasets_succeeded = False
                self.issues.add("ERROR", "trading_calendar", "", "", str(error))
                self.logger.exception("交易日历落表失败")
        if "kline_1d" in self.config.datasets or "finance_daily" in self.config.datasets:
            lifecycle = {}
            instrument_info = None
            instrument_failed = False
            if "kline_1d" in self.config.datasets:
                # 生命周期信息必须先于日线收集取得：逐日缺口检查据此在生成阶段就跳过
                # 未上市和已退市的证券日。全区间回溯下这类证券日占三成以上，若等到收集
                # 结束后再过滤，中间会堆积上千万条随即被丢弃的提示。
                instrument_info, instrument_failed = self._collect_instrument_info()
                if instrument_failed:
                    all_datasets_succeeded = False
                    self.issues.add(
                        "ERROR",
                        "instrument_info",
                        "",
                        "",
                        "上市退市信息获取失败，无法可靠过滤日线缺失提示",
                    )
                    self.logger.error(
                        "缺少上市退市信息，跳过本次日线收集以免产生无法过滤的缺失提示；"
                        "修复后重新运行会从断点继续"
                    )
                else:
                    lifecycle = _lifecycle_windows(instrument_info)
            if not instrument_failed:
                failed = self._collect_kline(trade_dates, lifecycle)
                if failed:
                    all_datasets_succeeded = False
                    self.issues.add("ERROR", "kline_1d", "", "", "存在失败批次，未生成最终日线分区；下次以相同配置运行会从断点继续")
                if instrument_info is not None:
                    # 生成阶段已按存续期过滤，这里作为兜底再扫一遍，正常情况下不会命中。
                    self._filter_kline_issues(instrument_info)

        if "finance_raw" in self.config.datasets or "finance_daily" in self.config.datasets:
            failed = self._collect_finance()
            if failed:
                all_datasets_succeeded = False
                self.issues.add("ERROR", "finance_raw", "", "", "存在失败批次，未生成最终财务分区；下次会从断点继续")
            else:
                try:
                    raw_dates = self._prepare_finance_fragments(trade_dates)
                    if "finance_raw" in self.config.datasets:
                        self._write_finance_raw_partitions(raw_dates)
                    if "finance_daily" in self.config.datasets:
                        self._write_finance_daily_partitions(trade_dates)
                except Exception as error:
                    all_datasets_succeeded = False
                    self.issues.add("ERROR", "finance", "", "", str(error))
                    self.logger.exception("财务分区流式生成失败")

        if "corporate_actions" in self.config.datasets:
            actions, failed = self._collect_actions()
            if failed:
                all_datasets_succeeded = False
                self.issues.add("ERROR", "corporate_actions", "", "", "存在失败批次，未生成最终除权分区；下次会从断点继续")
            else:
                self._write_action_partitions(actions, trade_dates)

        report_path = self.store.write_issue_report(self.issues.items, run_id)
        correction_report_path = self.store.write_line_correction_report(
            self.line_corrections, run_id
        )
        error_count = sum(1 for item in self.issues.items if item["level"] == "ERROR")
        warning_count = sum(1 for item in self.issues.items if item["level"] == "WARNING")
        if all_datasets_succeeded and error_count == 0:
            self._write_run_completion(trade_dates)
        summary = {
            "job_key": self.job_key,
            "symbols": len(self.symbols),
            "trade_dates": len(trade_dates),
            "errors": error_count,
            "warnings": warning_count,
            "corrections": len(self.line_corrections),
            "issue_report": str(report_path),
            "line_correct_report": str(correction_report_path),
        }
        self._with_elapsed(summary)
        self.logger.info(
            "任务结束 用时=%s summary=%s",
            summary["elapsed"],
            json.dumps(summary, ensure_ascii=False),
        )
        return summary
