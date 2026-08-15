# -*- coding: utf-8 -*-
"""自检器的编排层：建立共享状态并按固定顺序驱动各职责 mixin。"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from .discovery import _PartitionDiscoveryMixin
from .loading import _ReferenceLoadingMixin
from .models import ISSUE_COLUMNS, AuditIssue, AuditResult, SelfCheckConfig
from .progress import _ProgressLoggingMixin
from .scanning import _CalendarScanMixin
from .summary import _SummaryReportMixin
from .validation import _RowValidationMixin

#: 自检进度日志器名称；库层只取用不配置处理器，由命令行入口决定输出目的地。
LOGGER_NAME = "quant.qmt_downloader.self_check"


class QmtDataSelfChecker(
    _ProgressLoggingMixin,
    _PartitionDiscoveryMixin,
    _ReferenceLoadingMixin,
    _CalendarScanMixin,
    _RowValidationMixin,
    _SummaryReportMixin,
):
    """扫描 QMT 日分区并生成不修改原始数据的详细质量报告。"""

    def __init__(self, config: SelfCheckConfig):
        """初始化自检器及空问题集合。

        参数：
            config: 数据根目录、审计日期范围、交易日历和异常阈值配置。
        """

        self.config = config
        self.root = config.output_root.resolve()
        self.data_root = self.root
        self.source_mode = "partitions"
        self.issues: list[AuditIssue] = []
        self.partition_paths: dict[str, Path] = {}
        self.reference_scope: dict[str, Any] | None = None
        self._lifecycle_issue_keys: set[tuple[str, str]] = set()
        self.staging_daily_gap_dates: list[tuple[str, list[str]]] = []
        self.logger = logging.getLogger(LOGGER_NAME)
        self._run_started_at = time.monotonic()
        self._phase_name = "初始化"
        self._phase_started_at = self._run_started_at
        self._step_stage = ""
        self._step_started_at = self._run_started_at

    def run(self) -> AuditResult:
        """执行全样本自检并写出 Markdown、JSON 与 CSV 报告。

        返回：
            包含统计摘要、全部问题、缺失区间、覆盖率和报告目录的结果。
        """

        if not self.root.is_dir():
            raise FileNotFoundError("QMT 数据根目录不存在: {0}".format(self.root))
        self._run_started_at = time.monotonic()
        self.logger.info(
            "[自检] 开始 output_root=%s start_date=%s end_date=%s",
            self.root,
            self.config.start_date or "全部",
            self.config.end_date or "全部",
        )
        self._begin_phase("发现日线分区")
        self._discover_partitions()
        self._end_phase(
            "来源={0} 分区={1} 个".format(self.source_mode, len(self.partition_paths))
        )
        self._begin_phase("装载证券信息与证券池")
        instrument = self._load_instrument_info()
        symbols = self._resolve_symbols(instrument)
        self._end_phase(
            "证券快照 {0} 行，审计证券池 {1} 只".format(len(instrument), len(symbols))
        )
        self._begin_phase("装载交易日历")
        calendar, authoritative_calendar = self._load_calendar()
        self._end_phase(
            "交易日 {0} 个 区间 {1}..{2} 权威日历={3}".format(
                len(calendar),
                calendar[0] if calendar else "-",
                calendar[-1] if calendar else "-",
                authoritative_calendar,
            )
        )
        if not calendar:
            self.logger.warning("[自检] 审计区间内没有交易日，只写出空报告")
            report_dir = self._resolve_report_dir()
            issue_frame = pd.DataFrame(
                [asdict(item) for item in self.issues], columns=ISSUE_COLUMNS
            )
            empty_missing = pd.DataFrame(
                columns=["code", "start_date", "end_date", "trading_days", "evidence"]
            )
            empty_coverage_dates = pd.DataFrame(
                columns=[
                    "trade_date",
                    "expected_symbols",
                    "actual_expected_symbols",
                    "missing_symbols",
                    "coverage",
                    "missing_sample",
                    "partition_exists",
                ]
            )
            empty_coverage_symbols = pd.DataFrame(
                columns=[
                    "code",
                    "open_date",
                    "expire_date",
                    "expected_rows",
                    "actual_rows",
                    "missing_rows",
                    "coverage",
                    "suspended_rows",
                    "active_zero_volume_rows",
                    "invalid_rows",
                ]
            )
            summary = self._build_summary(
                calendar,
                symbols,
                authoritative_calendar,
                issue_frame,
                empty_missing,
                empty_coverage_dates,
            )
            self._write_reports(
                report_dir,
                summary,
                issue_frame,
                empty_missing,
                empty_coverage_dates,
                empty_coverage_symbols,
            )
            self._log_run_finished(summary, report_dir)
            return AuditResult(
                summary=summary,
                issues=issue_frame,
                missing_spans=empty_missing,
                coverage_by_date=empty_coverage_dates,
                coverage_by_symbol=empty_coverage_symbols,
                report_dir=report_dir,
            )
        self._begin_phase("解析证券生命周期与除权事件")
        lifecycle = self._build_lifecycle(instrument, symbols, calendar)
        corporate_actions = self._load_corporate_action_keys()
        self._end_phase(
            "生命周期 {0} 只，除权事件 {1} 条".format(
                len(lifecycle), len(corporate_actions)
            )
        )
        self._begin_phase("审计日历外分区")
        self._audit_calendar_extra_partitions(
            calendar, symbols, lifecycle, corporate_actions
        )
        self._end_phase()
        self._begin_phase("逐日扫描行情")
        missing_spans, coverage_dates, coverage_symbols = self._scan_calendar(
            calendar,
            symbols,
            lifecycle,
            corporate_actions,
        )
        self._end_phase(
            "缺失区间 {0} 段，累计问题 {1} 条".format(len(missing_spans), len(self.issues))
        )
        self._begin_phase("汇总并写出报告")
        report_dir = self._resolve_report_dir()
        issue_frame = pd.DataFrame(
            [asdict(item) for item in self.issues], columns=ISSUE_COLUMNS
        )
        summary = self._build_summary(
            calendar,
            symbols,
            authoritative_calendar,
            issue_frame,
            missing_spans,
            coverage_dates,
        )
        self._write_reports(
            report_dir,
            summary,
            issue_frame,
            missing_spans,
            coverage_dates,
            coverage_symbols,
        )
        self._end_phase("报告目录 {0}".format(report_dir))
        self._log_run_finished(summary, report_dir)
        return AuditResult(
            summary=summary,
            issues=issue_frame,
            missing_spans=missing_spans,
            coverage_by_date=coverage_dates,
            coverage_by_symbol=coverage_symbols,
            report_dir=report_dir,
        )

    def _log_run_finished(self, summary: dict[str, Any], report_dir: Path) -> None:
        """记录本次自检的总耗时、问题数量和报告目录。

        参数：
            summary: 已生成的审计摘要，用于读取错误、告警和交易日统计。
            report_dir: 本次报告实际写入的目录。

        返回：
            无返回值。
        """

        self.logger.info(
            "[自检] 结束 状态=%s 总耗时 %.1fs errors=%s warnings=%s report=%s",
            summary.get("status", ""),
            time.monotonic() - self._run_started_at,
            summary.get("errors", 0),
            summary.get("warnings", 0),
            report_dir,
        )


def run_full_sample_self_check(config: SelfCheckConfig) -> AuditResult:
    """使用给定配置执行一次 QMT 全样本数据内部质量自检。

    参数：
        config: 数据根目录、交易日历、日期范围、报告路径和阈值配置。

    返回：
        包含退出状态、统计摘要和报告路径的完整审计结果。
    """

    return QmtDataSelfChecker(config).run()
