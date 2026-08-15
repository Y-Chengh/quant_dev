# -*- coding: utf-8 -*-
"""自检器的编排层：建立共享状态并按固定顺序驱动各职责 mixin。"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from .discovery import _PartitionDiscoveryMixin
from .loading import _ReferenceLoadingMixin
from .models import ISSUE_COLUMNS, AuditIssue, AuditResult, SelfCheckConfig
from .scanning import _CalendarScanMixin
from .summary import _SummaryReportMixin
from .validation import _RowValidationMixin


class QmtDataSelfChecker(
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

    def run(self) -> AuditResult:
        """执行全样本自检并写出 Markdown、JSON 与 CSV 报告。

        返回：
            包含统计摘要、全部问题、缺失区间、覆盖率和报告目录的结果。
        """

        if not self.root.is_dir():
            raise FileNotFoundError("QMT 数据根目录不存在: {0}".format(self.root))
        self._discover_partitions()
        instrument = self._load_instrument_info()
        symbols = self._resolve_symbols(instrument)
        calendar, authoritative_calendar = self._load_calendar()
        if not calendar:
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
            return AuditResult(
                summary=summary,
                issues=issue_frame,
                missing_spans=empty_missing,
                coverage_by_date=empty_coverage_dates,
                coverage_by_symbol=empty_coverage_symbols,
                report_dir=report_dir,
            )
        lifecycle = self._build_lifecycle(instrument, symbols, calendar)
        corporate_actions = self._load_corporate_action_keys()
        self._audit_calendar_extra_partitions(
            calendar, symbols, lifecycle, corporate_actions
        )
        missing_spans, coverage_dates, coverage_symbols = self._scan_calendar(
            calendar,
            symbols,
            lifecycle,
            corporate_actions,
        )
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
        return AuditResult(
            summary=summary,
            issues=issue_frame,
            missing_spans=missing_spans,
            coverage_by_date=coverage_dates,
            coverage_by_symbol=coverage_symbols,
            report_dir=report_dir,
        )


def run_full_sample_self_check(config: SelfCheckConfig) -> AuditResult:
    """使用给定配置执行一次 QMT 全样本数据内部质量自检。

    参数：
        config: 数据根目录、交易日历、日期范围、报告路径和阈值配置。

    返回：
        包含退出状态、统计摘要和报告路径的完整审计结果。
    """

    return QmtDataSelfChecker(config).run()
