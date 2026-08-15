# -*- coding: utf-8 -*-
"""审计摘要汇总与报告目录落盘。"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .base import _CheckerState
from .writers import (
    _summary_markdown,
    _write_csv_atomic,
    _write_json_atomic,
    _write_text_atomic,
)


class _SummaryReportMixin(_CheckerState):
    """汇总统计摘要并把 Markdown、JSON 与 CSV 报告写入报告目录。"""

    def _resolve_report_dir(self) -> Path:
        """解析本次报告目录并创建父目录。

        返回：
            已创建的唯一报告目录路径。
        """

        if self.config.report_dir is not None:
            report_dir = self.config.report_dir.resolve()
        else:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            report_dir = self.root / "reports" / "self_check" / run_id
        report_dir.mkdir(parents=True, exist_ok=False)
        return report_dir

    def _build_summary(
        self,
        calendar: list[str],
        symbols: list[str],
        authoritative_calendar: bool,
        issues: pd.DataFrame,
        missing_spans: pd.DataFrame,
        coverage_dates: pd.DataFrame,
    ) -> dict[str, Any]:
        """汇总审计范围、覆盖率和各严重级别问题数量。

        参数：
            calendar: 本次实际扫描的交易日列表。
            symbols: 本次实际扫描的完整证券池。
            authoritative_calendar: 是否使用独立 QMT 交易日历。
            issues: 全部详细问题表。
            missing_spans: 按证券压缩后的连续缺失区间表。
            coverage_dates: 按交易日统计的证券覆盖率表。

        返回：
            可直接序列化为 JSON 的审计摘要字典。
        """

        levels = Counter(issues["level"].tolist()) if not issues.empty else Counter()
        issue_codes = Counter(issues["issue_code"].tolist()) if not issues.empty else Counter()
        expected_rows = int(coverage_dates["expected_symbols"].sum()) if not coverage_dates.empty else 0
        actual_rows = int(coverage_dates["actual_expected_symbols"].sum()) if not coverage_dates.empty else 0
        return {
            "status": "failed" if levels.get("ERROR", 0) else "passed",
            "output_root": str(self.root),
            "data_root": str(self.data_root),
            "source_mode": self.source_mode,
            "start_date": calendar[0] if calendar else "",
            "end_date": calendar[-1] if calendar else "",
            "trading_days": len(calendar),
            "authoritative_calendar": authoritative_calendar,
            "symbols": len(symbols),
            "expected_rows": expected_rows,
            "actual_expected_rows": actual_rows,
            "missing_rows": max(expected_rows - actual_rows, 0),
            "missing_spans": len(missing_spans),
            "errors": levels.get("ERROR", 0),
            "warnings": levels.get("WARNING", 0),
            "info": levels.get("INFO", 0),
            "issue_codes": dict(sorted(issue_codes.items())),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _write_reports(
        self,
        report_dir: Path,
        summary: dict[str, Any],
        issues: pd.DataFrame,
        missing_spans: pd.DataFrame,
        coverage_dates: pd.DataFrame,
        coverage_symbols: pd.DataFrame,
    ) -> None:
        """原子写出 JSON、Markdown 和各类 CSV 审计报告。

        参数：
            report_dir: 本次自检唯一报告目录。
            summary: 可序列化的整体统计摘要。
            issues: 含完整定位、证据、原因和建议的全部问题表。
            missing_spans: 按证券压缩后的连续缺失区间表。
            coverage_dates: 每个交易日的应有、实有和缺失证券统计。
            coverage_symbols: 每只证券的应有、实有、停牌和异常统计。

        返回：
            无返回值；任一文件写入失败时抛出对应文件系统异常。
        """

        _write_json_atomic(report_dir / "summary.json", summary)
        _write_text_atomic(report_dir / "summary.md", _summary_markdown(summary))
        _write_csv_atomic(report_dir / "issues.csv", issues)
        _write_csv_atomic(report_dir / "missing_spans.csv", missing_spans)
        _write_csv_atomic(report_dir / "coverage_by_date.csv", coverage_dates)
        _write_csv_atomic(report_dir / "coverage_by_symbol.csv", coverage_symbols)
        lifecycle = issues[issues["issue_code"].isin(
            ["DATA_BEFORE_LISTING", "DATA_AFTER_DELISTING", "OPEN_DATE_MISSING", "INVALID_LIFECYCLE_RANGE"]
        )]
        volume = issues[issues["issue_code"].isin(
            ["ACTIVE_ZERO_VOLUME", "ACTIVE_ZERO_AMOUNT", "SUSPENDED_WITH_TURNOVER", "ABNORMAL_VOLUME_SCALE"]
        )]
        partition = issues[issues["issue_code"].str.startswith("PARTITION_")]
        statistical = issues[issues["issue_code"].isin(
            ["ABNORMAL_PRICE_JUMP", "ABNORMAL_VOLUME_SCALE", "PRE_CLOSE_DISCONTINUITY"]
        )]
        _write_csv_atomic(report_dir / "lifecycle_violations.csv", lifecycle)
        _write_csv_atomic(report_dir / "volume_violations.csv", volume)
        _write_csv_atomic(report_dir / "partition_violations.csv", partition)
        _write_csv_atomic(report_dir / "statistical_anomalies.csv", statistical)
