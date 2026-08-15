# -*- coding: utf-8 -*-
"""自检使用的问题记录、审计结果与配置数据类。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import _optional_date

ISSUE_COLUMNS = [
    "issue_code",
    "level",
    "dataset",
    "message",
    "code",
    "date",
    "start_date",
    "end_date",
    "field",
    "expected",
    "actual",
    "evidence",
    "possible_causes",
    "suggested_action",
    "source_file",
    "source_row",
]


@dataclass(frozen=True)
class AuditIssue:
    """保存一条可定位、可解释且可执行修复的数据质量问题。"""

    issue_code: str
    level: str
    dataset: str
    message: str
    code: str = ""
    date: str = ""
    start_date: str = ""
    end_date: str = ""
    field: str = ""
    expected: str = ""
    actual: str = ""
    evidence: str = ""
    possible_causes: str = ""
    suggested_action: str = ""
    source_file: str = ""
    source_row: int | str = ""


@dataclass(frozen=True)
class AuditResult:
    """保存全样本自检摘要、明细表和最终报告目录。"""

    summary: dict[str, Any]
    issues: pd.DataFrame
    missing_spans: pd.DataFrame
    coverage_by_date: pd.DataFrame
    coverage_by_symbol: pd.DataFrame
    report_dir: Path

    @property
    def exit_code(self) -> int:
        """根据是否存在硬错误返回命令行退出码。"""

        return 1 if (self.issues["level"] == "ERROR").any() else 0


@dataclass(frozen=True)
class SelfCheckConfig:
    """保存全样本自检所需路径、日期范围和统计告警阈值。"""

    output_root: Path
    start_date: str | None = None
    end_date: str | None = None
    calendar_csv: Path | None = None
    report_dir: Path | None = None
    coverage_error_threshold: float = 0.95
    price_jump_warning_ratio: float = 0.50
    volume_scale_warning_ratio: float = 100.0
    volume_history_window: int = 20
    volume_history_min_periods: int = 10
    verify_staging_hash: bool = False
    staging_root: Path | None = None
    #: 长循环每前进多少个元素输出一条 INFO 进度；0 表示按总量的 5% 自动选择。
    progress_every: int = 0
    #: 人工核实的源数据勘误表路径；``None`` 或文件不存在时视为没有勘误记录。
    #: 格式见 docs/qmt_source_data_errata.md，读入某交易日分区后、校验前应用。
    errata_csv: Path | None = None

    def __post_init__(self) -> None:
        """校验日期、覆盖率和统计异常阈值。

        返回：
            无返回值；配置非法时抛出 ``ValueError``。
        """

        _optional_date(self.start_date, "start_date")
        _optional_date(self.end_date, "end_date")
        if self.staging_root is not None and not str(self.staging_root).strip():
            raise ValueError("staging_root 不能是空路径")
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("start_date 不能晚于 end_date")
        if not 0.0 < self.coverage_error_threshold <= 1.0:
            raise ValueError("coverage_error_threshold 必须位于 (0, 1] 范围")
        if self.price_jump_warning_ratio <= 0:
            raise ValueError("price_jump_warning_ratio 必须大于 0")
        if self.volume_scale_warning_ratio <= 1:
            raise ValueError("volume_scale_warning_ratio 必须大于 1")
        if self.volume_history_window < 1:
            raise ValueError("volume_history_window 必须大于 0")
        if not 1 <= self.volume_history_min_periods <= self.volume_history_window:
            raise ValueError("volume_history_min_periods 必须位于 1 和窗口长度之间")
        if self.progress_every < 0:
            raise ValueError("progress_every 不能为负数")


@dataclass
class _SymbolStats:
    """累计单只证券在审计区间内的应有、实有、停牌和异常行数。"""

    expected_rows: int = 0
    actual_rows: int = 0
    suspended_rows: int = 0
    active_zero_volume_rows: int = 0
    invalid_rows: int = 0


@dataclass
class _ScanState:
    """保存跨交易日连续性、缺口区间和滚动成交量状态。"""

    last_row: dict[str, dict[str, Any]] = field(default_factory=dict)
    missing_open: dict[str, tuple[int, str]] = field(default_factory=dict)
    volume_history: dict[str, deque[float]] = field(default_factory=dict)
