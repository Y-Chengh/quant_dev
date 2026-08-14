# -*- coding: utf-8 -*-
"""对大 QMT 按日落盘行情执行全样本内部质量审计。"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .dates import normalize_date as _qmt_normalize_date


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

    def __post_init__(self) -> None:
        """校验日期、覆盖率和统计异常阈值。

        返回：
            无返回值；配置非法时抛出 ``ValueError``。
        """

        _optional_date(self.start_date, "start_date")
        _optional_date(self.end_date, "end_date")
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


class QmtDataSelfChecker:
    """扫描 QMT 日分区并生成不修改原始数据的详细质量报告。"""

    def __init__(self, config: SelfCheckConfig):
        """初始化自检器及空问题集合。

        参数：
            config: 数据根目录、审计日期范围、交易日历和异常阈值配置。
        """

        self.config = config
        self.root = config.output_root.resolve()
        self.issues: list[AuditIssue] = []
        self.partition_paths: dict[str, Path] = {}
        self.reference_scope: dict[str, Any] | None = None

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

    def _discover_partitions(self) -> None:
        """发现日线分区，并校验完成标记、哈希、行数和证券池口径。

        返回：
            无返回值；发现的问题追加到当前自检器的问题集合。
        """

        kline_root = self.root / "kline_1d"
        if not kline_root.is_dir():
            self._add_issue(
                "KLINE_DATASET_MISSING",
                "ERROR",
                "kline_1d",
                "日线数据集目录不存在，无法执行行情完整性检查。",
                expected="存在 kline_1d/date=YYYYMMDD/data.csv 日分区",
                actual="目录不存在: {0}".format(kline_root),
                possible_causes="日线尚未下载、输出目录配置错误或目录被移动",
                suggested_action="核对下载配置 output_root，并先完成 kline_1d 下载",
                source_file=str(kline_root),
            )
            return
        for directory in sorted(kline_root.glob("date=*")):
            if not directory.is_dir():
                continue
            raw_date = directory.name.split("=", 1)[-1]
            if re.fullmatch(r"\d{8}", raw_date) is None:
                self._add_issue(
                    "INVALID_PARTITION_DATE",
                    "ERROR",
                    "kline_1d",
                    "日线分区目录名不是合法的 YYYYMMDD 日期。",
                    actual="分区名={0}".format(directory.name),
                    expected="date=YYYYMMDD",
                    suggested_action="修复目录名或重新生成该分区，不要把非法目录纳入正式数据",
                    source_file=str(directory),
                )
                continue
            date_value = raw_date
            if date_value in self.partition_paths:
                self._add_issue(
                    "DUPLICATE_PARTITION_DATE",
                    "ERROR",
                    "kline_1d",
                    "多个日线目录映射到同一个交易日期。",
                    date=date_value,
                    expected="每个 YYYYMMDD 日期恰好一个 date=YYYYMMDD 目录",
                    actual="重复目录包含: {0}, {1}".format(
                        self.partition_paths[date_value], directory
                    ),
                    possible_causes="分区目录复制、大小写/路径同步冲突或人工移动文件",
                    suggested_action="保留唯一可信分区并使用 repair 重建重复日期",
                    source_file=str(directory),
                )
                continue
            self.partition_paths[date_value] = directory
            metadata = self._read_partition_metadata(directory, date_value)
            if metadata is not None:
                scope = metadata.get("partition_scope")
                if isinstance(scope, dict):
                    if self.reference_scope is None:
                        self.reference_scope = scope
                    elif scope != self.reference_scope:
                        self._add_issue(
                            "PARTITION_SCOPE_MISMATCH",
                            "ERROR",
                            "kline_1d",
                            "该日期分区的证券池或抽取口径与其他分区不一致。",
                            date=date_value,
                            expected=json.dumps(self.reference_scope, ensure_ascii=False, sort_keys=True),
                            actual=json.dumps(scope, ensure_ascii=False, sort_keys=True),
                            evidence="不同口径的数据不能视为同一个完整全样本",
                            possible_causes="曾使用不同证券池或数据集配置写入同一输出目录",
                            suggested_action="按统一配置使用 repair 重建异常分区，或改用独立输出目录",
                            source_file=str(directory / "_SUCCESS.json"),
                        )

    def _read_partition_metadata(
        self, directory: Path, date_value: str
    ) -> dict[str, Any] | None:
        """读取并验证一个日线分区完成标记。

        参数：
            directory: 当前 ``date=YYYYMMDD`` 日线分区目录。
            date_value: 从目录名解析得到的八位交易日期。

        返回：
            成功解析的完成元数据；文件不存在或 JSON 损坏时返回 ``None``。
        """

        data_path = directory / "data.csv"
        marker_path = directory / "_SUCCESS.json"
        if not data_path.is_file() or not marker_path.is_file():
            missing = [
                path.name for path in (data_path, marker_path) if not path.is_file()
            ]
            self._add_issue(
                "PARTITION_FILE_MISSING",
                "ERROR",
                "kline_1d",
                "日线分区缺少数据文件或完成标记。",
                date=date_value,
                expected="data.csv 与 _SUCCESS.json 同时存在",
                actual="缺少: {0}".format(", ".join(missing)),
                evidence="不完整分区不能证明写入已经原子完成",
                possible_causes="下载中断、人工删除文件或磁盘写入失败",
                suggested_action="使用 repair 模式重新生成整个日期分区",
                source_file=str(directory),
            )
            return None
        try:
            with marker_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self._add_issue(
                "PARTITION_MARKER_INVALID",
                "ERROR",
                "kline_1d",
                "日线分区完成标记无法解析。",
                date=date_value,
                expected="合法的 UTF-8 JSON 对象",
                actual="{0}: {1}".format(type(exc).__name__, exc),
                possible_causes="完成标记写入中断或文件被修改",
                suggested_action="使用 repair 模式重建分区，不要手工伪造完成标记",
                source_file=str(marker_path),
            )
            return None
        if not isinstance(metadata, dict):
            self._add_issue(
                "PARTITION_MARKER_SCHEMA_INVALID",
                "ERROR",
                "kline_1d",
                "日线分区完成标记的 JSON 根节点不是对象。",
                date=date_value,
                expected="JSON object containing sha256, rows and partition_scope",
                actual="根节点类型={0}".format(type(metadata).__name__),
                possible_causes="完成标记被截断、人工替换或写入了错误格式",
                suggested_action="使用 repair 模式重建该日期分区，不要手工修改 _SUCCESS.json",
                source_file=str(marker_path),
            )
            return None
        expected_hash = str(metadata.get("sha256", ""))
        actual_hash = _file_sha256(data_path)
        if expected_hash != actual_hash:
            self._add_issue(
                "PARTITION_HASH_MISMATCH",
                "ERROR",
                "kline_1d",
                "日线 CSV 内容与完成标记记录的 SHA-256 不一致。",
                date=date_value,
                field="sha256",
                expected=expected_hash or "_SUCCESS.json 中应存在 sha256",
                actual=actual_hash,
                evidence="数据文件在完成标记生成后发生修改或损坏",
                possible_causes="人工编辑、磁盘损坏、非原子覆盖或错误同步",
                suggested_action="停止使用该分区并通过 repair 模式重新下载",
                source_file=str(data_path),
            )
        expected_rows = metadata.get("rows")
        actual_rows = _csv_row_count(data_path)
        try:
            rows_match = int(expected_rows) == actual_rows
        except (TypeError, ValueError):
            rows_match = False
        if not rows_match:
            self._add_issue(
                "PARTITION_ROW_COUNT_MISMATCH",
                "ERROR",
                "kline_1d",
                "日线 CSV 记录数与完成标记不一致。",
                date=date_value,
                field="rows",
                expected=str(expected_rows),
                actual=str(actual_rows),
                evidence="_SUCCESS.json 与实际 CSV 行数核对失败",
                possible_causes="文件被截断、追加、人工编辑或完成标记损坏",
                suggested_action="使用 repair 模式重新生成该日期分区",
                source_file=str(data_path),
            )
        return metadata if isinstance(metadata, dict) else None

    def _load_instrument_info(self) -> pd.DataFrame:
        """读取证券上市退市快照并验证文件级完整性。

        返回：
            含证券代码、上市日期和退市日期的原始快照；不可读时返回空表。
        """

        directory = self.root / "instrument_info" / "snapshot=latest"
        data_path = directory / "data.csv"
        if not data_path.is_file():
            self._add_issue(
                "INSTRUMENT_INFO_MISSING",
                "ERROR",
                "instrument_info",
                "缺少证券上市退市信息快照，无法判断理论应有行情日期。",
                expected="instrument_info/snapshot=latest/data.csv",
                actual="文件不存在",
                possible_causes="证券详情下载失败或快照目录被删除",
                suggested_action="重新运行 QMT 下载器生成 instrument_info 快照",
                source_file=str(data_path),
            )
            return pd.DataFrame(columns=["code", "open_date", "expire_date"])
        marker = self._read_auxiliary_marker(directory, "instrument_info", "snapshot=latest")
        try:
            frame = pd.read_csv(str(data_path), encoding="utf-8-sig", dtype=str)
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            self._add_issue(
                "INSTRUMENT_INFO_UNREADABLE",
                "ERROR",
                "instrument_info",
                "证券上市退市快照无法读取。",
                expected="包含 code、open_date、expire_date 的合法 CSV",
                actual="{0}: {1}".format(type(exc).__name__, exc),
                suggested_action="重新运行下载器生成证券信息快照",
                source_file=str(data_path),
            )
            return pd.DataFrame(columns=["code", "open_date", "expire_date"])
        marker_rows = _optional_int(marker.get("rows")) if marker is not None else None
        if marker is not None and marker_rows != len(frame):
            self._add_issue(
                "INSTRUMENT_INFO_ROW_COUNT_MISMATCH",
                "ERROR",
                "instrument_info",
                "证券信息快照行数与完成标记不一致。",
                expected=str(marker.get("rows")),
                actual=str(len(frame)),
                suggested_action="使用下载器重新生成 instrument_info 快照",
                source_file=str(data_path),
            )
        missing = sorted({"code", "open_date", "expire_date"}.difference(frame.columns))
        if missing:
            self._add_issue(
                "INSTRUMENT_INFO_COLUMNS_MISSING",
                "ERROR",
                "instrument_info",
                "证券信息快照缺少生命周期必需字段。",
                expected="code, open_date, expire_date",
                actual="缺少: {0}".format(", ".join(missing)),
                suggested_action="确认下载器版本并重新生成快照",
                source_file=str(data_path),
            )
            for column in missing:
                frame[column] = ""
        return frame

    def _read_auxiliary_marker(
        self, directory: Path, dataset: str, partition: str
    ) -> dict[str, Any] | None:
        """校验非日线分区的完成标记和文件哈希。

        参数：
            directory: 同时包含 ``data.csv`` 与 ``_SUCCESS.json`` 的目录。
            dataset: 错误报告使用的数据集名称。
            partition: 错误报告使用的分区名称。

        返回：
            合法 JSON 元数据；缺失或损坏时返回 ``None``。
        """

        data_path = directory / "data.csv"
        marker_path = directory / "_SUCCESS.json"
        if not marker_path.is_file():
            self._add_issue(
                "AUXILIARY_MARKER_MISSING",
                "ERROR",
                dataset,
                "辅助数据文件缺少完成标记，无法确认写入完整。",
                date=partition,
                expected="_SUCCESS.json",
                actual="文件不存在",
                suggested_action="重新运行下载器生成该数据集",
                source_file=str(directory),
            )
            return None
        try:
            with marker_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self._add_issue(
                "AUXILIARY_MARKER_INVALID",
                "ERROR",
                dataset,
                "辅助数据完成标记无法解析。",
                date=partition,
                actual="{0}: {1}".format(type(exc).__name__, exc),
                suggested_action="重新运行下载器生成该数据集",
                source_file=str(marker_path),
            )
            return None
        if not isinstance(metadata, dict):
            self._add_issue(
                "AUXILIARY_MARKER_SCHEMA_INVALID",
                "ERROR",
                dataset,
                "辅助数据完成标记的 JSON 根节点不是对象。",
                date=partition,
                expected="JSON object containing rows and sha256",
                actual="根节点类型={0}".format(type(metadata).__name__),
                possible_causes="完成标记被截断、人工替换或写入了错误格式",
                suggested_action="重新运行下载器生成该辅助数据集",
                source_file=str(marker_path),
            )
            return None
        expected_hash = str(metadata.get("sha256", ""))
        if data_path.is_file() and expected_hash != _file_sha256(data_path):
            self._add_issue(
                "AUXILIARY_HASH_MISMATCH",
                "ERROR",
                dataset,
                "辅助数据文件与完成标记的 SHA-256 不一致。",
                date=partition,
                expected=expected_hash,
                actual=_file_sha256(data_path),
                suggested_action="停止使用该快照并重新运行下载器",
                source_file=str(data_path),
            )
        return metadata if isinstance(metadata, dict) else None

    def _resolve_symbols(self, instrument: pd.DataFrame) -> list[str]:
        """从分区口径和证券快照解析理论证券池并报告不一致。

        参数：
            instrument: 证券上市、退市及当前状态快照。

        返回：
            稳定排序且去重后的完整证券代码列表。
        """

        scope_symbols: list[str] = []
        if self.reference_scope is not None:
            raw = self.reference_scope.get("symbols", [])
            if isinstance(raw, list):
                scope_symbols = [str(item).strip().upper() for item in raw if str(item).strip()]
        info_symbols = []
        if "code" in instrument:
            info_symbols = [
                str(item).strip().upper()
                for item in instrument["code"].tolist()
                if str(item).strip() and str(item).lower() != "nan"
            ]
        symbols = sorted(set(scope_symbols or info_symbols))
        if not symbols:
            self._add_issue(
                "EXPECTED_SYMBOLS_EMPTY",
                "ERROR",
                "instrument_info",
                "无法从分区口径或证券快照解析应有证券池。",
                expected="partition_scope.symbols 或 instrument_info.code 至少提供一个证券",
                actual="证券池为空",
                suggested_action="重新生成至少一个完整分区及 instrument_info 快照",
            )
            return []
        missing_info = sorted(set(symbols).difference(info_symbols))
        if missing_info:
            self._add_issue(
                "SYMBOLS_MISSING_INSTRUMENT_INFO",
                "ERROR",
                "instrument_info",
                "证券池中的部分股票缺少上市退市信息。",
                expected="每个 partition_scope.symbols 代码均有 instrument_info 记录",
                actual="缺少 {0} 只: {1}".format(len(missing_info), _sample(missing_info)),
                evidence="缺少生命周期时无法准确区分未上市、退市与行情缺失",
                possible_causes="证券详情接口失败或快照只写入了部分批次",
                suggested_action="重新获取证券详情并生成完整 instrument_info 快照",
                source_file=str(self.root / "instrument_info" / "snapshot=latest" / "data.csv"),
            )
        extra_info = sorted(set(info_symbols).difference(symbols))
        if extra_info:
            self._add_issue(
                "INSTRUMENT_INFO_OUTSIDE_SCOPE",
                "WARNING",
                "instrument_info",
                "证券信息快照含有不属于日线分区证券池的股票。",
                expected="instrument_info.code 与 partition_scope.symbols 一致",
                actual="额外 {0} 只: {1}".format(len(extra_info), _sample(extra_info)),
                possible_causes="证券池调整后复用了旧快照或分区口径不一致",
                suggested_action="确认目标证券池；不要把额外股票自动加入审计范围",
                source_file=str(self.root / "instrument_info" / "snapshot=latest" / "data.csv"),
            )
        return symbols

    def _load_calendar(self) -> tuple[list[str], bool]:
        """读取权威交易日历，缺失时退化为已观察分区日期。

        返回：
            ``(交易日列表, 是否为权威日历)``；日期均为升序八位字符串。
        """

        calendar_path = self.config.calendar_csv
        if calendar_path is None:
            default_path = self.root / "trading_calendar" / "data.csv"
            if default_path.is_file():
                calendar_path = default_path
        authoritative = calendar_path is not None
        dates: list[str] = []
        if calendar_path is not None:
            dates = _read_calendar_csv(calendar_path)
        else:
            dates = sorted(self.partition_paths)
            self._add_issue(
                "CALENDAR_INFERRED_FROM_PARTITIONS",
                "ERROR",
                "trading_calendar",
                "未提供独立 QMT 交易日历，只能使用现有日线分区日期推导审计日历。",
                expected="--calendar-csv 或 trading_calendar/data.csv",
                actual="使用 {0} 个已观察分区日期".format(len(dates)),
                evidence="这种退化模式可以发现单只股票缺失，但不能发现整个交易日分区完全缺失",
                possible_causes="下载器尚未持久化交易日历或命令行未指定日历文件",
                suggested_action="从 QMT get_trading_dates 导出 trade_date 列后重新执行自检",
            )
        start = self.config.start_date
        end = self.config.end_date
        filtered = [
            value
            for value in sorted(set(dates))
            if (start is None or value >= start) and (end is None or value <= end)
        ]
        return filtered, authoritative

    def _audit_calendar_extra_partitions(
        self,
        calendar: list[str],
        symbols: list[str],
        lifecycle: dict[str, tuple[str, str | None]],
        corporate_actions: set[tuple[str, str]],
    ) -> None:
        """审计不属于交易日历范围的日线分区，避免异常日期被静默跳过。

        参数：
            calendar: 本次审计使用的交易日列表。
            symbols: 分区口径声明的完整证券池。
            lifecycle: 每只证券的上市日期和退市日期。
            corporate_actions: 可解释价格断层的公司行为主键集合。

        返回：
            无返回值；额外分区和其中的行级生命周期问题追加到问题集合。
        """

        calendar_set = set(calendar)
        extra_dates = sorted(
            date_value
            for date_value in set(self.partition_paths).difference(calendar_set)
            if (self.config.start_date is None or date_value >= self.config.start_date)
            and (self.config.end_date is None or date_value <= self.config.end_date)
        )
        for date_value in extra_dates:
            directory = self.partition_paths[date_value]
            self._add_issue(
                "DATA_ON_NON_TRADING_DATE",
                "ERROR",
                "kline_1d",
                "日线分区日期不属于本次 QMT 交易日历。",
                date=date_value,
                expected="trade_date 必须属于提供的 QMT 交易日历",
                actual="存在分区: {0}".format(directory),
                evidence="该日期不在 calendar_csv 的完整交易日集合中",
                possible_causes="日历文件范围错误、误把自然日当交易日或分区日期写错",
                suggested_action="核对 QMT get_trading_dates 与分区目录，确认后移出或重建异常分区",
                source_file=str(directory / "data.csv"),
            )
            frame = self._read_kline_partition(directory, date_value)
            stats = {code: _SymbolStats() for code in symbols}
            self._validate_rows(
                frame,
                date_value,
                directory,
                set(symbols),
                lifecycle,
                corporate_actions,
                stats,
                _ScanState(),
            )

    def _build_lifecycle(
        self,
        instrument: pd.DataFrame,
        symbols: list[str],
        calendar: list[str],
    ) -> dict[str, tuple[str, str | None]]:
        """规范化证券生命周期并校验重复、缺失和日期先后关系。

        参数：
            instrument: 从 QMT 证券详情快照读取的原始表。
            symbols: 本次审计使用的权威证券池。
            calendar: 本次审计使用的升序交易日列表。

        返回：
            证券代码到 ``(上市日期, 可选退市日期)`` 的映射。
        """

        lifecycle: dict[str, tuple[str, str | None]] = {}
        source = self.root / "instrument_info" / "snapshot=latest" / "data.csv"
        if instrument.empty:
            for code in symbols:
                lifecycle[code] = (calendar[0], None)
            return lifecycle
        normalized = instrument.copy()
        normalized["code"] = normalized["code"].astype(str).str.strip().str.upper()
        duplicates = normalized[normalized.duplicated("code", keep=False)]
        for index, row in duplicates.iterrows():
            self._add_issue(
                "INSTRUMENT_INFO_DUPLICATE",
                "ERROR",
                "instrument_info",
                "同一证券在上市退市快照中出现多次。",
                code=row["code"],
                expected="每只证券恰好一条生命周期记录",
                actual="存在重复 code",
                possible_causes="批次合并重复或快照文件被追加",
                suggested_action="重新生成证券快照，不要直接保留任意一条重复记录",
                source_file=str(source),
                source_row=int(index) + 2,
            )
        first_rows = normalized.drop_duplicates("code", keep="first").set_index("code")
        for code in symbols:
            if code not in first_rows.index:
                lifecycle[code] = (calendar[0], None)
                continue
            row = first_rows.loc[code]
            row_number = int(normalized.index[normalized["code"] == code][0]) + 2
            open_date, open_invalid = _parse_lifecycle_value(row.get("open_date"))
            expire_date, expire_invalid = _parse_lifecycle_value(row.get("expire_date"))
            if open_invalid:
                self._add_issue(
                    "OPEN_DATE_INVALID",
                    "ERROR",
                    "instrument_info",
                    "证券上市日期不是空值、合法日期或受支持的无期限哨兵。",
                    code=code,
                    field="open_date",
                    expected="合法 YYYYMMDD 日期或明确空值",
                    actual=str(row.get("open_date", "")),
                    possible_causes="QMT OpenDate 字段损坏、日期格式变化或快照被人工修改",
                    suggested_action="核对 QMT OpenDate 原始值并重新生成证券详情快照",
                    source_file=str(source),
                    source_row=row_number,
                )
            if expire_invalid:
                self._add_issue(
                    "EXPIRE_DATE_INVALID",
                    "ERROR",
                    "instrument_info",
                    "证券退市日期非空但不是合法日期或受支持的无期限哨兵。",
                    code=code,
                    field="expire_date",
                    expected="合法 YYYYMMDD 日期、空值或 99999999 无期限哨兵",
                    actual=str(row.get("expire_date", "")),
                    possible_causes="QMT ExpireDate 字段损坏、日期格式变化或快照被人工修改",
                    suggested_action="核对 QMT ExpireDate 原始值并重新生成证券详情快照",
                    source_file=str(source),
                    source_row=row_number,
                )
            if open_date is None:
                self._add_issue(
                    "OPEN_DATE_MISSING",
                    "ERROR",
                    "instrument_info",
                    "证券缺少合法上市日期，无法准确确定应有行情起点。",
                    code=code,
                    field="open_date",
                    expected="合法的 YYYYMMDD 上市日期",
                    actual=str(row.get("open_date", "")),
                    possible_causes="QMT 合约详情缺失、无日期哨兵未正确转换或快照损坏",
                    suggested_action="核对 QMT OpenDate 并重新生成证券详情快照",
                    source_file=str(source),
                    source_row=row_number,
                )
                open_date = calendar[0]
            if expire_date is not None and open_date > expire_date:
                self._add_issue(
                    "INVALID_LIFECYCLE_RANGE",
                    "ERROR",
                    "instrument_info",
                    "证券上市日期晚于退市日期。",
                    code=code,
                    field="open_date,expire_date",
                    expected="open_date <= expire_date",
                    actual="open_date={0}, expire_date={1}".format(open_date, expire_date),
                    possible_causes="QMT 日期字段映射错误或证券详情快照损坏",
                    suggested_action="核对 OpenDate 与 ExpireDate，确认前不要删除任何行情",
                    source_file=str(source),
                    source_row=row_number,
                )
            lifecycle[code] = (open_date, expire_date)
        return lifecycle

    def _load_corporate_action_keys(self) -> set[tuple[str, str]]:
        """读取公司行为分区中的证券和除权日期主键。

        返回：
            ``(证券代码, 除权日期)`` 集合，用于解释跨日价格连续性差异。
        """

        keys: set[tuple[str, str]] = set()
        root = self.root / "corporate_actions"
        if not root.is_dir():
            return keys
        for directory in sorted(root.glob("ex_date=*")):
            path = directory / "data.csv"
            if not path.is_file():
                continue
            try:
                frame = pd.read_csv(str(path), encoding="utf-8-sig", dtype=str)
            except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
                continue
            if "code" not in frame:
                continue
            fallback_date = _normalize_date_text(directory.name.split("=", 1)[-1])
            for _, row in frame.iterrows():
                date_value = _normalize_date_text(row.get("ex_date")) or fallback_date
                if date_value:
                    keys.add((str(row["code"]).strip().upper(), date_value))
        return keys

    def _scan_calendar(
        self,
        calendar: list[str],
        symbols: list[str],
        lifecycle: dict[str, tuple[str, str | None]],
        corporate_actions: set[tuple[str, str]],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """逐交易日扫描行情并统计缺口、生命周期、成交量和异常跳变。

        参数：
            calendar: 本次审计范围内的权威或退化交易日列表。
            symbols: 分区口径声明的完整证券池。
            lifecycle: 每只证券的上市日期和可选退市日期。
            corporate_actions: 可解释跨日价格断层的证券除权日期集合。

        返回：
            缺失区间、逐日覆盖率和逐证券覆盖率三个表格。
        """

        add_events: dict[int, list[str]] = defaultdict(list)
        remove_events: dict[int, list[str]] = defaultdict(list)
        for code in symbols:
            open_date, expire_date = lifecycle.get(code, (calendar[0], None))
            first = bisect.bisect_left(calendar, open_date)
            last = len(calendar) - 1 if expire_date is None else bisect.bisect_right(calendar, expire_date) - 1
            if first < len(calendar) and last >= first:
                add_events[first].append(code)
                remove_events[last + 1].append(code)

        active: set[str] = set()
        state = _ScanState()
        stats = {code: _SymbolStats() for code in symbols}
        missing_spans: list[dict[str, Any]] = []
        coverage_rows: list[dict[str, Any]] = []
        for position, date_value in enumerate(calendar):
            previous_date = calendar[position - 1] if position > 0 else ""
            for code in remove_events.get(position, []):
                self._close_missing_span(
                    code, position - 1, previous_date, missing_spans, state
                )
                active.discard(code)
            active.update(add_events.get(position, []))
            directory = self.partition_paths.get(date_value)
            if directory is None:
                frame = pd.DataFrame()
                self._add_issue(
                    "MISSING_DATE_PARTITION",
                    "ERROR",
                    "kline_1d",
                    "QMT 交易日历包含该日期，但整个日线分区不存在。",
                    date=date_value,
                    expected=str(self.root / "kline_1d" / ("date=" + date_value) / "data.csv"),
                    actual="日期目录不存在",
                    evidence="理论应有证券 {0} 只，实际记录 0 条".format(len(active)),
                    possible_causes="当日任务未运行、下载失败、水位错误推进或目录被删除",
                    suggested_action="检查 downloader.log 与 run_complete 水位，并使用 repair 补齐该日期",
                    source_file=str(self.root / "kline_1d" / ("date=" + date_value)),
                )
            else:
                frame = self._read_kline_partition(directory, date_value)
            present = self._validate_rows(
                frame,
                date_value,
                directory,
                set(symbols),
                lifecycle,
                corporate_actions,
                stats,
                state,
            )
            expected = set(active)
            missing = sorted(expected.difference(present))
            for code in expected:
                stats[code].expected_rows += 1
            for code in expected.intersection(present):
                stats[code].actual_rows += 1
            for code in missing:
                if code not in state.missing_open:
                    state.missing_open[code] = (position, date_value)
            for code in expected.intersection(present):
                self._close_missing_span(
                    code, position - 1, previous_date, missing_spans, state
                )
            coverage = 1.0 if not expected else len(expected.intersection(present)) / len(expected)
            coverage_rows.append(
                {
                    "trade_date": date_value,
                    "expected_symbols": len(expected),
                    "actual_expected_symbols": len(expected.intersection(present)),
                    "missing_symbols": len(missing),
                    "coverage": coverage,
                    "missing_sample": _sample(missing),
                    "partition_exists": directory is not None,
                }
            )
            if missing:
                level = "ERROR"
                self._add_issue(
                    "DATE_SYMBOLS_MISSING",
                    level,
                    "kline_1d",
                    "该交易日分区缺少生命周期内理论应有的证券行情。",
                    date=date_value,
                    expected="{0} 只证券均有一条日线，覆盖率 100%".format(len(expected)),
                    actual="实有 {0} 只，缺少 {1} 只，覆盖率 {2:.2%}".format(
                        len(expected.intersection(present)), len(missing), coverage
                    ),
                    evidence="缺失示例: {0}".format(_sample(missing)),
                    possible_causes="部分下载批次失败、本地缓存不完整或分区错误标记为完成",
                    suggested_action="按 missing_spans.csv 定位证券和区间，并使用 repair 重建相关分区",
                    source_file=str(
                        (directory or self.root / "kline_1d" / ("date=" + date_value))
                        / "data.csv"
                    ),
                )
            if coverage < self.config.coverage_error_threshold:
                self._add_issue(
                    "LOW_DATE_COVERAGE",
                    "ERROR",
                    "kline_1d",
                    "该交易日证券覆盖率低于配置阈值，疑似整批或大范围数据缺失。",
                    date=date_value,
                    expected="覆盖率 >= {0:.2%}".format(self.config.coverage_error_threshold),
                    actual="覆盖率 {0:.2%}，缺少 {1} 只".format(coverage, len(missing)),
                    evidence="缺失示例: {0}".format(_sample(missing)),
                    possible_causes="下载任务中断、多个批次失败或证券池口径不一致",
                    suggested_action="优先核查该日期日志和 staging 批次，再执行整日 repair",
                    source_file=str(directory or ""),
                )

        if calendar:
            for code in list(state.missing_open):
                self._close_missing_span(
                    code, len(calendar) - 1, calendar[-1], missing_spans, state
                )
        missing_frame = pd.DataFrame(
            missing_spans,
            columns=["code", "start_date", "end_date", "trading_days", "evidence"],
        )
        coverage_date_frame = pd.DataFrame(coverage_rows)
        symbol_rows = []
        for code in symbols:
            item = stats[code]
            missing_rows = max(item.expected_rows - item.actual_rows, 0)
            coverage = 1.0 if item.expected_rows == 0 else item.actual_rows / item.expected_rows
            symbol_rows.append(
                {
                    "code": code,
                    "open_date": lifecycle.get(code, ("", None))[0],
                    "expire_date": lifecycle.get(code, ("", None))[1] or "",
                    "expected_rows": item.expected_rows,
                    "actual_rows": item.actual_rows,
                    "missing_rows": missing_rows,
                    "coverage": coverage,
                    "suspended_rows": item.suspended_rows,
                    "active_zero_volume_rows": item.active_zero_volume_rows,
                    "invalid_rows": item.invalid_rows,
                }
            )
            if item.expected_rows > 0 and item.actual_rows == 0:
                audit_start = max(
                    lifecycle.get(code, (calendar[0], None))[0], calendar[0]
                )
                lifecycle_end = lifecycle.get(code, ("", None))[1]
                audit_end = min(lifecycle_end, calendar[-1]) if lifecycle_end else calendar[-1]
                self._add_issue(
                    "SYMBOL_ALL_DATA_MISSING",
                    "ERROR",
                    "kline_1d",
                    "证券在整个有效生命周期审计区间内完全没有行情数据。",
                    code=code,
                    start_date=audit_start,
                    end_date=audit_end,
                    expected="应有 {0} 个交易日日线记录".format(item.expected_rows),
                    actual="实际 0 条",
                    evidence="证券属于 partition_scope.symbols 且生命周期与审计区间相交",
                    possible_causes="整只股票未下载、证券代码映射失败或所有相关批次缺失",
                    suggested_action="核对 QMT 证券代码和本地缓存，并对该证券全区间重新下载",
                )
        return missing_frame, coverage_date_frame, pd.DataFrame(symbol_rows)

    def _read_kline_partition(self, directory: Path, date_value: str) -> pd.DataFrame:
        """读取单个日线 CSV，并在解析失败时报告详细错误。

        参数：
            directory: 当前日线日期分区目录。
            date_value: 该分区理论对应的八位交易日期。

        返回：
            原始日线表；文件不存在、为空或无法解析时返回空表。
        """

        path = directory / "data.csv"
        if not path.is_file():
            return pd.DataFrame()
        try:
            return pd.read_csv(str(path), encoding="utf-8-sig", dtype={"code": str, "trade_date": str})
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            self._add_issue(
                "KLINE_PARTITION_UNREADABLE",
                "ERROR",
                "kline_1d",
                "日线分区 CSV 无法读取。",
                date=date_value,
                expected="字段和行结构合法的 UTF-8 CSV",
                actual="{0}: {1}".format(type(exc).__name__, exc),
                possible_causes="文件截断、编码错误、列数错位或空文件",
                suggested_action="使用 repair 模式重新生成该日期分区",
                source_file=str(path),
            )
            return pd.DataFrame()

    def _validate_rows(
        self,
        frame: pd.DataFrame,
        date_value: str,
        directory: Path | None,
        expected_symbols: set[str],
        lifecycle: dict[str, tuple[str, str | None]],
        corporate_actions: set[tuple[str, str]],
        stats: dict[str, _SymbolStats],
        state: _ScanState,
    ) -> set[str]:
        """验证单日全部行情行及跨日价格、成交量连续性。

        参数：
            frame: 当前交易日日线表。
            date_value: 当前八位交易日期。
            directory: 当前分区目录；整日缺失时为 ``None``。
            expected_symbols: 分区口径声明的完整证券池集合。
            lifecycle: 证券代码到上市退市日期的映射。
            corporate_actions: 可解释价格断层的公司行为主键集合。
            stats: 按证券累计覆盖率和异常数量的可变统计字典。
            state: 跨日期保留上一行、缺口和成交量历史的状态。

        返回：
            当前分区实际出现且主键可解析的证券代码集合。
        """

        if frame.empty:
            return set()
        required = {
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
        }
        missing_columns = sorted(required.difference(frame.columns))
        path = (directory / "data.csv") if directory is not None else Path("")
        if missing_columns:
            self._add_issue(
                "KLINE_COLUMNS_MISSING",
                "ERROR",
                "kline_1d",
                "日线分区缺少必需字段。",
                date=date_value,
                expected=", ".join(sorted(required)),
                actual="缺少: {0}".format(", ".join(missing_columns)),
                possible_causes="下载器版本不一致、CSV 表头损坏或人工编辑",
                suggested_action="确认字段版本并使用 repair 重建分区",
                source_file=str(path),
            )
            for column in missing_columns:
                frame[column] = math.nan
        normalized_codes = frame["code"].astype(str).str.strip().str.upper()
        duplicated = frame.assign(_code=normalized_codes).duplicated(
            ["_code", "trade_date"], keep=False
        )
        for index in frame.index[duplicated]:
            code = normalized_codes.loc[index]
            self._row_issue(
                "KLINE_DUPLICATE_KEY",
                "ERROR",
                "同一证券和交易日出现重复日线主键。",
                code,
                date_value,
                "code,trade_date",
                "每个主键恰好一行",
                "存在重复记录",
                "批次合并重复、文件被追加或修复时未覆盖旧记录",
                "重建该日期分区，不要任意保留重复行",
                path,
                index,
            )
        present: set[str] = set()
        for index, row in frame.iterrows():
            code = normalized_codes.loc[index]
            row_date = _normalize_date_text(row.get("trade_date"))
            if not code or code.lower() == "nan":
                self._row_issue(
                    "KLINE_CODE_MISSING",
                    "ERROR",
                    "日线记录缺少证券代码。",
                    "",
                    date_value,
                    "code",
                    "非空的 code.market",
                    str(row.get("code", "")),
                    "CSV 字段错位或写入前代码丢失",
                    "重新下载该日期分区",
                    path,
                    index,
                )
                continue
            if code not in expected_symbols:
                self._row_issue(
                    "KLINE_SYMBOL_OUTSIDE_SCOPE",
                    "ERROR",
                    "日线记录的证券不属于分区声明的证券池。",
                    code,
                    date_value,
                    "code",
                    "code 属于 partition_scope.symbols",
                    code,
                    "证券池切换后混写分区或代码映射错误",
                    "核对证券池并使用统一口径重建分区",
                    path,
                    index,
                )
                continue
            if row_date != date_value:
                self._row_issue(
                    "KLINE_DATE_PARTITION_MISMATCH",
                    "ERROR",
                    "日线记录的 trade_date 与所在日期分区不一致。",
                    code,
                    date_value,
                    "trade_date",
                    date_value,
                    str(row.get("trade_date", "")),
                    "分区合并错误、日期标准化错误或文件被移动",
                    "按记录真实日期核对源数据后重建相关分区",
                    path,
                    index,
                )
                continue
            present.add(code)
            open_date, expire_date = lifecycle.get(code, (date_value, None))
            in_lifecycle = date_value >= open_date and (
                expire_date is None or date_value <= expire_date
            )
            if date_value < open_date:
                self._row_issue(
                    "DATA_BEFORE_LISTING",
                    "ERROR",
                    "证券上市前出现日线行情。",
                    code,
                    date_value,
                    "trade_date",
                    "trade_date >= {0}".format(open_date),
                    date_value,
                    "上市日期错误、证券代码复用或行情归属错误",
                    "核对 QMT OpenDate 和证券代码，确认前不要自动删除行情",
                    path,
                    index,
                )
            if expire_date is not None and date_value > expire_date:
                self._row_issue(
                    "DATA_AFTER_DELISTING",
                    "ERROR",
                    "证券退市后仍存在日线行情。",
                    code,
                    date_value,
                    "trade_date",
                    "trade_date <= {0}".format(expire_date),
                    date_value,
                    "退市日期口径错误、证券代码复用或历史数据混入",
                    "核对 QMT ExpireDate 和证券代码，确认前不要自动删除行情",
                    path,
                    index,
                )
            invalid = self._validate_numeric_row(row, code, date_value, path, index, stats)
            if invalid:
                stats[code].invalid_rows += 1
            if in_lifecycle:
                self._validate_continuity(
                    row,
                    code,
                    date_value,
                    path,
                    index,
                    corporate_actions,
                    state,
                )
        return present

    def _validate_numeric_row(
        self,
        row: pd.Series,
        code: str,
        date_value: str,
        path: Path,
        index: Any,
        stats: dict[str, _SymbolStats],
    ) -> bool:
        """验证一行 OHLC、成交量、成交额和停牌标志。

        参数：
            row: 当前证券日线原始字段。
            code: 当前证券代码。
            date_value: 当前交易日期。
            path: 当前记录来源 CSV 路径。
            index: 当前记录在 ``DataFrame`` 中的零基行索引。
            stats: 按证券累计停牌和零成交异常的可变统计字典。

        返回：
            任一确定性数值校验失败时返回 ``True``。
        """

        values = {name: _finite_float(row.get(name)) for name in (
            "open", "high", "low", "close", "pre_close", "volume", "amount", "suspend_flag"
        )}
        invalid = False
        for field_name in ("open", "high", "low", "close", "pre_close", "volume", "amount", "suspend_flag"):
            if values[field_name] is None:
                invalid = True
                self._row_issue(
                    "KLINE_FIELD_INVALID",
                    "ERROR",
                    "日线必需数值字段缺失、无穷或无法转换为有限数。",
                    code,
                    date_value,
                    field_name,
                    "有限数值",
                    str(row.get(field_name, "")),
                    "QMT 哨兵值、CSV 字段错位、空值或非数字文本",
                    "核对 QMT 原始返回并重新下载该证券日期",
                    path,
                    index,
                )
        if any(values[name] is None for name in values):
            return True
        open_price = values["open"]
        high = values["high"]
        low = values["low"]
        close = values["close"]
        pre_close = values["pre_close"]
        volume = values["volume"]
        amount = values["amount"]
        suspend = values["suspend_flag"]
        assert None not in (open_price, high, low, close, pre_close, volume, amount, suspend)
        if min(open_price, high, low, close, pre_close) <= 0:
            invalid = True
            self._row_issue(
                "NON_POSITIVE_PRICE",
                "ERROR",
                "日线价格字段包含零或负数。",
                code,
                date_value,
                "open,high,low,close,pre_close",
                "所有价格 > 0",
                "open={0}, high={1}, low={2}, close={3}, pre_close={4}".format(
                    open_price, high, low, close, pre_close
                ),
                "行情字段损坏、哨兵值未转换或代码映射错误",
                "核对 QMT 原始日线并重新下载该证券日期",
                path,
                index,
            )
        if high < max(open_price, low, close) or low > min(open_price, high, close):
            invalid = True
            self._row_issue(
                "INVALID_OHLC_RELATION",
                "ERROR",
                "日线最高价或最低价与开收盘价关系不合法。",
                code,
                date_value,
                "open,high,low,close",
                "high >= max(open, low, close) 且 low <= min(open, high, close)",
                "open={0}, high={1}, low={2}, close={3}".format(open_price, high, low, close),
                "字段错位、部分字段来自不同记录或源行情损坏",
                "核对 QMT 原始记录并重建该日期分区",
                path,
                index,
            )
        if volume < 0 or amount < 0:
            invalid = True
            self._row_issue(
                "NEGATIVE_TURNOVER",
                "ERROR",
                "成交量或成交额为负数。",
                code,
                date_value,
                "volume,amount",
                "volume >= 0 且 amount >= 0",
                "volume={0}, amount={1}".format(volume, amount),
                "数值溢出、字段转换错误或行情记录损坏",
                "核对 QMT 原始记录并重建该日期分区",
                path,
                index,
            )
        if suspend not in (0.0, 1.0):
            invalid = True
            self._row_issue(
                "INVALID_SUSPEND_FLAG",
                "ERROR",
                "停牌标志不是 0 或 1。",
                code,
                date_value,
                "suspend_flag",
                "0=未停牌，1=停牌",
                str(suspend),
                "QMT 字段映射错误、CSV 错位或未知接口版本",
                "核对 suspendFlag 原始值及下载器字段映射",
                path,
                index,
            )
        elif suspend == 0.0:
            if volume <= 0:
                invalid = True
                stats[code].active_zero_volume_rows += 1
                self._row_issue(
                    "ACTIVE_ZERO_VOLUME",
                    "ERROR",
                    "QMT 标记为未停牌，但成交量不大于零。",
                    code,
                    date_value,
                    "volume",
                    "suspend_flag=0 时 volume > 0",
                    "suspend_flag=0, volume={0}, amount={1}".format(volume, amount),
                    "停牌标志错误、行情补齐异常或本地行情未完整下载",
                    "核对 QMT 原始 suspendFlag 与 volume，并重新下载该证券日期",
                    path,
                    index,
                )
            if amount <= 0:
                invalid = True
                self._row_issue(
                    "ACTIVE_ZERO_AMOUNT",
                    "ERROR",
                    "QMT 标记为未停牌，但成交额不大于零。",
                    code,
                    date_value,
                    "amount",
                    "suspend_flag=0 时 amount > 0",
                    "suspend_flag=0, volume={0}, amount={1}".format(volume, amount),
                    "成交额字段缺失、停牌标志错误或行情补齐异常",
                    "核对 QMT 原始 amount，并重新下载该证券日期",
                    path,
                    index,
                )
        else:
            stats[code].suspended_rows += 1
            if volume != 0 or amount != 0:
                invalid = True
                self._row_issue(
                    "SUSPENDED_WITH_TURNOVER",
                    "ERROR",
                    "QMT 标记为停牌，但记录仍包含成交量或成交额。",
                    code,
                    date_value,
                    "suspend_flag,volume,amount",
                    "suspend_flag=1 时 volume=0 且 amount=0",
                    "suspend_flag=1, volume={0}, amount={1}".format(volume, amount),
                    "停牌字段错位、批次拼接错误或 QMT 返回口径异常",
                    "查看 QMT 原始返回；确认前不要自动修改停牌标志",
                    path,
                    index,
                )
            if not (open_price == high == low == close):
                self._row_issue(
                    "SUSPENDED_PRICE_NOT_FLAT",
                    "WARNING",
                    "停牌补齐行的开高低收并不完全相等。",
                    code,
                    date_value,
                    "open,high,low,close",
                    "fill_data=True 的停牌补齐行通常满足 open=high=low=close",
                    "open={0}, high={1}, low={2}, close={3}".format(open_price, high, low, close),
                    "不同 QMT 版本补齐口径、停牌状态边界或源数据异常",
                    "在 QMT 客户端核对该日原始停牌行情；确认口径后决定是否修复",
                    path,
                    index,
                )
        return invalid

    def _validate_continuity(
        self,
        row: pd.Series,
        code: str,
        date_value: str,
        path: Path,
        index: Any,
        corporate_actions: set[tuple[str, str]],
        state: _ScanState,
    ) -> None:
        """检查跨日昨收连续性、极端涨跌和成交量数量级突变。

        参数：
            row: 当前证券日线原始字段。
            code: 当前证券代码。
            date_value: 当前交易日期。
            path: 当前记录来源 CSV 路径。
            index: 当前记录在 ``DataFrame`` 中的零基行索引。
            corporate_actions: 可解释昨收断层的公司行为主键集合。
            state: 保存上一行及滚动成交量历史的跨日状态。

        返回：
            无返回值；统计异常以 ``WARNING`` 或 ``INFO`` 追加到问题集合。
        """

        current = {
            name: _finite_float(row.get(name))
            for name in ("close", "pre_close", "volume", "suspend_flag")
        }
        previous = state.last_row.get(code)
        pre_close = current["pre_close"]
        if previous is not None and pre_close and previous.get("close"):
            relative = abs(pre_close / previous["close"] - 1.0)
            if relative > 1e-6:
                has_action = (code, date_value) in corporate_actions
                self._row_issue(
                    "PRE_CLOSE_DISCONTINUITY_EXPLAINED" if has_action else "PRE_CLOSE_DISCONTINUITY",
                    "INFO" if has_action else "WARNING",
                    "当日 pre_close 与上一条实际日线 close 不一致。",
                    code,
                    date_value,
                    "pre_close",
                    "上一实际日线 close={0}".format(previous["close"]),
                    "pre_close={0}, 相对差异={1:.4%}".format(pre_close, relative),
                    "当日存在公司行为记录" if has_action else "除权除息、数据缺口、复权口径变化或行情异常",
                    "无需修复，保留公司行为解释" if has_action else "核对公司行为和 QMT 复权口径，再判断是否重下数据",
                    path,
                    index,
                )
        close = current["close"]
        if close and pre_close:
            change = close / pre_close - 1.0
            if abs(change) > self.config.price_jump_warning_ratio:
                self._row_issue(
                    "ABNORMAL_PRICE_JUMP",
                    "WARNING",
                    "单日收盘相对昨收变化超过统计告警阈值。",
                    code,
                    date_value,
                    "close,pre_close",
                    "abs(close / pre_close - 1) <= {0:.2%}".format(
                        self.config.price_jump_warning_ratio
                    ),
                    "close={0}, pre_close={1}, 涨跌={2:.2%}".format(close, pre_close, change),
                    "真实极端行情、除权口径、价格单位变化或数据损坏",
                    "结合涨跌停规则和公司行为核对，不应仅凭统计告警修改数据",
                    path,
                    index,
                )
        volume = current["volume"]
        suspend = current["suspend_flag"]
        history = state.volume_history.setdefault(
            code, deque(maxlen=self.config.volume_history_window)
        )
        if volume is not None and volume > 0 and suspend == 0.0:
            if len(history) >= self.config.volume_history_min_periods:
                median = statistics.median(history)
                if median > 0:
                    ratio = max(volume / median, median / volume)
                    if ratio >= self.config.volume_scale_warning_ratio:
                        self._row_issue(
                            "ABNORMAL_VOLUME_SCALE",
                            "WARNING",
                            "成交量相对近期中位数发生异常数量级变化。",
                            code,
                            date_value,
                            "volume",
                            "与前 {0} 个有效交易日中位数的倍数 < {1:g}".format(
                                len(history), self.config.volume_scale_warning_ratio
                            ),
                            "volume={0}, 历史中位数={1}, 最大倍数={2:.2f}".format(
                                volume, median, ratio
                            ),
                            "真实放量/缩量、成交量单位改变、重复累计或数据源版本变化",
                            "核对异常日期边界两侧的 QMT 原始成交量单位，不要自动缩放",
                            path,
                            index,
                        )
            history.append(volume)
        state.last_row[code] = {
            "date": date_value,
            "close": close,
        }

    def _close_missing_span(
        self,
        code: str,
        end_position: int,
        end_date: str,
        output: list[dict[str, Any]],
        state: _ScanState,
    ) -> None:
        """关闭一段连续交易日缺口并同时生成详细错误。

        参数：
            code: 存在连续缺口的证券代码。
            end_position: 缺失区间末日在审计日历中的零基位置。
            end_date: 缺失区间末日的八位交易日期。
            output: 累积缺失区间报告行的可变列表。
            state: 保存每只证券当前缺口起点的跨日状态。

        返回：
            无返回值；若该证券没有打开的缺口则不做任何操作。
        """

        opened = state.missing_open.pop(code, None)
        if opened is None or end_position < opened[0]:
            return
        start_position, start_date = opened
        days = end_position - start_position + 1
        evidence = "证券属于分区证券池，且这些日期位于上市退市有效区间；停牌日也应有补齐行"
        output.append(
            {
                "code": code,
                "start_date": start_date,
                "end_date": end_date,
                "trading_days": days,
                "evidence": evidence,
            }
        )
        self._add_issue(
            "KLINE_MISSING_SPAN",
            "ERROR",
            "kline_1d",
            "证券在生命周期内存在连续交易日日线缺失。",
            code=code,
            start_date=start_date,
            end_date=end_date,
            expected="连续 {0} 个交易日均有一条日线记录".format(days),
            actual="{0} 至 {1} 共缺失 {2} 个交易日".format(start_date, end_date, days),
            evidence=evidence,
            possible_causes="QMT 本地行情未下载完整、下载批次失败或日期分区写入不完整",
            suggested_action="核对 missing_spans.csv，并使用 repair 模式重建涉及的日期分区",
            source_file=str(self.root / "kline_1d"),
        )

    def _row_issue(
        self,
        issue_code: str,
        level: str,
        message: str,
        code: str,
        date_value: str,
        field_name: str,
        expected: str,
        actual: str,
        possible_causes: str,
        suggested_action: str,
        path: Path,
        index: Any,
    ) -> None:
        """生成包含证券、日期、字段、源文件和 CSV 行号的逐行问题。

        参数：
            issue_code: 稳定的机器可读错误编号。
            level: ``ERROR``、``WARNING`` 或 ``INFO`` 严重级别。
            message: 对问题本身的简明中文说明。
            code: 当前证券代码。
            date_value: 当前八位交易日期。
            field_name: 触发问题的一个或多个字段名称。
            expected: 业务规则要求的理论结果。
            actual: 当前记录的实际值和必要上下文。
            possible_causes: 不擅自修正数据前可供排查的可能原因。
            suggested_action: 推荐的人工核对或可恢复修复动作。
            path: 当前记录来源 CSV 文件路径。
            index: 当前记录在 ``DataFrame`` 中的零基行索引。

        返回：
            无返回值；问题会追加到当前自检器的问题集合。
        """

        try:
            source_row: int | str = int(index) + 2
        except (TypeError, ValueError):
            source_row = ""
        self._add_issue(
            issue_code,
            level,
            "kline_1d",
            message,
            code=code,
            date=date_value,
            field=field_name,
            expected=expected,
            actual=actual,
            evidence="源记录已按证券代码、交易日期和字段规则逐项校验",
            possible_causes=possible_causes,
            suggested_action=suggested_action,
            source_file=str(path),
            source_row=source_row,
        )

    def _add_issue(
        self,
        issue_code: str,
        level: str,
        dataset: str,
        message: str,
        **details: Any,
    ) -> None:
        """追加一条字段完整且严重级别合法的问题。

        参数：
            issue_code: 稳定的机器可读错误编号。
            level: ``ERROR``、``WARNING`` 或 ``INFO`` 严重级别。
            dataset: 发生问题的数据集名称。
            message: 面向使用者的中文错误说明。
            **details: ``AuditIssue`` 其余定位、证据、原因和建议字段。

        返回：
            无返回值；严重级别非法时抛出 ``ValueError``。
        """

        if level not in {"ERROR", "WARNING", "INFO"}:
            raise ValueError("未知问题级别: {0}".format(level))
        if level == "ERROR":
            details.setdefault("expected", "请结合 issue_code 核对对应业务规则")
            details.setdefault("actual", "未提供原始值；请查看 source_file 并复核源数据")
            details.setdefault(
                "evidence", "该问题由全样本结构、字段或生命周期规则触发"
            )
            details.setdefault(
                "possible_causes", "数据下载失败、文件损坏、字段口径变化或配置不一致"
            )
            details.setdefault(
                "suggested_action", "查看 issues.csv 的证据后重新核对或 repair 相关数据"
            )
            details.setdefault("source_file", str(self.root))
        self.issues.append(
            AuditIssue(
                issue_code=issue_code,
                level=level,
                dataset=dataset,
                message=message,
                **details,
            )
        )

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


def run_full_sample_self_check(config: SelfCheckConfig) -> AuditResult:
    """使用给定配置执行一次 QMT 全样本数据内部质量自检。

    参数：
        config: 数据根目录、交易日历、日期范围、报告路径和阈值配置。

    返回：
        包含退出状态、统计摘要和报告路径的完整审计结果。
    """

    return QmtDataSelfChecker(config).run()


def _read_calendar_csv(path: Path) -> list[str]:
    """从一列或含标准日期列的 CSV 读取去重交易日。

    参数：
        path: QMT 导出的交易日历 CSV，优先读取 ``trade_date``、``date`` 或 ``time`` 列。

    返回：
        严格升序且去重的八位交易日期列表。
    """

    calendar_path = Path(path)
    try:
        frame = pd.read_csv(str(calendar_path), encoding="utf-8-sig", dtype=str)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError("无法读取交易日历 {0}: {1}".format(calendar_path, exc)) from exc
    column = next((name for name in ("trade_date", "date", "time") if name in frame.columns), None)
    if column is None and len(frame.columns) == 1:
        column = str(frame.columns[0])
    if column is None:
        raise ValueError("交易日历必须包含 trade_date、date、time 之一或仅有一列")
    dates = [_normalize_date_text(value) for value in frame[column].tolist()]
    invalid = [str(value) for value, normalized in zip(frame[column].tolist(), dates) if normalized is None]
    if invalid:
        raise ValueError("交易日历包含非法日期，示例: {0}".format(_sample(invalid)))
    return sorted(set(value for value in dates if value is not None))


def _normalize_date_text(value: Any) -> str | None:
    """将 CSV 日期、时间戳或数字文本规范为八位日期。

    参数：
        value: 可能来自 QMT CSV、JSON 或 pandas 的日期值。

    返回：
        合法的 ``YYYYMMDD`` 日期；空值、无日期哨兵或非法值返回 ``None``。
    """

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none", "99999999", "0"}:
        return None
    qmt_date = _qmt_normalize_date(value)
    if qmt_date is not None:
        try:
            datetime.strptime(qmt_date, "%Y%m%d")
            return qmt_date
        except ValueError:
            return None
    digits = "".join(character for character in text if character.isdigit())
    if len(digits) >= 8:
        candidate = digits[:8]
        try:
            datetime.strptime(candidate, "%Y%m%d")
            return candidate
        except ValueError:
            pass
    try:
        return pd.Timestamp(text).strftime("%Y%m%d")
    except (TypeError, ValueError, OverflowError):
        return None


def _parse_lifecycle_value(value: Any) -> tuple[str | None, bool]:
    """解析上市或退市日期并区分合法空值与非法非空文本。

    参数：
        value: instrument_info 中的上市日期或退市日期原始值。

    返回：
        ``(日期, 是否为非法非空值)``；空值和 99999999 返回 ``(None, False)``。
    """

    if value is None:
        return None, False
    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none", "99999999", "0"}:
        return None, False
    normalized = _normalize_date_text(value)
    return normalized, normalized is None


def _optional_int(value: Any) -> int | None:
    """把完成标记中的行数转换为整数并识别非法值。

    参数：
        value: ``_SUCCESS.json`` 中的 rows 原始值。

    返回：
        非负整数；缺失、布尔值或非法文本返回 ``None``。
    """

    if isinstance(value, bool) or value is None:
        return None
    try:
        converted = int(value)
    except (TypeError, ValueError):
        return None
    return converted if converted >= 0 else None


def _optional_date(value: str | None, field_name: str) -> None:
    """校验可选八位日期参数。

    参数：
        value: 可为空的 ``YYYYMMDD`` 日期字符串。
        field_name: 用于错误消息的配置字段名称。

    返回：
        无返回值；日期非法时抛出 ``ValueError``。
    """

    if value is not None and _normalize_date_text(value) != value:
        raise ValueError("{0} 必须是 YYYYMMDD 日期".format(field_name))


def _finite_float(value: Any) -> float | None:
    """把行情字段转换为有限浮点数并排除 QMT 最大双精度哨兵值。

    参数：
        value: CSV 中的价格、成交量、成交额或停牌标志原始值。

    返回：
        有限浮点数；空值、无穷、非数字或哨兵值返回 ``None``。
    """

    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(converted) or abs(converted) >= 1.7976931348623157e308:
        return None
    return converted


def _file_sha256(path: Path) -> str:
    """分块计算本地文件 SHA-256。

    参数：
        path: 需要校验内容是否与完成标记一致的文件路径。

    返回：
        六十四位小写十六进制摘要。
    """

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _csv_row_count(path: Path) -> int:
    """读取 CSV 物理行数并扣除一行表头。

    参数：
        path: 不应含嵌入换行字段的 QMT 标准分区 CSV。

    返回：
        不含表头的非负物理行数。
    """

    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def _sample(values: Iterable[Any], limit: int = 10) -> str:
    """把较长代码或日期序列压缩为可读示例。

    参数：
        values: 需要在错误证据中展示的任意可迭代值。
        limit: 最多展示的元素数量，缺省为十个。

    返回：
        逗号分隔的示例文本；超出上限时追加省略说明。
    """

    items = [str(value) for value in values]
    shown = items[:limit]
    suffix = " ... 共 {0} 项".format(len(items)) if len(items) > limit else ""
    return ", ".join(shown) + suffix


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    """先写临时文件再原子替换 JSON 报告。

    参数：
        path: 最终 JSON 报告路径。
        value: 需要以 UTF-8 中文格式序列化的摘要字典。

    返回：
        无返回值。
    """

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
    temporary.replace(path)


def _write_text_atomic(path: Path, value: str) -> None:
    """先写临时文件再原子替换 UTF-8 文本报告。

    参数：
        path: 最终 Markdown 或文本报告路径。
        value: 需要完整写入的文本内容。

    返回：
        无返回值。
    """

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    """先写临时文件再原子替换 UTF-8 BOM CSV 报告。

    参数：
        path: 最终 CSV 报告路径。
        frame: 需要写出的结构化报告表。

    返回：
        无返回值。
    """

    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(str(temporary), index=False, encoding="utf-8-sig")
    temporary.replace(path)


def _summary_markdown(summary: dict[str, Any]) -> str:
    """把机器可读摘要转换为简洁 Markdown。

    参数：
        summary: 包含审计范围、覆盖率、问题数量和错误编号统计的摘要。

    返回：
        可直接写入 ``summary.md`` 的中文 Markdown 文本。
    """

    lines = [
        "# QMT 全样本数据自检报告",
        "",
        "- 结论：{0}".format("未通过" if summary["status"] == "failed" else "通过"),
        "- 数据根目录：`{0}`".format(summary["output_root"]),
        "- 审计区间：{0} 至 {1}".format(summary["start_date"], summary["end_date"]),
        "- 交易日：{0}".format(summary["trading_days"]),
        "- 证券数：{0}".format(summary["symbols"]),
        "- 理论应有记录：{0}".format(summary["expected_rows"]),
        "- 实际有效范围记录：{0}".format(summary["actual_expected_rows"]),
        "- 缺失记录：{0}".format(summary["missing_rows"]),
        "- 连续缺失区间：{0}".format(summary["missing_spans"]),
        "- ERROR：{0}".format(summary["errors"]),
        "- WARNING：{0}".format(summary["warnings"]),
        "- INFO：{0}".format(summary["info"]),
        "- 使用独立 QMT 交易日历：{0}".format("是" if summary["authoritative_calendar"] else "否"),
        "",
        "## 问题类型统计",
        "",
        "| 错误编号 | 数量 |",
        "|---|---:|",
    ]
    for code, count in summary["issue_codes"].items():
        lines.append("| `{0}` | {1} |".format(code, count))
    lines.extend(
        [
            "",
            "完整定位、理论值、实际值、证据、可能原因和处理建议见 `issues.csv`。",
            "连续缺失日期见 `missing_spans.csv`，逐日和逐证券覆盖率见对应 CSV。",
            "脚本只报告问题，不会修改、填充或删除任何原始数据。",
            "",
        ]
    )
    return "\n".join(lines)
