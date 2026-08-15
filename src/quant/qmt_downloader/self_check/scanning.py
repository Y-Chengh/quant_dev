# -*- coding: utf-8 -*-
"""按交易日扫描分区并读取日线与 staging 数据。"""

from __future__ import annotations

import bisect
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from .base import _CheckerState
from .models import _ScanState, _SymbolStats
from .utils import _file_sha256, _sample


class _CalendarScanMixin(_CheckerState):
    """按交易日逐日扫描分区，读取正式日线与 staging 片段。"""

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
        self.logger.info(
            "[自检] 待扫描交易日 %d 个，证券池 %d 只", len(calendar), len(symbols)
        )
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
                    expected=str(self._expected_partition_path(date_value)),
                    actual="日期目录不存在",
                    evidence="理论应有证券 {0} 只，实际记录 0 条".format(len(active)),
                    possible_causes="当日任务未运行、下载失败、水位错误推进或目录被删除",
                    suggested_action="检查 downloader.log 与 run_complete 水位，并使用 repair 补齐该日期",
                    source_file=str(self._expected_partition_path(date_value)),
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
                        (directory or self._expected_partition_path(date_value))
                        / ("" if self.source_mode == "staging" else "data.csv")
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
                    source_file=str(directory or self._expected_partition_path(date_value)),
                )
            self._log_step_progress(
                "逐日扫描",
                position,
                len(calendar),
                "date={0} 行数={1} 覆盖率={2:.2%} 累计问题={3}".format(
                    date_value, len(frame), coverage, len(self.issues)
                ),
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

        if self.source_mode == "staging":
            return self._read_staging_partition(directory, date_value)
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

    def _expected_partition_path(self, date_value: str) -> Path:
        """返回指定交易日在当前数据布局中的预期路径。

        参数：
            date_value: 八位交易日期。
        返回：
            最终分区的 ``data.csv`` 父目录，或 staging 的按日目录。
        """

        if self.source_mode == "staging":
            return self.data_root / ("kline_daily_" + date_value)
        return self.data_root / "kline_1d" / ("date=" + date_value)

    def _read_staging_partition(self, directory: Path, date_value: str) -> pd.DataFrame:
        """读取 staging 日目录下的批次 CSV，并校验批次元数据行数。

        参数：
            directory: ``kline_daily_YYYYMMDD`` staging 日目录。
            date_value: 目录对应的八位交易日期。
        返回：
            合并后的日线 DataFrame；没有可读批次时返回空表并记录详细错误。
        """

        paths = sorted(directory.glob("batch_*.csv"), key=lambda item: item.name)
        if not paths:
            self._add_issue(
                "STAGING_BATCHES_MISSING",
                "ERROR",
                "kline_1d",
                "staging 日目录存在但没有 batch_*.csv 数据批次。",
                date=date_value,
                expected="至少存在一个 batch_*.csv 及其可选 .meta.json",
                actual="未找到 CSV 批次",
                evidence=str(directory),
                possible_causes="下载任务中断、批次被删除或目录尚未写完",
                suggested_action="检查 downloader 日志并重新下载该交易日",
                source_file=str(directory),
            )
            return pd.DataFrame()
        frames: list[pd.DataFrame] = []
        for path in paths:
            try:
                frame = pd.read_csv(
                    str(path), encoding="utf-8-sig", dtype={"code": str, "trade_date": str}
                )
            except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
                self._add_issue(
                    "KLINE_PARTITION_UNREADABLE",
                    "ERROR",
                    "kline_1d",
                    "staging 行情批次 CSV 无法读取。",
                    date=date_value,
                    expected="合法 UTF-8 CSV",
                    actual="{0}: {1}".format(type(exc).__name__, exc),
                    evidence="批次文件读取失败",
                    possible_causes="文件截断、编码错误、列数错位或空文件",
                    suggested_action="删除并重新生成该交易日 staging 批次",
                    source_file=str(path),
                )
                continue
            metadata_path = path.with_suffix(".meta.json")
            if not metadata_path.is_file():
                self._add_issue(
                    "STAGING_BATCH_METADATA_MISSING",
                    "ERROR",
                    "kline_1d",
                    "staging 批次缺少对应的 .meta.json 完成元数据。",
                    date=date_value,
                    expected=str(metadata_path),
                    actual="文件不存在",
                    evidence=str(path),
                    possible_causes="批次写入尚未完成、元数据被删除或文件被单独复制",
                    suggested_action="重新生成该批次并确认 CSV 与 .meta.json 同时落盘",
                    source_file=str(path),
                )
            else:
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    self._add_issue(
                        "STAGING_BATCH_METADATA_INVALID",
                        "ERROR",
                        "kline_1d",
                        "staging 批次元数据无法解析。",
                        date=date_value,
                        expected="包含 rows 和 sha256 的 JSON 对象",
                        actual="{0}: {1}".format(type(exc).__name__, exc),
                        evidence=str(metadata_path),
                        possible_causes="元数据写入中断或文件被修改",
                        suggested_action="重新生成该批次的 CSV 和 .meta.json",
                        source_file=str(metadata_path),
                    )
                else:
                    if not isinstance(metadata, dict):
                        self._add_issue(
                            "STAGING_BATCH_METADATA_INVALID",
                            "ERROR",
                            "kline_1d",
                            "staging 批次元数据不是 JSON 对象。",
                            date=date_value,
                            expected="JSON object containing rows and sha256",
                            actual=type(metadata).__name__,
                            evidence=str(metadata_path),
                            possible_causes="元数据文件格式损坏",
                            suggested_action="重新生成该批次的 CSV 和 .meta.json",
                            source_file=str(metadata_path),
                        )
                    else:
                        expected_rows = metadata.get("rows")
                        try:
                            rows_match = int(expected_rows) == len(frame)
                        except (TypeError, ValueError):
                            rows_match = False
                        if not rows_match:
                            self._add_issue(
                                "STAGING_BATCH_ROW_COUNT_MISMATCH",
                                "ERROR",
                                "kline_1d",
                                "staging 批次实际行数与元数据不一致。",
                                date=date_value,
                                field="rows",
                                expected=str(expected_rows),
                                actual=str(len(frame)),
                                evidence=str(path),
                                possible_causes="批次写入中断、文件被追加或元数据过期",
                                suggested_action="重新生成该批次并确认写入完成标记",
                                source_file=str(metadata_path),
                            )
                        expected_hash = str(metadata.get("sha256", ""))
                        if not expected_hash:
                            self._add_issue(
                                "STAGING_BATCH_HASH_MISSING",
                                "ERROR",
                                "kline_1d",
                                "staging 批次元数据缺少 sha256。",
                                date=date_value,
                                field="sha256",
                                expected="非空 SHA-256 字符串",
                                actual=repr(metadata.get("sha256")),
                                evidence=str(metadata_path),
                                possible_causes="元数据写入不完整或使用了旧版下载器",
                                suggested_action="重新生成批次元数据，或使用 --verify-staging-hash 做完整校验",
                                source_file=str(metadata_path),
                            )
                        elif self.config.verify_staging_hash:
                            actual_hash = _file_sha256(path)
                            if expected_hash != actual_hash:
                                self._add_issue(
                                    "STAGING_BATCH_HASH_MISMATCH",
                                    "ERROR",
                                    "kline_1d",
                                    "staging 批次内容与 .meta.json 的 SHA-256 不一致。",
                                    date=date_value,
                                    field="sha256",
                                    expected=expected_hash,
                                    actual=actual_hash,
                                    evidence=str(path),
                                    possible_causes="CSV 被追加/修改、磁盘损坏或元数据来自另一份批次",
                                    suggested_action="丢弃该批次并重新下载，确认写入完成后再审计",
                                    source_file=str(metadata_path),
                                )
            frame = frame.copy()
            frame["_source_file"] = str(path)
            frame["_source_row"] = range(2, len(frame) + 2)
            frames.append(frame)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)
