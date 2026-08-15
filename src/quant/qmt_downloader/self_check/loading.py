# -*- coding: utf-8 -*-
"""证券生命周期、交易日历、证券池与除权事件的装载。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .base import _CheckerState
from .models import _ScanState, _SymbolStats
from .utils import (
    _file_sha256,
    _normalize_date_text,
    _optional_int,
    _parse_lifecycle_value,
    _read_calendar_csv,
    _sample,
    _select_paths_in_range,
)


class _ReferenceLoadingMixin(_CheckerState):
    """装载上市退市信息、交易日历、证券池与除权事件主键。"""

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
        if extra_dates:
            self.logger.info("[自检] 日历外分区 %d 个待审计", len(extra_dates))
        for index, date_value in enumerate(extra_dates):
            self._log_step_progress(
                "日历外分区", index, len(extra_dates), "date=" + date_value, pending=True
            )
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
                source_file=str(
                    directory
                    if self.source_mode == "staging"
                    else directory / "data.csv"
                ),
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
        # 逐只证券扫一遍整列求首次出现的行号是 O(证券数 × 快照行数)，五千只证券要比较
        # 两千五百万次；改为一次遍历建立代码到首个索引标签的映射，结果完全相同。
        first_labels: dict[Any, Any] = {}
        for label, value in zip(normalized.index, normalized["code"]):
            first_labels.setdefault(value, label)
        for code in symbols:
            if code not in first_rows.index:
                lifecycle[code] = (calendar[0], None)
                continue
            row = first_rows.loc[code]
            row_number = int(first_labels[code]) + 2
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
            ``(证券代码, 除权日期)`` 集合，用于解释跨日价格连续性差异。行级校验只按
            ``(代码, 当日日期)`` 查询该集合，因此只读取审计区间内的除权分区即可，
            区间外的事件不会被查到。
        """

        keys: set[tuple[str, str]] = set()
        root = self.root / "corporate_actions"
        if not root.is_dir():
            return keys
        directories = _select_paths_in_range(
            sorted(root.glob("ex_date=*")),
            r"ex_date=(\d{8})",
            self.config.start_date,
            self.config.end_date,
        )
        self.logger.info("[自检] 待读取除权事件分区 %d 个", len(directories))
        for index, directory in enumerate(directories):
            self._log_step_progress(
                "除权事件", index, len(directories), directory.name, pending=True
            )
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
