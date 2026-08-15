# -*- coding: utf-8 -*-
"""原始财务抓取、日级快照生成与两类财务分区落盘。"""

import pandas as pd

from ..finance import finance_daily_columns, materialize_finance_daily
from ..gateway import FINANCE_FIELDS
from .base import _RunnerState
from .helpers import (
    _contains_error,
    _contains_finance_missing,
    _ensure_columns,
    _finance_templates,
)


class _FinanceCollectionMixin(_RunnerState):
    """抓取原始财务、按公告日生成日级快照并写出财务分区。"""

    def _collect_finance(self):
        """按证券批次读取所有财务表并分别保存 staging。

        返回：
            至少一个财务批次未完成时返回 ``True``，全部 staging 可靠时返回 ``False``。
        """
        state_dataset = "finance_raw"
        failed = False
        for batch_id, symbols in enumerate(self.batches):
            fragments_exist = all(
                self.store.fragment_exists(self.job_key, "finance_{0}".format(name), batch_id)
                for name in FINANCE_FIELDS
            )
            if self.checkpoints.is_completed(self.job_key, state_dataset, batch_id) and fragments_exist:
                self._log_batch_progress(state_dataset, batch_id, symbols, "断点跳过")
                self.logger.info("断点命中 dataset=%s batch=%d", state_dataset, batch_id)
                continue
            self._log_batch_progress(state_dataset, batch_id, symbols, "开始")
            self.checkpoints.mark_running(self.job_key, state_dataset, batch_id)
            try:
                frames, issues = self.gateway.fetch_finance(
                    symbols,
                    self.config.finance_lookback_start,
                    self.config.end_date,
                )
                self.issues.extend(issues)
                self._log_issue_details(state_dataset, batch_id, symbols, issues)
                row_count = 0
                for table_name in FINANCE_FIELDS:
                    frame = frames.get(table_name, pd.DataFrame())
                    self.store.write_fragment(
                        self.job_key,
                        "finance_{0}".format(table_name),
                        batch_id,
                        frame,
                    )
                    row_count += len(frame)
                finance_missing = _contains_finance_missing(issues)
                if (
                    row_count == 0
                    or _contains_error(issues)
                    or (finance_missing and not self.config.allow_partial_finance)
                ):
                    failed = True
                    reason = "批次财务数据全空、包含接口错误或不允许的部分缺失"
                    self.checkpoints.mark_failed(self.job_key, state_dataset, batch_id, reason)
                    self._log_batch_progress(state_dataset, batch_id, symbols, "失败")
                    self.logger.error("批次未完成 dataset=%s batch=%d reason=%s", state_dataset, batch_id, reason)
                    continue
                self.checkpoints.mark_completed(self.job_key, state_dataset, batch_id, row_count)
                self._log_batch_progress(state_dataset, batch_id, symbols, "完成")
                self.logger.info("批次完成 dataset=%s batch=%d rows=%d", state_dataset, batch_id, row_count)
            except Exception as error:
                failed = True
                self.checkpoints.mark_failed(self.job_key, state_dataset, batch_id, error)
                self._log_batch_progress(state_dataset, batch_id, symbols, "失败")
                self.logger.exception("批次失败 dataset=%s batch=%d", state_dataset, batch_id)
        return failed


    def _prepare_finance_fragments(self, trade_dates):
        """逐证券批次生成原始财务日片段和日快照片段。

        参数：
            trade_dates: 大 QMT 交易日历返回的实际交易日序列。

        返回：
            按财务表映射的待写原始公告日集合；只包含尚未完成或范围不符的分区。
        """
        templates = _finance_templates()
        raw_dates = {table_name: set() for table_name in FINANCE_FIELDS}
        raw_scope_cache = {table_name: {} for table_name in FINANCE_FIELDS}
        needed_daily_dates = [
            trade_date
            for trade_date in trade_dates
            if not self._can_reuse_partition(
                "finance_daily", "date", trade_date, self.partition_scope
            )
        ] if "finance_daily" in self.config.datasets else []
        if "finance_raw" in self.config.datasets:
            for table_name in FINANCE_FIELDS:
                for trade_date in trade_dates:
                    matches = self._can_reuse_partition(
                        "finance_raw/table={0}".format(table_name),
                        "announce_date",
                        trade_date,
                        self.partition_scope,
                    )
                    raw_scope_cache[table_name][trade_date] = matches
                    if not matches:
                        raw_dates[table_name].add(trade_date)

        for batch_id, symbols in enumerate(self.batches):
            table_frames = {}
            for table_name in FINANCE_FIELDS:
                frame = self.store.read_fragments(
                    self.job_key, "finance_{0}".format(table_name), [batch_id]
                )
                frame = _ensure_columns(frame, list(templates[table_name].columns))
                table_frames[table_name] = frame
                if "finance_raw" in self.config.datasets:
                    grouped = frame.copy()
                    grouped["_partition_date"] = grouped["announce_date"].where(
                        grouped["announce_date"].notna(), "unknown"
                    )
                    for announce_date, daily in grouped.groupby("_partition_date"):
                        date_value = str(announce_date)
                        dataset_path = "finance_raw/table={0}".format(table_name)
                        matches = raw_scope_cache[table_name].get(date_value)
                        if matches is None:
                            matches = self._can_reuse_partition(
                                dataset_path,
                                "announce_date",
                                date_value,
                                self.partition_scope,
                            )
                            raw_scope_cache[table_name][date_value] = matches
                        if matches:
                            continue
                        raw_dates[table_name].add(date_value)
                        self.store.write_fragment(
                            self.job_key,
                            "finance_raw_{0}_{1}".format(table_name, date_value),
                            batch_id,
                            daily.drop(columns=["_partition_date"]),
                        )
            if needed_daily_dates:
                snapshot = materialize_finance_daily(
                    table_frames, needed_daily_dates, symbols
                )
                expected = finance_daily_columns(templates)
                snapshot = _ensure_columns(snapshot, expected)
                for trade_date in needed_daily_dates:
                    daily = snapshot[snapshot["trade_date"].astype(str) == trade_date]
                    self.store.write_fragment(
                        self.job_key,
                        "finance_daily_{0}".format(trade_date),
                        batch_id,
                        daily,
                    )
        return raw_dates

    def _write_finance_raw_partitions(self, raw_dates):
        """逐公告日合并批次片段并保存原始财务分区。

        参数：
            raw_dates: 按财务表映射的待写公告日集合。

        返回：
            无返回值；每次只把一个公告日的数据加载到内存。
        """
        templates = _finance_templates()
        for table_name, values in raw_dates.items():
            sorted_dates = sorted(values)
            for date_index, announce_date in enumerate(sorted_dates):
                self._log_date_progress(
                    "finance_raw/{0}".format(table_name),
                    date_index,
                    len(sorted_dates),
                    announce_date,
                    "开始写入",
                )
                frame = self.store.read_fragments(
                    self.job_key,
                    "finance_raw_{0}_{1}".format(table_name, announce_date),
                    range(len(self.batches)),
                )
                frame = _ensure_columns(frame, list(templates[table_name].columns))
                self._write_partition(
                    "finance_raw/table={0}".format(table_name),
                    "announce_date",
                    announce_date,
                    frame,
                    list(templates[table_name].columns),
                    ["code", "report_date", "announce_date"],
                    ["code", "report_date", "announce_date"],
                )
                self._log_date_progress(
                    "finance_raw/{0}".format(table_name),
                    date_index,
                    len(sorted_dates),
                    announce_date,
                    "完成",
                )

    def _write_finance_daily_partitions(self, trade_dates):
        """逐交易日合并证券批次快照并保存日财务分区。

        参数：
            trade_dates: 大 QMT 交易日历返回的实际交易日序列。

        返回：
            无返回值；已完成且范围一致的日期不会重新生成。
        """
        expected = finance_daily_columns(_finance_templates())
        for date_index, trade_date in enumerate(trade_dates):
            if self._can_reuse_partition(
                "finance_daily", "date", trade_date, self.partition_scope
            ):
                self._log_date_progress(
                    "finance_daily", date_index, len(trade_dates), trade_date, "断点跳过"
                )
                continue
            self._log_date_progress(
                "finance_daily", date_index, len(trade_dates), trade_date, "开始写入"
            )
            daily = self.store.read_fragments(
                self.job_key,
                "finance_daily_{0}".format(trade_date),
                range(len(self.batches)),
            )
            daily = _ensure_columns(daily, expected)
            missing = daily[daily["has_finance"] == 0]
            for code in missing["code"].astype(str):
                self.issues.add("WARNING", "finance_daily", code, trade_date, "该日收盘前没有可见财务记录")
            self._write_partition(
                "finance_daily",
                "date",
                trade_date,
                daily,
                expected,
                ["code", "trade_date"],
                ["code"],
            )
            self._log_date_progress(
                "finance_daily", date_index, len(trade_dates), trade_date, "完成"
            )
