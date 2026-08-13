# -*- coding: utf-8 -*-
"""编排批量回溯、日增量、断点恢复和按天分区落盘。"""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pandas as pd

from .finance import finance_daily_columns, materialize_finance_daily
from .gateway import (
    CORPORATE_ACTION_COLUMNS,
    FINANCE_FIELDS,
    INSTRUMENT_INFO_COLUMNS,
    KLINE_COLUMNS,
)
from .storage import IssueCollector
from .validation import find_missing_kline, validate_kline


class QmtDailyDownloader(object):
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
        self.symbols = []
        self.batches = []
        self.job_key = ""
        self.partition_scope = {}

    def run(self):
        """依次执行证券池解析、数据抽取、质量检查和日分区保存。

        返回：
            本次运行摘要字典，包含任务标识、证券数、交易日数、问题数和报告路径。
        """
        self.symbols = self.gateway.resolve_symbols(self.config.symbols, self.config.sector)
        self.partition_scope = {
            "schema_version": 2,
            "datasets": sorted(self.config.datasets),
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
            self.logger.info("增量数据已是最新，无需下载 summary=%s", json.dumps(summary, ensure_ascii=False))
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
        if "kline_1d" in self.config.datasets or "finance_daily" in self.config.datasets:
            failed = self._collect_kline(trade_dates)
            if failed:
                all_datasets_succeeded = False
                self.issues.add("ERROR", "kline_1d", "", "", "存在失败批次，未生成最终日线分区；下次以相同配置运行会从断点继续")
            if "kline_1d" in self.config.datasets:
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
                else:
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
            "issue_report": str(report_path),
        }
        self.logger.info("任务结束 summary=%s", json.dumps(summary, ensure_ascii=False))
        return summary

    def _collect_instrument_info(self):
        """读取全部证券的上市退市信息并保存当前快照表。

        返回：
            ``(DataFrame, failed)``；快照写入 ``instrument_info/snapshot=latest``，失败时
            返回已读取的数据和 ``True``。
        """
        self.logger.info("[instrument_info] 开始获取上市退市信息 证券 1/%d", len(self.symbols))
        frame, issues = self.gateway.fetch_instrument_info(self.symbols)
        self.issues.extend(issues)
        for issue in issues:
            log_method = self.logger.error if issue.get("level") == "ERROR" else self.logger.warning
            log_method(
                "详细问题 dataset=instrument_info symbols=%s issue_level=%s code=%s date=%s message=%s",
                ",".join(self.symbols),
                issue.get("level", "WARNING"),
                issue.get("code", ""),
                issue.get("date", ""),
                issue.get("message", ""),
            )
        failed = bool(issues) or len(frame) != len(self.symbols)
        if len(self.symbols):
            self.logger.info(
                "[instrument_info] 完成获取上市退市信息 证券 %d/%d (%.1f%%)",
                len(frame),
                len(self.symbols),
                len(frame) * 100.0 / len(self.symbols),
            )
        if not failed:
            self._write_partition(
                "instrument_info",
                "snapshot",
                "latest",
                frame,
                INSTRUMENT_INFO_COLUMNS,
                ["code"],
                ["code"],
                overwrite=True,
            )
        return frame, failed

    def _filter_kline_issues(self, instrument_info):
        """按上市和退市日期过滤日线缺失警告。

        参数：
            instrument_info: ``fetch_instrument_info`` 返回的证券生命周期信息表。

        返回：
            无返回值；未上市或已退市期间的缺失提示从问题报告中移除，其他缺失保留。
        """
        lifecycle = {}
        for row in instrument_info.to_dict("records"):
            lifecycle[str(row.get("code"))] = (
                str(row.get("open_date") or ""),
                str(row.get("expire_date") or ""),
            )
        kept = []
        removed = 0
        for issue in self.issues.items:
            if issue.get("dataset") != "kline_1d" or "填充后仍无日线" not in issue.get("message", ""):
                kept.append(issue)
                continue
            open_date, expire_date = lifecycle.get(str(issue.get("code")), ("", ""))
            date_value = str(issue.get("date") or "")
            if (open_date and date_value < open_date) or (expire_date and date_value > expire_date):
                removed += 1
                continue
            kept.append(issue)
        self.issues.items = kept
        self.logger.info("[instrument_info] 已过滤未上市/已退市期间 K 线缺失提示 %d 条", removed)

    def _finish_without_download(self, run_id, status):
        """在交易日历失败或区间无交易日时生成摘要并安全结束。

        参数：
            run_id: 当前运行的报告文件标识。
            status: ``calendar_error`` 或 ``no_trading_dates`` 状态文本。

        返回：
            不包含任何业务分区写入的任务摘要字典。
        """
        report_path = self.store.write_issue_report(self.issues.items, run_id)
        error_count = sum(1 for item in self.issues.items if item["level"] == "ERROR")
        summary = {
            "job_key": self.job_key,
            "symbols": len(self.symbols),
            "trade_dates": 0,
            "errors": error_count,
            "warnings": 0,
            "issue_report": str(report_path),
            "status": status,
        }
        self.logger.info("任务无业务下载 summary=%s", json.dumps(summary, ensure_ascii=False))
        return summary

    def _log_batch_progress(self, dataset, batch_id, symbols, status):
        """记录当前证券批次的序号、总数和百分比。

        参数：
            dataset: 当前数据集名称，如 ``kline_1d``、``finance_raw`` 或 ``corporate_actions``。
            batch_id: 当前批次的零基编号。
            symbols: 当前批次包含的证券代码序列。
            status: 进度事件状态，如“开始”、“完成”、“断点跳过”或“失败”。

        返回：
            无返回值；进度同时输出到 QMT 终端和日志文件。
        """
        total = max(len(self.batches), 1)
        current = min(batch_id + 1, total)
        percent = current * 100.0 / total
        first_index = batch_id * self.config.batch_size + 1
        last_index = first_index + len(symbols) - 1
        self.logger.info(
            "[%s] %s 批次 %d/%d (%.1f%%)，证券序号 %d-%d/%d",
            dataset,
            status,
            current,
            total,
            percent,
            first_index,
            last_index,
            len(self.symbols),
        )

    def _log_date_progress(self, dataset, index, total, date_value, status):
        """记录按交易日或公告日分区写入的进度。

        参数：
            dataset: 当前数据集名称。
            index: 当前日期在本次日期序列中的零基索引。
            total: 本次待处理的日期总数。
            date_value: 当前分区日期或 ``unknown``。
            status: 进度事件状态，如“开始”、“完成”或“跳过”。

        返回：
            无返回值。
        """
        total = max(int(total), 1)
        current = min(index + 1, total)
        percent = current * 100.0 / total
        self.logger.info(
            "[%s] %s 日期 %d/%d (%.1f%%) date=%s",
            dataset,
            status,
            current,
            total,
            percent,
            date_value,
        )

    def _parallel_save(self, tasks, stage, labels=None):
        """使用受控线程池并行执行互不冲突的本地文件写入任务。

        参数：
            tasks: 无参数可调用对象序列；每个任务只能写入独立文件路径，不得调用 QMT 接口。
            stage: 保存阶段名称，用于错误日志定位。
            labels: 与 ``tasks`` 对齐的目标日期或路径描述序列；缺省使用任务序号。

        返回：
            按输入任务顺序排列的返回值列表；任一任务失败时抛出包含任务序号和异常类型的
            ``RuntimeError``。QMT 接口请求始终在调用线程串行执行。
        """
        task_list = list(tasks)
        if labels is None:
            label_list = ["task-{0}".format(index + 1) for index in range(len(task_list))]
        else:
            label_list = list(labels)
        if len(label_list) != len(task_list):
            raise ValueError("并行保存任务 labels 数量必须与 tasks 一致")
        if not task_list:
            return []
        if self.config.save_workers == 1 or len(task_list) == 1:
            results = []
            for index, task in enumerate(task_list):
                try:
                    results.append(task())
                except Exception as error:
                    self.logger.exception(
                        "保存失败 stage=%s task=%d/%d label=%s error_type=%s error=%s",
                        stage,
                        index + 1,
                        len(task_list),
                        label_list[index],
                        type(error).__name__,
                        error,
                    )
                    raise RuntimeError(
                        "保存阶段失败 stage={0} task={1} label={2} {3}: {4}".format(
                            stage, index + 1, label_list[index], type(error).__name__, error
                        )
                    )
            return results
        self.logger.info(
            "[%s] 并行保存任务数=%d workers=%d",
            stage,
            len(task_list),
            self.config.save_workers,
        )
        results = [None] * len(task_list)
        errors = []
        with ThreadPoolExecutor(max_workers=self.config.save_workers) as executor:
            futures = [executor.submit(task) for task in task_list]
            for index, future in enumerate(futures):
                try:
                    results[index] = future.result()
                except Exception as error:
                    errors.append((index, error))
                    self.logger.exception(
                        "并行保存失败 stage=%s task=%d/%d label=%s error_type=%s error=%s",
                        stage,
                        index + 1,
                        len(task_list),
                        label_list[index],
                        type(error).__name__,
                        error,
                    )
        if errors:
            details = "; ".join(
                "task={0} label={1} {2}: {3}".format(
                    index + 1, label_list[index], type(error).__name__, error
                )
                for index, error in errors
            )
            raise RuntimeError("并行保存阶段失败 stage={0}: {1}".format(stage, details))
        return results

    def _log_issue_details(self, dataset, batch_id, symbols, issues):
        """将批次返回的每条接口或质量问题写入详细终端日志。

        参数：
            dataset: 当前数据集名称。
            batch_id: 当前证券批次编号。
            symbols: 当前批次的证券代码序列。
            issues: 问题字典序列，包含级别、代码、日期和消息。

        返回：
            无返回值；问题仍会保留在外部 CSV 报告中。
        """
        for issue in issues:
            level = str(issue.get("level", "WARNING")).upper()
            log_method = self.logger.error if level == "ERROR" else self.logger.warning
            log_method(
                "详细问题 dataset=%s batch=%d/%d symbols=%s issue_level=%s code=%s date=%s message=%s",
                dataset,
                batch_id + 1,
                max(len(self.batches), 1),
                ",".join(str(code) for code in symbols),
                level,
                issue.get("code", ""),
                issue.get("date", ""),
                issue.get("message", ""),
            )

    def _log_global_issue_details(self, dataset, issues):
        """将跨证券汇总的质量问题写入详细日志。

        参数：
            dataset: 当前数据集名称。
            issues: 不属于单个证券批次的问题字典序列。

        返回：
            无返回值；问题仍会保留在外部 CSV 报告中。
        """
        for issue in issues:
            level = str(issue.get("level", "WARNING")).upper()
            log_method = self.logger.error if level == "ERROR" else self.logger.warning
            log_method(
                "详细问题 dataset=%s batch=all issue_level=%s code=%s date=%s message=%s",
                dataset,
                level,
                issue.get("code", ""),
                issue.get("date", ""),
                issue.get("message", ""),
            )

    def _collect_kline(self, expected_trade_dates):
        """按证券批次下载日线并写入可恢复 staging。

        参数：
            expected_trade_dates: 大 QMT 交易日历返回的预期交易日序列。

        返回：
            至少一个批次或预期交易日未完成时返回 ``True``；成功时逐日写最终分区。
        """
        dataset = "kline_1d"
        failed = False
        for batch_id, symbols in enumerate(self.batches):
            daily_fragments_exist = all(
                self.store.fragment_exists(
                    self.job_key, "kline_daily_{0}".format(trade_date), batch_id
                )
                for trade_date in expected_trade_dates
            )
            if self._can_resume(dataset, batch_id) and daily_fragments_exist:
                self._log_batch_progress(dataset, batch_id, symbols, "断点跳过")
                self.logger.info("断点命中 dataset=%s batch=%d", dataset, batch_id)
                continue
            self._log_batch_progress(dataset, batch_id, symbols, "开始")
            self.checkpoints.mark_running(self.job_key, dataset, batch_id)
            try:
                frame, issues = self.gateway.fetch_kline(
                    symbols,
                    self.config.start_date,
                    self.config.end_date,
                    self.config.download_kline,
                )
                self.issues.extend(issues)
                quality_issues = validate_kline(frame)
                self.issues.extend(quality_issues)
                self._log_issue_details(dataset, batch_id, symbols, issues + quality_issues)
                self.store.write_fragment(self.job_key, dataset, batch_id, frame)
                if frame.empty or _contains_error(issues) or _contains_error(quality_issues):
                    failed = True
                    reason = "批次日线为空、包含接口错误或未通过质量检查"
                    self.checkpoints.mark_failed(self.job_key, dataset, batch_id, reason)
                    self._log_batch_progress(dataset, batch_id, symbols, "失败")
                    self.logger.error(
                        "批次未完成 dataset=%s batch=%d/%d symbols=%s date_range=%s..%s rows=%d issue_count=%d reason=%s",
                        dataset,
                        batch_id + 1,
                        len(self.batches),
                        ",".join(symbols),
                        self.config.start_date,
                        self.config.end_date,
                        len(frame),
                        len(issues) + len(quality_issues),
                        reason,
                    )
                    continue
                chunk_size = max(self.config.save_workers * 4, 1)
                for chunk_start in range(0, len(expected_trade_dates), chunk_size):
                    save_tasks = []
                    save_labels = []
                    for trade_date in expected_trade_dates[chunk_start : chunk_start + chunk_size]:
                        daily = frame[frame["trade_date"].astype(str) == trade_date]
                        save_tasks.append(
                            lambda current_date=trade_date, current_daily=_ensure_columns(daily, KLINE_COLUMNS): self.store.write_fragment(
                                self.job_key,
                                "kline_daily_{0}".format(current_date),
                                batch_id,
                                current_daily,
                            )
                        )
                        save_labels.append(trade_date)
                    self._parallel_save(
                        save_tasks,
                        "kline_daily_staging batch={0}".format(batch_id + 1),
                        save_labels,
                    )
                self.checkpoints.mark_completed(self.job_key, dataset, batch_id, len(frame))
                self._log_batch_progress(dataset, batch_id, symbols, "完成")
                self.logger.info("批次完成 dataset=%s batch=%d rows=%d", dataset, batch_id, len(frame))
            except Exception as error:
                failed = True
                self.checkpoints.mark_failed(self.job_key, dataset, batch_id, error)
                self._log_batch_progress(dataset, batch_id, symbols, "失败")
                self.logger.exception(
                    "批次异常 dataset=%s batch=%d/%d symbols=%s date_range=%s..%s error_type=%s error=%s",
                    dataset,
                    batch_id + 1,
                    len(self.batches),
                    ",".join(symbols),
                    self.config.start_date,
                    self.config.end_date,
                    type(error).__name__,
                    error,
                )
        if failed:
            return True
        missing_dates = []
        for trade_date in expected_trade_dates:
            daily = self.store.read_fragments(
                self.job_key,
                "kline_daily_{0}".format(trade_date),
                range(len(self.batches)),
            )
            daily = _ensure_columns(daily, KLINE_COLUMNS)
            if daily.empty:
                missing_dates.append(trade_date)
            else:
                quality_issues = find_missing_kline(daily, self.symbols, [trade_date])
                self.issues.extend(quality_issues)
                self._log_global_issue_details(dataset, quality_issues)
        if missing_dates:
            failed = True
            for trade_date in missing_dates:
                self.issues.add("ERROR", dataset, "", trade_date, "全部证券均缺少该预期交易日，日线批次保持为可重试状态")
                self._log_global_issue_details(
                    dataset,
                    [
                        {
                            "level": "ERROR",
                            "code": "",
                            "date": trade_date,
                            "message": "全部证券均缺少该预期交易日，日线批次保持为可重试状态",
                        }
                    ],
                )
            for batch_id in range(len(self.batches)):
                self.checkpoints.mark_failed(
                    self.job_key,
                    dataset,
                    batch_id,
                    "缺少预期交易日: {0}".format(",".join(missing_dates)),
                )
            return True
        if "kline_1d" in self.config.datasets:
            chunk_size = max(self.config.save_workers * 4, 1)
            all_dates = tuple(expected_trade_dates)
            for chunk_start in range(0, len(all_dates), chunk_size):
                date_chunk = all_dates[chunk_start : chunk_start + chunk_size]
                save_tasks = [
                    lambda current_date=trade_date, dates=all_dates: self._write_kline_date_partition(
                        current_date, dates
                    )
                    for trade_date in date_chunk
                ]
                self._parallel_save(save_tasks, "kline_daily_final", date_chunk)
        return False

    def _write_kline_date_partition(self, trade_date, trade_dates):
        """读取一个交易日的日线 staging 并写入最终 CSV 分区。
        ``trade_dates`` 是本次任务完整交易日序列，仅用于计算并行写入进度。

        参数：
            trade_date: 当前交易日的八位日期字符串。
            trade_dates: 本次任务完整交易日序列，用于计算进度日志中的位置和总数。

        返回：
            最终日线分区的写入结果字典。
        """
        date_index = self._date_index(trade_dates, trade_date)
        self._log_date_progress("kline_1d", date_index, len(trade_dates), trade_date, "开始写入")
        daily = self.store.read_fragments(
            self.job_key,
            "kline_daily_{0}".format(trade_date),
            range(len(self.batches)),
        )
        result = self._write_partition(
            "kline_1d",
            "date",
            trade_date,
            _ensure_columns(daily, KLINE_COLUMNS),
            KLINE_COLUMNS,
            ["code", "trade_date"],
            ["code"],
        )
        self._log_date_progress("kline_1d", date_index, len(trade_dates), trade_date, "完成")
        return result

    @staticmethod
    def _date_index(values, target):
        """查找日期在本次任务日期序列中的位置。

        参数：
            values: 有序日期序列。
            target: 待查找的日期字符串。

        返回：
            零基索引；找不到时返回零以保证错误日志仍可输出。
        """
        try:
            return list(values).index(target)
        except ValueError:
            return 0

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

    def _collect_actions(self):
        """按证券批次读取全部除权送转记录并保存 staging。

        返回：
            ``(DataFrame, failed)``；数据已限定在配置日期范围。
        """
        dataset = "corporate_actions"
        failed = False
        for batch_id, symbols in enumerate(self.batches):
            if self._can_resume(dataset, batch_id):
                self._log_batch_progress(dataset, batch_id, symbols, "断点跳过")
                self.logger.info("断点命中 dataset=%s batch=%d", dataset, batch_id)
                continue
            self._log_batch_progress(dataset, batch_id, symbols, "开始")
            self.checkpoints.mark_running(self.job_key, dataset, batch_id)
            try:
                frame, issues = self.gateway.fetch_corporate_actions(
                    symbols, self.config.start_date, self.config.end_date
                )
                self.issues.extend(issues)
                self._log_issue_details(dataset, batch_id, symbols, issues)
                self.store.write_fragment(self.job_key, dataset, batch_id, frame)
                if _contains_error(issues):
                    failed = True
                    reason = "批次除权接口包含错误"
                    self.checkpoints.mark_failed(self.job_key, dataset, batch_id, reason)
                    self._log_batch_progress(dataset, batch_id, symbols, "失败")
                    self.logger.error("批次未完成 dataset=%s batch=%d reason=%s", dataset, batch_id, reason)
                    continue
                self.checkpoints.mark_completed(self.job_key, dataset, batch_id, len(frame))
                self._log_batch_progress(dataset, batch_id, symbols, "完成")
                self.logger.info("批次完成 dataset=%s batch=%d rows=%d", dataset, batch_id, len(frame))
            except Exception as error:
                failed = True
                self.checkpoints.mark_failed(self.job_key, dataset, batch_id, error)
                self._log_batch_progress(dataset, batch_id, symbols, "失败")
                self.logger.exception("批次失败 dataset=%s batch=%d", dataset, batch_id)
        frame = self.store.read_fragments(self.job_key, dataset, range(len(self.batches)))
        return _ensure_columns(frame, CORPORATE_ACTION_COLUMNS), failed

    def _can_resume(self, dataset, batch_id):
        """判断普通单表批次能否直接使用已有 staging。

        参数：
            dataset: 批次所属数据集名称。
            batch_id: 零基批次编号。

        返回：
            SQLite 状态完成且 staging 文件存在时返回 ``True``。
        """
        return self.checkpoints.is_completed(self.job_key, dataset, batch_id) and self.store.fragment_exists(
            self.job_key, dataset, batch_id
        )

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

    def _write_action_partitions(self, frame, trade_dates):
        """将除权送转记录按除权日保存，并为无事件交易日创建空分区。

        参数：
            frame: 配置区间内的除权送转记录。
            trade_dates: 从日线得到的实际交易日序列；用于写出明确的无事件分区。

        返回：
            无返回值。
        """
        action_dates = set(frame["ex_date"].dropna().astype(str))
        partition_dates = sorted(action_dates | set(trade_dates))
        if not partition_dates:
            self.issues.add("WARNING", "corporate_actions", "", "", "没有交易日或除权日可供分区")
        tasks = [
            lambda current_date=ex_date, dates=tuple(partition_dates): self._write_action_date_partition(
                frame, current_date, dates
            )
            for ex_date in partition_dates
        ]
        self._parallel_save(tasks, "corporate_actions_final", partition_dates)

    def _write_action_date_partition(self, frame, ex_date, partition_dates):
        """写入一个除权日分区；每个任务只操作自己的目标路径。

        参数：
            frame: 已获取的除权转送明细总表。
            ex_date: 当前除权日的八位日期字符串。
            partition_dates: 本次任务的完整分区日期序列，用于进度日志。
        返回：
            最终除权分区的写入结果字典。
        """
        date_index = self._date_index(partition_dates, ex_date)
        self._log_date_progress(
            "corporate_actions", date_index, len(partition_dates), ex_date, "开始写入"
        )
        daily = frame[frame["ex_date"].astype(str) == ex_date]
        result = self._write_partition(
            "corporate_actions",
            "ex_date",
            ex_date,
            daily,
            CORPORATE_ACTION_COLUMNS,
            ["code", "ex_date"],
            ["code"],
        )
        self._log_date_progress(
            "corporate_actions", date_index, len(partition_dates), ex_date, "完成"
        )
        return result

    def _write_partition(
        self,
        dataset_path,
        partition_name,
        partition_value,
        frame,
        expected_columns,
        identity_columns,
        sort_columns,
        overwrite=None,
    ):
        """写入分区并输出统一日志。

        参数：
            dataset_path: 相对输出根目录的数据集路径。
            partition_name: 分区字段名。
            partition_value: 八位日期或 ``unknown``。
            frame: 当前分区记录表。
            expected_columns: 即使空分区也必须保存的列。
            identity_columns: 用于拒绝重复记录的业务主键列。
            sort_columns: 输出 CSV 的稳定排序列。
            overwrite: 是否强制覆盖已有分区；缺省跟随配置，上市退市快照可强制刷新。

        返回：
            分区存储层返回的状态、行数和文件路径字典。
        """
        result = self.store.write_partition(
            dataset_path,
            partition_name,
            partition_value,
            frame,
            expected_columns,
            identity_columns,
            sort_columns,
            {
                "job_key": self.job_key,
                "mode": self.config.mode,
                "partition_scope": self.partition_scope,
            },
            overwrite=(
                self.config.overwrite_completed_partition
                if overwrite is None
                else bool(overwrite)
            ),
        )
        self.logger.info(
            "分区%s dataset=%s %s=%s rows=%s path=%s",
            "写入" if result["status"] == "written" else "跳过",
            dataset_path,
            partition_name,
            partition_value,
            result["rows"],
            result["path"],
        )
        return result

    def _can_reuse_partition(
        self, dataset_path, partition_name, partition_value, partition_scope
    ):
        """判断当前模式能否复用一个完整业务分区。

        参数：
            dataset_path: 相对输出根目录的数据集路径。
            partition_name: 分区字段名。
            partition_value: 当前日期或 ``unknown``。
            partition_scope: 当前证券池和抽取口径范围。

        返回：
            非覆盖模式且分区完整、范围一致时返回 ``True``；repair 模式始终返回 ``False``。
        """
        if self.config.overwrite_completed_partition:
            return False
        return self.store.partition_matches_scope(
            dataset_path, partition_name, partition_value, partition_scope
        )

    def _write_run_completion(self, trade_dates):
        """为全部所选数据集成功的交易日写全局增量水位。

        参数：
            trade_dates: 已通过交易日历和数据质量校验的实际交易日序列。

        返回：
            无返回值；任一引用业务分区损坏时抛出异常并拒绝推进水位。
        """
        for trade_date in trade_dates:
            required = []
            if "kline_1d" in self.config.datasets:
                required.append("kline_1d/date={0}".format(trade_date))
            if "finance_daily" in self.config.datasets:
                required.append("finance_daily/date={0}".format(trade_date))
            if "corporate_actions" in self.config.datasets:
                required.append("corporate_actions/ex_date={0}".format(trade_date))
            if "finance_raw" in self.config.datasets:
                for table_name in sorted(FINANCE_FIELDS):
                    required.append(
                        "finance_raw/table={0}/announce_date={1}".format(
                            table_name, trade_date
                        )
                    )
            path = self.store.write_run_date_complete(
                trade_date,
                required,
                {
                    "job_key": self.job_key,
                    "mode": self.config.mode,
                    "watermark_scope": self.config.watermark_scope,
                },
            )
            self.logger.info("整日完成水位写入 date=%s path=%s", trade_date, path)


def _iter_batches(values, batch_size):
    """按固定大小顺序切分证券池。

    参数：
        values: 已稳定排序的证券代码列表。
        batch_size: 每批最多证券数量。

    返回：
        逐批产生证券代码列表的生成器。
    """
    for start in range(0, len(values), int(batch_size)):
        yield values[start : start + int(batch_size)]


def _make_job_key(config, symbols):
    """根据影响抽取结果的配置生成稳定任务标识。

    参数：
        config: 当前 ``DownloaderConfig``。
        symbols: 已解析并排序的最终证券池。

    返回：
        ``qmt_`` 前缀加 SHA-256 前二十位的稳定字符串。
    """
    payload = {
        "schema_version": 2,
        "mode": config.mode,
        "start_date": config.start_date,
        "end_date": config.end_date,
        "finance_lookback_start": config.finance_lookback_start,
        "symbols": list(symbols),
        "datasets": list(config.datasets),
        "download_kline": config.download_kline,
        "batch_size": config.batch_size,
        "allow_partial_finance": config.allow_partial_finance,
        "calendar_symbol": config.calendar_symbol,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return "qmt_{0}".format(hashlib.sha256(encoded).hexdigest()[:20])


def _ensure_columns(frame, columns):
    """为 staging 读取结果补齐规范列并保持额外列。

    参数：
        frame: 从一个或多个 staging CSV 合并的数据表。
        columns: 当前数据集必须包含的规范列顺序。

    返回：
        规范列在前、额外列在后的新 ``DataFrame``。
    """
    output = frame.copy()
    for column in columns:
        if column not in output.columns:
            output[column] = pd.Series(index=output.index, dtype="object")
    extras = [column for column in output.columns if column not in columns]
    return output[list(columns) + sorted(extras)]


def _finance_templates():
    """构造所有财务表的空表和稳定列结构。

    返回：
        按逻辑财务表名映射的空 ``DataFrame``，用于流式片段补列和最终表头。
    """
    output = {}
    for table_name, fields in FINANCE_FIELDS.items():
        columns = ["code", "report_date", "announce_date"] + [
            field.split(".", 1)[1] for field in fields[2:]
        ]
        output[table_name] = pd.DataFrame(columns=columns)
    return output


def _contains_error(issues):
    """判断一次网关调用是否返回接口级错误。

    参数：
        issues: 网关返回的结构化问题字典列表。

    返回：
        任一问题严重级别为 ``ERROR`` 时返回 ``True``。
    """
    return any(item.get("level") == "ERROR" for item in issues)


def _contains_finance_missing(issues):
    """判断财务问题中是否存在部分表、证券或公告日缺失。

    参数：
        issues: 财务网关返回的结构化问题字典列表。

    返回：
        存在 ``finance_raw`` 警告时返回 ``True``，用于严格完整性模式保持可重试。
    """
    return any(
        item.get("level") == "WARNING"
        and str(item.get("dataset", "")).startswith("finance_raw/")
        for item in issues
    )
