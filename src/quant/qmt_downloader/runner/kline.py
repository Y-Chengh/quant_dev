# -*- coding: utf-8 -*-
"""日线收集、缺口探测与定向补下载，以及按日分区落盘。"""

import time

import pandas as pd

from ..gateway import KLINE_COLUMNS
from ..validation import (
    correct_kline_prices,
    find_missing_kline,
    missing_dates_by_code,
    validate_kline,
)
from .base import _RunnerState
from .helpers import _codes_alive_on, _contains_error, _ensure_columns


class _KlineCollectionMixin(_RunnerState):
    """收集日线、探测并修复中间缺口，并按交易日写出分区。"""

    @staticmethod
    def _find_internal_kline_gaps(frame, symbols, expected_trade_dates):
        """查找每只证券首尾已有行情之间缺失的交易日区间。

        参数：
            frame: 当前证券批次已经取得的标准日线表。
            symbols: 当前批次证券代码序列。
            expected_trade_dates: 大 QMT 交易日历返回的有序交易日序列。

        返回：
            ``code -> [(start_date, end_date, missing_dates)]`` 映射；缺失判定复用
            ``validation.find_missing_kline``（停牌补齐行不算缺失），再裁剪到证券
            已有首日和末日之间并合并连续区间，因此不包含上市前或退市后的日期。
        """
        dates = list(expected_trade_dates)
        positions = {date_value: index for index, date_value in enumerate(dates)}
        output = {}
        if frame is None or frame.empty or not dates:
            return output
        missing_by_code = missing_dates_by_code(frame, symbols, dates)
        span_by_code = {}
        for code_value, date_value in zip(
            frame["code"].astype(str), frame["trade_date"].astype(str)
        ):
            position = positions.get(date_value)
            if position is None:
                continue
            span = span_by_code.get(code_value)
            if span is None:
                span_by_code[code_value] = [position, position]
            else:
                span[0] = min(span[0], position)
                span[1] = max(span[1], position)
        for code in symbols:
            code_key = str(code)
            span = span_by_code.get(code_key)
            code_missing = missing_by_code.get(code_key, ())
            if span is None or span[0] == span[1] or not code_missing:
                continue
            missing = [
                dates[index]
                for index in range(span[0] + 1, span[1])
                if dates[index] in code_missing
            ]
            if not missing:
                continue
            ranges = []
            current = [missing[0]]
            for date_value in missing[1:]:
                if positions[date_value] == positions[current[-1]] + 1:
                    current.append(date_value)
                else:
                    ranges.append((current[0], current[-1], tuple(current)))
                    current = [date_value]
            ranges.append((current[0], current[-1], tuple(current)))
            output[code_key] = ranges
        return output

    @classmethod
    def _resolved_gap_issue_warnings(cls, issues):
        """把补下载过程中已恢复的接口错误转换为审计警告。

        参数：
            issues: 大 QMT 补下载接口返回、但对应缺口最终已经补齐的问题序列。

        返回：
            保留原始数据集、代码、日期和消息，并把级别改为 ``WARNING`` 的新字典列表。
        """
        return [
            cls._downgrade_issue(issue, "补下载过程中接口曾报错但缺口最终已补齐")
            for issue in issues
        ]

    def _repair_kline_gaps(self, frame, symbols, expected_trade_dates, batch_id):
        """按连续日期区间合并证券后对中间日线缺口执行有限次数补下载。

        本方法只读写内存中的批次数据，不触碰最终分区目录；缺口补齐后由分区
        写入层按业务主键差异决定是否重写旧分区。

        参数：
            frame: 首轮下载得到的标准日线表。
            symbols: 当前证券批次的代码序列。
            expected_trade_dates: 大 QMT 交易日历返回的有序交易日序列。
            batch_id: 当前证券批次的零基编号，用于详细日志定位。

        返回：
            ``(merged_frame, issues)``；前者合并补下载结果并按业务主键去重，后者
            在缺口全部补齐时只含降级 ``WARNING``，否则包含最终仍缺失的 ``ERROR``
            和未补齐证券最后一轮的接口问题。
        """
        merged = _ensure_columns(frame, KLINE_COLUMNS)
        # round_issues 只保留最近一轮的接口问题，循环结束后即最后一轮视图。
        round_issues = {}
        issue_history_by_code = {}
        gaps = self._find_internal_kline_gaps(merged, symbols, expected_trade_dates)
        attempt = 0
        while gaps and attempt < self.config.kline_gap_retry_count:
            attempt += 1
            round_issues = {}
            gap_count = sum(
                len(missing_dates)
                for ranges in gaps.values()
                for _, _, missing_dates in ranges
            )
            self.logger.warning(
                "日线缺口补下载 batch=%d/%d attempt=%d/%d symbols=%d gaps=%d",
                batch_id + 1,
                len(self.batches),
                attempt,
                self.config.kline_gap_retry_count,
                len(gaps),
                gap_count,
            )
            codes_by_range = {}
            for code, ranges in sorted(gaps.items()):
                for start_date, end_date, missing_dates in ranges:
                    codes_by_range.setdefault((start_date, end_date), []).append(
                        (code, missing_dates)
                    )
            additions = []
            for (start_date, end_date), entries in sorted(codes_by_range.items()):
                codes = [code for code, _ in entries]
                self.logger.info(
                    "日线缺口请求 stage=gap_redownload batch=%d/%d codes=%s start=%s end=%s missing_dates=%s attempt=%d/%d",
                    batch_id + 1,
                    len(self.batches),
                    ",".join(codes),
                    start_date,
                    end_date,
                    ",".join(sorted(set(
                        trade_date for _, missing in entries for trade_date in missing
                    ))),
                    attempt,
                    self.config.kline_gap_retry_count,
                )
                # 补缺口的目的就是刷新本地缓存，因此始终先执行定向历史下载；
                # download_kline 只控制首轮批量读取，不影响这里。
                repaired, repair_issues = self.gateway.fetch_kline(
                    codes, start_date, end_date, True
                )
                self._log_issue_details(
                    "kline_1d/gap_redownload", batch_id, codes, repair_issues
                )
                for issue in repair_issues:
                    issue_codes = [str(issue.get("code") or "")]
                    if not issue_codes[0]:
                        issue_codes = codes
                    for code in issue_codes:
                        round_issues.setdefault(code, []).append(issue)
                        issue_history_by_code.setdefault(code, []).append(issue)
                if not repaired.empty:
                    additions.append(repaired)
            if additions:
                merged = pd.concat([merged] + additions, ignore_index=True)
                merged = _ensure_columns(merged, KLINE_COLUMNS)
                merged = merged.drop_duplicates(
                    ["code", "trade_date"], keep="last"
                ).sort_values(["code", "trade_date"], kind="mergesort")
                merged = merged.reset_index(drop=True)
            # 只有本轮补下载的证券会新增行，其余证券的缺口状态不可能改变，
            # 因此复查范围收敛到本轮缺口证券，避免整批全量重扫。
            gaps = self._find_internal_kline_gaps(
                merged, sorted(gaps), expected_trade_dates
            )

        issues = []
        for code, code_issues in issue_history_by_code.items():
            if code not in gaps:
                issues.extend(self._resolved_gap_issue_warnings(code_issues))
        for code, ranges in sorted(gaps.items()):
            issues.extend(round_issues.get(code, []))
            for _, _, missing_dates in ranges:
                for trade_date in missing_dates:
                    issues.append(
                        {
                            "level": "ERROR",
                            "dataset": "kline_1d",
                            "code": code,
                            "date": trade_date,
                            "message": "中间日线缺口补下载 {0} 次后仍缺失".format(
                                self.config.kline_gap_retry_count
                            ),
                        }
                    )
        return merged, issues

    def _fail_kline_batch(self, batch_id, symbols, frame, batch_issues, reason):
        """统一记录一个日线批次失败的问题、staging、断点状态和日志。

        参数：
            batch_id: 当前证券批次的零基编号。
            symbols: 当前批次证券代码序列。
            frame: 本次实际取得的日线表，允许为空。
            batch_issues: 本批次收集到的全部问题字典。
            reason: 写入断点状态和错误日志的失败原因。

        返回：
            无返回值。
        """
        dataset = "kline_1d"
        self.issues.extend(batch_issues)
        self._log_issue_details(dataset, batch_id, symbols, batch_issues)
        if not frame.empty or not self.store.fragment_exists(
            self.job_key, dataset, batch_id
        ):
            self.store.write_fragment(self.job_key, dataset, batch_id, frame)
        else:
            # 上次运行留下的完好批次片段比本次空帧更有恢复价值，保留供续跑使用。
            self.logger.warning(
                "批次返回空帧，保留上次运行的 staging 片段 dataset=%s batch=%d/%d",
                dataset,
                batch_id + 1,
                len(self.batches),
            )
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
            len(batch_issues),
            reason,
        )

    def _collect_kline(self, expected_trade_dates, lifecycle=None):
        """按证券批次下载日线并写入可恢复 staging。

        参数：
            expected_trade_dates: 大 QMT 交易日历返回的预期交易日序列。
            lifecycle: ``_lifecycle_windows`` 生成的证券生命周期映射；逐日缺口检查
                据此跳过未上市和已退市的证券日。缺省或为空时不做筛选。

        返回：
            至少一个批次或预期交易日未完成时返回 ``True``；成功时逐日写最终分区。
        """
        lifecycle = lifecycle or {}
        dataset = "kline_1d"
        failed = False
        for batch_id, symbols in enumerate(self.batches):
            batch_started_at = time.perf_counter()
            # 完成状态只在缺口检查通过后写入，因此完成断点可直接信任，无需重读 staging。
            # 断点未命中时短路，避免为必然要重下的批次逐日校验 staging 片段。
            if self._can_resume(dataset, batch_id) and all(
                self.store.fragment_exists(
                    self.job_key, "kline_daily_{0}".format(trade_date), batch_id
                )
                for trade_date in expected_trade_dates
            ):
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
                if frame.empty or _contains_error(issues):
                    failed = True
                    self._fail_kline_batch(
                        batch_id,
                        symbols,
                        frame,
                        issues + validate_kline(frame),
                        "批次日线为空或包含接口错误",
                    )
                    continue
                frame, gap_issues = self._repair_kline_gaps(
                    frame, symbols, expected_trade_dates, batch_id
                )
                frame, correction_records, correction_issues = correct_kline_prices(frame)
                if correction_records:
                    self.line_corrections.extend(correction_records)
                    self.logger.warning(
                        "日线价格修正 dataset=%s batch=%d rows=%d，明细将写入 reports/line_correct",
                        dataset,
                        batch_id + 1,
                        len(correction_records),
                    )
                batch_issues = issues + gap_issues + correction_issues + validate_kline(frame)
                if _contains_error(batch_issues):
                    failed = True
                    self._fail_kline_batch(
                        batch_id,
                        symbols,
                        frame,
                        batch_issues,
                        "批次包含补下载后仍未恢复的日线缺口或质量错误",
                    )
                    continue
                self.issues.extend(batch_issues)
                self._log_issue_details(dataset, batch_id, symbols, batch_issues)
                self.store.write_fragment(self.job_key, dataset, batch_id, frame)
                staging_started_at = time.perf_counter()
                # 按日分组一次完成，避免在交易日循环里对全批次做逐日全表扫描。
                daily_groups = {
                    key: group
                    for key, group in frame.groupby(frame["trade_date"].astype(str))
                }
                empty_daily = frame.iloc[0:0]
                chunk_size = max(self.config.save_workers * 4, 1)
                for chunk_start in range(0, len(expected_trade_dates), chunk_size):
                    save_tasks = []
                    save_labels = []
                    for trade_date in expected_trade_dates[chunk_start : chunk_start + chunk_size]:
                        daily = daily_groups.get(trade_date, empty_daily)
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
                self.logger.info(
                    "日线staging写入完成 dataset=%s batch=%d dates=%d rows=%d elapsed=%.2fs",
                    dataset,
                    batch_id + 1,
                    len(expected_trade_dates),
                    len(frame),
                    time.perf_counter() - staging_started_at,
                )
                self.checkpoints.mark_completed(self.job_key, dataset, batch_id, len(frame))
                self._log_batch_progress(dataset, batch_id, symbols, "完成")
                self.logger.info(
                    "批次完成 dataset=%s batch=%d rows=%d elapsed=%.2fs",
                    dataset,
                    batch_id,
                    len(frame),
                    time.perf_counter() - batch_started_at,
                )
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
                self.logger.error(
                    "批次失败耗时 dataset=%s batch=%d elapsed=%.2fs",
                    dataset,
                    batch_id + 1,
                    time.perf_counter() - batch_started_at,
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
                quality_issues = find_missing_kline(
                    daily, _codes_alive_on(self.symbols, lifecycle, trade_date), [trade_date]
                )
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
        daily = _ensure_columns(daily, KLINE_COLUMNS)
        # 已完成分区由存储层先校验证券池范围，再按业务主键集合决定跳过或重写。
        result = self._write_partition(
            "kline_1d",
            "date",
            trade_date,
            daily,
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
