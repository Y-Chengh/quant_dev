# -*- coding: utf-8 -*-
"""除权除息事件的收集与按除权日分区落盘。"""

from ..gateway import CORPORATE_ACTION_COLUMNS
from .base import _RunnerState
from .helpers import _contains_error, _ensure_columns


class _CorporateActionMixin(_RunnerState):
    """收集除权除息事件并按除权日写出分区。"""

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
