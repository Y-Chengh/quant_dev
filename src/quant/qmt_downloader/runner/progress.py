# -*- coding: utf-8 -*-
"""批次与日期进度日志、并行落盘以及问题明细输出。"""

from concurrent.futures import ThreadPoolExecutor

from .base import _RunnerState


class _ProgressLoggingMixin(_RunnerState):
    """输出批次与日期进度、并行执行落盘任务并打印问题明细。"""

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
            "[%s] %s 批次 %d/%d (%.1f%%)，证券序号 %d-%d/%d，requested_date_range=%s..%s",
            dataset,
            status,
            current,
            total,
            percent,
            first_index,
            last_index,
            len(self.symbols),
            self.config.start_date,
            self.config.end_date,
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
