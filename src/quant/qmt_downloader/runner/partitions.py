# -*- coding: utf-8 -*-
"""断点复用判定、通用分区写入与整日水位标记。"""

from ..gateway import FINANCE_FIELDS
from .base import _RunnerState


class _PartitionWriteMixin(_RunnerState):
    """判定断点可否复用，写出通用分区并登记整日完成水位。"""

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
        # 长区间任务每个交易日都会写一个分区，逐个输出会淹没日志；INFO 级别的
        # 进度由 _log_date_progress 按里程碑给出，这里保留 DEBUG 供逐分区排查。
        self.logger.debug(
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
            无返回值；任一引用业务分区损坏时抛出异常并拒绝推进水位。逐日明细写
            DEBUG，INFO 只汇总一条，避免长区间任务刷屏。
        """
        written = []
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
            if not required:
                # 只落交易日历时没有任何按日业务分区，写出空引用的水位既无法被增量识别
                # （引用为空的水位一律忽略），又会凭空生成成千上万个空目录。
                continue
            path = self.store.write_run_date_complete(
                trade_date,
                required,
                {
                    "job_key": self.job_key,
                    "mode": self.config.mode,
                    "watermark_scope": self.config.watermark_scope,
                },
            )
            self.logger.debug("整日完成水位写入 date=%s path=%s", trade_date, path)
            written.append(trade_date)
        if written:
            self.logger.info(
                "整日完成水位写入 dates=%d first=%s last=%s",
                len(written),
                written[0],
                written[-1],
            )
