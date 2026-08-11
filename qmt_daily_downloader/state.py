# -*- coding: utf-8 -*-
"""使用 SQLite 保存批次断点，不保存任何业务行情数据。"""

import sqlite3
from datetime import datetime
from pathlib import Path


class CheckpointStore(object):
    """管理可幂等恢复的股票批次状态。"""

    def __init__(self, path):
        """初始化 SQLite 断点库并创建任务表。

        参数：
            path: SQLite 文件路径；父目录不存在时自动创建。
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def is_completed(self, job_key, dataset, batch_id):
        """判断指定任务批次是否已经成功落盘。

        参数：
            job_key: 由模式、日期区间和股票池生成的稳定任务标识。
            dataset: 数据集名称，如 ``kline_1d`` 或 ``finance_raw``。
            batch_id: 当前股票批次的零基编号。

        返回：
            状态为 ``completed`` 时返回 ``True``，否则返回 ``False``。
        """
        connection = sqlite3.connect(str(self.path))
        try:
            row = connection.execute(
                "SELECT status FROM batch_state WHERE job_key=? AND dataset=? AND batch_id=?",
                (job_key, dataset, int(batch_id)),
            ).fetchone()
        finally:
            connection.close()
        return row is not None and row[0] == "completed"

    def mark_running(self, job_key, dataset, batch_id):
        """将批次置为运行中并累计尝试次数。

        参数：
            job_key: 当前稳定任务标识。
            dataset: 当前处理的数据集名称。
            batch_id: 股票批次编号。

        返回：
            无返回值；状态更新在独立事务中提交。
        """
        now = _now_text()
        connection = sqlite3.connect(str(self.path))
        try:
            connection.execute(
                """
                INSERT OR IGNORE INTO batch_state(
                    job_key,dataset,batch_id,status,attempt_count,row_count,last_error,updated_at
                ) VALUES(?,?,?,'pending',0,0,'',?)
                """,
                (job_key, dataset, int(batch_id), now),
            )
            connection.execute(
                """
                UPDATE batch_state SET status='running',attempt_count=attempt_count+1,
                    last_error='',updated_at=?
                WHERE job_key=? AND dataset=? AND batch_id=?
                """,
                (now, job_key, dataset, int(batch_id)),
            )
            connection.commit()
        finally:
            connection.close()

    def mark_completed(self, job_key, dataset, batch_id, row_count):
        """在批次文件成功写入后记录完成状态和行数。

        参数：
            job_key: 当前稳定任务标识。
            dataset: 当前处理的数据集名称。
            batch_id: 股票批次编号。
            row_count: 当前批次成功写入 staging 的记录数。

        返回：
            无返回值。
        """
        self._update(job_key, dataset, batch_id, "completed", row_count, "")

    def mark_failed(self, job_key, dataset, batch_id, error):
        """记录失败批次及可供用户定位的异常文本。

        参数：
            job_key: 当前稳定任务标识。
            dataset: 当前处理的数据集名称。
            batch_id: 股票批次编号。
            error: 捕获到的异常对象或错误文本。

        返回：
            无返回值。
        """
        self._update(job_key, dataset, batch_id, "failed", 0, str(error))

    def _initialize(self):
        """创建批次状态表及其复合主键。

        返回：
            无返回值。
        """
        connection = sqlite3.connect(str(self.path))
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS batch_state(
                    job_key TEXT NOT NULL,
                    dataset TEXT NOT NULL,
                    batch_id INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    row_count INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(job_key,dataset,batch_id)
                )
                """
            )
            connection.commit()
        finally:
            connection.close()

    def _update(self, job_key, dataset, batch_id, status, row_count, error):
        """更新既有批次的终态字段。

        参数：
            job_key: 当前稳定任务标识。
            dataset: 当前处理的数据集名称。
            batch_id: 股票批次编号。
            status: ``completed`` 或 ``failed`` 终态。
            row_count: 已可靠保存的记录数，失败时为零。
            error: 失败原因；成功时为空字符串。

        返回：
            无返回值。
        """
        connection = sqlite3.connect(str(self.path))
        try:
            connection.execute(
                """
                UPDATE batch_state SET status=?,row_count=?,last_error=?,updated_at=?
                WHERE job_key=? AND dataset=? AND batch_id=?
                """,
                (
                    status,
                    int(row_count),
                    error,
                    _now_text(),
                    job_key,
                    dataset,
                    int(batch_id),
                ),
            )
            connection.commit()
        finally:
            connection.close()


def _now_text():
    """生成无时区歧义的本地秒级日志时间文本。

    返回：
        ``YYYY-MM-DD HH:MM:SS`` 格式字符串。
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
