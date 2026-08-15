# -*- coding: utf-8 -*-
"""问题级别降级、日线问题过滤与无下载收尾。"""

import json

from .base import _RunnerState
from .helpers import _lifecycle_windows


class _IssueHandlingMixin(_RunnerState):
    """按证券生命周期降级或过滤问题，并处理无需下载时的收尾。"""

    @classmethod
    def _downgrade_carried_issue(cls, issue, carried_codes):
        """把仅涉及历史快照代码的详情错误降级为警告。

        参数：
            issue: 网关返回的结构化问题字典。
            carried_codes: 来自上一快照、不在当前证券池中的代码集合。

        返回：
            当前证券池代码的问题原样返回；历史快照代码的 ``ERROR`` 返回降级
            副本，使已被大 QMT 清除的代码在下一次快照重写后自动消失。
        """
        if issue.get("level") != "ERROR" or issue.get("code") not in carried_codes:
            return issue
        return cls._downgrade_issue(issue, "历史快照代码详情读取失败，本次从快照移除")

    @staticmethod
    def _downgrade_issue(issue, prefix):
        """把一条问题降级为保留原始定位信息的审计警告。

        参数：
            issue: 结构化问题字典。
            prefix: 说明降级原因的中文前缀。

        返回：
            级别改为 ``WARNING``、消息加上前缀的新字典；原字典不被修改。
        """
        downgraded = dict(issue)
        downgraded["level"] = "WARNING"
        downgraded["message"] = "{0}: {1}".format(prefix, issue.get("message", ""))
        return downgraded

    def _filter_kline_issues(self, instrument_info):
        """按上市和退市日期过滤日线缺失警告。

        参数：
            instrument_info: ``fetch_instrument_info`` 返回的证券生命周期信息表。

        返回：
            无返回值；未上市或已退市期间的缺失提示从问题报告中移除，其他缺失保留。
        """
        lifecycle = _lifecycle_windows(instrument_info)
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
        self._with_elapsed(summary)
        self.logger.info(
            "任务无业务下载 用时=%s summary=%s",
            summary["elapsed"],
            json.dumps(summary, ensure_ascii=False),
        )
        return summary
