# -*- coding: utf-8 -*-
"""上市退市信息的收集与证券池推导。"""

from datetime import datetime, timedelta

import pandas as pd

from ..gateway import INSTRUMENT_INFO_COLUMNS
from .base import _RunnerState
from .helpers import _contains_error


class _InstrumentInfoMixin(_RunnerState):
    """收集上市退市快照并据此推导实际证券池。"""

    def _collect_instrument_info(self):
        """读取证券池及历史存续证券的上市退市信息并保存当前快照表。

        返回：
            ``(DataFrame, failed)``；快照写入 ``instrument_info/snapshot=latest``。当前
            证券池任一代码详情缺失时失败；仅历史快照代码缺失时降级为警告并从新快照
            移除，避免已被大 QMT 清除的退市代码永久阻塞任务。
        """
        info_symbols = self._instrument_info_symbols()
        pool = set(self.symbols)
        carried = {code for code in info_symbols if code not in pool}
        self.logger.info("[instrument_info] 开始获取上市退市信息 证券 1/%d", len(info_symbols))
        frame, issues = self.gateway.fetch_instrument_info(info_symbols)
        issues = [self._downgrade_carried_issue(issue, carried) for issue in issues]
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
        returned = set(frame["code"].astype(str)) if not frame.empty else set()
        failed = _contains_error(issues) or not pool <= returned
        if len(info_symbols):
            self.logger.info(
                "[instrument_info] 完成获取上市退市信息 证券 %d/%d (%.1f%%)",
                len(frame),
                len(info_symbols),
                len(frame) * 100.0 / len(info_symbols),
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

    def _instrument_info_symbols(self):
        """合并当前证券池与上一快照中前一日尚未退市的证券代码。

        返回：
            去重并按代码排序的证券代码列表；历史快照不可读时仅返回当前证券池。
        """
        current = {str(code).strip().upper() for code in self.symbols if str(code).strip()}
        if not self.store.is_partition_complete("instrument_info", "snapshot", "latest"):
            return sorted(current)
        previous = self.store.read_partition(
            "instrument_info",
            "snapshot",
            "latest",
            dtype={"code": str, "expire_date": str},
        )
        if previous is None or not {"code", "expire_date"}.issubset(previous.columns):
            return sorted(current)

        reference_date = datetime.strptime(self.config.end_date, "%Y%m%d") - timedelta(days=1)
        reference_text = reference_date.strftime("%Y%m%d")
        for row in previous.to_dict("records"):
            code = str(row.get("code") or "").strip().upper()
            if not code or "." not in code:
                continue
            raw_expire_date = row.get("expire_date")
            expire_date = "" if pd.isna(raw_expire_date) else str(raw_expire_date).strip()
            # 空值表示尚未退市；退市日等于前一日时仍属于前一日的证券池。
            if not expire_date:
                current.add(code)
                continue
            try:
                datetime.strptime(expire_date, "%Y%m%d")
            except ValueError:
                # 单行退市日期损坏不应静默丢弃其余行；保留该代码继续跟踪生命周期。
                self.logger.warning(
                    "上市退市快照存在无法解析的退市日期 code=%s expire_date=%s，仍保留该代码",
                    code,
                    expire_date,
                )
                current.add(code)
                continue
            if expire_date >= reference_text:
                current.add(code)
        return sorted(current)
