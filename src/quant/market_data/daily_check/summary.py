"""审计结果的汇总与报告落盘。

问题记录复用 ``qmt_downloader.self_check`` 的 ``AuditIssue`` 与 ``ISSUE_COLUMNS``，
因此日线库报告与 CSV 源侧报告是同一套表结构，可以直接拼在一起看；原子写入函数
也直接复用那边公开的实现，不再复制一份。摘要 Markdown 则各写各的，因为两侧的
统计口径本来就不一样。
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any

import pandas as pd

from quant.qmt_downloader.self_check import ISSUE_COLUMNS, AuditResult
from quant.qmt_downloader.self_check.writers import (
    write_csv_atomic,
    write_json_atomic,
    write_text_atomic,
)

from .base import _DailyCheckerState


class _SummaryMixin(_DailyCheckerState):
    """把问题列表与明细表汇总成报告并落盘。"""

    def _build_result(self, report_dir, started_at: datetime, extras: dict) -> AuditResult:
        """汇总本次审计并写出全部报告文件。

        参数：
            report_dir: 本次报告目录；不存在时创建。
            started_at: 审计开始时间，写进摘要便于追溯。
            extras: 需要一并写进摘要的附加信息，例如成交量单位倍数与是否做了
                交叉校验。

        返回：
            与 ``python -m quant.cli.qmt_self_check`` 同构的
            ``AuditResult``，其 ``exit_code`` 在存在 ERROR 时为 1。
        """
        report_dir.mkdir(parents=True, exist_ok=True)
        issues = _issues_frame(self.issues)
        counts = Counter(issue.level for issue in self.issues)
        summary: dict[str, Any] = {
            "status": "failed" if counts.get("ERROR", 0) else "passed",
            "database": str(self.config.database),
            "start_date": str(self.start_date),
            "end_date": str(self.end_date),
            "trading_days": len(self.calendar),
            "authoritative_calendar": self.calendar_is_authoritative,
            "symbols": len(self.codes),
            "expected_rows": int(self.coverage_by_symbol["expected"].sum())
            if not self.coverage_by_symbol.empty
            else 0,
            "actual_rows": int(self.coverage_by_symbol["actual"].sum())
            if not self.coverage_by_symbol.empty
            else 0,
            "missing_rows": int(self.coverage_by_symbol["missing"].sum())
            if not self.coverage_by_symbol.empty
            else 0,
            "missing_spans": int(len(self.missing_spans)),
            "price_limit_violations": int(len(self.price_limit_violations)),
            "cross_check_differences": int(len(self.cross_check_diff)),
            "errors": int(counts.get("ERROR", 0)),
            "warnings": int(counts.get("WARNING", 0)),
            "info": int(counts.get("INFO", 0)),
            "issue_codes": dict(
                sorted(Counter(issue.issue_code for issue in self.issues).items())
            ),
            "started_at": started_at.strftime("%Y-%m-%d %H:%M:%S"),
            "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        summary.update(extras)

        write_json_atomic(report_dir / "summary.json", summary)
        write_text_atomic(report_dir / "summary.md", _summary_markdown(summary))
        write_csv_atomic(report_dir / "issues.csv", issues)
        write_csv_atomic(report_dir / "missing_spans.csv", self.missing_spans)
        write_csv_atomic(report_dir / "coverage_by_date.csv", self.coverage_by_date)
        write_csv_atomic(report_dir / "coverage_by_symbol.csv", self.coverage_by_symbol)
        write_csv_atomic(report_dir / "adjust_factor_audit.csv", self.adjust_audit)
        write_csv_atomic(report_dir / "price_limit_violations.csv", self.price_limit_violations)
        write_csv_atomic(report_dir / "cross_5m_diff.csv", self.cross_check_diff)

        return AuditResult(
            summary=summary,
            issues=issues,
            missing_spans=self.missing_spans,
            coverage_by_date=self.coverage_by_date,
            coverage_by_symbol=self.coverage_by_symbol,
            report_dir=report_dir,
        )


def _issues_frame(issues) -> pd.DataFrame:
    """把问题记录列表转换为固定列顺序的数据表。

    参数：
        issues: ``AuditIssue`` 列表，可以为空。

    返回：
        列顺序为 ``ISSUE_COLUMNS`` 的数据表；没有问题时返回只有表头的空表。
    """
    if not issues:
        return pd.DataFrame(columns=ISSUE_COLUMNS)
    records = [
        {column: getattr(issue, column) for column in ISSUE_COLUMNS} for issue in issues
    ]
    return pd.DataFrame(records, columns=ISSUE_COLUMNS)


def _summary_markdown(summary: dict[str, Any]) -> str:
    """把机器可读摘要渲染成简洁的中文 Markdown。

    参数：
        summary: ``_build_result`` 组装出的摘要字典。

    返回：
        可直接写入 ``summary.md`` 的文本。
    """
    lines = [
        "# 日线库数据合法性审计报告",
        "",
        "- 结论：{0}".format("未通过" if summary["status"] == "failed" else "通过"),
        "- 日线库：`{0}`".format(summary["database"]),
        "- 审计区间：{0} 至 {1}".format(summary["start_date"], summary["end_date"]),
        "- 交易日：{0}".format(summary["trading_days"]),
        "- 使用权威交易日历：{0}".format("是" if summary["authoritative_calendar"] else "否"),
        "- 证券数：{0}".format(summary["symbols"]),
        "- 理论应有记录：{0}".format(summary["expected_rows"]),
        "- 实际记录：{0}".format(summary["actual_rows"]),
        "- 缺失记录：{0}".format(summary["missing_rows"]),
        "- 连续缺失区间：{0}".format(summary["missing_spans"]),
        "- 涨跌停越界：{0}".format(summary["price_limit_violations"]),
        "- 5 分钟交叉校验差异：{0}".format(summary["cross_check_differences"]),
        "- 成交量单位倍数：{0}".format(summary.get("volume_multiplier", "未标定")),
        "- ERROR：{0}".format(summary["errors"]),
        "- WARNING：{0}".format(summary["warnings"]),
        "- INFO：{0}".format(summary["info"]),
        "",
        "## 问题类型统计",
        "",
        "| 错误编号 | 数量 |",
        "|---|---:|",
    ]
    for code, count in summary["issue_codes"].items():
        lines.append(f"| `{code}` | {count} |")
    lines.extend(
        [
            "",
            "完整定位、理论值、实际值、证据、可能原因和处理建议见 `issues.csv`。",
            "连续缺失区间见 `missing_spans.csv`，逐日和逐证券覆盖率见对应 CSV，",
            "除权因子自洽性见 `adjust_factor_audit.csv`，涨跌停越界见 `price_limit_violations.csv`，",
            "5 分钟交叉校验差异见 `cross_5m_diff.csv`。",
            "",
            "本命令只读取并报告，不会修改、填充或删除日线库中的任何数据。",
            "",
            "> 注意：涨跌停校验默认按板块基础幅度判定，**不**按当前 ST 状态收紧到 5%——",
            "> 证券简称只有当前快照，历史某一天是否处于风险警示无从得知。因此这条校验是",
            "> 「有没有超过该板块可能的最大幅度」这一必要条件，需要更严格时加 `--apply-st-limit`。",
            "",
        ]
    )
    return "\n".join(lines)
