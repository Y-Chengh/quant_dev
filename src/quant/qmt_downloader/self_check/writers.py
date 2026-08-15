# -*- coding: utf-8 -*-
"""报告文件的原子写入与摘要 Markdown 渲染。

三个原子写入函数不含任何 QMT 语义，``quant.market_data.daily_check`` 直接复用它们，
以保证两侧审计报告的编码、缩进和落盘方式完全一致。它们的下划线别名保留给包内
既有调用方，行为与公开名完全相同。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    """先写临时文件再原子替换 JSON 报告。

    参数：
        path: 最终 JSON 报告路径。
        value: 需要以 UTF-8 中文格式序列化的摘要字典。

    返回：
        无返回值。
    """

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
    temporary.replace(path)


def write_text_atomic(path: Path, value: str) -> None:
    """先写临时文件再原子替换 UTF-8 文本报告。

    参数：
        path: 最终 Markdown 或文本报告路径。
        value: 需要完整写入的文本内容。

    返回：
        无返回值。
    """

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    """先写临时文件再原子替换 UTF-8 BOM CSV 报告。

    参数：
        path: 最终 CSV 报告路径。
        frame: 需要写出的结构化报告表。

    返回：
        无返回值。
    """

    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(str(temporary), index=False, encoding="utf-8-sig")
    temporary.replace(path)


# 包内既有调用方仍使用下划线名字；保留别名，公开名与私有名指向同一实现。
_write_json_atomic = write_json_atomic
_write_text_atomic = write_text_atomic
_write_csv_atomic = write_csv_atomic


def _summary_markdown(summary: dict[str, Any]) -> str:
    """把机器可读摘要转换为简洁 Markdown。

    参数：
        summary: 包含审计范围、覆盖率、问题数量和错误编号统计的摘要。

    返回：
        可直接写入 ``summary.md`` 的中文 Markdown 文本。
    """

    lines = [
        "# QMT 全样本数据自检报告",
        "",
        "- 结论：{0}".format("未通过" if summary["status"] == "failed" else "通过"),
        "- 数据根目录：`{0}`".format(summary["output_root"]),
        "- 审计区间：{0} 至 {1}".format(summary["start_date"], summary["end_date"]),
        "- 交易日：{0}".format(summary["trading_days"]),
        "- 证券数：{0}".format(summary["symbols"]),
        "- 理论应有记录：{0}".format(summary["expected_rows"]),
        "- 实际有效范围记录：{0}".format(summary["actual_expected_rows"]),
        "- 缺失记录：{0}".format(summary["missing_rows"]),
        "- 连续缺失区间：{0}".format(summary["missing_spans"]),
        "- ERROR：{0}（详见 `errors.csv`）".format(summary["errors"]),
        "- WARNING：{0}（详见 `warnings.csv`）".format(summary["warnings"]),
        "- INFO：{0}（仅供参考，不计入结论，详见 `info.csv`）".format(summary["info"]),
        "- 使用独立 QMT 交易日历：{0}".format("是" if summary["authoritative_calendar"] else "否"),
        "",
        "## 问题类型统计（不含 INFO 级问题）",
        "",
        "| 错误编号 | 数量 |",
        "|---|---:|",
    ]
    for code, count in summary["issue_codes"].items():
        lines.append("| `{0}` | {1} |".format(code, count))
    lines.extend(
        [
            "",
            "完整定位、理论值、实际值、证据、可能原因和处理建议按级别分别列在"
            " `errors.csv`、`warnings.csv`、`info.csv` 三个文件中，互不混杂。",
            "连续缺失日期见 `missing_spans.csv`，逐日和逐证券覆盖率见对应 CSV。",
            "脚本只报告问题，不会修改、填充或删除任何原始数据。",
            "",
        ]
    )
    return "\n".join(lines)
