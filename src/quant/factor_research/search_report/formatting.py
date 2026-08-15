"""搜索报告的数值、单元格与 Markdown 表格格式化。"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def _json_default(value: object) -> object:
    """把报告元数据中的常见非 JSON 类型转换为稳定文本或标量。

    参数：
        value: 待序列化的路径、日期、NumPy 标量或其他业务对象。

    返回：
        可由标准 ``json`` 模块编码的值。
    """

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"无法序列化类型 {type(value).__name__}")


def _format_cell(value: object, *, column: str | None = None) -> str:
    """把指标值格式化为紧凑且不会破坏 Markdown 表格的文本。

    参数：
        value: 指标、布尔值、日期、表达式或缺失值。
        column: 当前值所属的报告列名；以 ``_rank`` 结尾的名次列会移除无意义的
            小数尾零，缺省为空时沿用普通指标格式。

    返回：
        适合写入 Markdown 单元格的转义文本。
    """

    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return "—"
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            return "—"
        if column is not None and column.endswith("_rank"):
            return f"{number:.6f}".rstrip("0").rstrip(".")
        return f"{number:.6f}"
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.strftime("%Y-%m-%d")
    return str(value).replace("|", "\\|").replace("\n", " ")


def _powershell_single_quoted(value: str) -> str:
    """把文本编码为不会展开美元符号或反引号的 PowerShell 单引号参数。

    参数：
        value: 要作为单个 PowerShell 命令行参数展示的因子表达式文本。

    返回：
        已包围单引号且把内部单引号加倍的 PowerShell 字面量。
    """

    return "'" + value.replace("'", "''") + "'"


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    """生成不依赖 ``tabulate`` 的 Markdown 指标表。

    参数：
        frame: 报告中需要展示的数据表，可为空或缺少部分可选列。
        columns: 期望展示的列名及顺序；不存在的列会被跳过。

    返回：
        Markdown 表格文本；无可展示数据时返回明确提示。
    """

    selected = [column for column in columns if column in frame.columns]
    if frame.empty or not selected:
        return "无数据。"
    header = "| " + " | ".join(selected) + " |"
    divider = "| " + " | ".join("---" for _ in selected) + " |"
    rows = [
        "| "
        + " | ".join(
            _format_cell(row[column], column=column) for column in selected
        )
        + " |"
        for _, row in frame.loc[:, selected].iterrows()
    ]
    return "\n".join([header, divider, *rows])
