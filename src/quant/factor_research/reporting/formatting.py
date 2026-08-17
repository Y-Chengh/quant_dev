"""数值格式化与 Markdown 行内标记的渲染helpers。"""

from __future__ import annotations

import re
from html import escape
from typing import Any

import numpy as np


def _format_number(value: Any) -> str:
    number = float(value)
    return "N/A" if not np.isfinite(number) else f"{number:.6f}"


def _format_percentage(value: Any) -> str:
    """将有限比率格式化为百分比，并将缺失值显示为 ``N/A``。

    参数：
        value: 以一为单位的比率，例如单只证券的开盘至收盘收益率、组合最大回撤
            或处于回撤的交易日占比；允许 NaN。

    返回：
        保留两位小数的百分比文本，或缺失值标记 ``N/A``。
    """

    number = float(value)
    return "N/A" if not np.isfinite(number) else f"{number:.2%}"


def _format_integer(value: Any) -> str:
    """将交易日数或区间计数格式化为整数文本，并将缺失值显示为 ``N/A``。

    参数：
        value: 回撤区间的交易日数量或区间个数；未修复等无定义场景允许为 NaN。

    返回：
        四舍五入到整数的计数文本，或缺失值标记 ``N/A``。
    """

    number = float(value)
    return "N/A" if not np.isfinite(number) else f"{int(round(number))}"


def _render_inline_markdown(text: str, allow_breaks: bool = False) -> str:
    """安全渲染报告使用的图片和行内代码 Markdown 子集。

    参数：
        text: 单个标题、段落或表格单元格的 Markdown 文本。
        allow_breaks: 是否保留报告生成器写入的 ``<br>`` 换行标签；仅表格
            单元格应启用，其他原始 HTML 一律转义。

    返回：
        已完成 HTML 转义并替换受支持行内标记的文本。
    """

    token_pattern = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)|`([^`]+)`")

    def escaped_text(value: str) -> str:
        """转义普通文本，并按调用口径选择性保留换行标签。

        参数：
            value: 不包含图片或行内代码 token 的普通 Markdown 片段。

        返回：
            可安全嵌入 HTML 的文本片段。
        """

        if not allow_breaks:
            return escape(value)
        return "<br>".join(escape(part) for part in value.split("<br>"))

    rendered: list[str] = []
    cursor = 0
    for match in token_pattern.finditer(text):
        rendered.append(escaped_text(text[cursor : match.start()]))
        if match.group(1) is not None:
            alt = escape(match.group(1), quote=True)
            source = escape(match.group(2), quote=True)
            rendered.append(
                f'<img src="{source}" alt="{alt}" loading="lazy">'
            )
        else:
            rendered.append(f"<code>{escape(match.group(3))}</code>")
        cursor = match.end()
    rendered.append(escaped_text(text[cursor:]))
    return "".join(rendered)


def _split_markdown_table_row(line: str) -> list[str]:
    """拆分一行 Markdown 表格并保留转义后的竖线字符。

    参数：
        line: 以竖线分隔的 Markdown 表头、分隔行或数据行。

    返回：
        去除单元格两侧空白后的字段列表。
    """

    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    cells: list[str] = []
    current: list[str] = []
    index = 0
    while index < len(stripped):
        character = stripped[index]
        if character == "\\" and index + 1 < len(stripped):
            following = stripped[index + 1]
            if following == "|":
                current.append("|")
                index += 2
                continue
        if character == "|":
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(character)
        index += 1
    cells.append("".join(current).strip())
    return cells


def _is_markdown_table_separator(line: str) -> bool:
    """判断文本行是否为 Markdown 表格的对齐分隔行。

    参数：
        line: 待识别的 Markdown 文本行。

    返回：
        所有单元格都只含冒号和至少三个连字符时返回 ``True``。
    """

    cells = _split_markdown_table_row(line)
    return bool(cells) and all(
        re.fullmatch(r":?-{3,}:?", cell) is not None for cell in cells
    )
