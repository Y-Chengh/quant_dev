"""Markdown 报告正文拼装、HTML 转换与候选表格渲染。"""

from __future__ import annotations

import re
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from .formatting import (
    _format_percentage,
    _is_markdown_table_separator,
    _render_inline_markdown,
    _split_markdown_table_row,
)


def _markdown_report_body(
    markdown_text: str,
) -> tuple[str, list[tuple[int, str, str]], str]:
    """将框架生成的 Markdown 报告转换为安全的语义化 HTML 主体。

    参数：
        markdown_text: ``write_evaluation_report`` 生成的完整 Markdown 文本。

    返回：
        HTML 主体、目录标题元组列表及文档标题。
    """

    lines = markdown_text.splitlines()
    body: list[str] = []
    headings: list[tuple[int, str, str]] = []
    document_title = "量化因子研究报告"
    index = 0
    heading_number = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            index += 1
            continue

        fence_match = re.fullmatch(r"(`{3,})([A-Za-z0-9_-]*)", stripped)
        if fence_match is not None:
            delimiter = fence_match.group(1)
            language = fence_match.group(2)
            code_lines: list[str] = []
            index += 1
            while index < len(lines) and lines[index].strip() != delimiter:
                code_lines.append(lines[index])
                index += 1
            if index < len(lines):
                index += 1
            language_class = (
                f' class="language-{escape(language, quote=True)}"'
                if language
                else ""
            )
            body.append(
                f"<pre><code{language_class}>"
                f"{escape(chr(10).join(code_lines))}</code></pre>"
            )
            continue

        heading_match = re.fullmatch(r"(#{1,6})\s+(.+)", stripped)
        if heading_match is not None:
            level = len(heading_match.group(1))
            label = heading_match.group(2)
            heading_number += 1
            anchor = f"section-{heading_number}"
            plain_label = re.sub(r"`([^`]+)`", r"\1", label)
            if level == 1:
                document_title = plain_label
            headings.append((level, plain_label, anchor))
            body.append(
                f'<h{level} id="{anchor}">'
                f'<a class="heading-anchor" href="#{anchor}">#</a>'
                f"{_render_inline_markdown(label)}</h{level}>"
            )
            index += 1
            continue

        if (
            stripped.startswith("|")
            and index + 1 < len(lines)
            and _is_markdown_table_separator(lines[index + 1])
        ):
            headers = _split_markdown_table_row(line)
            alignments = _split_markdown_table_row(lines[index + 1])
            body.append('<div class="table-scroll"><table><thead><tr>')
            for cell, alignment in zip(headers, alignments):
                align_class = " align-right" if alignment.endswith(":") else ""
                body.append(
                    f'<th class="{align_class.strip()}">'
                    f"{_render_inline_markdown(cell, allow_breaks=True)}</th>"
                )
            body.append("</tr></thead><tbody>")
            index += 2
            while index < len(lines) and lines[index].strip().startswith("|"):
                cells = _split_markdown_table_row(lines[index])
                body.append("<tr>")
                for position, cell in enumerate(cells):
                    alignment = (
                        alignments[position] if position < len(alignments) else ""
                    )
                    align_class = " align-right" if alignment.endswith(":") else ""
                    body.append(
                        f'<td class="{align_class.strip()}">'
                        f"{_render_inline_markdown(cell, allow_breaks=True)}</td>"
                    )
                body.append("</tr>")
                index += 1
            body.append("</tbody></table></div>")
            continue

        if stripped.startswith("- "):
            body.append("<ul>")
            while index < len(lines) and lines[index].strip().startswith("- "):
                item = lines[index].strip()[2:]
                body.append(f"<li>{_render_inline_markdown(item)}</li>")
                index += 1
            body.append("</ul>")
            continue

        body.append(f"<p>{_render_inline_markdown(stripped)}</p>")
        index += 1
    return "\n".join(body), headings, document_title


def _render_table_of_contents(headings: list[tuple[int, str, str]]) -> str:
    """把报告标题列表渲染为带默认折叠分组的侧边目录。

    一至四级标题都会进入目录。四级标题（每日 Top N 选股明细下的逐月表格）会连同
    其上级标题收进默认折叠的 ``<details>`` 分组，避免长月份列表把其它章节挤出
    可视区域；点击分组标题仍可直接跳转到对应章节。

    参数：
        headings: ``_markdown_report_body`` 返回的（层级, 纯文本标题, 锚点）列表，
            顺序与正文标题出现顺序一致。

    返回：
        可直接嵌入侧边栏 ``<nav>`` 的 HTML 片段。
    """

    items = [item for item in headings if item[0] <= 4]
    parts: list[str] = []
    index = 0
    while index < len(items):
        level, label, anchor = items[index]
        link = f'<a class="toc-level-{level}" href="#{anchor}">{escape(label)}</a>'
        children: list[tuple[int, str, str]] = []
        cursor = index + 1
        if level < 4:
            while cursor < len(items) and items[cursor][0] == 4:
                children.append(items[cursor])
                cursor += 1
        if not children:
            parts.append(link)
            index += 1
            continue
        child_links = "".join(
            f'<a class="toc-level-{child_level}" href="#{child_anchor}">'
            f"{escape(child_label)}</a>"
            for child_level, child_label, child_anchor in children
        )
        parts.append(
            f'<details class="toc-group"><summary>{link}'
            f'<span class="toc-count">{len(children)}</span></summary>'
            f'<div class="toc-children">{child_links}</div></details>'
        )
        index = cursor
    return "\n".join(parts)


def render_markdown_report_html(markdown_text: str, output_path: Path) -> None:
    """将评估 Markdown 转换为带目录和响应式样式的自包含 HTML。

    HTML 使用同目录的 SVG 图表相对路径；宽表支持横向滚动并冻结首列，目录收录
    一至四级标题，在桌面端固定显示、窄屏设备上自动收起。四级标题按上级章节收进
    默认折叠的分组，详见 ``_render_table_of_contents``。所有报告动态文本默认
    进行 HTML 转义，避免 YAML 或参数快照被解释为可执行标签。

    参数：
        markdown_text: 已生成的完整 Markdown 报告内容。
        output_path: HTML 报告写入路径，通常与 Markdown 同名且后缀为 ``.html``。

    返回：
        无；函数将完整 HTML 文档写入指定路径。
    """

    body, headings, document_title = _markdown_report_body(markdown_text)
    navigation = _render_table_of_contents(headings)
    source_name = escape(output_path.with_suffix(".md").name, quote=True)
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(document_title)}</title>
<style>
:root{{--bg:#f4f7fb;--panel:#fff;--ink:#172033;--muted:#64748b;--line:#dbe3ee;--brand:#2563eb;--brand-soft:#eff6ff;--shadow:0 14px 40px rgba(15,23,42,.08)}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,"Microsoft YaHei","PingFang SC",Arial,sans-serif;line-height:1.65}}
.layout{{display:grid;grid-template-columns:260px minmax(0,1fr);min-height:100vh}}aside{{position:sticky;top:0;height:100vh;overflow:auto;padding:28px 22px;background:#0f172a;color:#e2e8f0}}aside h2{{margin:0 0 18px;font-size:17px;color:#fff}}nav{{display:flex;flex-direction:column;gap:4px}}nav a{{padding:7px 10px;border-radius:7px;color:#cbd5e1;text-decoration:none;font-size:13px}}nav a:hover{{background:#1e293b;color:#fff}}nav .toc-level-3{{padding-left:24px;font-size:12px;color:#94a3b8}}nav .toc-level-4{{padding-left:40px;font-size:12px;color:#94a3b8}}
details.toc-group{{display:flex;flex-direction:column;gap:4px}}details.toc-group>summary{{display:flex;align-items:center;gap:6px;padding:0 8px 0 6px;border-radius:7px;list-style:none;cursor:pointer}}details.toc-group>summary::-webkit-details-marker{{display:none}}details.toc-group>summary::before{{content:"▸";color:#94a3b8;font-size:11px}}details.toc-group[open]>summary::before{{content:"▾"}}details.toc-group>summary:hover{{background:#1e293b}}details.toc-group>summary a{{flex:1;padding-left:4px}}details.toc-group>summary:hover a{{color:#fff}}.toc-count{{padding:1px 7px;border-radius:999px;background:#1e293b;color:#94a3b8;font-size:11px}}.toc-children{{display:flex;flex-direction:column;gap:4px}}.source{{display:block;margin-top:24px;padding:9px 12px;border:1px solid #334155;border-radius:8px;color:#bfdbfe;text-align:center;text-decoration:none;font-size:12px}}
main{{min-width:0;padding:34px}}article{{max-width:1500px;margin:0 auto;padding:38px 42px 70px;background:var(--panel);border:1px solid #e7edf5;border-radius:16px;box-shadow:var(--shadow)}}h1{{margin:0 0 22px;font-size:30px;line-height:1.25}}h2{{margin:42px 0 16px;padding-bottom:9px;border-bottom:2px solid var(--line);font-size:22px}}h3{{margin:30px 0 12px;font-size:18px}}h4{{margin:25px 0 10px;color:#334155}}.heading-anchor{{margin-left:-20px;padding-right:6px;color:#94a3b8;text-decoration:none;opacity:0}}h1:hover .heading-anchor,h2:hover .heading-anchor,h3:hover .heading-anchor,h4:hover .heading-anchor{{opacity:1}}p{{margin:10px 0;color:#334155}}ul{{margin:8px 0 20px;padding-left:22px}}code{{padding:.12em .38em;border-radius:5px;background:#eef2f7;color:#be123c;font-family:"Cascadia Code",Consolas,monospace;font-size:.9em}}pre{{overflow:auto;padding:18px;border-radius:10px;background:#111827;color:#e5e7eb}}pre code{{padding:0;background:transparent;color:inherit}}img{{display:block;max-width:100%;height:auto;margin:18px auto;border:1px solid var(--line);border-radius:10px;background:#fff}}
.table-scroll{{max-width:100%;margin:14px 0 24px;overflow:auto;border:1px solid var(--line);border-radius:10px;background:#fff}}table{{width:max-content;min-width:100%;border-collapse:separate;border-spacing:0;font-size:13px;line-height:1.45}}th,td{{min-width:108px;padding:10px 12px;border-right:1px solid var(--line);border-bottom:1px solid var(--line);vertical-align:top;white-space:nowrap}}th{{position:sticky;top:0;z-index:2;background:#eaf1fb;color:#1e3a5f;font-weight:700}}th:first-child,td:first-child{{position:sticky;left:0;z-index:1;min-width:120px;background:#f8fafc}}th:first-child{{z-index:3;background:#dfeafb}}tbody tr:nth-child(even) td{{background:#f8fafc}}tbody tr:nth-child(even) td:first-child{{background:#eef2f7}}tbody tr:hover td{{background:#fff7ed}}tbody tr:hover td:first-child{{background:#ffedd5}}tr:last-child td{{border-bottom:0}}th:last-child,td:last-child{{border-right:0}}.align-right{{text-align:right;font-variant-numeric:tabular-nums}}
@media(max-width:900px){{.layout{{display:block}}aside{{position:relative;width:auto;height:auto;padding:18px}}nav{{display:none}}.source{{margin-top:8px}}main{{padding:12px}}article{{padding:24px 18px;border-radius:10px}}h1{{font-size:25px}}h2{{font-size:20px}}}}
@media print{{body{{background:#fff}}.layout{{display:block}}aside{{display:none}}main{{padding:0}}article{{max-width:none;padding:0;border:0;box-shadow:none}}.table-scroll{{overflow:visible}}th,td{{white-space:normal}}}}
</style>
</head>
<body><div class="layout"><aside><h2>报告目录</h2><nav>{navigation}</nav><a class="source" href="{source_name}">查看原始 Markdown</a></aside><main><article>{body}</article></main></div></body>
</html>"""
    output_path.write_text(html, encoding="utf-8")


def _render_top_selection_tables(
    selections: pd.DataFrame,
    score_column: str,
) -> list[str]:
    """按月份生成以交易日期为横轴的每日 Top N Markdown 明细表。

    每个排名占一行，单元格依次以百分比显示模型预估值、当日实际涨幅、当日
    收盘相对前日收盘的“t涨幅”，以及前日收盘相对前前日收盘的“t-1涨幅”
    和前日收盘相对前日开盘的“t-1日内涨幅”。
    按月份拆表以限制单表宽度，但日期始终位于横轴；某日不足 N 只时对应排名
    显示为 ``-``。

    参数：
        selections: 含目标日期、日内排名、证券代码、预估值、实际收益、目标日
            收盘相对前日收盘收益及两个前日收益口径的 Top N 选股明细；旧数据可仅含
            ``previous_actual_return``，此时按前日开盘至收盘口径兼容显示。
        score_column: 原始模型分数字段名，用于在表格说明中标明预估值口径。

    返回：
        可直接追加到评估报告的 Markdown 文本行。
    """

    if selections.empty:
        return ["", "暂无 Top N 选股明细。"]

    required = {
        "target_date",
        "top_rank",
        "code",
        "predicted_value",
        "actual_return",
    }
    missing = required.difference(selections.columns)
    if missing:
        raise ValueError(f"Top N 选股明细缺少列: {sorted(missing)}")

    frame = selections.loc[:, list(required)].copy()
    if "previous_open_to_close_return" in selections.columns:
        frame["previous_open_to_close_return"] = selections[
            "previous_open_to_close_return"
        ]
    elif "previous_actual_return" in selections.columns:
        frame["previous_open_to_close_return"] = selections[
            "previous_actual_return"
        ]
    else:
        frame["previous_open_to_close_return"] = np.nan
    if "previous_close_to_close_return" in selections.columns:
        frame["previous_close_to_close_return"] = selections[
            "previous_close_to_close_return"
        ]
    else:
        frame["previous_close_to_close_return"] = np.nan
    if "target_close_to_previous_close_return" in selections.columns:
        frame["target_close_to_previous_close_return"] = selections[
            "target_close_to_previous_close_return"
        ]
    else:
        frame["target_close_to_previous_close_return"] = np.nan
    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="coerce")
    if frame["target_date"].isna().any():
        raise ValueError("Top N 选股明细的 target_date 包含缺失或无效日期")
    frame = frame.sort_values(
        ["target_date", "top_rank"], kind="mergesort"
    ).reset_index(drop=True)
    frame["month"] = frame["target_date"].dt.to_period("M")
    lines = [
        "",
        "### 每日 Top N 选股明细",
        "",
        f"日期为横轴；预估结果使用 `{score_column}`。实际为当日收盘价相对开盘价的收益率；t涨幅为当日收盘价相对前日收盘价的收益率；两个 t-1 指标均按同一股票的上一可见交易日计算。",
    ]
    for month, month_frame in frame.groupby("month", sort=True):
        dates = pd.Index(month_frame["target_date"].drop_duplicates().sort_values())
        ranks = range(1, int(month_frame["top_rank"].max()) + 1)
        lines.extend(
            [
                "",
                f"#### {month}",
                "",
                "| Top N 排名 | "
                + " | ".join(pd.Timestamp(date).strftime("%Y-%m-%d") for date in dates)
                + " |",
                "| ---: | " + " | ".join("---" for _ in dates) + " |",
            ]
        )
        indexed = month_frame.set_index(["top_rank", "target_date"])
        for rank in ranks:
            cells: list[str] = []
            for target_date in dates:
                key = (rank, pd.Timestamp(target_date))
                if key not in indexed.index:
                    cells.append("-")
                    continue
                row = indexed.loc[key]
                code = str(row["code"]).replace("|", r"\|")
                cells.append(
                    f"`{code}`<br>预估：{_format_percentage(row['predicted_value'])}"
                    f"<br>实际：{_format_percentage(row['actual_return'])}"
                    f"<br>t涨幅：{_format_percentage(row['target_close_to_previous_close_return'])}"
                    f"<br>t-1涨幅：{_format_percentage(row['previous_close_to_close_return'])}"
                    f"<br>t-1日内涨幅：{_format_percentage(row['previous_open_to_close_return'])}"
                )
            lines.append(f"| Top {rank} | " + " | ".join(cells) + " |")
    return lines
