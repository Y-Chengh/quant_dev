from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from .backtesting import TopNBacktestResult
from .experiment import ExperimentResult


METRIC_LABELS = {
    "samples": "样本数",
    "positive_rate": "实际上涨比例",
    "accuracy": "准确率",
    "balanced_accuracy": "平衡准确率",
    "auc": "ROC AUC",
    "ic": "IC",
    "rank_ic": "Rank IC",
    "icir": "ICIR",
    "ic_win_rate": "IC 胜率",
    "pooled_ic": "全样本 Pearson IC",
    "ic_dates": "有效 IC 交易日数",
    "rank_ic_dates": "有效 Rank IC 交易日数",
    "brier_score": "Brier 分数",
    "log_loss": "Log Loss",
    "mae": "MAE",
    "rmse": "RMSE",
    "r2": "R²",
    "direction_accuracy": "方向准确率",
}


def _format_number(value: Any) -> str:
    number = float(value)
    return "N/A" if not np.isfinite(number) else f"{number:.6f}"


def _format_percentage(value: Any) -> str:
    """将有限收益率格式化为百分比，并将缺失值显示为 ``N/A``。

    参数：
        value: 单只证券的开盘至收盘收益率，单位为一；允许 NaN。

    返回：
        保留两位小数的百分比文本，或缺失值标记 ``N/A``。
    """

    number = float(value)
    return "N/A" if not np.isfinite(number) else f"{number:.2%}"


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


def render_markdown_report_html(markdown_text: str, output_path: Path) -> None:
    """将评估 Markdown 转换为带目录和响应式样式的自包含 HTML。

    HTML 使用同目录的 SVG 图表相对路径；宽表支持横向滚动并冻结首列，目录收录
    一至四级标题，在桌面端固定显示、窄屏设备上自动收起。所有报告动态文本默认
    进行 HTML 转义，避免 YAML 或参数快照被解释为可执行标签。

    参数：
        markdown_text: 已生成的完整 Markdown 报告内容。
        output_path: HTML 报告写入路径，通常与 Markdown 同名且后缀为 ``.html``。

    返回：
        无；函数将完整 HTML 文档写入指定路径。
    """

    body, headings, document_title = _markdown_report_body(markdown_text)
    navigation = "\n".join(
        f'<a class="toc-level-{level}" href="#{anchor}">{escape(label)}</a>'
        for level, label, anchor in headings
        if level <= 4
    )
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
.layout{{display:grid;grid-template-columns:260px minmax(0,1fr);min-height:100vh}}aside{{position:sticky;top:0;height:100vh;overflow:auto;padding:28px 22px;background:#0f172a;color:#e2e8f0}}aside h2{{margin:0 0 18px;font-size:17px;color:#fff}}nav{{display:flex;flex-direction:column;gap:4px}}nav a{{padding:7px 10px;border-radius:7px;color:#cbd5e1;text-decoration:none;font-size:13px}}nav a:hover{{background:#1e293b;color:#fff}}nav .toc-level-3{{padding-left:24px;font-size:12px;color:#94a3b8}}nav .toc-level-4{{padding-left:40px;font-size:12px;color:#94a3b8}}.source{{display:block;margin-top:24px;padding:9px 12px;border:1px solid #334155;border-radius:8px;color:#bfdbfe;text-align:center;text-decoration:none;font-size:12px}}
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

    每个排名占一行，单元格依次显示证券代码、模型预估值、当日实际涨幅，以及
    前日收盘相对前前日收盘、前日收盘相对前日开盘的收益率。按月份拆表以限制
    单表宽度，但日期始终位于横轴；某日不足 N 只时对应排名显示为 ``-``。

    参数：
        selections: 含目标日期、日内排名、证券代码、预估值、实际收益及两个
            前日收益口径的 Top N 选股明细；旧数据可仅含
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
        f"日期为横轴；预估结果使用 `{score_column}`。实际为当日收盘价相对开盘价的收益率；两个前日指标均按同一股票的上一可见交易日计算。",
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
                    f"`{code}`<br>预估：{_format_number(row['predicted_value'])}"
                    f"<br>实际：{_format_percentage(row['actual_return'])}"
                    f"<br>前日收盘价对比前前日收盘价：{_format_percentage(row['previous_close_to_close_return'])}"
                    f"<br>前日收盘价对比前日开盘价：{_format_percentage(row['previous_open_to_close_return'])}"
                )
            lines.append(f"| Top {rank} | " + " | ".join(cells) + " |")
    return lines


def render_accuracy_trend_svg(trend: pd.DataFrame, output_path: Path) -> None:
    """Render daily accuracy and its 20-day moving average as a standalone SVG."""
    if trend.empty:
        raise ValueError("Cannot render an accuracy trend without daily observations")

    width, height = 1000, 440
    left, right, top, bottom = 72, 28, 38, 66
    plot_width = width - left - right
    plot_height = height - top - bottom
    accuracy = trend["accuracy"].to_numpy(dtype=float)
    rolling = trend["accuracy"].rolling(window=20, min_periods=1).mean().to_numpy(dtype=float)
    count = len(trend)

    def x_at(index: int) -> float:
        return left + (plot_width * index / max(1, count - 1))

    def y_at(value: float) -> float:
        return top + (1.0 - value) * plot_height

    def points(values: np.ndarray) -> str:
        return " ".join(f"{x_at(index):.2f},{y_at(value):.2f}" for index, value in enumerate(values))

    dates = pd.to_datetime(trend["target_date"])
    tick_indices = np.unique(np.linspace(0, count - 1, min(7, count), dtype=int))
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">日级预估准确率趋势</title>',
        '<desc id="desc">每日准确率及二十日移动平均折线图</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,"Microsoft YaHei",sans-serif;fill:#334155}.grid{stroke:#e2e8f0;stroke-width:1}.axis{stroke:#64748b;stroke-width:1.2}.daily{fill:none;stroke:#93c5fd;stroke-width:1.5;opacity:.9}.rolling{fill:none;stroke:#2563eb;stroke-width:3}</style>',
    ]
    for value in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = y_at(value)
        svg.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}"/>')
        svg.append(f'<text x="{left - 12}" y="{y + 4:.2f}" font-size="12" text-anchor="end">{value:.0%}</text>')
    svg.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}"/>',
            f'<line class="axis" x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}"/>',
            f'<polyline class="daily" points="{points(accuracy)}"/>',
            f'<polyline class="rolling" points="{points(rolling)}"/>',
        ]
    )
    for index in tick_indices:
        x = x_at(int(index))
        label = escape(dates.iloc[int(index)].strftime("%Y-%m-%d"))
        svg.append(f'<text x="{x:.2f}" y="{height - bottom + 25}" font-size="12" text-anchor="middle">{label}</text>')
    svg.extend(
        [
            f'<line class="daily" x1="{width - 278}" y1="19" x2="{width - 242}" y2="19"/>',
            f'<text x="{width - 234}" y="23" font-size="12">每日准确率</text>',
            f'<line class="rolling" x1="{width - 142}" y1="19" x2="{width - 106}" y2="19"/>',
            f'<text x="{width - 98}" y="23" font-size="12">20 日均线</text>',
            '</svg>',
        ]
    )
    output_path.write_text("\n".join(svg), encoding="utf-8")


def render_equity_curve_svg(
    daily_returns: pd.DataFrame,
    output_path: Path,
) -> None:
    """将 Top N 及三组横截面对照的日度净值曲线渲染为独立 SVG。

    参数：
        daily_returns: 按目标交易日排序且至少含日期与 Top N 累计净值的日度回测
            结果；全市场平均、Bottom N 和 Mid N 净值列可选，以兼容旧结果。
        output_path: SVG 收益曲线的写入路径。

    返回：
        无；函数将 SVG 内容写入 ``output_path``。
    """

    equity_columns = {
        "equity": ("Top N", "equity"),
        "universe_equity": ("全市场平均", "universe"),
        "bottom_equity": ("Bottom N", "bottom"),
        "mid_equity": ("Mid N", "mid"),
    }
    required = {"target_date", "equity"}
    missing = required.difference(daily_returns.columns)
    if missing:
        raise ValueError(f"收益曲线缺少列: {sorted(missing)}")
    if daily_returns.empty:
        raise ValueError("无法为没有日度收益的回测绘制收益曲线")

    width, height = 1000, 440
    left, right, top, bottom = 82, 28, 38, 66
    plot_width = width - left - right
    plot_height = height - top - bottom
    available_equity_columns = {
        column: metadata
        for column, metadata in equity_columns.items()
        if column in daily_returns.columns
    }
    closing_equities = daily_returns.loc[
        :, list(available_equity_columns)
    ].to_numpy(
        dtype=float
    )
    if not np.isfinite(closing_equities).all():
        raise ValueError("收益曲线净值包含 NaN 或无穷值")
    # 显式加入期初净值，确保单日回测也能画出一条可见线段。
    equities = {
        column: np.concatenate(
            ([1.0], daily_returns[column].to_numpy(dtype=float))
        )
        for column in available_equity_columns
    }
    count = len(next(iter(equities.values())))
    all_equities = np.concatenate(list(equities.values()))
    lower = min(1.0, float(all_equities.min()))
    upper = max(1.0, float(all_equities.max()))
    padding = max((upper - lower) * 0.08, max(abs(lower), abs(upper), 1.0) * 0.01)
    y_min, y_max = lower - padding, upper + padding

    def x_at(index: int) -> float:
        """将净值观测序号映射为绘图区横坐标。

        参数：
            index: 从期初零开始的净值观测序号。

        返回：
            当前观测在 SVG 绘图区内的像素横坐标。
        """

        return left + plot_width * index / max(1, count - 1)

    def y_at(value: float) -> float:
        """将策略净值映射为绘图区纵坐标。

        参数：
            value: 需要绘制的累计净值。

        返回：
            当前净值在 SVG 绘图区内的像素纵坐标。
        """

        return top + (y_max - value) / (y_max - y_min) * plot_height

    curve_points = {
        column: " ".join(
            f"{x_at(index):.2f},{y_at(value):.2f}"
            for index, value in enumerate(values)
        )
        for column, values in equities.items()
    }
    date_labels = [
        "期初",
        *(date.strftime("%Y-%m-%d") for date in pd.to_datetime(daily_returns["target_date"])),
    ]
    tick_indices = np.unique(np.linspace(0, count - 1, min(7, count), dtype=int))
    y_ticks = np.linspace(y_min, y_max, 5)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Top N 与横截面对照收益曲线</title>',
        '<desc id="desc">Top N、全市场平均、Bottom N 和 Mid N 计入相同双边成本后的累计净值</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,"Microsoft YaHei",sans-serif;fill:#334155}.grid{stroke:#e2e8f0;stroke-width:1}.axis{stroke:#64748b;stroke-width:1.2}.baseline{stroke:#94a3b8;stroke-width:1;stroke-dasharray:5 4}.equity,.universe,.bottom,.mid{fill:none;stroke-width:2.3}.equity{stroke:#059669}.universe{stroke:#2563eb}.bottom{stroke:#dc2626}.mid{stroke:#d97706}</style>',
    ]
    for value in y_ticks:
        y = y_at(float(value))
        svg.append(
            f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}"/>'
        )
        svg.append(
            f'<text x="{left - 12}" y="{y + 4:.2f}" font-size="12" text-anchor="end">{value:.3f}</text>'
        )
    svg.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}"/>',
            f'<line class="axis" x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}"/>',
            f'<line class="baseline" x1="{left}" y1="{y_at(1.0):.2f}" x2="{width - right}" y2="{y_at(1.0):.2f}"/>',
        ]
    )
    for column, (_, css_class) in available_equity_columns.items():
        svg.append(
            f'<polyline class="{css_class}" points="{curve_points[column]}"/>'
        )
    for index in tick_indices:
        x = x_at(int(index))
        label = escape(date_labels[int(index)])
        svg.append(
            f'<text x="{x:.2f}" y="{height - bottom + 25}" font-size="12" text-anchor="middle">{label}</text>'
        )
    legend_x = width - 430
    for position, (_, (label, css_class)) in enumerate(
        available_equity_columns.items()
    ):
        x = legend_x + position * 108
        svg.append(
            f'<line class="{css_class}" x1="{x}" y1="19" x2="{x + 28}" y2="19"/>'
        )
        svg.append(f'<text x="{x + 34}" y="23" font-size="12">{label}</text>')
    svg.append("</svg>")
    output_path.write_text("\n".join(svg), encoding="utf-8")


def write_evaluation_report(
    result: ExperimentResult,
    report_path: Path,
    chart_path: Path,
    run_id: str,
    run_arguments: dict[str, Any],
    yaml_config: str | None = None,
    backtest: TopNBacktestResult | None = None,
    equity_chart_path: Path | None = None,
) -> None:
    """写入模型评估、可选 Top N 回测、HTML 副本以及对应 SVG 图表。

    参数：
        result: 模型验证结果及逐证券预测。
        report_path: Markdown 评估报告写入路径。
        chart_path: 日级预测准确率 SVG 图表写入路径。
        run_id: 当前实验的唯一运行标识。
        run_arguments: 已生效的命令行及 YAML 合并参数。
        yaml_config: 原始 YAML 配置文本；未使用配置文件时为 ``None``。
        backtest: Top N 日内策略结果；缺省时不输出回测章节。
        equity_chart_path: 收益曲线 SVG 路径；提供回测结果时必须同时提供。

    返回：
        无；函数写入 Markdown、同名 HTML 报告及配置的 SVG 图表。
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)
    render_accuracy_trend_svg(result.daily_accuracy_trend, chart_path)
    if backtest is not None:
        if equity_chart_path is None:
            raise ValueError("提供 backtest 时必须同时提供 equity_chart_path")
        render_equity_curve_svg(backtest.daily_returns, equity_chart_path)

    predictions = result.predictions.copy()
    if "prediction" in predictions:
        predictions["predicted_up"] = predictions["prediction"].astype(bool)
    else:
        predictions["predicted_up"] = predictions["up_probability"] >= 0.5
    daily_summary = (
        predictions.groupby("target_date", as_index=False, sort=True)
        .agg(
            samples=("label", "size"),
            actual_up_rate=("label", "mean"),
            predicted_up_rate=("predicted_up", "mean"),
        )
        .merge(result.daily_accuracy_trend, on=["target_date", "samples"], how="left")
    )

    lines = [
        (
            "# 涨跌幅预测评估报告"
            if result.task == "regression"
            else "# 方向预测评估报告"
        ),
        "",
        f"- 运行 ID：`{run_id}`",
        f"- 生成时间：{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"- 验证区间：{pd.Timestamp(daily_summary['target_date'].min()).date()} 至 {pd.Timestamp(daily_summary['target_date'].max()).date()}",
        "",
        "## 汇总指标",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
    ]
    for key, value in result.metrics.items():
        lines.append(f"| {METRIC_LABELS.get(key, key)} | {_format_number(value)} |")

    lines.extend(
        [
            "",
            "## 准确率趋势",
            "",
            f"![日级预估准确率趋势]({chart_path.name})",
            "",
            "浅色线为每日准确率，深色线为 20 日移动平均。",
        ]
    )

    if backtest is not None:
        metric_labels = {
            "trading_days": "交易日数",
            "total_return": "累计收益率",
            "annualized_return": "年化收益率",
            "annualized_volatility": "年化波动率",
            "sharpe_ratio": "夏普比率",
        }
        lines.extend(
            [
                "",
                "## Top N 日内策略回测",
                "",
                f"- 每日选股数：最多 {backtest.top_n} 只",
                f"- 排序分数：`{backtest.score_column}`",
                f"- 单边滑点：{backtest.slippage_bps:.4f} bps",
                f"- 单边手续费：{backtest.commission_bps:.4f} bps",
                "- 交易口径：目标日开盘等权买入、收盘全部卖出，成本在买卖两边分别计取。",
                "",
                "| 回测指标 | 数值 |",
                "| --- | ---: |",
            ]
        )
        for key, value in backtest.metrics.items():
            lines.append(
                f"| {metric_labels.get(key, key)} | {_format_number(value)} |"
            )
        lines.extend(
            _render_top_selection_tables(
                backtest.top_selections,
                backtest.score_column,
            )
        )
        lines.extend(
            [
                "",
                "### Top N 与横截面对照收益曲线",
                "",
                f"![Top N、全市场平均、Bottom N 与 Mid N 收益曲线]({equity_chart_path.name})",
                "",
                "四组均按目标日开盘等权买入、收盘卖出并采用相同双边成本；Mid N 为预测排序居中的最多 N 只。",
                "",
                "### Top N 基准与横截面对照",
                "",
                "等权及随机组合采用与 Top N 相同的双边成本；随机基准按固定种子独立逐日抽样。",
                "",
                "| 基准指标 | 数值 |",
                "| --- | ---: |",
            ]
        )
        benchmark_labels = {
            "equal_weight_total_return": "全股票等权累计收益率",
            "equal_weight_annualized_return": "全股票等权年化收益率",
            "equal_weight_sharpe_ratio": "全股票等权夏普比率",
            "random_simulations": "随机 Top N 模拟次数",
            "random_annualized_p05": "随机 Top N 年化收益率 P05",
            "random_annualized_median": "随机 Top N 年化收益率中位数",
            "random_annualized_p95": "随机 Top N 年化收益率 P95",
            "strategy_random_percentile": "策略在随机基准中的百分位",
        }
        for key, value in backtest.benchmark_metrics.items():
            lines.append(
                f"| {benchmark_labels.get(key, key)} | {_format_number(value)} |"
            )

        lines.extend(
            [
                "",
                "### Top N 超额与多空价差",
                "",
                "收益差采用每日毛收益之差和算术年化；该口径用于检验选股能力，不作为可复利长仓净值。",
                "",
                "| 价差指标 | 数值 |",
                "| --- | ---: |",
            ]
        )
        relative_labels = {
            "top_minus_universe_annualized_return": "Top N - 全股票等权：算术年化收益",
            "top_minus_universe_annualized_volatility": "Top N - 全股票等权：年化波动率",
            "top_minus_universe_sharpe_ratio": "Top N - 全股票等权：夏普比率",
            "top_minus_bottom_annualized_return": "Top N - Bottom N：算术年化收益",
            "top_minus_bottom_annualized_volatility": "Top N - Bottom N：年化波动率",
            "top_minus_bottom_sharpe_ratio": "Top N - Bottom N：夏普比率",
        }
        for key, value in backtest.relative_metrics.items():
            lines.append(
                f"| {relative_labels.get(key, key)} | {_format_number(value)} |"
            )

        lines.extend(
            [
                "",
                "### 预测分数十分位收益",
                "",
                "十分位 1 为最低预测分数组，十分位 10 为最高预测分数组；收益未扣成本。",
                "",
                "| 十分位 | 样本数 | 交易日数 | 平均日收益 | 算术年化收益 | 上涨比例 |",
                "| ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in backtest.decile_returns.itertuples(index=False):
            lines.append(
                f"| {row.predicted_decile} | {row.samples} | {row.trading_days} | "
                f"{_format_number(row.average_return)} | "
                f"{_format_number(row.annualized_arithmetic_return)} | "
                f"{_format_number(row.hit_rate)} |"
            )

        selection_labels = {
            "top_samples": "Top N 样本数",
            "top_hit_rate": "Top N 上涨比例",
            "top_average_return": "Top N 平均日收益",
            "other_samples": "其余股票样本数",
            "other_hit_rate": "其余股票上涨比例",
            "other_average_return": "其余股票平均日收益",
            "top_minus_other_average_return": "Top N 相对其余股票平均日收益差",
        }
        lines.extend(
            [
                "",
                "### Top N 与其余股票命中对照",
                "",
                "| 选股指标 | 数值 |",
                "| --- | ---: |",
            ]
        )
        for key, value in backtest.selection_metrics.items():
            lines.append(
                f"| {selection_labels.get(key, key)} | {_format_number(value)} |"
            )

    lines.extend(["", "## 因子重要性", ""])
    if result.feature_importance is None:
        lines.append("当前模型未提供因子重要性。")
    else:
        lines.extend(["| 因子 | 重要性 |", "| --- | ---: |"])
        for factor, importance in result.feature_importance.items():
            lines.append(f"| `{factor}` | {float(importance):.6f} |")

    lines.extend(["", "## 运行参数", "", "```text"])
    lines.extend(f"{key}={value}" for key, value in sorted(run_arguments.items()))
    lines.extend(["```", "", "## YAML 配置", ""])
    if yaml_config is None:
        lines.append("未使用 YAML 配置文件。")
    else:
        config_text = yaml_config.rstrip("\r\n")
        fence = "```"
        while fence in config_text:
            fence += "`"
        lines.extend([f"{fence}yaml", config_text, fence])

    if not result.daily_ic_trend.empty:
        lines.extend(
            [
                "",
                "## 每日横截面 IC",
                "",
                "| 目标日期 | 有效样本数 | IC | Rank IC |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for row in result.daily_ic_trend.itertuples(index=False):
            lines.append(
                f"| {pd.Timestamp(row.target_date).date()} | {row.samples} | "
                f"{_format_number(row.ic)} | {_format_number(row.rank_ic)} |"
            )

    lines.extend(
        [
            "",
            "## 每日预估汇总",
            "",
            "| 目标日期 | 样本数 | 实际上涨比例 | 预测上涨比例 | 准确率 | 较前日变化 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in daily_summary.itertuples(index=False):
        change = "-" if pd.isna(row.accuracy_change) else f"{row.accuracy_change:+.2%}"
        lines.append(
            f"| {pd.Timestamp(row.target_date).date()} | {row.samples} | {row.actual_up_rate:.2%} | "
            f"{row.predicted_up_rate:.2%} | {row.accuracy:.2%} | {change} |"
        )

    lines.append("")
    markdown_text = "\n".join(lines)
    report_path.write_text(markdown_text, encoding="utf-8")
    render_markdown_report_html(markdown_text, report_path.with_suffix(".html"))
