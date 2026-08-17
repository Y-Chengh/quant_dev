"""不同滑点假设下 Top N 净值对比曲线的 SVG 渲染。

与 ``charts`` 分开成模块，避免图表模块超过 AGENTS.md 约定的 700 行拆分线；
本模块只依赖标准库、numpy 与 pandas，不反向依赖报告正文或回测层。
"""

from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

# 候选滑点曲线的描边颜色与线型，颜色用尽后换下一种线型，保证同色曲线仍可区分；
# 基准曲线固定使用绿色实线。
BASELINE_COLOR = "#059669"
CANDIDATE_COLORS = (
    "#2563eb",
    "#d97706",
    "#7c3aed",
    "#dc2626",
    "#0891b2",
    "#db2777",
)
CANDIDATE_DASHES = ("", "7 4", "2 3")


def _format_bps(value: float) -> str:
    """把单边滑点基点数格式化为图例文本。

    小数位数与评估报告的滑点表保持一致，避免同一档滑点在图和表里显示不同。

    参数：
        value: 单边滑点，单位为基点。

    返回：
        保留四位小数的基点数文本，例如 ``2.5000``。
    """

    return f"{float(value):.4f}"


def render_slippage_curves_svg(
    curves: pd.DataFrame,
    output_path: Path,
) -> None:
    """把各档滑点下的 Top N 累计净值画成同一坐标系内的对比曲线。

    每条曲线都用同一批每日选股与同一手续费率，只替换单边滑点，因此曲线之间的
    差异可以完全归因于滑点。净值统一从期初 1.0 起算，图像只用于结果诊断，不参与
    训练、选股或候选排序。

    参数：
        curves: 长表形式的对比曲线，必须含 ``slippage_bps``、``target_date`` 与
            ``equity``；``is_baseline`` 列可选，用于把基准滑点标注为「基准」。
            同一档滑点的日期必须升序，且各档滑点共用同一组目标交易日。
        output_path: SVG 对比图的写入路径，父目录必须已经存在。

    返回：
        无；函数将可独立打开的 UTF-8 SVG 写入 ``output_path``。
    """

    required = {"slippage_bps", "target_date", "equity"}
    missing = required.difference(curves.columns)
    if missing:
        raise ValueError(f"滑点对比曲线缺少列: {sorted(missing)}")
    if curves.empty:
        raise ValueError("无法为没有对比曲线的回测绘制滑点对比图")

    frame = curves.copy()
    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="coerce")
    if frame["target_date"].isna().any():
        raise ValueError("滑点对比曲线的 target_date 包含缺失或无效日期")
    if not np.isfinite(frame["equity"].to_numpy(dtype=float)).all():
        raise ValueError("滑点对比曲线净值包含 NaN 或无穷值")

    series: list[tuple[float, np.ndarray, bool]] = []
    reference_dates: pd.DatetimeIndex | None = None
    for level, level_frame in frame.groupby("slippage_bps", sort=False):
        dates = pd.DatetimeIndex(level_frame["target_date"])
        if not dates.is_monotonic_increasing:
            raise ValueError("滑点对比曲线的 target_date 必须按目标交易日升序排列")
        if reference_dates is None:
            reference_dates = dates
        elif not dates.equals(reference_dates):
            raise ValueError("各档滑点的对比曲线必须使用同一组目标交易日")
        is_baseline = bool(level_frame.get("is_baseline", pd.Series([False])).any())
        # 显式补上期初净值，保证单日回测也能画出可见线段。
        series.append(
            (
                float(level),
                np.concatenate(([1.0], level_frame["equity"].to_numpy(dtype=float))),
                is_baseline,
            )
        )
    if reference_dates is None:
        raise ValueError("滑点对比曲线没有可绘制的滑点分组")

    # 基准曲线固定绿色实线；候选先按顺序取色，颜色用尽后换线型，与图例一一对应。
    styles: list[tuple[str, str]] = []
    candidate_index = 0
    for _, _, is_baseline in series:
        if is_baseline:
            styles.append((BASELINE_COLOR, ""))
            continue
        color = CANDIDATE_COLORS[candidate_index % len(CANDIDATE_COLORS)]
        dash = CANDIDATE_DASHES[
            (candidate_index // len(CANDIDATE_COLORS)) % len(CANDIDATE_DASHES)
        ]
        styles.append((color, dash))
        candidate_index += 1

    count = len(series[0][1])
    legend_columns = min(4, len(series))
    legend_rows = (len(series) + legend_columns - 1) // legend_columns
    width = 1000
    left, right = 82, 30
    plot_width = width - left - right
    plot_top = 80 + legend_rows * 20
    plot_height = 320
    height = plot_top + plot_height + 66

    all_equities = np.concatenate([values for _, values, _ in series])
    lower = min(1.0, float(all_equities.min()))
    upper = max(1.0, float(all_equities.max()))
    padding = max((upper - lower) * 0.08, max(abs(lower), abs(upper), 1.0) * 0.01)
    y_min, y_max = lower - padding, upper + padding

    def x_at(index: int) -> float:
        """把净值观测序号映射为绘图区横坐标。

        参数：
            index: 从期初零开始的净值观测序号。

        返回：
            当前观测在绘图区内的像素横坐标。
        """

        return left + plot_width * index / max(1, count - 1)

    def y_at(value: float) -> float:
        """把累计净值映射为绘图区纵坐标。

        参数：
            value: 需要绘制的某档滑点下的累计净值。

        返回：
            当前净值在绘图区内的像素纵坐标。
        """

        return plot_top + (y_max - value) / (y_max - y_min) * plot_height

    date_labels = ["期初", *(date.strftime("%Y-%m-%d") for date in reference_dates)]
    tick_indices = np.unique(np.linspace(0, count - 1, min(7, count), dtype=int))
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">不同滑点下的 Top N 收益曲线</title>',
        '<desc id="desc">同一批每日选股在不同单边滑点假设下的累计净值对比</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,"Microsoft YaHei",sans-serif;fill:#334155}.title{font-size:18px;font-weight:600}.subtitle{font-size:12px;fill:#64748b}.grid{stroke:#e2e8f0;stroke-width:1}.axis{stroke:#64748b;stroke-width:1.2}.baseline-mark{stroke:#94a3b8;stroke-width:1;stroke-dasharray:5 4}.curve{fill:none;stroke-width:2.1}.curve-baseline{stroke-width:2.9}</style>',
        '<text x="20" y="30" class="title">不同滑点下的 Top N 收益曲线</text>',
        '<text x="20" y="50" class="subtitle">同一批每日选股、同一手续费率，只替换单边滑点；净值均自期初 1.0 起算</text>',
    ]
    for position, (level, _, is_baseline) in enumerate(series):
        color, dash = styles[position]
        dash_attribute = f' stroke-dasharray="{dash}"' if dash else ""
        column = position % legend_columns
        row = position // legend_columns
        legend_x = left + column * (plot_width / legend_columns)
        legend_y = 74 + row * 20
        label = f"滑点 {_format_bps(level)} bps" + ("（基准）" if is_baseline else "")
        svg.append(
            f'<line class="curve{" curve-baseline" if is_baseline else ""}" '
            f'stroke="{color}"{dash_attribute} x1="{legend_x:.2f}" y1="{legend_y}" '
            f'x2="{legend_x + 28:.2f}" y2="{legend_y}"/>'
        )
        svg.append(
            f'<text x="{legend_x + 34:.2f}" y="{legend_y + 4}" font-size="12">'
            f"{escape(label)}</text>"
        )
    for value in np.linspace(y_min, y_max, 5):
        y = y_at(float(value))
        svg.append(
            f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}"/>'
        )
        svg.append(
            f'<text x="{left - 12}" y="{y + 4:.2f}" font-size="12" text-anchor="end">{float(value):.3f}</text>'
        )
    svg.extend(
        [
            f'<line class="axis" x1="{left}" y1="{plot_top}" x2="{left}" y2="{plot_top + plot_height}"/>',
            f'<line class="axis" x1="{left}" y1="{plot_top + plot_height}" x2="{width - right}" y2="{plot_top + plot_height}"/>',
            f'<line class="baseline-mark" x1="{left}" y1="{y_at(1.0):.2f}" x2="{width - right}" y2="{y_at(1.0):.2f}"/>',
        ]
    )
    for position, (_, values, is_baseline) in enumerate(series):
        color, dash = styles[position]
        dash_attribute = f' stroke-dasharray="{dash}"' if dash else ""
        points = " ".join(
            f"{x_at(index):.2f},{y_at(float(value)):.2f}"
            for index, value in enumerate(values)
        )
        svg.append(
            f'<polyline class="curve{" curve-baseline" if is_baseline else ""}" '
            f'stroke="{color}"{dash_attribute} points="{points}"/>'
        )
    for index in tick_indices:
        x = x_at(int(index))
        label = escape(date_labels[int(index)])
        svg.append(
            f'<text x="{x:.2f}" y="{plot_top + plot_height + 25}" font-size="12" text-anchor="middle">{label}</text>'
        )
    svg.append("</svg>")
    output_path.write_text("\n".join(svg), encoding="utf-8")
