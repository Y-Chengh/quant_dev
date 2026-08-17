"""准确率趋势、IC 趋势与资金曲线的 SVG 图表渲染。"""

from __future__ import annotations

import math
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from .labels import (
    IC_TREND_LONG_MIN_PERIODS,
    IC_TREND_LONG_WINDOW,
    IC_TREND_SHORT_MIN_PERIODS,
    IC_TREND_SHORT_WINDOW,
)


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


def render_ic_trend_svg(
    trend: pd.DataFrame,
    output_path: Path,
    *,
    title: str = "IC / Rank IC 滚动趋势",
    short_window: int = IC_TREND_SHORT_WINDOW,
    short_min_periods: int = IC_TREND_SHORT_MIN_PERIODS,
    long_window: int = IC_TREND_LONG_WINDOW,
    long_min_periods: int = IC_TREND_LONG_MIN_PERIODS,
    boundary_date: pd.Timestamp | None = None,
) -> None:
    """绘制动态纵轴的短期及长期 IC/Rank IC 滚动趋势 SVG。

    两条均线均为以当日为右端点的尾随窗口，不引用未来观测。每个指标面板的纵轴
    只依据自身滚动值动态缩放，避免常见的小幅 IC 波动被固定相关系数全范围压成
    近似直线。图像只用于结果诊断，不参与模型训练、因子定向或候选排序。

    参数：
        trend: 按目标交易日记录的横截面指标表，必须包含 ``target_date``、``ic``
            和 ``rank_ic``；允许指标缺失，但日期必须可解析。
        output_path: SVG 图像写入路径，父目录必须已经存在。
        title: 图像主标题；缺省为通用 IC/Rank IC 滚动趋势标题。
        short_window: 短期尾随窗口包含的目标交易日行数，缺省为 20 日。
        short_min_periods: 短期均线出值所需的最少有限观测数，缺省为 5 日。
        long_window: 长期尾随窗口包含的目标交易日行数，缺省为 60 日。
        long_min_periods: 长期均线出值所需的最少有限观测数，缺省为 20 日。
        boundary_date: 可选的阶段分界日期，例如首个 holdout 目标日；缺省不绘制
            分界线。

    返回：
        无；函数将可独立打开的 UTF-8 SVG 写入 ``output_path``。
    """

    for window_name, window, min_periods in (
        ("short", short_window, short_min_periods),
        ("long", long_window, long_min_periods),
    ):
        if window <= 0:
            raise ValueError(f"{window_name}_window 必须为正整数")
        if min_periods <= 0 or min_periods > window:
            raise ValueError(
                f"{window_name}_min_periods 必须在 1 到对应 window 之间"
            )
    required = {"target_date", "ic", "rank_ic"}
    missing = required.difference(trend.columns)
    if missing:
        raise ValueError(f"IC 趋势缺少列: {sorted(missing)}")

    frame = trend.loc[:, ["target_date", "ic", "rank_ic"]].copy()
    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="coerce")
    if frame["target_date"].isna().any():
        raise ValueError("IC 趋势的 target_date 包含缺失或无效日期")
    frame = frame.sort_values("target_date", kind="stable").reset_index(drop=True)
    for metric in ("ic", "rank_ic"):
        values = pd.to_numeric(frame[metric], errors="coerce").to_numpy(
            dtype=float,
            na_value=np.nan,
        ).copy()
        values[~np.isfinite(values)] = np.nan
        frame[metric] = values
        frame[f"short_{metric}"] = frame[metric].rolling(
            window=short_window,
            min_periods=short_min_periods,
        ).mean()
        frame[f"long_{metric}"] = frame[metric].rolling(
            window=long_window,
            min_periods=long_min_periods,
        ).mean()

    width = 1200
    height = 760
    left = 80
    right = 30
    panel_gap = 70
    panel_width = (width - left - right - panel_gap) / 2.0
    panel_height = 220
    panel_lefts = (float(left), left + panel_width + panel_gap)
    panel_tops = (105, 420)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{escape(title)}</title>',
        f'<desc id="desc">IC 与 Rank IC 的 {short_window} 日及 {long_window} 日尾随均线，四个面板分别按自身滚动值动态缩放</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,"Microsoft YaHei",sans-serif;fill:#334155}.title{font-size:18px;font-weight:600}.subtitle{font-size:12px;fill:#64748b}.axis{font-size:11px;fill:#64748b}.panel{font-size:14px;font-weight:600}.short{fill:none;stroke:#0ea5e9;stroke-width:2}.long{fill:none;stroke:#1d4ed8;stroke-width:3}.grid{stroke:#e2e8f0;stroke-width:1}.zero{stroke:#94a3b8;stroke-width:1;stroke-dasharray:4 4}.boundary{stroke:#dc2626;stroke-width:1.5;stroke-dasharray:6 4}</style>',
        f'<text x="20" y="30" class="title">{escape(title)}</text>',
        f'<text x="20" y="52" class="subtitle">四面板独立动态纵轴 · {short_window} 日观察近期变化 · {long_window} 日观察长期稳定性</text>',
    ]
    if frame.empty:
        svg.extend(
            [
                '<rect x="20" y="75" width="1160" height="645" rx="6" fill="#f8fafc" stroke="#d1d5db"/>',
                '<text x="600" y="390" text-anchor="middle" class="subtitle">暂无逐日 IC 数据</text>',
                "</svg>",
            ]
        )
        output_path.write_text("\n".join(svg), encoding="utf-8")
        return

    first_date = pd.Timestamp(frame["target_date"].iloc[0])
    last_date = pd.Timestamp(frame["target_date"].iloc[-1])
    date_span = max((last_date - first_date).total_seconds(), 1.0)

    def x_at(value: pd.Timestamp, panel_left: float) -> float:
        """把目标日期映射到指定小面板的时间轴横坐标。

        参数：
            value: 当前滚动指标对应的目标交易日。
            panel_left: 当前短期或长期趋势小面板的左边界横坐标。

        返回：
            日期在首尾目标日之间按实际时间比例映射得到的 SVG 横坐标。
        """

        return (
            panel_left
            + (value - first_date).total_seconds() / date_span * panel_width
        )

    def panel_range(column: str, fallback_metric: str) -> tuple[float, float]:
        """按单条滚动趋势计算独立动态纵轴范围。

        参数：
            column: 当前小面板的短期或长期滚动指标列名。
            fallback_metric: 滚动历史不足时用于确定占位纵轴的原始指标列名。

        返回：
            加入比例边距后的纵轴下界和上界；滚动值不足时退回原始逐日值。
        """

        rolling_values = frame[column].to_numpy(dtype=float)
        finite = rolling_values[np.isfinite(rolling_values)]
        if finite.size == 0:
            raw = frame[fallback_metric].to_numpy(dtype=float)
            finite = raw[np.isfinite(raw)]
        if finite.size == 0:
            return -0.01, 0.01
        lower = float(finite.min())
        upper = float(finite.max())
        scale = max(abs(lower), abs(upper), 0.01)
        padding = max((upper - lower) * 0.15, scale * 0.02)
        return lower - padding, upper + padding

    def y_at(value: float, panel_top: float, y_min: float, y_max: float) -> float:
        """把滚动指标值映射到当前动态纵轴面板。

        参数：
            value: 当前短期或长期滚动 IC 数值。
            panel_top: 当前指标面板顶边的 SVG 纵坐标。
            y_min: 当前面板动态纵轴下界。
            y_max: 当前面板动态纵轴上界。

        返回：
            指标值在当前面板内的 SVG 纵坐标。
        """

        clipped = min(y_max, max(y_min, value))
        return panel_top + (y_max - clipped) / (y_max - y_min) * panel_height

    def polyline_segments(
        column: str,
        panel_left: float,
        panel_top: float,
        y_min: float,
        y_max: float,
        css_class: str,
    ) -> list[str]:
        """把含缺失值的滚动序列拆成不跨越缺口的 SVG 折线。

        参数：
            column: ``frame`` 中要绘制的短期或长期滚动指标列。
            panel_left: 当前趋势小面板左边界的 SVG 横坐标。
            panel_top: 当前指标面板顶边的 SVG 纵坐标。
            y_min: 当前面板动态纵轴下界。
            y_max: 当前面板动态纵轴上界。
            css_class: 短期或长期均线的 SVG CSS 类名。

        返回：
            每个连续有限值区间对应的一段 SVG ``polyline`` 标记。
        """

        output: list[str] = []
        points: list[str] = []
        for target_date, raw_value in frame[["target_date", column]].itertuples(
            index=False,
            name=None,
        ):
            if pd.notna(raw_value) and math.isfinite(float(raw_value)):
                points.append(
                    f"{x_at(pd.Timestamp(target_date), panel_left):.2f},"
                    f"{y_at(float(raw_value), panel_top, y_min, y_max):.2f}"
                )
            elif points:
                if len(points) >= 2:
                    output.append(
                        f'<polyline class="{css_class}" points="{" ".join(points)}"/>'
                    )
                points = []
        if len(points) >= 2:
            output.append(
                f'<polyline class="{css_class}" points="{" ".join(points)}"/>'
            )
        return output

    parsed_boundary = (
        pd.to_datetime(boundary_date, errors="coerce")
        if boundary_date is not None
        else None
    )
    if parsed_boundary is not None and pd.isna(parsed_boundary):
        raise ValueError("boundary_date 必须是有效日期")
    panel_specs = (
        (panel_lefts[0], panel_tops[0], "short_ic", "ic", f"IC · {short_window} 日近期趋势", "short"),
        (panel_lefts[1], panel_tops[0], "long_ic", "ic", f"IC · {long_window} 日长期趋势", "long"),
        (panel_lefts[0], panel_tops[1], "short_rank_ic", "rank_ic", f"Rank IC · {short_window} 日近期趋势", "short"),
        (panel_lefts[1], panel_tops[1], "long_rank_ic", "rank_ic", f"Rank IC · {long_window} 日长期趋势", "long"),
    )
    for panel_left, panel_top, column, metric, label, css_class in panel_specs:
        y_min, y_max = panel_range(column, metric)
        svg.append(
            f'<text x="{panel_left}" y="{panel_top - 12}" class="panel">{label}</text>'
        )
        for tick_value in np.linspace(y_min, y_max, 5):
            tick = float(tick_value)
            y_value = y_at(tick, panel_top, y_min, y_max)
            svg.append(
                f'<line x1="{panel_left:.2f}" y1="{y_value:.2f}" x2="{panel_left + panel_width:.2f}" y2="{y_value:.2f}" class="grid"/>'
            )
            svg.append(
                f'<text x="{panel_left - 8:.2f}" y="{y_value + 4:.2f}" text-anchor="end" class="axis">{tick:.4f}</text>'
            )
        if y_min <= 0.0 <= y_max:
            zero_y = y_at(0.0, panel_top, y_min, y_max)
            svg.append(
                f'<line x1="{panel_left:.2f}" y1="{zero_y:.2f}" x2="{panel_left + panel_width:.2f}" y2="{zero_y:.2f}" class="zero"/>'
            )
        svg.extend(
            polyline_segments(
                column,
                panel_left,
                panel_top,
                y_min,
                y_max,
                css_class,
            )
        )
        if (
            parsed_boundary is not None
            and first_date <= parsed_boundary <= last_date
        ):
            boundary_x = x_at(parsed_boundary, panel_left)
            svg.append(
                f'<line x1="{boundary_x:.2f}" y1="{panel_top}" x2="{boundary_x:.2f}" y2="{panel_top + panel_height}" class="boundary"/>'
            )
        for fraction in (0.0, 0.5, 1.0):
            tick_date = first_date + (last_date - first_date) * fraction
            tick_x = panel_left + panel_width * fraction
            svg.append(
                f'<text x="{tick_x:.2f}" y="{panel_top + panel_height + 20}" text-anchor="middle" class="axis">{tick_date:%Y-%m-%d}</text>'
            )
    svg.extend(
        [
            *(
                [
                    '<line x1="1070" y1="48" x2="1105" y2="48" class="boundary"/><text x="1112" y="52" class="subtitle">holdout</text>'
                ]
                if parsed_boundary is not None
                else []
            ),
            "</svg>",
        ]
    )
    output_path.write_text("\n".join(svg), encoding="utf-8")


def render_equity_curve_svg(
    daily_returns: pd.DataFrame,
    output_path: Path,
) -> None:
    """将 Top N 及三组横截面对照的日度净值曲线渲染为上下双面板 SVG。

    上面板为全区间累计净值；下面板把同一批净值按自然年重定基：每个自然年
    以上一年末净值（首年为期初 1.0）为基准重新从 1.0 起算，因此段末净值减一
    即该自然年的累计收益，并在每个年份区段内直接标注年份与各组当年收益。

    参数：
        daily_returns: 按目标交易日升序排列且至少含日期与 Top N 累计净值的
            日度回测结果；全市场平均、Bottom N 和 Mid N 净值列可选，以兼容
            旧结果。
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
    # 逐组曲线颜色，与 <style> 中各 CSS 类的描边颜色一一对应，
    # 用于把年度收益标注渲染成与曲线同色的文字。
    series_colors = {
        "equity": "#059669",
        "universe": "#2563eb",
        "bottom": "#dc2626",
        "mid": "#d97706",
    }
    required = {"target_date", "equity"}
    missing = required.difference(daily_returns.columns)
    if missing:
        raise ValueError(f"收益曲线缺少列: {sorted(missing)}")
    if daily_returns.empty:
        raise ValueError("无法为没有日度收益的回测绘制收益曲线")
    dates = pd.to_datetime(daily_returns["target_date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("收益曲线的 target_date 包含缺失或无效日期")
    if not dates.is_monotonic_increasing:
        raise ValueError("收益曲线的 target_date 必须按目标交易日升序排列")

    width = 1000
    left, right = 82, 28
    panel_a_top, panel_a_height = 44, 330
    panel_b_top, panel_b_height = 450, 300
    height = 780
    plot_width = width - left - right
    panel_a_bottom = panel_a_top + panel_a_height
    panel_b_bottom = panel_b_top + panel_b_height
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
    if (closing_equities <= 0.0).any():
        raise ValueError("收益曲线净值必须为正数，才能按自然年重定基")
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

    # 观测序号 0 为期初，行号 j 对应观测序号 j + 1；日期升序保证同一自然年
    # 的行必然连续，这里按年份变化点切出各年的观测序号区间（含首尾）。
    years = dates.dt.year.to_numpy()
    year_starts = [0, *(int(pos) + 1 for pos in np.flatnonzero(np.diff(years) != 0))]
    year_segments = []
    for order, start_row in enumerate(year_starts):
        end_row = (
            year_starts[order + 1] - 1
            if order + 1 < len(year_starts)
            else len(years) - 1
        )
        year_segments.append((int(years[start_row]), start_row + 1, end_row + 1))
    # 每段以上一观测（上一年末或期初）为基准重定基，首元素恒为 1.0，
    # 段末元素减一即该自然年累计收益。
    yearly_rebased = {
        column: [
            values[start - 1 : end + 1] / values[start - 1]
            for _, start, end in year_segments
        ]
        for column, values in equities.items()
    }
    rebased_all = np.concatenate(
        [segment for segments in yearly_rebased.values() for segment in segments]
    )
    rebased_lower = min(1.0, float(rebased_all.min()))
    rebased_upper = max(1.0, float(rebased_all.max()))
    rebased_padding = max(
        (rebased_upper - rebased_lower) * 0.08,
        max(abs(rebased_lower), abs(rebased_upper), 1.0) * 0.01,
    )
    rebased_min = rebased_lower - rebased_padding
    rebased_max = rebased_upper + rebased_padding

    def x_at(index: int) -> float:
        """将净值观测序号映射为绘图区横坐标。

        参数：
            index: 从期初零开始的净值观测序号。

        返回：
            当前观测在 SVG 绘图区内的像素横坐标。
        """

        return left + plot_width * index / max(1, count - 1)

    def y_at(value: float, panel_top: float, span: float, floor: float, ceiling: float) -> float:
        """将净值映射为指定面板内的纵坐标。

        参数：
            value: 需要绘制的累计或按年重定基净值。
            panel_top: 当前面板绘图区顶边的 SVG 纵坐标。
            span: 当前面板绘图区的像素高度。
            floor: 当前面板纵轴下界净值。
            ceiling: 当前面板纵轴上界净值。

        返回：
            当前净值在对应面板绘图区内的像素纵坐标。
        """

        return panel_top + (ceiling - value) / (ceiling - floor) * span

    def y_at_a(value: float) -> float:
        """将累计净值映射为上面板纵坐标。

        参数：
            value: 需要绘制的全区间累计净值。

        返回：
            当前净值在上面板绘图区内的像素纵坐标。
        """

        return y_at(value, panel_a_top, panel_a_height, y_min, y_max)

    def y_at_b(value: float) -> float:
        """将按年重定基净值映射为下面板纵坐标。

        参数：
            value: 需要绘制的自然年重定基净值。

        返回：
            当前净值在下面板绘图区内的像素纵坐标。
        """

        return y_at(value, panel_b_top, panel_b_height, rebased_min, rebased_max)

    curve_points = {
        column: " ".join(
            f"{x_at(index):.2f},{y_at_a(value):.2f}"
            for index, value in enumerate(values)
        )
        for column, values in equities.items()
    }
    date_labels = [
        "期初",
        *(date.strftime("%Y-%m-%d") for date in dates),
    ]
    tick_indices = np.unique(np.linspace(0, count - 1, min(7, count), dtype=int))
    y_ticks = np.linspace(y_min, y_max, 5)
    rebased_ticks = np.linspace(rebased_min, rebased_max, 5)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Top N 与横截面对照收益曲线</title>',
        '<desc id="desc">上面板为计入相同双边成本的累计净值，下面板按自然年重定基并标注各年份与当年收益</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,"Microsoft YaHei",sans-serif;fill:#334155}.grid{stroke:#e2e8f0;stroke-width:1}.axis{stroke:#64748b;stroke-width:1.2}.baseline{stroke:#94a3b8;stroke-width:1;stroke-dasharray:5 4}.caption{font-size:13px;font-weight:600}.year-label{font-size:13px;font-weight:600}.year-return{font-size:11px}.year-boundary{stroke:#cbd5e1;stroke-width:1;stroke-dasharray:3 3}.equity,.universe,.bottom,.mid{fill:none;stroke-width:2.3}.equity{stroke:#059669}.universe{stroke:#2563eb}.bottom{stroke:#dc2626}.mid{stroke:#d97706}</style>',
        f'<text x="{left}" y="23" class="caption">全区间累计净值</text>',
    ]
    for value in y_ticks:
        y = y_at_a(float(value))
        svg.append(
            f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}"/>'
        )
        svg.append(
            f'<text x="{left - 12}" y="{y + 4:.2f}" font-size="12" text-anchor="end">{value:.3f}</text>'
        )
    svg.extend(
        [
            f'<line class="axis" x1="{left}" y1="{panel_a_top}" x2="{left}" y2="{panel_a_bottom}"/>',
            f'<line class="axis" x1="{left}" y1="{panel_a_bottom}" x2="{width - right}" y2="{panel_a_bottom}"/>',
            f'<line class="baseline" x1="{left}" y1="{y_at_a(1.0):.2f}" x2="{width - right}" y2="{y_at_a(1.0):.2f}"/>',
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
            f'<text x="{x:.2f}" y="{panel_a_bottom + 25}" font-size="12" text-anchor="middle">{label}</text>'
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

    svg.append(
        f'<text x="{left}" y="{panel_b_top - 16}" class="caption">每自然年收益曲线（每年以上一年末重定基为 1，段末净值减一即当年收益）</text>'
    )
    for value in rebased_ticks:
        y = y_at_b(float(value))
        svg.append(
            f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}"/>'
        )
        svg.append(
            f'<text x="{left - 12}" y="{y + 4:.2f}" font-size="12" text-anchor="end">{value:.3f}</text>'
        )
    svg.extend(
        [
            f'<line class="axis" x1="{left}" y1="{panel_b_top}" x2="{left}" y2="{panel_b_bottom}"/>',
            f'<line class="axis" x1="{left}" y1="{panel_b_bottom}" x2="{width - right}" y2="{panel_b_bottom}"/>',
            f'<line class="baseline" x1="{left}" y1="{y_at_b(1.0):.2f}" x2="{width - right}" y2="{y_at_b(1.0):.2f}"/>',
        ]
    )
    for _, start, _ in year_segments[1:]:
        boundary_x = x_at(start - 1)
        svg.append(
            f'<line class="year-boundary" x1="{boundary_x:.2f}" y1="{panel_b_top}" x2="{boundary_x:.2f}" y2="{panel_b_bottom}"/>'
        )
    for column, (_, css_class) in available_equity_columns.items():
        for order, (_, start, _end) in enumerate(year_segments):
            points = " ".join(
                f"{x_at(start - 1 + offset):.2f},{y_at_b(float(value)):.2f}"
                for offset, value in enumerate(yearly_rebased[column][order])
            )
            svg.append(f'<polyline class="{css_class}" points="{points}"/>')
    for order, (year, start, end) in enumerate(year_segments):
        mid_x = (x_at(start - 1) + x_at(end)) / 2.0
        svg.append(
            f'<text x="{mid_x:.2f}" y="{panel_b_top + 18}" class="year-label" text-anchor="middle">{year}</text>'
        )
        for position, (column, (label, css_class)) in enumerate(
            available_equity_columns.items()
        ):
            year_return = float(yearly_rebased[column][order][-1]) - 1.0
            svg.append(
                f'<text x="{mid_x:.2f}" y="{panel_b_top + 34 + position * 15}" class="year-return" text-anchor="middle" style="fill:{series_colors[css_class]}">{label} {year_return:+.1%}</text>'
            )
    svg.append("</svg>")
    output_path.write_text("\n".join(svg), encoding="utf-8")
