"""搜索目标值排名图与滚动 IC 稳定性图的 SVG 渲染。"""

from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from ..reporting import (
    IC_TREND_SHORT_MIN_PERIODS,
    IC_TREND_SHORT_WINDOW,
    render_ic_trend_svg,
)
from .constants import ROLLING_IC_MIN_PERIODS, ROLLING_IC_WINDOW


def _write_objective_chart(
    leaderboard: pd.DataFrame,
    path: Path,
    objective: str,
    top_n: int = 20,
) -> None:
    """用纯 SVG 绘制前若干合格候选的 selection 排名指标条形图。

    参数：
        leaderboard: 已按 selection 目标降序排列的完整候选榜单。
        path: SVG 输出路径，父目录必须已存在。
        objective: 排名使用的 selection 指标列名。
        top_n: 图中最多展示的合格候选数量，缺省为 20。

    返回：
        无；函数把 UTF-8 SVG 文件写入 ``path``。
    """

    chart = leaderboard.loc[leaderboard["eligible"], ["factor_id", objective]].head(top_n)
    chart = chart.loc[np.isfinite(pd.to_numeric(chart[objective], errors="coerce"))]
    width = 920
    row_height = 28
    height = max(150, 95 + row_height * len(chart))
    label_width = 180
    plot_width = 650
    maximum = float(chart[objective].max()) if not chart.empty else 1.0
    maximum = maximum if maximum > 0 else 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Segoe UI,Microsoft YaHei,sans-serif;fill:#202124}.label{font-size:12px}.value{font-size:11px}.title{font-size:18px;font-weight:600}</style>',
        f'<text x="20" y="30" class="title">Top {len(chart)} selection candidates — {escape(objective)}</text>',
    ]
    for row_index, (_, row) in enumerate(chart.iterrows()):
        y = 60 + row_index * row_height
        value = float(row[objective])
        bar_width = max(1.0, value / maximum * plot_width)
        parts.extend(
            [
                f'<text x="20" y="{y + 14}" class="label">{escape(str(row["factor_id"]))}</text>',
                f'<rect x="{label_width}" y="{y}" width="{bar_width:.2f}" height="18" rx="3" fill="#3b82f6"/>',
                f'<text x="{label_width + bar_width + 8:.2f}" y="{y + 14}" class="value">{value:.6f}</text>',
            ]
        )
    if chart.empty:
        parts.append('<text x="20" y="75" class="label">No eligible finite candidates</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _write_rolling_ic_stability_chart(
    daily_ic: pd.DataFrame,
    path: Path,
    factor_id: str | None,
    *,
    window: int = ROLLING_IC_WINDOW,
    min_periods: int = ROLLING_IC_MIN_PERIODS,
) -> None:
    """调用共享渲染器绘制单个候选的双周期 IC/Rank IC 稳定性图。

    短期趋势固定使用项目通用的 20 日窗口，长期趋势由 ``window`` 指定。传入的
    逐日指标已经按 selection 锁定方向调整；holdout 起点只作为图中分界线，不参与
    方向确定或候选重排。

    参数：
        daily_ic: Top K 候选的逐日横截面指标，必须包含候选 ID、区间、目标日期、
            IC 和 Rank IC 列。
        path: SVG 输出路径，父目录必须已经存在。
        factor_id: 要绘图的候选 ID；为 ``None`` 或找不到对应明细时输出无数据占位图。
        window: 长期滚动均值包含的目标交易日行数，缺省为 60 日。
        min_periods: 长期均线出值所需的最少有限观测数，缺省为 20 日。

    返回：
        无；函数通过共享报告渲染器把 UTF-8 SVG 图像写入 ``path``。
    """

    required = {"factor_id", "period", "target_date", "ic", "rank_ic"}
    missing = required.difference(daily_ic.columns)
    if missing:
        raise ValueError(f"daily_ic 缺少列: {sorted(missing)}")
    selected = daily_ic.iloc[0:0].copy()
    if factor_id is not None:
        selected = daily_ic.loc[daily_ic["factor_id"].astype(str).eq(str(factor_id))].copy()
    holdout_dates = selected.loc[selected["period"].eq("holdout"), "target_date"]
    render_ic_trend_svg(
        selected,
        path,
        title=f"Rolling IC stability — {factor_id or 'no candidate'}",
        short_window=IC_TREND_SHORT_WINDOW,
        short_min_periods=IC_TREND_SHORT_MIN_PERIODS,
        long_window=window,
        long_min_periods=min_periods,
        boundary_date=(
            pd.to_datetime(holdout_dates, errors="coerce").min()
            if not holdout_dates.empty
            else None
        ),
    )
