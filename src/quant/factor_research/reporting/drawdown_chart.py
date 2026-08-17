"""Top N 净值历史回撤与修复过程的 SVG 渲染。

与 charts 分开成模块，避免图表模块超过 AGENTS.md 约定的 700 行拆分线；
本模块只依赖标准库、numpy 与 pandas，不反向依赖报告正文或回测层。
"""

from __future__ import annotations

from collections.abc import Callable
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd


def render_drawdown_curve_svg(
    daily_returns: pd.DataFrame,
    output_path: Path,
) -> None:
    """将 Top N 净值的历史峰值包络与水下回撤修复过程渲染为独立 SVG。

    上面板叠加累计净值与历史峰值，并把两者之间的水下区间填充出来；下面板画出
    逐日回撤深度，标注最大回撤谷底及其修复位置。回撤统一按期初净值 1.0 起算，
    取值不大于 0，等于 0 表示当日创出新高。存在全市场平均净值列时，下面板额外
    叠加其回撤折线作为横截面对照。图像只用于结果诊断，不参与训练或选股。

    历史峰值一律由净值就地累计；回测已经算好 ``drawdown`` 时直接复用该列，保证
    图与报告里的回撤指标严格同源，旧结果缺列时按同一口径重算，两条路径不会给出
    不同的最大回撤。``peak_equity``/``drawdown`` 列存在时都会与净值核对，口径
    不一致直接抛 ``ValueError``，避免画出与指标表矛盾的图；因此传入按区间切片后
    的日度结果时应一并去掉这两列，否则全局峰值与切片重算结果必然不一致。

    参数：
        daily_returns: 按目标交易日升序排列且至少含 ``target_date`` 与 ``equity``
            的日度回测结果；``peak_equity``、``drawdown``、``universe_equity``、
            ``universe_drawdown`` 均为可选列：回撤列缺省时就地重算，峰值列只参与
            一致性校验，全市场净值列缺省时不画横截面对照线。
        output_path: SVG 回撤图的写入路径，父目录必须已经存在。

    返回：
        无；函数将可独立打开的 UTF-8 SVG 写入 ``output_path``。
    """

    required = {"target_date", "equity"}
    missing = required.difference(daily_returns.columns)
    if missing:
        raise ValueError(f"回撤曲线缺少列: {sorted(missing)}")
    if daily_returns.empty:
        raise ValueError("无法为没有日度收益的回测绘制回撤曲线")
    dates = pd.to_datetime(daily_returns["target_date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("回撤曲线的 target_date 包含缺失或无效日期")
    # 历史峰值按位置累计，日期乱序会算出错误的回撤与修复位置，必须显式拒绝。
    if not dates.is_monotonic_increasing:
        raise ValueError("回撤曲线的 target_date 必须按目标交易日升序排列")

    def underwater(
        values: np.ndarray,
        peak_column: str,
        drawdown_column: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """取回测算好的回撤列，缺列时按同一口径就地重算。

        历史峰值一律由净值就地累计得到，外部峰值列只参与一致性校验，避免被
        篡改或口径不同的列直接决定图上的回撤深度。

        参数：
            values: 按交易日升序排列的累计净值序列，不含期初净值。
            peak_column: 该组净值对应的历史峰值列名；存在时校验其与净值同源。
            drawdown_column: 该组净值对应的水下回撤列名；不存在时就地重算。

        返回：
            含期初净值在内的历史峰值数组和回撤数组，长度均比输入多一；期初
            位置的峰值为 1.0、回撤为 0。
        """

        if not np.isfinite(values).all():
            raise ValueError("回撤曲线净值包含 NaN 或无穷值")
        extended = np.concatenate(([1.0], values))
        # 历史峰值是净值的纯函数，一律就地计算；外部列只用来核对口径是否同源。
        peaks = np.maximum.accumulate(extended)
        if peak_column in daily_returns.columns:
            stored_peaks = np.concatenate(
                ([1.0], daily_returns[peak_column].to_numpy(dtype=float))
            )
            if not np.isfinite(stored_peaks).all():
                raise ValueError(f"{peak_column} 包含 NaN 或无穷值")
            if not np.allclose(stored_peaks, peaks, rtol=1e-9, atol=1e-12):
                raise ValueError(f"{peak_column} 与净值不一致，无法绘制回撤曲线")
        if drawdown_column not in daily_returns.columns:
            return peaks, extended / peaks - 1.0
        drawdowns = np.concatenate(
            ([0.0], daily_returns[drawdown_column].to_numpy(dtype=float))
        )
        if not np.isfinite(drawdowns).all():
            raise ValueError(f"{drawdown_column} 包含 NaN 或无穷值")
        # 复用回测算好的回撤时校验其与净值同源，避免图与指标表口径静默分叉。
        if not np.allclose(
            drawdowns, extended / peaks - 1.0, rtol=1e-9, atol=1e-12
        ):
            raise ValueError(f"{drawdown_column} 与净值不一致，无法绘制回撤曲线")
        return peaks, drawdowns

    equities = np.concatenate(
        ([1.0], daily_returns["equity"].to_numpy(dtype=float))
    )
    peaks, drawdowns = underwater(
        daily_returns["equity"].to_numpy(dtype=float), "peak_equity", "drawdown"
    )
    universe_drawdowns: np.ndarray | None = None
    if "universe_equity" in daily_returns.columns:
        _, universe_drawdowns = underwater(
            daily_returns["universe_equity"].to_numpy(dtype=float),
            "universe_peak_equity",
            "universe_drawdown",
        )

    width, height = 1000, 660
    left, right = 86, 30
    plot_width = width - left - right
    equity_top, equity_height = 62, 200
    drawdown_top, drawdown_height = 336, 190
    count = len(equities)

    def x_at(index: int) -> float:
        """把净值观测序号映射为绘图区横坐标。

        参数：
            index: 从期初零开始的净值观测序号。

        返回：
            当前观测在两个面板共用时间轴上的像素横坐标。
        """

        return left + plot_width * index / max(1, count - 1)

    equity_lower = float(equities.min())
    equity_upper = float(equities.max())
    equity_padding = max(
        (equity_upper - equity_lower) * 0.08,
        max(abs(equity_lower), abs(equity_upper), 1.0) * 0.01,
    )
    equity_min = equity_lower - equity_padding
    equity_max = equity_upper + equity_padding

    def y_equity(value: float) -> float:
        """把累计净值映射为上面板纵坐标。

        参数：
            value: 需要绘制的累计净值或同期历史峰值。

        返回：
            当前净值在上面板内的像素纵坐标。
        """

        return equity_top + (equity_max - value) / (equity_max - equity_min) * equity_height

    all_drawdowns = (
        drawdowns
        if universe_drawdowns is None
        else np.concatenate((drawdowns, universe_drawdowns))
    )
    drawdown_lower = float(all_drawdowns.min())
    drawdown_min = min(-0.01, drawdown_lower - max(abs(drawdown_lower) * 0.12, 0.002))

    def y_drawdown(value: float) -> float:
        """把回撤深度映射为下面板纵坐标。

        参数：
            value: 需要绘制的回撤值，取值不大于 0。

        返回：
            当前回撤在下面板内的像素纵坐标；零回撤位于面板顶边。
        """

        clipped = min(0.0, max(drawdown_min, value))
        return drawdown_top + (0.0 - clipped) / (0.0 - drawdown_min) * drawdown_height

    def polyline_points(
        values: np.ndarray,
        mapper: Callable[[float], float],
    ) -> str:
        """把等长序列拼成 SVG 折线的点串。

        参数：
            values: 与时间轴一一对应的净值或回撤序列。
            mapper: 把序列取值换算为像素纵坐标的函数。

        返回：
            以空格分隔的 SVG ``points`` 坐标串。
        """

        return " ".join(
            f"{x_at(index):.2f},{mapper(float(value)):.2f}"
            for index, value in enumerate(values)
        )

    date_labels = ["期初", *(date.strftime("%Y-%m-%d") for date in dates)]
    tick_indices = np.unique(np.linspace(0, count - 1, min(7, count), dtype=int))
    trough_index = int(np.argmin(drawdowns))
    max_drawdown = float(drawdowns[trough_index])
    recovery_index: int | None = None
    if max_drawdown < 0.0:
        after_trough = np.nonzero(drawdowns[trough_index + 1 :] >= 0.0)[0]
        if after_trough.size:
            recovery_index = trough_index + 1 + int(after_trough[0])

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Top N 历史回撤与修复</title>',
        '<desc id="desc">上面板为累计净值与历史峰值，下面板为逐日水下回撤及最大回撤修复位置</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,"Microsoft YaHei",sans-serif;fill:#334155}.title{font-size:18px;font-weight:600}.subtitle{font-size:12px;fill:#64748b}.panel{font-size:14px;font-weight:600}.grid{stroke:#e2e8f0;stroke-width:1}.axis{stroke:#64748b;stroke-width:1.2}.equity{fill:none;stroke:#059669;stroke-width:2.3}.peak{fill:none;stroke:#94a3b8;stroke-width:1.6;stroke-dasharray:5 4}.underwater{fill:#fecaca;fill-opacity:.55;stroke:none}.drawdown{fill:#ef4444;fill-opacity:.28;stroke:#dc2626;stroke-width:2}.universe{fill:none;stroke:#2563eb;stroke-width:1.8}.recovery{stroke:#0f766e;stroke-width:1.5;stroke-dasharray:6 4}.marker{fill:#b91c1c}</style>',
        '<text x="20" y="30" class="title">Top N 历史回撤与修复</text>',
        '<text x="20" y="48" class="subtitle">上：累计净值与历史峰值，阴影为水下区间 · 下：逐日回撤深度与最大回撤修复位置 · 均已扣除双边成本</text>',
    ]

    for value in np.linspace(equity_min, equity_max, 5):
        y = y_equity(float(value))
        svg.append(
            f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}"/>'
        )
        svg.append(
            f'<text x="{left - 12}" y="{y + 4:.2f}" font-size="12" text-anchor="end">{float(value):.3f}</text>'
        )
    underwater_polygon = " ".join(
        [
            polyline_points(peaks, y_equity),
            *(
                f"{x_at(index):.2f},{y_equity(float(equities[index])):.2f}"
                for index in range(count - 1, -1, -1)
            ),
        ]
    )
    svg.extend(
        [
            f'<polygon class="underwater" points="{underwater_polygon}"/>',
            f'<polyline class="peak" points="{polyline_points(peaks, y_equity)}"/>',
            f'<polyline class="equity" points="{polyline_points(equities, y_equity)}"/>',
            f'<line class="axis" x1="{left}" y1="{equity_top}" x2="{left}" y2="{equity_top + equity_height}"/>',
            f'<line class="axis" x1="{left}" y1="{equity_top + equity_height}" x2="{width - right}" y2="{equity_top + equity_height}"/>',
            f'<text x="{left}" y="{equity_top - 12}" class="panel">累计净值与历史峰值</text>',
            f'<line class="equity" x1="{width - 300}" y1="{equity_top - 16}" x2="{width - 268}" y2="{equity_top - 16}"/>',
            f'<text x="{width - 260}" y="{equity_top - 12}" font-size="12">累计净值</text>',
            f'<line class="peak" x1="{width - 180}" y1="{equity_top - 16}" x2="{width - 148}" y2="{equity_top - 16}"/>',
            f'<text x="{width - 140}" y="{equity_top - 12}" font-size="12">历史峰值</text>',
        ]
    )

    for value in np.linspace(drawdown_min, 0.0, 5):
        y = y_drawdown(float(value))
        svg.append(
            f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}"/>'
        )
        svg.append(
            f'<text x="{left - 12}" y="{y + 4:.2f}" font-size="12" text-anchor="end">{float(value):.2%}</text>'
        )
    drawdown_polygon = " ".join(
        [
            f"{x_at(0):.2f},{y_drawdown(0.0):.2f}",
            polyline_points(drawdowns, y_drawdown),
            f"{x_at(count - 1):.2f},{y_drawdown(0.0):.2f}",
        ]
    )
    svg.append(f'<polygon class="drawdown" points="{drawdown_polygon}"/>')
    if universe_drawdowns is not None:
        svg.append(
            f'<polyline class="universe" points="{polyline_points(universe_drawdowns, y_drawdown)}"/>'
        )
    svg.extend(
        [
            f'<line class="axis" x1="{left}" y1="{drawdown_top}" x2="{left}" y2="{drawdown_top + drawdown_height}"/>',
            f'<line class="axis" x1="{left}" y1="{drawdown_top + drawdown_height}" x2="{width - right}" y2="{drawdown_top + drawdown_height}"/>',
            f'<text x="{left}" y="{drawdown_top - 12}" class="panel">水下回撤与修复</text>',
        ]
    )
    if universe_drawdowns is not None:
        svg.extend(
            [
                f'<line class="universe" x1="{width - 210}" y1="{drawdown_top - 16}" x2="{width - 178}" y2="{drawdown_top - 16}"/>',
                f'<text x="{width - 170}" y="{drawdown_top - 12}" font-size="12">全市场平均回撤</text>',
            ]
        )
    if max_drawdown < 0.0:
        trough_x = x_at(trough_index)
        trough_y = y_drawdown(max_drawdown)
        anchor = "end" if trough_x > left + plot_width * 0.6 else "start"
        offset = -8 if anchor == "end" else 8
        recovery_text = (
            f"{date_labels[recovery_index]} 修复，用时 {recovery_index - trough_index} 个交易日"
            if recovery_index is not None
            else "截至验证区间末尾尚未修复"
        )
        svg.extend(
            [
                f'<circle class="marker" cx="{trough_x:.2f}" cy="{trough_y:.2f}" r="4"/>',
                f'<text x="{trough_x + offset:.2f}" y="{trough_y + 18:.2f}" font-size="12" text-anchor="{anchor}">'
                f"最大回撤 {max_drawdown:.2%} · {escape(date_labels[trough_index])}</text>",
                f'<text x="{trough_x + offset:.2f}" y="{trough_y + 34:.2f}" font-size="12" text-anchor="{anchor}">'
                f"{escape(recovery_text)}</text>",
            ]
        )
        if recovery_index is not None:
            recovery_x = x_at(recovery_index)
            svg.append(
                f'<line class="recovery" x1="{recovery_x:.2f}" y1="{drawdown_top}" x2="{recovery_x:.2f}" y2="{drawdown_top + drawdown_height}"/>'
            )
            svg.append(
                f'<line class="recovery" x1="{recovery_x:.2f}" y1="{equity_top}" x2="{recovery_x:.2f}" y2="{equity_top + equity_height}"/>'
            )
    for index in tick_indices:
        x = x_at(int(index))
        label = escape(date_labels[int(index)])
        svg.append(
            f'<text x="{x:.2f}" y="{drawdown_top + drawdown_height + 25}" font-size="12" text-anchor="middle">{label}</text>'
        )
    svg.append("</svg>")
    output_path.write_text("\n".join(svg), encoding="utf-8")
