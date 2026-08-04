from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .experiment import ExperimentResult


METRIC_LABELS = {
    "samples": "样本数",
    "positive_rate": "实际上涨比例",
    "accuracy": "准确率",
    "balanced_accuracy": "平衡准确率",
    "auc": "ROC AUC",
    "brier_score": "Brier 分数",
    "log_loss": "Log Loss",
}


def _format_number(value: Any) -> str:
    number = float(value)
    return "N/A" if not np.isfinite(number) else f"{number:.6f}"


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


def write_evaluation_report(
    result: ExperimentResult,
    report_path: Path,
    chart_path: Path,
    run_id: str,
    run_arguments: dict[str, Any],
) -> None:
    """Write a Markdown evaluation summary and its linked accuracy chart."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    render_accuracy_trend_svg(result.daily_accuracy_trend, chart_path)

    predictions = result.predictions.copy()
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
        "# 方向预测评估报告",
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

    lines.extend(
        [
            "",
            "## 因子重要性",
            "",
            "| 因子 | 重要性 |",
            "| --- | ---: |",
        ]
    )
    for factor, importance in result.feature_importance.items():
        lines.append(f"| `{factor}` | {float(importance):.6f} |")

    lines.extend(["", "## 运行参数", "", "```text"])
    lines.extend(f"{key}={value}" for key, value in sorted(run_arguments.items()))
    lines.extend(["```", ""])
    report_path.write_text("\n".join(lines), encoding="utf-8")
