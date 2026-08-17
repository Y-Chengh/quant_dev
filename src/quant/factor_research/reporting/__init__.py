"""Markdown/HTML 双格式评估报告与图表输出。

原先的单文件 ``reporting.py``（1174 行）按职责拆为：

- ``labels``：指标中文名与 IC 趋势窗口常量。
- ``formatting``：数值格式化与 Markdown 行内标记渲染。
- ``markdown``：报告正文拼装、HTML 转换与候选表格渲染。
- ``charts``：准确率趋势、IC 趋势与资金曲线的 SVG 渲染。
- ``drawdown_chart``：Top N 净值历史回撤与修复过程的 SVG 渲染。
- ``slippage_chart``：不同滑点假设下 Top N 净值对比曲线的 SVG 渲染。
- ``report``：报告落盘编排。

本模块保持与拆分前完全一致的公开接口，调用方无需改动导入语句。
"""

from __future__ import annotations

from .charts import (
    render_accuracy_trend_svg,
    render_equity_curve_svg,
    render_ic_trend_svg,
)
from .drawdown_chart import render_drawdown_curve_svg
from .labels import (
    IC_TREND_LONG_MIN_PERIODS,
    IC_TREND_LONG_WINDOW,
    IC_TREND_SHORT_MIN_PERIODS,
    IC_TREND_SHORT_WINDOW,
    METRIC_LABELS,
)
from .markdown import render_markdown_report_html
from .report import write_evaluation_report
from .slippage_chart import render_slippage_curves_svg

__all__ = [
    "IC_TREND_LONG_MIN_PERIODS",
    "IC_TREND_LONG_WINDOW",
    "IC_TREND_SHORT_MIN_PERIODS",
    "IC_TREND_SHORT_WINDOW",
    "METRIC_LABELS",
    "render_accuracy_trend_svg",
    "render_drawdown_curve_svg",
    "render_equity_curve_svg",
    "render_ic_trend_svg",
    "render_markdown_report_html",
    "render_slippage_curves_svg",
    "write_evaluation_report",
]
