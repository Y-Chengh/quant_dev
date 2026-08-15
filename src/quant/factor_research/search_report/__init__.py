"""因子搜索结果的可审计报告生成。

这些逻辑原先内联在命令行入口 ``quant.cli.grid_search``（1026 行）里，
违反“CLI 只做参数解析与装配、不承载可复用业务逻辑”的分层约束，现下沉到库层：

- ``constants``：滚动 IC 稳定性图的窗口常量。
- ``formatting``：数值、单元格与 Markdown 表格格式化。
- ``reproduction``：由候选表达式生成主实验复现命令。
- ``summaries``：搜索区间摘要与候选逐日 IC 明细。
- ``charts``：目标值排名图与滚动 IC 稳定性图。
- ``report``：完整报告落盘编排。
"""

from __future__ import annotations

from .constants import ROLLING_IC_MIN_PERIODS, ROLLING_IC_WINDOW
from .report import write_grid_search_report

__all__ = [
    "ROLLING_IC_MIN_PERIODS",
    "ROLLING_IC_WINDOW",
    "write_grid_search_report",
]
