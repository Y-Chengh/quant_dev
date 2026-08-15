"""搜索报告图表使用的滚动 IC 稳定性窗口常量。"""

from __future__ import annotations

#: 滚动 IC 稳定性图的窗口长度，单位为交易日。
ROLLING_IC_WINDOW = 60

#: 滚动 IC 稳定性图允许的最小有效观测数，不足则该点留空。
ROLLING_IC_MIN_PERIODS = 20
