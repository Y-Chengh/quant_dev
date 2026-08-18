"""Top N 目标收益策略回测、横截面对照与成本敏感性诊断。

原先的单文件 ``backtesting.py`` 在加入滑点敏感性对比后达到 811 行，越过 700 行
拆分线，按职责拆为：

- ``metrics``：交易日年化常量与复利、价差两组汇总指标。
- ``drawdown``：净值回撤曲线、历次回撤区间切分与回撤诊断指标。
- ``costs``：买卖两端成交乘数与不同滑点下的净值敏感性对比。
- ``result``：回测结果数据类。
- ``top_n``：Top N 目标收益等权策略回测主流程。

本模块保持与拆分前完全一致的公开接口，调用方无需改动导入语句；包内私有实现
（``_compound_metrics``、``_drawdown_curve`` 等）由使用方从对应子模块直接导入，
不在包入口重导出。
"""

from __future__ import annotations

from .metrics import TRADING_DAYS_PER_YEAR
from .result import TopNBacktestResult
from .top_n import (
    DEFAULT_RANDOM_BASELINE_SEED,
    DEFAULT_RANDOM_BASELINE_SIMULATIONS,
    run_top_n_intraday_backtest,
)

__all__ = [
    "DEFAULT_RANDOM_BASELINE_SEED",
    "DEFAULT_RANDOM_BASELINE_SIMULATIONS",
    "TRADING_DAYS_PER_YEAR",
    "TopNBacktestResult",
    "run_top_n_intraday_backtest",
]
