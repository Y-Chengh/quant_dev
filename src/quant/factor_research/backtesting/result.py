"""Top N 目标收益策略回测结果的数据类。"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class TopNBacktestResult:
    """保存 Top N 目标收益策略及其横截面对照诊断。"""

    top_n: int
    score_column: str
    slippage_bps: float
    commission_bps: float
    daily_returns: pd.DataFrame
    metrics: dict[str, float]
    benchmark_metrics: dict[str, float] = field(default_factory=dict)
    relative_metrics: dict[str, float] = field(default_factory=dict)
    selection_metrics: dict[str, float] = field(default_factory=dict)
    decile_returns: pd.DataFrame = field(default_factory=pd.DataFrame)
    top_selections: pd.DataFrame = field(default_factory=pd.DataFrame)
    drawdown_metrics: dict[str, float] = field(default_factory=dict)
    drawdown_episodes: pd.DataFrame = field(default_factory=pd.DataFrame)
    slippage_curves: pd.DataFrame = field(default_factory=pd.DataFrame)
    slippage_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
