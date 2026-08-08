"""因子搜索与现有分钟聚合、正式因子工厂之间的薄适配层。"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from factor_research.factors import aggregate_daily_bars, build_daily_features

from .context import SearchContext


def prepare_search_context(
    bars: pd.DataFrame,
    *,
    fixed_features: Sequence[str] = (),
    cache_dir: str | Path | None = None,
    selection_start: str | pd.Timestamp | None = None,
    holdout_start: str | pd.Timestamp | None = None,
    holdout_end: str | pd.Timestamp | None = None,
) -> SearchContext:
    """从分钟行情准备一次性搜索上下文。

    有固定因子时只调用一次现有 ``build_daily_features``；没有固定因子时只聚合
    OHLCV。候选 worker 只接收返回的上下文，因此不会再次访问因子工厂。
    """

    fixed = tuple(fixed_features)
    daily = (
        build_daily_features(bars, feature_columns=fixed, cache_dir=cache_dir)
        if fixed
        else aggregate_daily_bars(bars)
    )
    return SearchContext.from_daily(
        daily,
        fixed_features=fixed,
        selection_start=selection_start,
        holdout_start=holdout_start,
        holdout_end=holdout_end,
    )
