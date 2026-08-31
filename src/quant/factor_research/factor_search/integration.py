"""因子搜索与现有分钟聚合、正式因子工厂之间的薄适配层。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from quant.factor_research.dataset import DEFAULT_LABEL_RETURN_THRESHOLD
from quant.factor_research.factors import aggregate_daily_bars, build_daily_features

from .context import SearchContext


def prepare_search_context(
    bars: pd.DataFrame,
    *,
    fixed_features: Sequence[str] = (),
    cache_dir: str | Path | None = None,
    selection_start: str | pd.Timestamp | None = None,
    holdout_start: str | pd.Timestamp | None = None,
    holdout_end: str | pd.Timestamp | None = None,
    label_return_threshold: float = DEFAULT_LABEL_RETURN_THRESHOLD,
) -> SearchContext:
    """从分钟行情准备一次性搜索上下文。

    有固定因子时只调用一次现有 ``build_daily_features``；没有固定因子时只聚合
    OHLCV。候选 worker 只接收返回的上下文，因此不会再次访问因子工厂。

    参数：
        bars: 已标准化的分钟行情表。
        fixed_features: 在搜索中与候选因子共同使用的正式因子名；缺省为空。
        cache_dir: 固定因子缓存根目录；为空时不持久化缓存。
        selection_start: 候选筛选区间的首个目标日期；为空时不限制起点。
        holdout_start: 样本外报告区间的首个目标日期；为空时不划分 holdout。
        holdout_end: 样本外报告区间的最后一个目标日期，包含该日；为空时不限制结束日。
        label_return_threshold: 二分类正类的最低目标收益率，单位为一；严格大于
            该值时标签为 1，缺省 ``0.005`` 表示 0.5%。
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
        label_return_threshold=label_return_threshold,
    )
