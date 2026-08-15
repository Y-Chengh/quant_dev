"""日内上涨分钟线占比因子。

先计算相邻分钟收盘价的简单收益率，再统计收益率严格大于 0 的区间占全部
有效相邻区间的比例。平盘区间计为非上涨；不足 2 根分钟线时返回缺失值。
"""

import numpy as np
import pandas as pd

from .base import FactorFactory, close_returns, intraday_values
from .registry import register_factor


@register_factor
class PositiveBarRatioFactory(FactorFactory):
    """计算日内相邻分钟收盘价上涨的频率。"""

    name = "positive_bar_ratio"

    def compute(self, bars: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
        return intraday_values(
            bars,
            daily,
            # close_returns 的首项为 NaN，切片 [1:] 后只保留可比较区间。
            lambda group: float(np.mean(close_returns(group)[1:] > 0)) if len(group) > 1 else np.nan,
        )
