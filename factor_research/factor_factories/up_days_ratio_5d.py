"""最近 5 日上涨天数占比因子。

先按证券计算日收盘收益率，再统计最近 5 个日收益率中严格大于 0 的比例。
平盘计为非上涨；必须有完整的 5 个日收益率观测，因而至少需要 6 日价格。
"""

import numpy as np
import pandas as pd

from .base import FactorFactory
from .registry import register_factor


@register_factor
class UpDaysRatio5dFactory(FactorFactory):
    """计算最近 5 个交易日收益率为正的频率。"""

    name = "up_days_ratio_5d"

    def compute(self, bars: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
        returns = daily.groupby("code", sort=False)["close"].pct_change()
        return returns.groupby(daily["code"], sort=False).transform(
            lambda values: values.rolling(5, min_periods=5).apply(
                lambda window: np.mean(window > 0), raw=True
            )
        )
