"""5 日成交量比率因子。

计算公式：当日成交量 / 最近 5 个交易日（包含当日）的平均成交量。滚动窗口
至少需要 3 个观测；比率大于 1 表示当日成交量高于近期均量。当前均量包含
当日成交量，因而不是“当日相对过去 5 日”的口径。
"""

import pandas as pd

from .base import FactorFactory
from .registry import register_factor


@register_factor
class VolumeRatio5dFactory(FactorFactory):
    """计算当日成交量相对含当日在内的 5 日滚动均量之比。"""

    name = "volume_ratio_5d"

    def compute(self, bars: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
        # 每只证券独立计算滚动均量，避免不同证券的成交量相互影响。
        mean = daily.groupby("code", sort=False)["volume"].transform(
            lambda value: value.rolling(5, min_periods=3).mean()
        )
        return daily["volume"] / mean
