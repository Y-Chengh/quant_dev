"""相对前 20 日高低区间的位置因子。

计算公式为 ``(当日收盘价 - 前20日最低价) / (前20日最高价 - 前20日最低价)``。
最高价和最低价窗口均先后移一个交易日，不包含当日；按证券独立计算，并要求
完整 20 日历史。历史区间宽度为 0 或历史不足时返回缺失值。突破历史区间时
结果可以小于 0 或大于 1，未年化或额外归一化。
"""

import numpy as np
import pandas as pd

from .base import FactorFactory
from .registry import register_factor


@register_factor
class ChannelPosition20dFactory(FactorFactory):
    """计算收盘价在当日以前 20 日最高价和最低价之间的位置。"""

    name = "channel_position_20d"

    def compute(self, bars: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
        previous_high = daily.groupby("code", sort=False)["high"].transform(
            lambda values: values.shift(1).rolling(20, min_periods=20).max()
        )
        previous_low = daily.groupby("code", sort=False)["low"].transform(
            lambda values: values.shift(1).rolling(20, min_periods=20).min()
        )
        width = previous_high - previous_low
        return pd.Series(
            np.where(width > 0, (daily["close"] - previous_low) / width, np.nan),
            index=daily.index,
            dtype=float,
        )
