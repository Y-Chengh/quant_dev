"""5 日历史波动率因子。

先按证券计算相邻交易日收盘价简单收益率，再取最近 5 个日收益率的滚动
样本标准差（pandas 默认 ddof=1）。窗口至少需要 3 个有效收益率，因此每只
证券通常从第 4 个日频观测起产生数值；结果未进行年化。
"""

import pandas as pd

from .base import FactorFactory
from .registry import register_factor


@register_factor
class Volatility5dFactory(FactorFactory):
    """计算最近 5 个交易日日收益率的滚动样本标准差。"""

    name = "volatility_5d"

    def compute(self, bars: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
        # pct_change 在每只证券内计算，避免跨证券串接价格。
        returns = daily.groupby("code", sort=False)["close"].pct_change()
        # transform 将滚动结果保持在 daily 的原始行粒度和索引上。
        return returns.groupby(daily["code"], sort=False).transform(
            lambda value: value.rolling(5, min_periods=3).std()
        )
