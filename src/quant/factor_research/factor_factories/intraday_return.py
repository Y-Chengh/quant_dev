"""日内收益率因子。

计算公式：当日收盘价 / 当日开盘价 - 1，衡量从开盘到收盘的价格涨跌幅。
"""

import pandas as pd

from .base import FactorFactory
from .registry import register_factor


@register_factor
class IntradayReturnFactory(FactorFactory):
    """计算单个交易日由开盘至收盘的简单收益率。"""

    name = "intraday_return"

    def compute(self, bars: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
        return daily["close"] / daily["open"] - 1
