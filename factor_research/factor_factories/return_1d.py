"""1 日收盘收益率因子。

计算公式：当日收盘价 / 前一交易日收盘价 - 1。按证券代码独立后移收盘价，
每只证券的首个交易日因没有前值而返回缺失值。
"""

import pandas as pd

from .base import FactorFactory, previous_close
from .registry import register_factor


@register_factor
class Return1dFactory(FactorFactory):
    """计算相邻两个交易日之间的收盘价简单收益率。"""

    name = "return_1d"

    def compute(self, bars: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
        return daily["close"] / previous_close(daily) - 1
