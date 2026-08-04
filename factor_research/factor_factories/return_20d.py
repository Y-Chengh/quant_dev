"""20 日收盘收益率因子。

计算公式：当日收盘价 / 同一证券 20 个交易日前的收盘价 - 1。窗口按交易日
观测数而非自然日计数；每只证券的前 20 个观测返回缺失值。
"""

import pandas as pd

from .base import FactorFactory
from .registry import register_factor


@register_factor
class Return20dFactory(FactorFactory):
    """计算当前收盘价相对 20 个交易日前的累计简单收益率。"""

    name = "return_20d"

    def compute(self, bars: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
        previous = daily.groupby("code", sort=False)["close"].shift(20)
        return daily["close"] / previous - 1
