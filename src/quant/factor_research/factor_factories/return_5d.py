"""5 日收盘收益率因子。

计算公式：当日收盘价 / 同一证券 5 个交易日前的收盘价 - 1。这里的窗口按
数据中的交易日行数计数，而非自然日；前 5 个观测因没有基准价而为缺失值。
"""

import pandas as pd

from .base import FactorFactory
from .registry import register_factor


@register_factor
class Return5dFactory(FactorFactory):
    """计算当前收盘价相对 5 个交易日前收盘价的累计简单收益率。"""

    name = "return_5d"

    def compute(self, bars: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
        return daily["close"] / daily.groupby("code", sort=False)["close"].shift(5) - 1
