"""日振幅因子。

计算公式：当日振幅 =（当日最高价 - 当日最低价）/ 前一交易日收盘价。
首个交易日没有前收盘价，因此结果自然为缺失值。
"""

import pandas as pd

from .base import FactorFactory, previous_close
from .registry import register_factor


@register_factor
class AmplitudeFactory(FactorFactory):
    """计算以前收盘价归一化的日内价格波动区间。"""

    name = "amplitude"

    def compute(self, bars: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
        # 先取同一证券的前一日收盘价，再用它消除绝对价格水平的影响。
        return (daily["high"] - daily["low"]) / previous_close(daily)
