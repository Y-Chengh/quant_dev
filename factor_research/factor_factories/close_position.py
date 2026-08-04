"""收盘价在当日价格区间中的相对位置因子。

计算公式：(收盘价 - 最低价) / (最高价 - 最低价)。结果通常位于 [0, 1]：
越接近 1 表示收盘越靠近当日最高价，越接近 0 表示越靠近最低价。
"""

import numpy as np
import pandas as pd

from .base import FactorFactory
from .registry import register_factor


@register_factor
class ClosePositionFactory(FactorFactory):
    """计算收盘价在当日高低价区间内的位置。"""

    name = "close_position"

    def compute(self, bars: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
        spread = daily["high"] - daily["low"]
        # 一字价等高低价相同的交易日没有可定义的区间位置，以中点 0.5 代替。
        return pd.Series(
            np.where(spread > 0, (daily["close"] - daily["low"]) / spread, 0.5),
            index=daily.index,
        )
