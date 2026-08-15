"""价格相对 5 日均线距离的单日变化因子。

先计算“收盘价 / 5 日均线 - 1”，再减去同一证券上一交易日的距离。正值表示
价格相对均线向上运动，负值表示向下运动，可连续表达上下穿越的方向和力度。
"""

import pandas as pd

from .base import FactorFactory, rolling_mean_by_code
from .registry import register_factor


@register_factor
class MaDistanceChange5dFactory(FactorFactory):
    """计算价格与 5 日均线相对距离的日变化。"""

    name = "ma_distance_change_5d"

    def compute(self, bars: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
        ma_5d = rolling_mean_by_code(daily, "close", 5)
        distance = daily["close"] / ma_5d - 1
        previous_distance = distance.groupby(daily["code"], sort=False).shift(1)
        return distance - previous_distance
