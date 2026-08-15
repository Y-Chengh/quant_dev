"""5 日与 20 日均线相对价差的单日变化因子。

先计算 MA5 / MA20 - 1，再减去同一证券上一交易日的均线价差。正值表示
短期均线相对长期均线上行，负值表示相对下行，可连续表达金叉或死叉力度。
"""

import pandas as pd

from .base import FactorFactory, rolling_mean_by_code
from .registry import register_factor


@register_factor
class MaSpreadChange5d20dFactory(FactorFactory):
    """计算 5 日与 20 日均线相对价差的日变化。"""

    name = "ma_spread_change_5d_20d"

    def compute(self, bars: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
        ma_5d = rolling_mean_by_code(daily, "close", 5)
        ma_20d = rolling_mean_by_code(daily, "close", 20)
        spread = ma_5d / ma_20d - 1
        previous_spread = spread.groupby(daily["code"], sort=False).shift(1)
        return spread - previous_spread
