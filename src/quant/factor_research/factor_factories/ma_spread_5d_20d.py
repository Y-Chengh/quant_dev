"""5 日与 20 日均线的相对价差因子。

计算公式：MA5 / MA20 - 1。正值表示短期均线位于长期均线上方，负值表示
位于下方，绝对值表示两条均线的相对距离；两条均线都包含当日收盘价。
"""

import pandas as pd

from .base import FactorFactory, rolling_mean_by_code
from .registry import register_factor


@register_factor
class MaSpread5d20dFactory(FactorFactory):
    """连续表达 5 日均线相对 20 日均线的位置。"""

    name = "ma_spread_5d_20d"

    def compute(self, bars: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
        ma_5d = rolling_mean_by_code(daily, "close", 5)
        ma_20d = rolling_mean_by_code(daily, "close", 20)
        return ma_5d / ma_20d - 1
