"""收盘价相对 5 日均线的距离因子。

计算公式：收盘价 / 5 日收盘均价 - 1。正值表示价格位于均线上方，负值表示
位于均线下方，绝对值表示偏离幅度。5 日均线包含当日并要求完整窗口。
"""

import pandas as pd

from .base import FactorFactory, rolling_mean_by_code
from .registry import register_factor


@register_factor
class CloseToMa5dFactory(FactorFactory):
    """连续表达收盘价位于 5 日均线上方或下方的程度。"""

    name = "close_to_ma_5d"

    def compute(self, bars: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
        ma_5d = rolling_mean_by_code(daily, "close", 5)
        return daily["close"] / ma_5d - 1
