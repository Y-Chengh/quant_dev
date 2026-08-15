"""5 日均线斜率因子。

计算公式：当日 MA5 / 上一交易日 MA5 - 1。正值表示短期均线向上，负值表示
短期均线向下；MA5 包含当日收盘价并要求完整的 5 日窗口。
"""

import pandas as pd

from .base import FactorFactory, rolling_mean_by_code
from .registry import register_factor


@register_factor
class Ma5dSlopeFactory(FactorFactory):
    """计算 5 日收盘均线的单日变化率。"""

    name = "ma_5d_slope"

    def compute(self, bars: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
        ma_5d = rolling_mean_by_code(daily, "close", 5)
        previous_ma = ma_5d.groupby(daily["code"], sort=False).shift(1)
        return ma_5d / previous_ma - 1
