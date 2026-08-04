"""相对前 20 日最高收盘价的突破强度因子。

计算公式：当日收盘价 / 前 20 个交易日最高收盘价 - 1。正值表示已经向上
突破并给出突破幅度，负值表示仍低于前高。基准窗口先后移一天，不包含当日。
"""

import pandas as pd

from .base import FactorFactory
from .registry import register_factor


@register_factor
class BreakoutStrength20dFactory(FactorFactory):
    """计算收盘价相对当日以前 20 日最高收盘价的距离。"""

    name = "breakout_strength_20d"

    def compute(self, bars: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
        previous_high = daily.groupby("code", sort=False)["close"].transform(
            lambda values: values.shift(1).rolling(20, min_periods=20).max()
        )
        return daily["close"] / previous_high - 1
