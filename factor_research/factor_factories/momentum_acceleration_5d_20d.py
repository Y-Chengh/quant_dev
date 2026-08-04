"""短期相对中期的动量加速度因子。

计算公式：log(C_t / C_{t-5}) / 5 - log(C_t / C_{t-20}) / 20。两个
累计对数收益率都换算为日均速度后再相减，正值表示近期上涨速度相对加快。
"""

import numpy as np
import pandas as pd

from .base import FactorFactory
from .registry import register_factor


@register_factor
class MomentumAcceleration5d20dFactory(FactorFactory):
    """比较最近 5 日和最近 20 日的日均对数收益率。"""

    name = "momentum_acceleration_5d_20d"

    def compute(self, bars: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
        grouped_close = daily.groupby("code", sort=False)["close"]
        close_5d_ago = grouped_close.shift(5)
        close_20d_ago = grouped_close.shift(20)
        short_speed = np.log(daily["close"] / close_5d_ago) / 5
        medium_speed = np.log(daily["close"] / close_20d_ago) / 20
        return short_speed - medium_speed
