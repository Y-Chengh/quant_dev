"""日内下行半波动率因子。

先计算各 5 分钟区间的简单收益率，首根线使用当日开盘价作为起点，之后使用
上一根收盘价；再计算 ``sqrt(mean(min(return, 0) ** 2))``。至少一根分钟线即可
计算；没有负收益时为 0。结果未年化，也未按日内区间数缩放。
"""

import numpy as np
import pandas as pd

from .base import FactorFactory, intraday_values
from .registry import register_factor


@register_factor
class DownsideSemivolFactory(FactorFactory):
    """计算日内 5 分钟收益率的下行半波动率。"""

    name = "downside_semivol"

    def compute(self, bars: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
        def calculate(group: pd.DataFrame) -> float:
            close = group["close"].to_numpy(dtype=float)
            previous = np.concatenate(([float(group["open"].iloc[0])], close[:-1]))
            returns = close / previous - 1
            downside = np.minimum(returns, 0.0)
            return float(np.sqrt(np.mean(np.square(downside))))

        return intraday_values(bars, daily, calculate)
