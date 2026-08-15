"""日内已实现波动率因子。

先计算相邻分钟收盘价的简单收益率，再计算这些分钟收益率的样本标准差
（ddof=1）。当前结果未做年化或按日内区间数缩放；不足 3 根分钟线时返回
缺失值，以确保至少有 2 个有效收益率观测。
"""

import numpy as np
import pandas as pd

from .base import FactorFactory, close_returns, intraday_values
from .registry import register_factor


@register_factor
class RealizedVolFactory(FactorFactory):
    """计算日内分钟收益率的样本标准差。"""

    name = "realized_vol"

    #: 本因子调用 ``intraday_values``，必须有分钟行情才能计算；
    #: 日频数据源会据此自动排除它。属性只声明在具体因子模块里，
    #: 不能上提到 ``FactorFactory`` 基类，否则会作废全部因子缓存。
    requires_intraday = True

    def compute(self, bars: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
        return intraday_values(
            bars,
            daily,
            # nanstd 忽略第一根分钟线因无前值而产生的 NaN。
            lambda group: float(np.nanstd(close_returns(group), ddof=1)) if len(group) > 2 else np.nan,
        )
