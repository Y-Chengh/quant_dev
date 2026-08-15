"""收盘价相对当日近似 VWAP 的距离因子。

使用 5 分钟收盘价和成交量计算 ``VWAP = sum(close * volume) / sum(volume)``，因子
为 ``daily_close / VWAP - 1``。该实现不依赖可选的成交额字段；当日总成交量为 0
时返回缺失值。窗口只包含当日已完成的分钟线，未年化或额外归一化。
"""

import numpy as np
import pandas as pd

from .base import FactorFactory, intraday_values
from .registry import register_factor


@register_factor
class CloseToVwapFactory(FactorFactory):
    """计算当日收盘价相对成交量加权 5 分钟收盘价的距离。"""

    name = "close_to_vwap"

    #: 本因子调用 ``intraday_values``，必须有分钟行情才能计算；
    #: 日频数据源会据此自动排除它。属性只声明在具体因子模块里，
    #: 不能上提到 ``FactorFactory`` 基类，否则会作废全部因子缓存。
    requires_intraday = True

    def compute(self, bars: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
        def calculate(group: pd.DataFrame) -> float:
            volume = group["volume"].to_numpy(dtype=float)
            total_volume = volume.sum()
            if total_volume <= 0:
                return np.nan
            close = group["close"].to_numpy(dtype=float)
            vwap = float(np.dot(close, volume) / total_volume)
            return float(close[-1] / vwap - 1)

        return intraday_values(bars, daily, calculate)
