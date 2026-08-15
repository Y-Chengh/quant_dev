"""日内带符号成交量失衡因子。

计算每根 5 分钟线相对上一价格节点的收益方向，其中首根线使用其开盘价作为
上一价格节点；再计算 ``sum(sign(return) * volume) / sum(volume)``。当日总成交量
为 0 时返回缺失值。该因子取值位于 [-1, 1]，未年化或额外归一化。
"""

import numpy as np
import pandas as pd

from .base import FactorFactory, intraday_values
from .registry import register_factor


@register_factor
class SignedVolumeImbalanceFactory(FactorFactory):
    """计算由每根 5 分钟线价格方向加权的成交量失衡。"""

    name = "signed_volume_imbalance"

    def compute(self, bars: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
        def calculate(group: pd.DataFrame) -> float:
            volume = group["volume"].to_numpy(dtype=float)
            total_volume = volume.sum()
            if total_volume <= 0:
                return np.nan
            close = group["close"].to_numpy(dtype=float)
            previous = np.concatenate(([float(group["open"].iloc[0])], close[:-1]))
            direction = np.sign(close / previous - 1)
            return float(np.dot(direction, volume) / total_volume)

        return intraday_values(bars, daily, calculate)
