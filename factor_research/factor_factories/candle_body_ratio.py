"""K 线实体占振幅比例因子。

计算公式：``abs(当日收盘价 - 当日开盘价) / (当日最高价 - 当日最低价)``。
计算仅使用当日已收盘的 OHLC 数据，窗口包含当日且最小观测数为 1；没有跨日
滚动计算。最高价、最低价、开盘价或收盘价缺失，以及最高价不大于最低价时，
结果为缺失值。该比例按当日价格区间归一化，不进行年化。
"""

import pandas as pd

from .base import FactorFactory
from .registry import register_factor


@register_factor
class CandleBodyRatioFactory(FactorFactory):
    """计算当日 K 线实体绝对长度占完整价格区间的比例。"""

    name = "candle_body_ratio"

    def compute(self, bars: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
        # 考虑停牌的情况
        price_range = daily["high"] - daily["low"] + 0.0001
        # 零振幅或高低价倒置时比例没有金融含义，保留为缺失值。
        # valid_range = price_range.where(price_range > 0)
        return (daily["close"] - daily["open"]) / price_range
