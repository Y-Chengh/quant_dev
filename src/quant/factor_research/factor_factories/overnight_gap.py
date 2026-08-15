"""隔夜跳空收益因子。

计算公式为 ``当日开盘价 / 前一有效交易日收盘价 - 1``。前收盘价按证券独立
后移一个交易日，因此不使用当日之后的数据；每只证券首个交易日返回缺失值。
结果未年化或额外归一化。
"""

import pandas as pd

from .base import FactorFactory, previous_close
from .registry import register_factor


@register_factor
class OvernightGapFactory(FactorFactory):
    """计算当日开盘价相对前一有效交易日收盘价的收益。"""

    name = "overnight_gap"

    def compute(self, bars: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
        return daily["open"] / previous_close(daily) - 1
