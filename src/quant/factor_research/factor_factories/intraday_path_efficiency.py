"""日内价格路径效率因子。

计算公式为 ``(最后收盘价 - 首根开盘价) / sum(abs(相邻价格节点之差))``，价格
节点依次为首根开盘价和每根 5 分钟线收盘价。它在 [-1, 1] 内连续表达方向与
路径效率：绝对值越接近 1，走势越单边。路径总变动为 0 时返回缺失值；至少
需要一根分钟线。仅使用当日数据，未年化或额外归一化。
"""

import numpy as np
import pandas as pd

from .base import FactorFactory, intraday_values
from .registry import register_factor


@register_factor
class IntradayPathEfficiencyFactory(FactorFactory):
    """计算日内净价格变化占价格路径总变化的比例。"""

    name = "intraday_path_efficiency"

    #: 本因子调用 ``intraday_values``，必须有分钟行情才能计算；
    #: 日频数据源会据此自动排除它。属性只声明在具体因子模块里，
    #: 不能上提到 ``FactorFactory`` 基类，否则会作废全部因子缓存。
    requires_intraday = True

    def compute(self, bars: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
        def calculate(group: pd.DataFrame) -> float:
            nodes = np.concatenate(
                ([float(group["open"].iloc[0])], group["close"].to_numpy(dtype=float))
            )
            total_movement = np.abs(np.diff(nodes)).sum()
            if total_movement <= 0:
                return np.nan
            return float((nodes[-1] - nodes[0]) / total_movement)

        return intraday_values(bars, daily, calculate)
