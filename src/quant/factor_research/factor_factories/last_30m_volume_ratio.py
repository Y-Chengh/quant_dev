"""尾盘 30 分钟成交量占比因子。

代码以最后 6 根分钟线的成交量之和除以全天成交量；因此“30 分钟”成立的
前提是输入 bars 为 5 分钟线。若当日不足 6 根则使用全部可用分钟线，全天
成交量为 0 时返回缺失值。
"""

import numpy as np
import pandas as pd

from .base import FactorFactory, intraday_values
from .registry import register_factor


@register_factor
class Last30mVolumeRatioFactory(FactorFactory):
    """计算最后 6 根分钟线成交量占全天成交量的比例。"""

    name = "last_30m_volume_ratio"

    #: 本因子调用 ``intraday_values``，必须有分钟行情才能计算；
    #: 日频数据源会据此自动排除它。属性只声明在具体因子模块里，
    #: 不能上提到 ``FactorFactory`` 基类，否则会作废全部因子缓存。
    requires_intraday = True

    def compute(self, bars: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
        def calculate(group: pd.DataFrame) -> float:
            volume = group["volume"].to_numpy(dtype=float)
            total = volume.sum()
            # 分母必须为正；零成交量交易日无法定义成交量分布。
            return float(volume[-min(6, len(volume)):].sum() / total) if total > 0 else np.nan

        return intraday_values(bars, daily, calculate)
