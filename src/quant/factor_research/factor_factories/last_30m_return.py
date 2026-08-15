"""尾盘 30 分钟收益率因子。

代码以最后 6 根分钟线作为尾盘窗口，计算窗口末根收盘价 / 首根收盘价 - 1；
因此“30 分钟”成立的前提是输入 bars 为 5 分钟线。若不足 6 根，则使用当日
全部可用分钟线；只有 1 根时返回 0。
"""

import pandas as pd

from .base import FactorFactory, intraday_values
from .registry import register_factor


@register_factor
class Last30mReturnFactory(FactorFactory):
    """按最后 6 根分钟线计算尾盘价格收益率。"""

    name = "last_30m_return"

    #: 本因子调用 ``intraday_values``，必须有分钟行情才能计算；
    #: 日频数据源会据此自动排除它。属性只声明在具体因子模块里，
    #: 不能上提到 ``FactorFactory`` 基类，否则会作废全部因子缓存。
    requires_intraday = True

    def compute(self, bars: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
        def calculate(group: pd.DataFrame) -> float:
            close = group["close"].to_numpy(dtype=float)
            # 取末尾至多 6 根 bar，保留短交易日或数据不完整时的可用观测。
            tail = close[-min(6, len(close)):]
            return float(tail[-1] / tail[0] - 1) if len(tail) > 1 else 0.0

        return intraday_values(bars, daily, calculate)
