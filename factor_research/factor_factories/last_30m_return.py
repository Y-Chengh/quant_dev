import pandas as pd

from .base import FactorFactory, intraday_values
from .registry import register_factor


@register_factor
class Last30mReturnFactory(FactorFactory):
    name = "last_30m_return"

    def compute(self, bars: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
        def calculate(group: pd.DataFrame) -> float:
            close = group["close"].to_numpy(dtype=float)
            tail = close[-min(6, len(close)):]
            return float(tail[-1] / tail[0] - 1) if len(tail) > 1 else 0.0

        return intraday_values(bars, daily, calculate)
