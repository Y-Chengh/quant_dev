import pandas as pd

from .base import FactorFactory
from .registry import register_factor


@register_factor
class IntradayReturnFactory(FactorFactory):
    name = "intraday_return"

    def compute(self, bars: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
        return daily["close"] / daily["open"] - 1
