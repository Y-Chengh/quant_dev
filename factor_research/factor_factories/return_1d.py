import pandas as pd

from .base import FactorFactory, previous_close
from .registry import register_factor


@register_factor
class Return1dFactory(FactorFactory):
    name = "return_1d"

    def compute(self, bars: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
        return daily["close"] / previous_close(daily) - 1
