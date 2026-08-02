import pandas as pd

from .base import FactorFactory
from .registry import register_factor


@register_factor
class Return5dFactory(FactorFactory):
    name = "return_5d"

    def compute(self, bars: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
        return daily["close"] / daily.groupby("code", sort=False)["close"].shift(5) - 1
