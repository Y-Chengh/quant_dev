import pandas as pd

from .base import FactorFactory
from .registry import register_factor


@register_factor
class Volatility5dFactory(FactorFactory):
    name = "volatility_5d"

    def compute(self, bars: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
        returns = daily.groupby("code", sort=False)["close"].pct_change()
        return returns.groupby(daily["code"], sort=False).transform(
            lambda value: value.rolling(5, min_periods=3).std()
        )
