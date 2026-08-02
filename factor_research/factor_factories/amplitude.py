import pandas as pd

from .base import FactorFactory, previous_close
from .registry import register_factor


@register_factor
class AmplitudeFactory(FactorFactory):
    name = "amplitude"

    def compute(self, bars: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
        return (daily["high"] - daily["low"]) / previous_close(daily)
