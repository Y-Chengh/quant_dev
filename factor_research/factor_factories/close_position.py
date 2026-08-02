import numpy as np
import pandas as pd

from .base import FactorFactory
from .registry import register_factor


@register_factor
class ClosePositionFactory(FactorFactory):
    name = "close_position"

    def compute(self, bars: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
        spread = daily["high"] - daily["low"]
        return pd.Series(
            np.where(spread > 0, (daily["close"] - daily["low"]) / spread, 0.5),
            index=daily.index,
        )
