import numpy as np
import pandas as pd

from .base import FactorFactory, close_returns, intraday_values
from .registry import register_factor


@register_factor
class PositiveBarRatioFactory(FactorFactory):
    name = "positive_bar_ratio"

    def compute(self, bars: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
        return intraday_values(
            bars,
            daily,
            lambda group: float(np.mean(close_returns(group)[1:] > 0)) if len(group) > 1 else np.nan,
        )
