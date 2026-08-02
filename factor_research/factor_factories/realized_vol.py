import numpy as np
import pandas as pd

from .base import FactorFactory, close_returns, intraday_values
from .registry import register_factor


@register_factor
class RealizedVolFactory(FactorFactory):
    name = "realized_vol"

    def compute(self, bars: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
        return intraday_values(
            bars,
            daily,
            lambda group: float(np.nanstd(close_returns(group), ddof=1)) if len(group) > 2 else np.nan,
        )
