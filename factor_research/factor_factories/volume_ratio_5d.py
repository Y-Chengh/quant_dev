import pandas as pd

from .base import FactorFactory
from .registry import register_factor


@register_factor
class VolumeRatio5dFactory(FactorFactory):
    name = "volume_ratio_5d"

    def compute(self, bars: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
        mean = daily.groupby("code", sort=False)["volume"].transform(
            lambda value: value.rolling(5, min_periods=3).mean()
        )
        return daily["volume"] / mean
