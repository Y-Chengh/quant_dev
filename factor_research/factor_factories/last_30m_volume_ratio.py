import numpy as np
import pandas as pd

from .base import FactorFactory, intraday_values
from .registry import register_factor


@register_factor
class Last30mVolumeRatioFactory(FactorFactory):
    name = "last_30m_volume_ratio"

    def compute(self, bars: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
        def calculate(group: pd.DataFrame) -> float:
            volume = group["volume"].to_numpy(dtype=float)
            total = volume.sum()
            return float(volume[-min(6, len(volume)):].sum() / total) if total > 0 else np.nan

        return intraday_values(bars, daily, calculate)
