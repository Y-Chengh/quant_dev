from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class FactorFactory(ABC):
    name: str

    @abstractmethod
    def compute(self, bars: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
        """Return one value for every row in daily, preserving its index."""


def previous_close(daily: pd.DataFrame) -> pd.Series:
    return daily.groupby("code", sort=False)["close"].shift(1)


def intraday_values(bars: pd.DataFrame, daily: pd.DataFrame, calculator) -> pd.Series:
    values = bars.groupby(["code", "trade_date"], sort=True).apply(
        calculator, include_groups=False
    )
    lookup = pd.MultiIndex.from_frame(daily[["code", "trade_date"]])
    return pd.Series(values.reindex(lookup).to_numpy(), index=daily.index, dtype=float)


def close_returns(group: pd.DataFrame) -> np.ndarray:
    return pd.Series(group["close"].to_numpy(dtype=float)).pct_change().to_numpy()
