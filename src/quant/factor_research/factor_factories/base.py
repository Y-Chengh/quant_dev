"""因子工厂的抽象接口与日频、分钟频计算辅助函数。"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class FactorFactory(ABC):
    """所有因子工厂的统一接口。"""

    name: str

    @abstractmethod
    def compute(self, bars: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
        """Return one value for every row in daily, preserving its index."""


def previous_close(daily: pd.DataFrame) -> pd.Series:
    """按证券代码分组，将收盘价后移一个交易日得到前收盘价。"""
    return daily.groupby("code", sort=False)["close"].shift(1)


def rolling_mean_by_code(
    daily: pd.DataFrame, column: str, window: int
) -> pd.Series:
    """按证券代码计算要求完整窗口的滚动均值。"""
    return daily.groupby("code", sort=False)[column].transform(
        lambda values: values.rolling(window, min_periods=window).mean()
    )


def intraday_values(bars: pd.DataFrame, daily: pd.DataFrame, calculator) -> pd.Series:
    """逐证券、逐交易日计算分钟因子，并对齐回日频数据的原始行索引。"""
    # calculator 每次只接收某一证券在某一交易日内的全部分钟线。
    values = bars.groupby(["code", "trade_date"], sort=True).apply(
        calculator, include_groups=False
    )
    # 以 daily 的证券和日期顺序重排结果，保证返回值能安全赋回 daily。
    lookup = pd.MultiIndex.from_frame(daily[["code", "trade_date"]])
    return pd.Series(values.reindex(lookup).to_numpy(), index=daily.index, dtype=float)


def close_returns(group: pd.DataFrame) -> np.ndarray:
    """计算一个证券交易日内相邻分钟收盘价的简单收益率序列。"""
    # 第一根分钟线没有上一根收盘价，对应收益率为 NaN。
    return pd.Series(group["close"].to_numpy(dtype=float)).pct_change().to_numpy()
