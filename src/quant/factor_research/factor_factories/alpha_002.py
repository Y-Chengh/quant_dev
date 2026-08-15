"""WorldQuant Alpha #2 量价背离因子。

公式为 ``-correlation(rank(delta(log(volume), 2)),
rank((close - open) / open), 6)``。成交量对数的 2 日差分使用当日与两个交易日
前的数据；两个 ``rank`` 均为同一交易日内的横截面百分位排名，采用平均名次；
相关系数按证券计算，窗口包含当日并要求连续 6 个有效排名观测。成交量小于等于
零、开盘价为零、输入价格缺失、相关窗口不完整或窗口内任一序列方差为零时，
结果为缺失值。因子不年化；除横截面排名外不做额外归一化。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import FactorFactory
from .registry import register_factor


@register_factor
class Alpha002Factory(FactorFactory):
    """计算成交量变化排名与日内收益排名的 6 日负相关系数。"""

    name = "alpha_002"

    def compute(self, bars: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
        # 先按证券和日期排序完成时序计算，最后恢复 daily 的原始行顺序。
        ordered = daily[["code", "trade_date", "open", "close", "volume"]].copy()
        ordered["_position"] = np.arange(len(ordered))
        ordered = ordered.sort_values(["code", "trade_date"], kind="stable")

        valid_volume = ordered["volume"].where(ordered["volume"] > 0)
        log_volume = np.log(valid_volume)
        volume_delta = log_volume.groupby(ordered["code"], sort=False).diff(2)

        valid_open = ordered["open"].where(ordered["open"] != 0)
        intraday_return = (ordered["close"] - valid_open) / valid_open

        volume_rank = volume_delta.groupby(
            ordered["trade_date"], sort=False
        ).rank(method="average", pct=True)
        return_rank = intraday_return.groupby(
            ordered["trade_date"], sort=False
        ).rank(method="average", pct=True)

        # 使用位置数组赋值，避免 daily 的自定义或重复索引影响分组滚动结果的对齐。
        correlation = np.full(len(ordered), np.nan, dtype=float)
        for positions in ordered.groupby("code", sort=False).indices.values():
            ranked_volume = pd.Series(volume_rank.iloc[positions].to_numpy())
            ranked_return = pd.Series(return_rank.iloc[positions].to_numpy())
            correlation[positions] = (
                ranked_volume.rolling(6, min_periods=6)
                .corr(ranked_return)
                .to_numpy()
            )

        result = np.full(len(daily), np.nan, dtype=float)
        result[ordered["_position"].to_numpy()] = -correlation
        return pd.Series(result, index=daily.index, name=self.name)
