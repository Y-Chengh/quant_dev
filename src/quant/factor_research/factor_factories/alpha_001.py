"""WorldQuant Alpha #1 因子。

公式为 ``rank(Ts_ArgMax(SignedPower(returns < 0 ? stddev(returns, 20) :
close, 2), 5))``。日收益率、20 日样本标准差和 5 日极值位置窗口均包含当日；
20 日标准差要求 20 个有效日收益率，极值位置要求连续 5 个有效输入。收益率为负
但波动率历史不足时输入为缺失值，任一 5 日窗口包含缺失值时结果也为缺失值。
波动率不年化；最终仅按交易日做横截面百分位排名，不减 0.5，也不做行业中性化。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import FactorFactory
from .registry import register_factor


def _first_argmax(values: np.ndarray) -> float:
    """返回完整窗口中首个最大值从旧到新的 1 基位置。"""
    return float(np.argmax(values) + 1)


@register_factor
class Alpha001Factory(FactorFactory):
    """计算波动率条件切换后的 5 日极值时点横截面排名。"""

    name = "alpha_001"

    #: 原式在 ``SignedPower`` 内混用了一次齐次的 ``close`` 与零次齐次的
    #: ``stddev(returns, 20)``，因此整体不满足任何齐次度：缩放价格会改变两支
    #: 分量的相对大小，进而改变 ``Ts_ArgMax`` 选中的位置。翻转门槛在
    #: ``c · close ≈ stddev(returns, 20)`` 处，即缩放系数约等于日收益率波动率
    #: 除以价格水平（十元级股价、2% 日波动对应 1e-3 量级）。真实后复权系数恒为
    #: 正且通常不小于 1，``close`` 项始终压倒波动率项，所以实践中取值稳定，
    #: 但不能声明为 ``k = 0``。属性只声明在具体因子模块里，不能上提到
    #: ``FactorFactory`` 基类，否则会作废全部因子缓存。
    price_homogeneity = None

    def compute(self, bars: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
        # 先按证券和日期排序完成全部时序计算，最后恢复 daily 的原始行顺序。
        ordered = daily[["code", "trade_date", "close"]].copy()
        ordered["_position"] = np.arange(len(ordered))
        ordered = ordered.sort_values(["code", "trade_date"], kind="stable")

        returns = ordered.groupby("code", sort=False)["close"].pct_change(
            fill_method=None
        )
        volatility = returns.groupby(ordered["code"], sort=False).transform(
            lambda values: values.rolling(20, min_periods=20).std()
        )

        # SignedPower(x, 2) = sign(x) * abs(x) ** 2；此处正常价格和波动率均非负。
        conditional = ordered["close"].where(returns >= 0, volatility)
        conditional = conditional.where(returns.notna(), ordered["close"])
        signed_power = np.sign(conditional) * conditional.abs().pow(2)

        argmax_position = signed_power.groupby(ordered["code"], sort=False).transform(
            lambda values: values.rolling(5, min_periods=5).apply(
                _first_argmax, raw=True
            )
        )
        ranked = argmax_position.groupby(ordered["trade_date"], sort=False).rank(
            method="average", pct=True
        )

        result = np.full(len(daily), np.nan, dtype=float)
        result[ordered["_position"].to_numpy()] = ranked.to_numpy()
        return pd.Series(result, index=daily.index, name=self.name)
