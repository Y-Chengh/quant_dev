"""交易日年化常量与复利、价差两组汇总指标。"""

from __future__ import annotations

import numpy as np

TRADING_DAYS_PER_YEAR = 252


def _compound_metrics(returns: np.ndarray) -> dict[str, float]:
    """计算可复利长仓日收益的累计、年化、波动和夏普指标。

    参数：
        returns: 按交易日排序的有限日收益率数组，单位为一。

    返回：
        含交易日数、累计收益率、复利年化收益率、年化波动率和夏普比率的字典。
    """

    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("复利指标要求非空的一维有限日收益率")
    periods = len(values)
    final_equity = float(np.prod(1.0 + values))
    daily_std = float(np.std(values, ddof=1)) if periods >= 2 else float("nan")
    return {
        "trading_days": float(periods),
        "total_return": final_equity - 1.0,
        "annualized_return": (
            float(final_equity ** (TRADING_DAYS_PER_YEAR / periods) - 1.0)
            if final_equity > 0.0
            else float("nan")
        ),
        "annualized_volatility": (
            float(daily_std * np.sqrt(TRADING_DAYS_PER_YEAR))
            if np.isfinite(daily_std)
            else float("nan")
        ),
        "sharpe_ratio": (
            float(np.sqrt(TRADING_DAYS_PER_YEAR) * np.mean(values) / daily_std)
            if np.isfinite(daily_std) and daily_std > 0.0
            else float("nan")
        ),
    }


def _spread_metrics(returns: np.ndarray, prefix: str) -> dict[str, float]:
    """计算横截面收益差的算术年化、波动率和夏普比率。

    参数：
        returns: Top N 减基准或 Bottom N 的逐日收益差，单位为一。
        prefix: 输出指标键的业务前缀，用于区分等权超额和多空价差。

    返回：
        使用算术年化收益的三项价差诊断指标。
    """

    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("价差指标要求非空的一维有限日收益率")
    daily_std = float(np.std(values, ddof=1)) if len(values) >= 2 else float("nan")
    return {
        f"{prefix}_annualized_return": float(np.mean(values) * TRADING_DAYS_PER_YEAR),
        f"{prefix}_annualized_volatility": (
            float(daily_std * np.sqrt(TRADING_DAYS_PER_YEAR))
            if np.isfinite(daily_std)
            else float("nan")
        ),
        f"{prefix}_sharpe_ratio": (
            float(np.sqrt(TRADING_DAYS_PER_YEAR) * np.mean(values) / daily_std)
            if np.isfinite(daily_std) and daily_std > 0.0
            else float("nan")
        ),
    }
