"""净值回撤曲线、历次回撤区间切分与回撤诊断指标。"""

from __future__ import annotations

import numpy as np
import pandas as pd

DRAWDOWN_EPISODE_COLUMNS = [
    "peak_date",
    "trough_date",
    "recovery_date",
    "max_drawdown",
    "decline_days",
    "recovery_days",
    "total_days",
    "recovered",
]


def _drawdown_curve(equity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """按期初净值 1.0 起算逐日历史峰值与水下回撤。

    期初净值一并参与峰值统计，因此第一个交易日就亏损时也能得到非零回撤，且
    历史峰值恒不小于 1.0，避免净值归零时出现除零。

    参数：
        equity: 按交易日升序排列的累计净值序列，首元素为第一个交易日收盘后的
            净值，单位为期初净值的倍数。

    返回：
        与输入等长的历史峰值数组和回撤数组；回撤为 ``净值 / 历史峰值 - 1``，
        取值不大于 0，等于 0 表示当日创出新高。
    """

    values = np.asarray(equity, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("回撤曲线要求非空的一维有限净值序列")
    extended = np.concatenate(([1.0], values))
    peaks = np.maximum.accumulate(extended)
    drawdowns = extended / peaks - 1.0
    return peaks[1:], drawdowns[1:]


def _drawdown_episodes(dates: pd.Series, equity: np.ndarray) -> pd.DataFrame:
    """把净值曲线切分为历次回撤区间并标注修复情况。

    一段回撤从最后一次创出新高的位置开始，到净值重新触及该峰值为止；样本末尾
    仍未回到峰值的区间记为未修复，其修复日期与修复交易日数为缺失值，区间交易
    日数只统计到样本末尾，是仍在进行中的下限而非最终时长。峰值出现在期初净值
    时，峰值日期记为缺失值，表示回撤自期初就开始。

    参数：
        dates: 与净值一一对应的目标交易日序列，必须已按升序排列；乱序会算出
            错误的峰值位置，因此显式拒绝。
        equity: 与 ``dates`` 等长的累计净值序列，单位为期初净值的倍数。

    返回：
        按时间先后排列的回撤区间表，含峰值、谷底与修复日期、回撤幅度（负值）、
        峰值至谷底交易日数、谷底至修复交易日数、区间交易日数及是否修复。
    """

    values = np.asarray(equity, dtype=float)
    parsed = pd.DatetimeIndex(pd.to_datetime(pd.Series(dates).to_numpy()))
    if len(parsed) != len(values):
        raise ValueError("回撤区间要求日期与净值长度一致")
    if not parsed.is_monotonic_increasing:
        raise ValueError("回撤区间要求目标交易日按升序排列")
    _, drawdowns = _drawdown_curve(values)
    # 下标 0 代表期初净值，回撤恒为 0，便于统一定位每段回撤之前的峰值位置。
    extended = np.concatenate(([0.0], drawdowns))
    total = len(values)

    def date_at(position: int) -> pd.Timestamp:
        """把扩展序列下标翻译为对应的目标交易日。

        参数：
            position: 扩展回撤序列的下标，0 表示期初净值。

        返回：
            对应交易日；期初位置返回 ``pd.NaT``。
        """

        return pd.NaT if position == 0 else parsed[position - 1]

    records: list[dict[str, object]] = []
    index = 1
    while index <= total:
        if extended[index] >= 0.0:
            index += 1
            continue
        peak_index = index - 1
        trough_index = index
        cursor = index
        while cursor <= total and extended[cursor] < 0.0:
            if extended[cursor] < extended[trough_index]:
                trough_index = cursor
            cursor += 1
        recovered = cursor <= total
        end_index = cursor if recovered else total
        records.append(
            {
                "peak_date": date_at(peak_index),
                "trough_date": date_at(trough_index),
                "recovery_date": date_at(cursor) if recovered else pd.NaT,
                "max_drawdown": float(extended[trough_index]),
                "decline_days": int(trough_index - peak_index),
                "recovery_days": (
                    float(cursor - trough_index) if recovered else float("nan")
                ),
                "total_days": int(end_index - peak_index),
                "recovered": bool(recovered),
            }
        )
        index = cursor
    episodes = pd.DataFrame(records, columns=DRAWDOWN_EPISODE_COLUMNS)
    if episodes.empty:
        return episodes.astype(
            {
                "peak_date": "datetime64[ns]",
                "trough_date": "datetime64[ns]",
                "recovery_date": "datetime64[ns]",
                "max_drawdown": float,
                "decline_days": int,
                "recovery_days": float,
                "total_days": int,
                "recovered": bool,
            }
        )
    return episodes


def _drawdown_metrics(
    equity: np.ndarray,
    episodes: pd.DataFrame,
    annualized_return: float,
) -> dict[str, float]:
    """汇总净值曲线的回撤深度、修复时长与 Calmar 比率。

    参数：
        equity: 按交易日升序排列的累计净值序列，单位为期初净值的倍数。
        episodes: ``_drawdown_episodes`` 切分出的历次回撤区间。
        annualized_return: 同一条净值曲线的复利年化收益率，用于计算 Calmar 比率。

    返回：
        含最大回撤（负值）、最大回撤各阶段交易日数、期末当前回撤、日均水下深度、
        处于回撤的交易日占比、最长回撤区间交易日数、回撤区间数量及 Calmar
        比率的字典；无回撤时全部时长为 0，Calmar 比率为 NaN。

        其中 ``average_drawdown`` 是逐日水下深度在全部交易日上的均值，创出新高
        的交易日按 0 计入，因此它比"历次回撤深度的平均"小；``longest_drawdown_days``
        把未修复区间按截至样本末尾的天数一并参与取最大值。
    """

    _, drawdowns = _drawdown_curve(equity)
    max_drawdown = float(drawdowns.min())
    if episodes.empty:
        deepest_decline = 0.0
        deepest_recovery = 0.0
        deepest_total = 0.0
        longest_total = 0.0
    else:
        deepest = episodes.loc[episodes["max_drawdown"].idxmin()]
        deepest_decline = float(deepest["decline_days"])
        deepest_recovery = float(deepest["recovery_days"])
        deepest_total = float(deepest["total_days"])
        longest_total = float(episodes["total_days"].max())
    return {
        "max_drawdown": max_drawdown,
        "max_drawdown_decline_days": deepest_decline,
        "max_drawdown_recovery_days": deepest_recovery,
        "max_drawdown_total_days": deepest_total,
        "current_drawdown": float(drawdowns[-1]),
        "average_drawdown": float(drawdowns.mean()),
        "drawdown_days_ratio": float(np.mean(drawdowns < 0.0)),
        "longest_drawdown_days": longest_total,
        "drawdown_episodes": float(len(episodes)),
        "recovered_drawdown_episodes": (
            float(episodes["recovered"].sum()) if not episodes.empty else 0.0
        ),
        "calmar_ratio": (
            float(annualized_return / abs(max_drawdown))
            if max_drawdown < 0.0 and np.isfinite(annualized_return)
            else float("nan")
        ),
    }
