"""由除权送转记录计算前复权与后复权系数。

入库永远只存原始不复权价，复权在读取时按需计算，因此原始值始终可审计、可对账。

系数口径已在真实数据上标定：2024 年全年 4524 个除权事件上，
``adjustment_factor(t)`` 与 ``close(t-1) / pre_close(t)`` 的比值中位数为 1.0，
10%~90% 分位落在 ``[1.0, 1.000001]``。于是：

- 原始价在除权日会被人为压低 ``1 / adjustment_factor``；
- 后复权系数 ``hfq(t) = Π{ adjustment_factor(s) : ex_date s <= t }`` 正好抵消，
  且历史值不随之后新增的分红送转改变；
- 前复权系数 ``qfq(t) = hfq(t) / hfq(anchor)``，每换一个基准日整段历史都会变，
  因此必须显式固定 anchor。

系数一律由 ``corporate_actions`` 全历史累乘得到，**不**依赖查询窗口内的
``pre_close``：否则窗口起点不同就会算出不同的复权价，因子缓存也会失去意义。
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from .models import AdjustMode

#: 复权后仍需保持原值的列。
_UNADJUSTED_COLUMNS = ("amount",)

#: 参与价格复权的列。
_PRICE_COLUMNS = ("open", "high", "low", "close", "pre_close")


def cumulative_factors(actions: pd.DataFrame) -> pd.DataFrame:
    """把逐次除权记录累乘成每只证券的后复权系数曲线。

    参数：
        actions: 除权送转表，至少含 ``code``、``ex_date`` 与
            ``adjustment_factor`` 三列。非正、缺失或非有限的系数按 1.0 处理，
            即视为「该次事件不改变价格连续性」。

    返回：
        含 ``code``、``ex_date``、``hfq_factor`` 三列、按代码与除权日升序排列的表；
        入参为空时返回具有相同列的空表。
    """
    columns = ["code", "ex_date", "hfq_factor"]
    if actions is None or actions.empty:
        return pd.DataFrame(columns=columns)
    frame = actions.loc[:, ["code", "ex_date", "adjustment_factor"]].copy()
    frame["code"] = frame["code"].astype(str)
    frame["ex_date"] = pd.to_datetime(frame["ex_date"], errors="coerce")
    factor = pd.to_numeric(frame["adjustment_factor"], errors="coerce")
    frame["adjustment_factor"] = factor.where(np.isfinite(factor) & (factor > 0), 1.0)
    frame = frame.dropna(subset=["ex_date"])
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame = frame.sort_values(["code", "ex_date"], kind="mergesort")
    # 同一天出现多条记录时按顺序连乘，等价于把当天的多次事件合成一次。
    frame["hfq_factor"] = frame.groupby("code", sort=False)["adjustment_factor"].cumprod()
    collapsed = (
        frame.groupby(["code", "ex_date"], sort=True)["hfq_factor"].last().reset_index()
    )
    return collapsed.loc[:, columns]


def factor_at(cumulative: pd.DataFrame, codes, moment: date) -> pd.Series:
    """取出各证券在某一时点的后复权系数。

    参数：
        cumulative: ``cumulative_factors`` 的返回值。
        codes: 需要取值的证券代码可迭代对象。
        moment: 目标时点；取除权日不晚于该时点的最后一条记录。

    返回：
        以证券代码为索引的浮点系数；该时点之前没有任何除权记录的证券取 1.0。
    """
    unique = list(dict.fromkeys(str(code) for code in codes))
    result = pd.Series(1.0, index=unique, dtype=float)
    if cumulative is None or cumulative.empty:
        return result
    limit = pd.Timestamp(moment)
    subset = cumulative[
        cumulative["code"].isin(unique) & (cumulative["ex_date"] <= limit)
    ]
    if subset.empty:
        return result
    latest = subset.sort_values("ex_date").groupby("code", sort=False)["hfq_factor"].last()
    result.update(latest.astype(float))
    return result


def attach_adjust_factor(
    bars: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    mode: AdjustMode,
    anchor: date | None = None,
) -> pd.DataFrame:
    """为每根日线附上复权系数列 ``adjust_factor``。

    参数：
        bars: 原始日线表，至少含 ``code`` 与 ``trade_date`` 两列。
        actions: 该批证券的**全历史**除权送转表；只截取窗口内的记录会让系数偏掉。
        mode: 复权口径。``NONE`` 时系数恒为 1.0。
        anchor: 前复权基准日；``mode`` 为 ``QFQ`` 时必须提供，其余口径忽略。

    返回：
        在输入基础上新增 ``adjust_factor`` 列的副本；输入为空时原样返回带该列的空表。
    """
    result = bars.copy()
    if result.empty:
        result["adjust_factor"] = pd.Series(dtype=float)
        return result
    if mode == AdjustMode.NONE:
        result["adjust_factor"] = 1.0
        return result
    if mode == AdjustMode.QFQ and anchor is None:
        raise ValueError("前复权必须显式提供基准日 anchor，否则历史价会随每次同步变化")

    cumulative = cumulative_factors(actions)
    result["code"] = result["code"].astype(str)
    result["trade_date"] = pd.to_datetime(result["trade_date"], errors="raise")
    if cumulative.empty:
        result["adjust_factor"] = 1.0
    else:
        ordered = result.sort_values(["trade_date", "code"], kind="mergesort")
        merged = pd.merge_asof(
            ordered,
            cumulative.sort_values(["ex_date", "code"], kind="mergesort"),
            left_on="trade_date",
            right_on="ex_date",
            by="code",
            direction="backward",
        )
        merged["hfq_factor"] = merged["hfq_factor"].fillna(1.0)
        result = merged.drop(columns=["ex_date"]).rename(columns={"hfq_factor": "adjust_factor"})
    if mode == AdjustMode.QFQ:
        base = factor_at(cumulative, result["code"].unique(), anchor)
        divisor = result["code"].map(base).astype(float)
        result["adjust_factor"] = result["adjust_factor"].astype(float) / divisor
    return result.sort_values(["code", "trade_date"], kind="mergesort").reset_index(drop=True)


def apply_adjustment(
    bars: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    mode: AdjustMode,
    anchor: date | None = None,
    adjust_volume: bool = False,
) -> pd.DataFrame:
    """按指定口径复权日线价格。

    参数：
        bars: 原始日线表，含 ``BAR_QUERY_COLUMNS`` 中的价格与成交列。
        actions: 该批证券的全历史除权送转表。
        mode: 复权口径，见 :class:`~quant.market_data.daily.models.AdjustMode`。
        anchor: 前复权基准日；``mode`` 为 ``QFQ`` 时必须提供。
        adjust_volume: 是否同步反向调整成交量。缺省 ``False``，保留原始股数；
            成交额在任何口径下都不调整，因为它本来就是真实成交金额。

    返回：
        复权后的新表，附带 ``adjust_factor`` 列；``mode`` 为 ``NONE`` 时价格
        与输入完全一致。
    """
    result = attach_adjust_factor(bars, actions, mode=mode, anchor=anchor)
    if result.empty or mode == AdjustMode.NONE:
        return result
    factor = pd.to_numeric(result["adjust_factor"], errors="coerce")
    factor = factor.where(np.isfinite(factor) & (factor > 0), 1.0)
    result["adjust_factor"] = factor
    for column in _PRICE_COLUMNS:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce") * factor
    if adjust_volume and "volume" in result.columns:
        result["volume"] = pd.to_numeric(result["volume"], errors="coerce") / factor
    for column in _UNADJUSTED_COLUMNS:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def observed_event_ratio(bars: pd.DataFrame) -> pd.DataFrame:
    """由行情自身推出每个交易日的除权比例，用于校验除权表。

    口径为 ``close(t-1) / pre_close(t)``，与 ``adjustment_factor`` 同向。

    参数：
        bars: 单一或多只证券的**连续**日线表，必须包含停牌行，否则
            ``shift(1)`` 会跨过日历空隙而算出错误的前收。至少含 ``code``、
            ``trade_date``、``close``、``pre_close`` 四列。

    返回：
        含 ``code``、``trade_date``、``observed_ratio`` 三列的表；前收或前一日
        收盘不可用时该行的比例为 ``NaN``。
    """
    columns = ["code", "trade_date", "observed_ratio"]
    if bars is None or bars.empty:
        return pd.DataFrame(columns=columns)
    frame = bars.loc[:, ["code", "trade_date", "close", "pre_close"]].copy()
    frame["code"] = frame["code"].astype(str)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame = frame.sort_values(["code", "trade_date"], kind="mergesort")
    previous_close = frame.groupby("code", sort=False)["close"].shift(1)
    pre_close = pd.to_numeric(frame["pre_close"], errors="coerce")
    previous_close = pd.to_numeric(previous_close, errors="coerce")
    valid = (
        np.isfinite(pre_close) & (pre_close > 0)
        & np.isfinite(previous_close) & (previous_close > 0)
    )
    frame["observed_ratio"] = (previous_close / pre_close).where(valid)
    return frame.loc[:, columns].reset_index(drop=True)


def theoretical_event_ratio(actions: pd.DataFrame, previous_close: pd.Series) -> pd.Series:
    """由分红送转配股字段推出理论除权比例。

    使用 A 股标准除权除息公式：

    ``pre_close = (前收 - 每股现金分红 + 配股比例 × 配股价) /
    (1 + 每股送股 + 每股转增 + 配股比例)``

    参数：
        actions: 除权送转表，含 ``cash_dividend_per_share``、
            ``bonus_share_per_share``、``capitalization_per_share``、
            ``rights_issue_per_share``、``rights_issue_price`` 五列。
        previous_close: 与 ``actions`` 逐行对齐的除权前一交易日收盘价。

    返回：
        与输入等长的理论比例序列，口径与 ``adjustment_factor`` 一致，即
        ``前收 / 理论前收``；分母非正或数据缺失时为 ``NaN``。
    """
    close = pd.to_numeric(previous_close, errors="coerce")
    cash = pd.to_numeric(actions["cash_dividend_per_share"], errors="coerce").fillna(0.0)
    bonus = pd.to_numeric(actions["bonus_share_per_share"], errors="coerce").fillna(0.0)
    capital = pd.to_numeric(actions["capitalization_per_share"], errors="coerce").fillna(0.0)
    rights = pd.to_numeric(actions["rights_issue_per_share"], errors="coerce").fillna(0.0)
    price = pd.to_numeric(actions["rights_issue_price"], errors="coerce").fillna(0.0)
    denominator = 1.0 + bonus + capital + rights
    theory = (close - cash + rights * price) / denominator
    valid = np.isfinite(theory) & (theory > 0) & np.isfinite(close) & (close > 0)
    return (close / theory).where(valid)
