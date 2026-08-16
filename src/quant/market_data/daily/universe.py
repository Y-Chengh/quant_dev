"""从证券静态信息推导无幸存者偏差的研究证券池。

关键点是**读 ``instruments`` 而不是 ``bars_1d``**：下载器会在 ``instrument_info``
里把已退市的代码一直带着，只有基于它筛选，历史某一天的证券池才包含那天还在、
今天已经退市的股票。若改用「库里有行情的代码」，就等于只保留了活到今天的证券，
回测结果会被幸存者偏差系统性抬高。

注意源数据的现状：实测当前快照里 ``expire_date`` 全部是空值或 1970 年哨兵，
即**一只退市股都没有**。这一点由
``python -m quant.cli.market_check`` 如实报告，本模块不做
任何补偿。
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd


def filter_universe(
    instruments: pd.DataFrame,
    as_of: date,
    *,
    board: str | None = None,
    exclude_st: bool = False,
    min_listed_days: int = 0,
    include_delisted: bool = True,
) -> list[str]:
    """筛出某一天仍在存续期内的证券代码。

    参数：
        instruments: 证券静态信息表，至少含 ``code``、``open_date``、
            ``expire_date``、``board``、``is_st`` 五列。
        as_of: 目标日期；上市日不晚于该日、退市日晚于该日的证券入池。
        board: 只保留该板块，取值为 ``main``/``star``/``chinext``/``bse``；
            ``None`` 表示不限板块。
        exclude_st: 是否剔除风险警示证券。注意 ``is_st`` 来自**当前**名称快照，
            无法还原历史某一天的状态，开启后会把「今天是 ST、当年不是」的证券
            也一并剔除。
        min_listed_days: 要求上市满多少个自然日，用于避开新股上市初期的异常
            波动；缺省 0 表示不限制。
        include_delisted: 是否保留在 ``as_of`` 之后才退市的证券。缺省 ``True``，
            这正是消除幸存者偏差的关键；置为 ``False`` 只保留至今仍在交易的证券，
            会重新引入偏差，仅供对照实验使用。

    返回：
        按代码升序排列、去重后的证券代码列表。
    """
    if instruments is None or instruments.empty:
        return []
    moment = pd.Timestamp(as_of)
    frame = instruments.copy()
    frame["code"] = frame["code"].astype(str)
    open_date = pd.to_datetime(frame.get("open_date"), errors="coerce")
    expire_date = pd.to_datetime(frame.get("expire_date"), errors="coerce")

    listed = open_date.notna() & (open_date <= moment)
    if min_listed_days > 0:
        listed &= open_date <= moment - timedelta(days=min_listed_days)
    alive = expire_date.isna() | (expire_date > moment)
    if not include_delisted:
        alive &= expire_date.isna()
    mask = listed & alive
    if board is not None:
        mask &= frame.get("board", pd.Series("", index=frame.index)).astype(str) == board
    if exclude_st:
        mask &= ~frame.get("is_st", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    return sorted(set(frame.loc[mask, "code"]))


def universe_over_window(
    instruments: pd.DataFrame,
    start: date,
    end: date,
    *,
    board: str | None = None,
    exclude_st: bool = False,
) -> list[str]:
    """筛出在给定窗口内任一时点上市过的证券代码。

    滚动回测应当用它而不是某一天的截面：只用窗口起点会漏掉期间新上市的证券，
    只用终点会漏掉期间退市的证券。

    参数：
        instruments: 证券静态信息表，列要求同 :func:`filter_universe`。
        start: 窗口起始日期，包含该日。
        end: 窗口结束日期，包含该日。
        board: 只保留该板块；``None`` 表示不限板块。
        exclude_st: 是否剔除风险警示证券，局限同 :func:`filter_universe`。

    返回：
        按代码升序排列、去重后的证券代码列表。
    """
    if instruments is None or instruments.empty:
        return []
    first = pd.Timestamp(start)
    last = pd.Timestamp(end)
    frame = instruments.copy()
    frame["code"] = frame["code"].astype(str)
    open_date = pd.to_datetime(frame.get("open_date"), errors="coerce")
    expire_date = pd.to_datetime(frame.get("expire_date"), errors="coerce")
    # 生命周期区间与查询窗口有交集即入池。
    mask = open_date.notna() & (open_date <= last)
    mask &= expire_date.isna() | (expire_date >= first)
    if board is not None:
        mask &= frame.get("board", pd.Series("", index=frame.index)).astype(str) == board
    if exclude_st:
        mask &= ~frame.get("is_st", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    return sorted(set(frame.loc[mask, "code"]))
