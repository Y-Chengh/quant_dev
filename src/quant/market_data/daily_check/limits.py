"""A 股各板块涨跌停幅度的口径推导。

**核心局限与由此决定的口径**：``is_st`` 来自 ``instrument_info`` 的**当前**名称
快照，无法还原历史某一天的风险警示状态——今天是 ST 的股票 2019 年未必是，反之
亦然。实测按当前 ST 状态套用 5% 限制，2024 一年就会产生 3860 条误报，而按板块
基础幅度只剩 36 条。

因此默认**不按 ST 收紧**，只校验「涨跌幅有没有超过该板块可能的最大幅度」。这是
一个必要条件而非精确判定：它不会漏掉真正不可能的行情，也几乎不误伤。确实想按
当前 ST 状态收紧的使用者可以显式打开 ``apply_st_limit``，并自行承担误报。
"""

from __future__ import annotations

from datetime import date

from ..daily.schema import BOARD_BSE, BOARD_CHINEXT, BOARD_MAIN, BOARD_STAR

#: 科创板开市日；此前不存在 688/689 代码。
_STAR_START = date(2019, 7, 22)

#: 创业板注册制改革日，涨跌停幅度由 10% 放宽到 20%。
_CHINEXT_WIDENED = date(2020, 8, 24)

#: 注册制新股上市后不设涨跌停的**交易日**数（含上市首日）。
_NEW_LISTING_FREE_SESSIONS = 5


def resolve_limit_ratio(
    board: str,
    is_st: bool,
    trade_date: date,
    sessions_since_listing: int | None,
    *,
    apply_st_limit: bool = False,
) -> float | None:
    """推导某只证券在某个交易日的涨跌停幅度。

    参数：
        board: 板块标识，取值为 ``main``/``star``/``chinext``/``bse``/``unknown``。
        is_st: 是否处于风险警示或退市整理状态；只反映**当前**名称快照，
            仅在 ``apply_st_limit`` 为 ``True`` 时才会被采纳。
        trade_date: 目标交易日，用于套用当日生效的监管口径。
        sessions_since_listing: 距上市日已经过去的**交易日**数，上市首日为 0；
            ``None`` 表示上市日未知，此时不做新股豁免。必须按交易日而不是自然日
            计数——遇上国庆长假时，5 个交易日会跨越 13 个自然日，按自然日算会
            把仍处于豁免期的新股误判为涨跌停越界。
        apply_st_limit: 是否按当前 ST 状态把主板与老创业板收紧到 5%。缺省
            ``False``：历史 ST 状态不可知，收紧会产生大量误报（实测 2024 年
            3860 条对 36 条）。

    返回：
        以小数表示的单边幅度，例如 ``0.10``；当日不设涨跌停、板块无法归类或
        处于新股豁免期时返回 ``None``，调用方应跳过该行。
    """
    tightened = 0.05 if (apply_st_limit and is_st) else None
    listing_day = sessions_since_listing == 0
    new_listing = (
        sessions_since_listing is not None
        and 0 <= sessions_since_listing < _NEW_LISTING_FREE_SESSIONS
    )
    if board == BOARD_STAR:
        if trade_date < _STAR_START or new_listing:
            return None
        return 0.20
    if board == BOARD_CHINEXT:
        if trade_date < _CHINEXT_WIDENED:
            return None if listing_day else (tightened or 0.10)
        return None if new_listing else 0.20
    if board == BOARD_BSE:
        return None if listing_day else 0.30
    if board == BOARD_MAIN:
        return None if listing_day else (tightened or 0.10)
    return None
