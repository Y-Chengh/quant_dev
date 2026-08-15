"""各审计 mixin 共享的实例状态契约。

按 AGENTS.md 的要求，跨 mixin 共享的实例属性集中在这里逐项声明业务含义。
``market_data`` 下允许使用变量注解（与 ``qmt_downloader`` 不同）。
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from quant.qmt_downloader.self_check import AuditIssue

from .config import DailyCheckConfig


class _DailyCheckerState:
    """日线库审计器在各 mixin 之间共享的可变状态。"""

    #: 本次审计的配置与阈值。
    config: DailyCheckConfig

    #: 累积的全部问题记录，按发现顺序追加。
    issues: list[AuditIssue]

    #: 审计区间起点，取自配置或库中最早交易日。
    start_date: date

    #: 审计区间终点，取自配置或库中最晚交易日。
    end_date: date

    #: 审计区间内的交易日历，升序排列；日历表为空时回落为库中出现过的日期。
    calendar: tuple[date, ...]

    #: 交易日历是否来自 ``trading_calendar`` 表。为 ``False`` 说明是从行情反推的，
    #: 此时无法发现「整天缺失」，必须在报告里显著提示。
    calendar_is_authoritative: bool

    #: 证券静态信息，以 ``code`` 为索引，含上市日、退市日、板块与风险警示标记。
    instruments: pd.DataFrame

    #: 本次审计涉及的证券代码，升序排列；配置未限定时为库中全部证券。
    codes: tuple[str, ...]

    #: 逐日覆盖率统计，列为 ``trade_date``/``expected``/``actual``/``coverage``。
    coverage_by_date: pd.DataFrame

    #: 逐证券覆盖率统计，列为 ``code``/``expected``/``actual``/``missing``/``coverage``。
    coverage_by_symbol: pd.DataFrame

    #: 连续缺失区间，列为 ``code``/``start_date``/``end_date``/``days``。
    missing_spans: pd.DataFrame

    #: 除权因子自洽性明细，列为 ``code``/``trade_date``/``observed``/``expected``/``deviation``。
    adjust_audit: pd.DataFrame

    #: 涨跌停越界明细，列为 ``code``/``trade_date``/``pct_change``/``limit_ratio``。
    price_limit_violations: pd.DataFrame

    #: 5 分钟聚合与日线的差异明细；未开启交叉校验时为空表。
    cross_check_diff: pd.DataFrame
