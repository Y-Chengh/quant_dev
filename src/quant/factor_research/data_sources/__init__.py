"""可切换的行情数据源。

主实验通过 ``--data-source`` 选择数据源，自身不为任何具体数据源写分支——
与模型工厂的约定完全一致。

各子模块职责：

- ``base``：``MarketDataSource`` 抽象与频率常量。
- ``registry``：注册表、自动发现与命令行装配。
- ``market_service``：本地 5 分钟行情库，项目原有的默认行为。
- ``qmt_daily``：大 QMT 日线库，带复权口径与自动增量同步。

对外只暴露抽象与注册表入口；具体数据源由注册表按名字装配，调用方不直接导入。
"""

from __future__ import annotations

from .base import FREQUENCY_DAILY, FREQUENCY_INTRADAY, MarketDataSource
from .registry import (
    DEFAULT_DATA_SOURCE,
    add_data_source_selection_argument,
    add_selected_data_source_arguments,
    available_data_sources,
    data_source_argument_names,
    data_source_from_args,
    register_data_source,
)

__all__ = [
    "DEFAULT_DATA_SOURCE",
    "FREQUENCY_DAILY",
    "FREQUENCY_INTRADAY",
    "MarketDataSource",
    "add_data_source_selection_argument",
    "add_selected_data_source_arguments",
    "available_data_sources",
    "data_source_argument_names",
    "data_source_from_args",
    "register_data_source",
]
