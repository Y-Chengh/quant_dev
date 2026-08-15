"""对入库后的日线库执行全样本合法性审计。

与 ``quant.qmt_downloader.self_check`` 的分工：那一侧审计大 QMT 落盘的 CSV 分区
本身是否完整，这一侧审计**转换入库之后**的数据在业务上是否合法，因此能发现转换
过程引入的问题，也能与 5 分钟库交叉对账。两侧共用同一套问题记录结构与报告列，
输出可以直接拼在一起看。

各子模块职责：

- ``config``：审计配置、阈值与报告目录解析。
- ``base``：各 mixin 共享的实例状态契约。
- ``loading``：审计区间、交易日历与证券生命周期的装载。
- ``rules_integrity``：主键、空值、价格关系等行级完整性规则。
- ``rules_volume``：停牌与成交的一致性，以及成交量单位标定。
- ``rules_coverage``：覆盖率统计与上市至退市之间的连续缺失区间。
- ``rules_prices``：前收连续性、除权因子自洽性与涨跌停校验。
- ``limits``：各板块涨跌停幅度的口径推导。
- ``cross_5m``：与 5 分钟行情库的聚合交叉校验。
- ``summary``：结果汇总与报告落盘。
- ``checker``：编排层。

对外只暴露配置与入口函数；具体规则实现由使用方从对应子模块直接导入。
"""

from __future__ import annotations

from .checker import DailyMarketChecker, run_daily_market_check
from .config import PRICE_LIMIT_LEVELS, DailyCheckConfig

__all__ = [
    "PRICE_LIMIT_LEVELS",
    "DailyCheckConfig",
    "DailyMarketChecker",
    "run_daily_market_check",
]
