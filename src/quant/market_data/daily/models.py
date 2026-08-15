"""日线行情查询所用的复权口径枚举与查询条件数据类。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from enum import StrEnum


class AdjustMode(StrEnum):
    """日线价格的复权口径。

    ``NONE`` 直接返回入库的原始不复权价；``HFQ`` 为后复权，历史值不随新的分红
    送转改变，是因子研究的推荐口径；``QFQ`` 为前复权，每次出现新的除权事件都会
    重算全部历史价格，只适合看图，用于研究时必须显式固定基准日。
    """

    NONE = "none"
    QFQ = "qfq"
    HFQ = "hfq"


@dataclass(frozen=True, slots=True)
class DailyKlineQuery:
    """一次日线查询的全部条件。

    参数：
        codes: 证券代码序列；六位沪深裸代码会自动补全交易所后缀，空串会被忽略。
        start: 查询起始交易日，包含该日。
        end: 查询结束交易日，包含该日。
        adjust: 复权口径，取值见 :class:`AdjustMode`，缺省为不复权。
        include_suspended: 是否保留停牌日。停牌行的开高低收全等于前收、成交量为
            零，留在因子输入里会造出成片的假零收益，因此缺省为 ``False``。
        adjust_anchor: 前复权基准日；只有 ``adjust`` 为 ``QFQ`` 时有意义，
            ``None`` 表示使用库中记录的 ``adjust_anchor_date``。
    """

    codes: Sequence[str]
    start: date
    end: date
    adjust: AdjustMode = AdjustMode.NONE
    include_suspended: bool = False
    adjust_anchor: date | None = None
