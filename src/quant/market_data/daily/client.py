"""QMT 日线数据的唯一公共入口。

与 ``market_data.client.MarketDataClient`` 保持同样的形状与命名习惯，调用方
无需了解 DuckDB、Parquet 或复权系数是怎么算出来的。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from . import adjust as adjust_module
from . import universe as universe_module
from .database import DailyMarketDatabase
from .models import AdjustMode, DailyKlineQuery


class DailyMarketClient:
    """QMT 日线行情的唯一公共入口。"""

    def __init__(self, database_path: str | Path | None = None) -> None:
        """绑定日线库文件。

        参数：
            database_path: ``qmt_daily.duckdb`` 文件路径；``None`` 表示按
                ``QMT_DAILY_DB_PATH`` 环境变量解析，未设置时使用内置回退路径。

        返回：
            无返回值。
        """
        self._repository = DailyMarketDatabase(database_path)

    def get_metadata(self) -> dict:
        """返回日线库的时间范围、行数与证券数。

        返回：
            与 5 分钟库形状对齐的元数据字典，含 ``first_time``、``last_time``、
            ``rows``、``symbols`` 以及日线特有的 ``adjust_anchor_date``。
        """
        return self._repository.metadata()

    def get_daily_bars(self, query: DailyKlineQuery) -> pd.DataFrame:
        """按查询条件读取日线，并按需复权。

        参数：
            query: 日线查询条件，见
                :class:`~quant.market_data.daily.models.DailyKlineQuery`。

        返回：
            按证券代码与交易日升序排列的日线表，附带 ``adjust_factor`` 列。
        """
        return self.get_klines_1d(
            query.codes,
            query.start,
            query.end,
            adjust=query.adjust,
            include_suspended=query.include_suspended,
            adjust_anchor=query.adjust_anchor,
        )

    def get_klines_1d(
        self,
        codes: Sequence[str],
        start: date,
        end: date,
        *,
        adjust: AdjustMode = AdjustMode.NONE,
        include_suspended: bool = False,
        adjust_anchor: date | None = None,
        adjust_volume: bool = False,
    ) -> pd.DataFrame:
        """为因子研究批量读取日线行情。

        刻意与 ``MarketDataClient.get_klines_5m`` 保持同样的调用形状，方便上层
        数据源在两种频率之间切换。

        参数：
            codes: 证券代码序列；空串会被忽略，六位沪深裸代码自动补全后缀，
                传空序列表示读取区间内的全市场行情。
            start: 起始交易日，包含该日。
            end: 结束交易日，包含该日。
            adjust: 复权口径。``NONE`` 返回原始不复权价；``HFQ`` 后复权，
                历史值不随之后新增分红改变，是研究推荐口径；``QFQ`` 前复权。
            include_suspended: 是否保留停牌行；缺省剔除，避免假零收益。
            adjust_anchor: 前复权基准日；``None`` 时取库中记录的
                ``adjust_anchor_date``，两者都没有则报错。
            adjust_volume: 是否同步反向调整成交量；缺省保留原始股数。

        返回：
            按证券代码与交易日升序排列的日线表，含 ``adjust_factor`` 列。
        """
        if start > end:
            raise ValueError("开始日期不能晚于结束日期")
        bars = self._repository.daily_bars(
            codes, start, end, include_suspended=include_suspended
        )
        if adjust == AdjustMode.NONE:
            result = bars.copy()
            result["adjust_factor"] = 1.0 if not result.empty else pd.Series(dtype=float)
            return result
        anchor = adjust_anchor
        if adjust == AdjustMode.QFQ and anchor is None:
            anchor = self._default_anchor()
        # 复权系数从上市起累乘，因此必须取全历史除权记录，只按上界截断。
        # 前复权还要把基准日之前的事件全部取到：基准日晚于查询窗口时若只截到
        # end，实际基准会退化成 min(anchor, end)，同一段历史就会随窗口变化。
        actions_end = end if anchor is None else max(end, anchor)
        actions = self._repository.corporate_actions(codes, end=actions_end)
        return adjust_module.apply_adjustment(
            bars,
            actions,
            mode=adjust,
            anchor=anchor,
            adjust_volume=adjust_volume,
        )

    def get_corporate_actions(
        self,
        codes: Sequence[str] = (),
        end: date | None = None,
    ) -> pd.DataFrame:
        """读取除权送转记录。

        参数：
            codes: 证券代码序列；空序列表示全部证券。
            end: 只取除权日不晚于该日的记录；``None`` 表示不限上界。

        返回：
            按证券代码与除权日升序排列的除权表。
        """
        return self._repository.corporate_actions(codes, end=end)

    def get_trading_calendar(
        self,
        start: date | None = None,
        end: date | None = None,
    ) -> list[date]:
        """读取交易日历。

        参数：
            start: 起始日期，包含该日；``None`` 表示不限下界。
            end: 结束日期，包含该日；``None`` 表示不限上界。

        返回：
            升序排列的交易日列表。
        """
        return self._repository.trading_dates(start, end)

    def get_instruments(self, codes: Sequence[str] = ()) -> pd.DataFrame:
        """读取证券静态信息。

        参数：
            codes: 证券代码序列；空序列表示全部证券。

        返回：
            含上市日、退市日、板块与风险警示标记的证券信息表。
        """
        return self._repository.instruments(codes)

    def list_universe(
        self,
        as_of: date,
        *,
        board: str | None = None,
        exclude_st: bool = False,
        min_listed_days: int = 0,
        include_delisted: bool = True,
        limit: int | None = None,
    ) -> list[str]:
        """返回某一天的无幸存者偏差证券池。

        参数：
            as_of: 目标日期，取该日仍在存续期内的证券。
            board: 只保留该板块；``None`` 表示不限板块。
            exclude_st: 是否剔除风险警示证券；标记来自当前名称快照，
                无法还原历史状态。
            min_listed_days: 要求上市满多少个自然日。
            include_delisted: 是否保留 ``as_of`` 之后才退市的证券；缺省保留，
                这是消除幸存者偏差的关键。
            limit: 返回条数上限；``None`` 表示不截断。

        返回：
            按代码升序排列的证券代码列表。
        """
        codes = universe_module.filter_universe(
            self._repository.instruments(),
            as_of,
            board=board,
            exclude_st=exclude_st,
            min_listed_days=min_listed_days,
            include_delisted=include_delisted,
        )
        return codes if limit is None else codes[:limit]

    def list_universe_over_window(
        self,
        start: date,
        end: date,
        *,
        board: str | None = None,
        exclude_st: bool = False,
        limit: int | None = None,
    ) -> list[str]:
        """返回窗口内任一时点上市过的证券池。

        参数：
            start: 窗口起始日期，包含该日。
            end: 窗口结束日期，包含该日。
            board: 只保留该板块；``None`` 表示不限板块。
            exclude_st: 是否剔除风险警示证券。
            limit: 返回条数上限；``None`` 表示不截断。

        返回：
            按代码升序排列的证券代码列表。
        """
        codes = universe_module.universe_over_window(
            self._repository.instruments(),
            start,
            end,
            board=board,
            exclude_st=exclude_st,
        )
        return codes if limit is None else codes[:limit]

    def search_symbols(self, text: str = "", limit: int = 20) -> list[str]:
        """按代码子串搜索库中有行情的证券。

        只反映「库里有没有这只证券的行情」，**不是**无幸存者偏差的证券池；
        研究选股请改用 :meth:`list_universe` 或 :meth:`list_universe_over_window`。

        参数：
            text: 代码子串，大小写不敏感；空串表示不过滤。
            limit: 返回条数上限。

        返回：
            按代码升序排列的证券代码列表。
        """
        return self._repository.symbols(text, limit)

    def _default_anchor(self) -> date:
        """读取库中记录的前复权缺省基准日。

        返回：
            ``dataset_metadata.adjust_anchor_date`` 对应的日期；未记录时抛出
            ``ValueError``，提醒调用方显式指定，避免静默使用一个会随同步漂移的基准。
        """
        recorded = self.get_metadata().get("adjust_anchor_date", "")
        if not recorded:
            raise ValueError(
                "日线库没有记录前复权基准日；请显式传入 adjust_anchor 或先完成一次同步"
            )
        return pd.Timestamp(recorded).date()
