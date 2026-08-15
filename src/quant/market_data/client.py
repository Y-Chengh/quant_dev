from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import duckdb  # noqa: F401  # 提前暴露缺失依赖，让报错指向 duckdb 而非下游调用
import pandas as pd

from .codes import normalize_security_code
from .database import MarketDatabase
from .models import KlineQuery, PagedBars, RawBarQuery


class MarketDataClient:
    """行情数据唯一公共入口；调用方无需了解 DuckDB 或 Parquet。"""

    def __init__(self, database_path: str | Path | None = None) -> None:
        """绑定行情库文件。

        参数：
            database_path: ``market.duckdb`` 文件路径；``None`` 表示按
                ``MARKET_DB_PATH`` 环境变量解析，未设置时使用内置回退路径。

        返回：
            无返回值。
        """
        self._repository = MarketDatabase(database_path)

    def get_metadata(self) -> dict:
        return self._repository.metadata()

    def get_kline(self, query: KlineQuery) -> pd.DataFrame:
        """读取单只证券的指定周期K线。

        参数：
            query: K线查询条件，包含证券代码、闭区间起止时间和目标周期；六位
                沪深裸代码由仓储层自动补全交易所后缀。

        返回：
            按时间升序排列的K线数据表，包含涨跌额、涨跌幅和日内涨跌幅字段。
        """
        self._validate_range(query.start, query.end)
        return self._repository.kline(query.code, query.start, query.end, query.period.value)

    def get_klines_5m(
        self,
        codes: Sequence[str],
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """为因子研究批量读取多个证券的5分钟行情。

        参数：
            codes: 证券代码序列；会去除空值与重复值，六位沪深裸代码会自动
                补全交易所后缀。
            start: 查询起始时间，包含该时刻。
            end: 查询结束时间，包含该时刻。

        返回：
            按证券代码和时间升序排列的原始5分钟行情数据表。
        """
        self._validate_range(start, end)
        normalized = list(dict.fromkeys(
            normalize_security_code(code) for code in codes if code.strip()
        ))
        return self._repository.klines_5m(normalized, start, end)

    def get_snapshot(self, at: datetime, codes: Sequence[str] | None = None) -> pd.DataFrame:
        """读取指定时刻的全市场或指定证券快照。

        参数：
            at: 快照对应的行情时刻，需与数据库分钟时间点一致。
            codes: 可选证券代码序列；六位沪深裸代码会自动补全后缀，缺省时
                返回该时刻的全部证券。

        返回：
            按证券代码排序的行情快照数据表。
        """
        normalized = [normalize_security_code(code) for code in codes] if codes else None
        return self._repository.snapshot(at, normalized)

    def get_raw_bars(self, query: RawBarQuery) -> PagedBars:
        """分页读取单只证券在一个交易日内的原始5分钟行情。

        参数：
            query: 原始行情查询条件，包含证券代码、交易日、可选价格/成交过滤
                条件及分页参数；六位沪深裸代码由仓储层自动补全后缀。

        返回：
            包含当前页数据、总记录数、页码和页大小的分页结果。
        """
        if query.page < 1 or not 1 <= query.page_size <= 1000:
            raise ValueError("page必须大于0，page_size必须在1到1000之间")
        frame, total = self._repository.raw(
            query.code, query.trade_date,
            start_time=query.start_time, end_time=query.end_time,
            min_close=query.min_close, max_close=query.max_close,
            min_volume=query.min_volume, max_volume=query.max_volume,
            min_amount=query.min_amount, max_amount=query.max_amount,
            page=query.page, page_size=query.page_size,
        )
        return PagedBars(frame, total, query.page, query.page_size)

    def search_symbols(self, text: str = "", limit: int = 20) -> list[str]:
        # if not 1 <= limit <= 100:
        #     raise ValueError("limit必须在1到100之间")
        return self._repository.symbols(text, limit)

    @staticmethod
    def _validate_range(start: datetime, end: datetime) -> None:
        if start > end:
            raise ValueError("开始时间不能晚于结束时间")
