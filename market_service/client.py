from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Sequence

import pandas as pd
import duckdb  # 明确暴露缺失依赖，避免被下面的兼容导入逻辑误判。

try:
    from .database import MarketDatabase
    from .models import KlineQuery, PagedBars, RawBarQuery
except ImportError:  # 支持直接从源码目录运行
    from database import MarketDatabase
    from models import KlineQuery, PagedBars, RawBarQuery


class MarketDataClient:
    """行情数据唯一公共入口；调用方无需了解 DuckDB 或 Parquet。"""

    def __init__(self, database_path: str | Path = r"D:\量化\market.duckdb") -> None:
        self._repository = MarketDatabase(database_path)

    def get_metadata(self) -> dict:
        return self._repository.metadata()

    def get_kline(self, query: KlineQuery) -> pd.DataFrame:
        self._validate_range(query.start, query.end)
        return self._repository.kline(query.code, query.start, query.end, query.period.value)

    def get_klines_5m(
        self,
        codes: Sequence[str],
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """为因子研究批量读取多个股票的5分钟行情。"""
        self._validate_range(start, end)
        normalized = list(dict.fromkeys(code.upper().strip() for code in codes if code.strip()))
        return self._repository.klines_5m(normalized, start, end)

    def get_snapshot(self, at: datetime, codes: Sequence[str] | None = None) -> pd.DataFrame:
        normalized = [code.upper().strip() for code in codes] if codes else None
        return self._repository.snapshot(at, normalized)

    def get_raw_bars(self, query: RawBarQuery) -> PagedBars:
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
