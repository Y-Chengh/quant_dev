from __future__ import annotations

from pathlib import Path
from typing import Sequence
import logging

import pandas as pd

from .timing import log_elapsed


logger = logging.getLogger(__name__)


REQUIRED_COLUMNS = {"code", "trade_time", "open", "high", "low", "close", "volume"}


def validate_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """校验并标准化5分钟行情，不修改调用方传入的数据。"""
    missing = REQUIRED_COLUMNS.difference(bars.columns)
    if missing:
        raise ValueError(f"行情缺少字段: {sorted(missing)}")
    result = bars.copy()
    result["trade_time"] = pd.to_datetime(result["trade_time"], errors="raise")
    result["code"] = result["code"].astype(str)
    if result.duplicated(["code", "trade_time"]).any():
        raise ValueError("行情存在重复的(code, trade_time)")
    numeric = ["open", "high", "low", "close", "volume"]
    result[numeric] = result[numeric].apply(pd.to_numeric, errors="coerce")
    invalid = (
        result[numeric].isna().any(axis=1)
        | (result[["open", "high", "low", "close"]] <= 0).any(axis=1)
        | (result["volume"] < 0)
        | (result["high"] < result[["open", "close", "low"]].max(axis=1))
        | (result["low"] > result[["open", "close", "high"]].min(axis=1))
    )
    if invalid.any():
        raise ValueError(f"行情包含{int(invalid.sum())}行非法OHLCV数据")
    result["trade_date"] = result["trade_time"].dt.normalize()
    return result.sort_values(["code", "trade_time"]).reset_index(drop=True)


def load_parquet(path: str | Path, codes: Sequence[str] | None = None) -> pd.DataFrame:
    """读取单个Parquet文件；适合样例和小规模研究。"""
    bars = pd.read_parquet(path)
    if codes:
        bars = bars[bars["code"].astype(str).isin(map(str, codes))]
    return validate_bars(bars)


def load_duckdb(
    database: str | Path,
    start: str,
    end: str,
    codes: Sequence[str] | None = None,
) -> pd.DataFrame:
    """从现有bars_5m视图按区间加载行情，duckdb为可选依赖。"""
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("使用DuckDB数据源前请安装duckdb") from exc

    clauses = ["trade_time >= ?", "trade_time < ?"]
    params: list[object] = [start, end]
    if codes:
        clauses.append("code IN (" + ",".join("?" for _ in codes) + ")")
        params.extend(map(str, codes))
    query = f"""
        SELECT code, trade_time, open, high, low, close, volume, amount
        FROM bars_5m WHERE {' AND '.join(clauses)}
        ORDER BY code, trade_time
    """
    connection = duckdb.connect(str(database), read_only=True)
    try:
        bars = connection.execute(query, params).df()
    finally:
        connection.close()
    return validate_bars(bars)


@log_elapsed(logger, "分钟行情加载")
def load_market_service(
    client: object,
    codes: Sequence[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    """通过MarketDataClient公共接口读取研究行情。"""
    start_time = pd.Timestamp(start).to_pydatetime()
    end_time = pd.Timestamp(end).to_pydatetime()
    get_klines = getattr(client, "get_klines_5m", None)
    if get_klines is None:
        raise TypeError("client必须提供get_klines_5m(codes, start, end)接口")
    return validate_bars(get_klines(codes, start_time, end_time))
