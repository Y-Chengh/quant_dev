from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

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


DAILY_REQUIRED_COLUMNS = {"code", "trade_date", "open", "high", "low", "close", "volume"}


def validate_daily_bars(daily: pd.DataFrame) -> pd.DataFrame:
    """校验并标准化日频行情，不修改调用方传入的数据。

    与 ``validate_bars`` 的分钟契约一一对应，区别只在于主键是
    ``(code, trade_date)`` 而不是 ``(code, trade_time)``，并且**不**伪造
    ``trade_time`` 列——日频数据源本来就没有日内时间，凭空造一个只会让下游
    误以为可以算分钟因子。

    参数：
        daily: 日频行情表，至少包含证券代码、交易日和开高低收量七列；
            允许携带 ``amount``、``pre_close``、``suspend_flag`` 等额外列。

    返回：
        按证券代码和交易日升序排列、索引重置后的新表；存在重复主键、
        非正价格、负成交量或最高最低价矛盾时抛出 ``ValueError``。
    """
    missing = DAILY_REQUIRED_COLUMNS.difference(daily.columns)
    if missing:
        raise ValueError(f"日频行情缺少字段: {sorted(missing)}")
    result = daily.copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"], errors="raise")
    result["code"] = result["code"].astype(str)
    if result.duplicated(["code", "trade_date"]).any():
        raise ValueError("日频行情存在重复的(code, trade_date)")
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
        # 日线库是原始落盘转换来的，出问题时必须能一眼定位到是哪只证券哪一天，
        # 否则只报一个总数根本无从排查。
        first = result.loc[invalid].iloc[0]
        raise ValueError(
            "日频行情包含{0}行非法OHLCV数据，首个问题行: code={1} trade_date={2}".format(
                int(invalid.sum()), first["code"], first["trade_date"]
            )
        )
    return result.sort_values(["code", "trade_date"]).reset_index(drop=True)


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
