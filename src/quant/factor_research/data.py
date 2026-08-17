from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
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
            可选的 ``adjust_factor`` 列会被一并校验：它是复权口径乘到价格上的
            那个正系数，取值必须有限且大于零。

    返回：
        按证券代码和交易日升序排列、索引重置后的新表；存在重复主键、
        非正价格、负成交量、最高最低价矛盾或非法复权系数时抛出 ``ValueError``。
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
    if "adjust_factor" in result.columns:
        # 复权系数会作为基础列一路传到因子层，``close / adjust_factor`` 必须能
        # 还原原始不复权价，因此这里宁可报错也不把缺失或非正的系数悄悄替换成
        # 1.0——那等于让这几行偷偷换成「未复权」口径，而且完全不可见。
        # 必须落到 numpy ``float64``：``pd.to_numeric`` 会保留可空扩展 dtype
        # （``Float64``），此时 ``np.isfinite`` 返回带 ``pd.NA`` 的 BooleanArray，
        # ``any()`` 默认 skipna 会把缺失位当 False 跳过，缺失系数就被静默放行。
        # ``astype`` 顺带把 ``pd.NA`` 归一成 ``np.nan``，并让写回列的 dtype 恒定，
        # 使 ``_daily_input_fingerprint`` 的 dtype 串不随输入 backend 变化。
        factor = pd.to_numeric(result["adjust_factor"], errors="coerce").astype("float64")
        bad_factor = ~np.isfinite(factor) | (factor <= 0)
        if bad_factor.any():
            first = result.loc[bad_factor].iloc[0]
            raise ValueError(
                "日频行情包含{0}行非法复权系数（必须有限且为正），"
                "首个问题行: code={1} trade_date={2} adjust_factor={3!r}".format(
                    int(bad_factor.sum()),
                    first["code"],
                    first["trade_date"],
                    first["adjust_factor"],
                )
            )
        result["adjust_factor"] = factor
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
