"""目录库的视图、库存表与运行记录刷新。

对 Parquet 的全量聚合一律在**内存连接**上完成，只把结果 ``DataFrame`` 注册进
目录库连接后写表，从而把日线库文件的写锁窗口压到亚秒级。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from ..schema import bars_parquet_glob, create_view_statement
from .shards import existing_shards

#: 月度库存与证券库存两张表的空表结构，供没有任何分片时使用。
_EMPTY_INVENTORY = pd.DataFrame(
    {
        "year": pd.Series(dtype="int64"),
        "month": pd.Series(dtype="int64"),
        "rows": pd.Series(dtype="int64"),
        "symbols": pd.Series(dtype="int64"),
        "trading_days": pd.Series(dtype="int64"),
        "min_date": pd.Series(dtype="datetime64[ns]"),
        "max_date": pd.Series(dtype="datetime64[ns]"),
    }
)
_EMPTY_SYMBOLS = pd.DataFrame(
    {
        "code": pd.Series(dtype="object"),
        "first_date": pd.Series(dtype="datetime64[ns]"),
        "last_date": pd.Series(dtype="datetime64[ns]"),
        "bar_count": pd.Series(dtype="int64"),
        "suspended_days": pd.Series(dtype="int64"),
    }
)


def collect_inventory(memory_connection, daily_root: Path | str) -> pd.DataFrame:
    """在内存连接上统计每个月度分片的行数、证券数与日期范围。

    用 hive 分区列一次聚合出全部月份，而不是逐个分片查询：实测一千六百万行的
    全量扫描不到一秒，比开三百多次查询快得多。

    参数：
        memory_connection: 内存 DuckDB 连接。
        daily_root: 日线库数据根目录。

    返回：
        与 ``monthly_inventory`` 表同结构的 ``DataFrame``；没有分片时返回空表。
    """
    if not existing_shards(daily_root):
        return _EMPTY_INVENTORY.copy()
    return memory_connection.execute(
        f"""
        SELECT CAST(year AS BIGINT) AS year, CAST(month AS BIGINT) AS month,
               CAST(count(*) AS BIGINT) AS rows,
               CAST(count(DISTINCT code) AS BIGINT) AS symbols,
               CAST(count(DISTINCT trade_date) AS BIGINT) AS trading_days,
               min(trade_date) AS min_date, max(trade_date) AS max_date
        FROM read_parquet('{bars_parquet_glob(daily_root)}', hive_partitioning=true, union_by_name=true)
        GROUP BY year, month ORDER BY year, month
        """
    ).df()


def collect_symbols(memory_connection, daily_root: Path | str) -> pd.DataFrame:
    """在内存连接上统计每只证券的首末交易日、行数与停牌天数。

    参数：
        memory_connection: 内存 DuckDB 连接。
        daily_root: 日线库数据根目录。

    返回：
        与 ``symbols`` 表同结构的 ``DataFrame``；没有分片时返回空表。
    """
    if not existing_shards(daily_root):
        return _EMPTY_SYMBOLS.copy()
    return memory_connection.execute(
        f"""
        SELECT code,
               min(trade_date) AS first_date,
               max(trade_date) AS last_date,
               CAST(count(*) AS BIGINT) AS bar_count,
               CAST(sum(CASE WHEN suspend_flag <> 0 THEN 1 ELSE 0 END) AS BIGINT) AS suspended_days
        FROM read_parquet('{bars_parquet_glob(daily_root)}', hive_partitioning=true, union_by_name=true)
        GROUP BY code ORDER BY code
        """
    ).df()


def refresh_catalog(
    connection,
    daily_root: Path | str,
    inventory: pd.DataFrame,
    symbols: pd.DataFrame,
) -> None:
    """把预先算好的库存结果写进目录库，并重建 ``bars_1d`` 视图。

    参数：
        connection: 以读写方式打开的目录库连接，调用方负责事务边界。
        daily_root: 日线库数据根目录，用于拼视图的 Parquet 通配路径。
        inventory: ``collect_inventory`` 的返回值。
        symbols: ``collect_symbols`` 的返回值。

    返回：
        无返回值。没有任何月度分片时会丢弃 ``bars_1d`` 视图，避免留下一个
        指向空通配、一查就报错的视图。
    """
    connection.register("inventory_snapshot", inventory)
    connection.register("symbols_snapshot", symbols)
    try:
        connection.execute("DELETE FROM monthly_inventory")
        connection.execute(
            """
            INSERT INTO monthly_inventory(year, month, rows, symbols, trading_days, min_date, max_date)
            SELECT CAST(year AS INTEGER), CAST(month AS INTEGER), rows, symbols, trading_days,
                   CAST(min_date AS DATE), CAST(max_date AS DATE)
            FROM inventory_snapshot
            """
        )
        connection.execute("DELETE FROM symbols")
        connection.execute(
            """
            INSERT INTO symbols(code, first_date, last_date, bar_count, suspended_days)
            SELECT code, CAST(first_date AS DATE), CAST(last_date AS DATE),
                   bar_count, suspended_days
            FROM symbols_snapshot
            """
        )
    finally:
        connection.unregister("inventory_snapshot")
        connection.unregister("symbols_snapshot")
    if existing_shards(daily_root):
        connection.execute(create_view_statement(daily_root))
    else:
        connection.execute("DROP VIEW IF EXISTS bars_1d")


def record_sync_run(connection, values: dict) -> None:
    """把一次同步的执行情况写入 ``sync_runs``。

    参数：
        connection: 以读写方式打开的目录库连接。
        values: 至少包含 ``run_id``、``started_at``、``finished_at``、``mode``、
            ``source_root``、``status``、``scanned``、``dirty``、``removed``、
            ``rewritten_shards``、``rows_after``、``message`` 十二个键的字典。

    返回：
        无返回值。
    """
    connection.execute(
        """
        INSERT OR REPLACE INTO sync_runs(
            run_id, started_at, finished_at, mode, source_root, status,
            scanned, dirty, removed, rewritten_shards, rows_after, message)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            values["run_id"],
            values["started_at"],
            values["finished_at"],
            values["mode"],
            values["source_root"],
            values["status"],
            values["scanned"],
            values["dirty"],
            values["removed"],
            values["rewritten_shards"],
            values["rows_after"],
            values["message"],
        ],
    )


def make_run_id(started_at: datetime) -> str:
    """按开始时间生成本次同步的运行标识。

    参数：
        started_at: 同步开始时间。

    返回：
        形如 ``sync_20260815_183045_123456`` 的稳定标识。
    """
    return "sync_" + started_at.strftime("%Y%m%d_%H%M%S_%f")
