from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path

import duckdb
import pandas as pd

from quant.config import default_market_database

from .codes import normalize_security_code

PERIODS = {
    "5m": None,
    "15m": "15 minutes",
    "30m": "30 minutes",
    "60m": "60 minutes",
    "1d": "1 day",
    "1w": "1 week",
    "1mo": "1 month",
}


class MarketDatabase:
    def __init__(self, path: str | Path | None = None) -> None:
        """绑定只读的 DuckDB 行情库文件。

        参数：
            path: ``market.duckdb`` 文件路径；``None`` 表示按
                ``MARKET_DB_PATH`` 环境变量解析，未设置时使用内置回退路径。

        返回：
            无返回值；连接在 ``connect()`` 中按需建立。
        """
        self.path = Path(path) if path is not None else default_market_database()

    @contextmanager
    def connect(self) -> Iterator[duckdb.DuckDBPyConnection]:
        con = duckdb.connect(str(self.path), read_only=True)
        try:
            con.execute("SET threads = 4")
            yield con
        finally:
            con.close()

    def metadata(self) -> dict:
        with self.connect() as con:
            first, last, rows = con.execute(
                "SELECT min(min_time), max(max_time), sum(row_count)::BIGINT FROM monthly_inventory"
            ).fetchone()
            symbols = con.execute("SELECT count(*) FROM symbols").fetchone()[0]
        return {
            "first_time": first.isoformat(sep=" "),
            "last_time": last.isoformat(sep=" "),
            "rows": rows,
            "symbols": symbols,
            "periods": list(PERIODS),
        }

    def kline(self, code: str, start: datetime, end: datetime, period: str) -> pd.DataFrame:
        """读取单只证券K线，并按相邻有效K线收盘价计算涨跌额和涨跌幅。

        参数：
            code: 证券代码；查询前会去除首尾空白并转为大写，六位沪深裸代码
                会自动补全交易所后缀。
            start: 查询起始时间，包含该时刻。
            end: 查询结束时间，包含该时刻。
            period: K线周期，可选值为 ``5m``、``15m``、``30m``、``60m``、
                ``1d``、``1w`` 或 ``1mo``。

        返回：
            按时间升序排列的K线；``pre_close`` 为上一根有效K线收盘价，
            ``change`` 和 ``pct_change`` 分别为涨跌额和百分比涨跌幅，
            ``intraday_pct_change`` 为本根K线开盘至收盘的百分比涨跌幅。查询区间
            首根K线会向前查找最近收盘价，若不存在则前收相关派生字段为空。
        """
        if period not in PERIODS:
            raise ValueError(f"不支持的周期：{period}")
        code = normalize_security_code(code)
        with self.connect() as con:
            if period == "5m":
                frame = con.execute(
                    """
                    SELECT trade_time AS time, open, high, low, close, volume, amount
                    FROM bars_5m
                    WHERE code = ? AND trade_time BETWEEN ? AND ?
                    ORDER BY trade_time
                    """,
                    [code, start, end],
                ).df()
            else:
                bucket = PERIODS[period]
                frame = con.execute(
                    f"""
                    WITH source AS (
                        SELECT *, time_bucket(INTERVAL '{bucket}', trade_time) AS bucket
                        FROM bars_5m
                        WHERE code = ?
                              AND trade_time BETWEEN
                                  time_bucket(INTERVAL '{bucket}', CAST(? AS TIMESTAMP)) AND ?
                    )
                    SELECT bucket AS time,
                           arg_min(open, trade_time) AS open,
                           max(high) AS high,
                           min(low) AS low,
                           arg_max(close, trade_time) AS close,
                           sum(volume)::BIGINT AS volume,
                           sum(amount) AS amount
                    FROM source GROUP BY bucket ORDER BY bucket
                    """,
                    [code, start, end],
                ).df()
            previous = None
            if not frame.empty:
                first_bucket = frame.iloc[0]["time"]
                if period == "5m":
                    previous = con.execute(
                        """
                        SELECT close
                        FROM bars_5m
                        WHERE code = ? AND trade_time < ?
                              AND close IS NOT NULL AND isfinite(close) AND close <> 0
                        ORDER BY trade_time DESC
                        LIMIT 1
                        """,
                        [code, first_bucket],
                    ).fetchone()
                else:
                    previous = con.execute(
                        f"""
                        WITH aggregated AS (
                            SELECT time_bucket(INTERVAL '{bucket}', trade_time) AS bucket,
                                   arg_max(close, trade_time) AS close
                            FROM bars_5m
                            WHERE code = ? AND trade_time < ?
                            GROUP BY bucket
                        )
                        SELECT close
                        FROM aggregated
                        WHERE close IS NOT NULL AND isfinite(close) AND close <> 0
                        ORDER BY bucket DESC
                        LIMIT 1
                        """,
                        [code, first_bucket],
                    ).fetchone()

        close = pd.to_numeric(frame["close"], errors="coerce")
        valid_close = close.notna() & close.abs().lt(float("inf")) & close.ne(0)
        frame["pre_close"] = close.where(valid_close).ffill().shift(1)
        if previous is not None and not frame.empty:
            frame.at[frame.index[0], "pre_close"] = previous[0]
            frame["pre_close"] = frame["pre_close"].ffill()
        pre_close = pd.to_numeric(frame["pre_close"], errors="coerce")
        valid = (
            valid_close
            & pre_close.notna()
            & pre_close.abs().lt(float("inf"))
            & pre_close.ne(0)
        )
        change = (close - pre_close).where(valid)
        frame["change"] = change
        frame["pct_change"] = (change / pre_close * 100).where(valid)
        open_price = pd.to_numeric(frame["open"], errors="coerce")
        valid_open = open_price.notna() & open_price.abs().lt(float("inf")) & open_price.ne(0)
        frame["intraday_pct_change"] = ((close / open_price - 1) * 100).where(
            valid_close & valid_open
        )
        return frame

    def klines_5m(self, codes: Sequence[str], start: datetime, end: datetime) -> pd.DataFrame:
        """批量读取研究所需的原始5分钟行情。

        参数：
            codes: 证券代码序列；忽略空字符串，六位沪深裸代码自动补全后缀。
            start: 查询起始时间，包含该时刻。
            end: 查询结束时间，包含该时刻。

        返回：
            按证券代码和时间升序排列的5分钟行情；代码序列为空时返回具有标准
            行情列的空数据表。
        """
        normalized = [normalize_security_code(code) for code in codes if code.strip()]
        if not normalized:
            return pd.DataFrame(
                columns=["code", "trade_time", "open", "high", "low", "close", "volume", "amount"]
            )
        placeholders = ",".join("?" for _ in normalized)
        with self.connect() as con:
            return con.execute(
                f"""
                SELECT code, trade_time, open, high, low, close, volume, amount
                FROM bars_5m
                WHERE code IN ({placeholders}) AND trade_time BETWEEN ? AND ?
                ORDER BY code, trade_time
                """,
                [*normalized, start, end],
            ).df()

    def snapshot(self, at: datetime, codes: Sequence[str] | None = None) -> pd.DataFrame:
        """读取指定时刻的市场快照，并兼容不带交易所后缀的沪深代码。

        参数：
            at: 快照对应的行情时刻，精确到数据库中已有的分钟时间点。
            codes: 可选证券代码序列；六位裸代码会自动补全沪深交易所后缀，
                缺省时查询该时刻的全部证券。

        返回：
            按证券代码排序的快照数据表；没有匹配行情时返回空表。
        """
        clauses = ["trade_time = ?"]
        params: list = [at]
        if codes:
            normalized = [normalize_security_code(code) for code in codes]
            placeholders = ",".join("?" for _ in normalized)
            clauses.append(f"code IN ({placeholders})")
            params.extend(normalized)
        with self.connect() as con:
            return con.execute(
                f"""
                SELECT code, trade_time, open, high, low, close, volume, amount,
                       pre_close, change, pct_change
                FROM bars_5m WHERE {' AND '.join(clauses)} ORDER BY code
                """,
                params,
            ).df()

    def raw(
        self,
        code: str,
        trade_date: date,
        *,
        start_time: str | None = None,
        end_time: str | None = None,
        min_close: float | None = None,
        max_close: float | None = None,
        min_volume: int | None = None,
        max_volume: int | None = None,
        min_amount: float | None = None,
        max_amount: float | None = None,
        page: int = 1,
        page_size: int = 100,
    ) -> tuple[pd.DataFrame, int]:
        """分页读取单只证券在指定交易日内的原始5分钟行情。

        参数：
            code: 证券代码；六位沪深裸代码会自动补全交易所后缀。
            trade_date: 目标交易日，不包含其他日期的行情。
            start_time: 可选日内起始时间，包含该时刻，缺省时不限制下界。
            end_time: 可选日内结束时间，包含该时刻，缺省时不限制上界。
            min_close: 可选最低收盘价，价格单位与行情源一致。
            max_close: 可选最高收盘价，价格单位与行情源一致。
            min_volume: 可选最低成交量，单位与行情源一致。
            max_volume: 可选最高成交量，单位与行情源一致。
            min_amount: 可选最低成交额，金额单位与行情源一致。
            max_amount: 可选最高成交额，金额单位与行情源一致。
            page: 从1开始的页码，缺省为第一页。
            page_size: 每页记录数，缺省为100。

        返回：
            当前页行情数据表与全部匹配记录数构成的二元组。
        """
        clauses = ["code = ?", "trade_date = ?"]
        params: list = [normalize_security_code(code), trade_date]
        filters = [
            ("CAST(trade_time AS TIME) >= CAST(? AS TIME)", start_time),
            ("CAST(trade_time AS TIME) <= CAST(? AS TIME)", end_time),
            ("close >= ?", min_close),
            ("close <= ?", max_close),
            ("volume >= ?", min_volume),
            ("volume <= ?", max_volume),
            ("amount >= ?", min_amount),
            ("amount <= ?", max_amount),
        ]
        for clause, value in filters:
            if value is not None and value != "":
                clauses.append(clause)
                params.append(value)
        where = " AND ".join(clauses)
        with self.connect() as con:
            total = con.execute(f"SELECT count(*) FROM bars_5m WHERE {where}", params).fetchone()[0]
            frame = con.execute(
                f"""
                SELECT code, trade_time, open, high, low, close, volume, amount,
                       pre_close, change, pct_change
                FROM bars_5m WHERE {where}
                ORDER BY trade_time LIMIT ? OFFSET ?
                """,
                [*params, page_size, (page - 1) * page_size],
            ).df()
        return frame, total

    def symbols(self, query: str = "", limit: int = 20) -> list[str]:
        with self.connect() as con:
            rows = con.execute(
                "SELECT code FROM symbols WHERE code LIKE ? ORDER BY code LIMIT ?",
                [f"%{query.upper().strip()}%", limit],
            ).fetchall()
        return [row[0] for row in rows]
