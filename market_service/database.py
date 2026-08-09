from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Iterator, Sequence

import duckdb
import pandas as pd


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
    def __init__(self, path: str | Path = r"D:\量化\market.duckdb") -> None:
        self.path = Path(path)

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
            code: 证券代码；查询前会转为大写并去除首尾空白。
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
        code = code.upper().strip()
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
        """批量读取研究所需的原始5分钟行情。"""
        normalized = [code.upper().strip() for code in codes if code.strip()]
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
        clauses = ["trade_time = ?"]
        params: list = [at]
        if codes:
            placeholders = ",".join("?" for _ in codes)
            clauses.append(f"code IN ({placeholders})")
            params.extend(codes)
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
        clauses = ["code = ?", "trade_date = ?"]
        params: list = [code.upper().strip(), trade_date]
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
