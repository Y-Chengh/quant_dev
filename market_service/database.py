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
        if period not in PERIODS:
            raise ValueError(f"不支持的周期：{period}")
        code = code.upper().strip()
        with self.connect() as con:
            if period == "5m":
                return con.execute(
                    """
                    SELECT trade_time AS time, open, high, low, close, volume, amount
                    FROM bars_5m
                    WHERE code = ? AND trade_time BETWEEN ? AND ?
                    ORDER BY trade_time
                    """,
                    [code, start, end],
                ).df()
            bucket = PERIODS[period]
            return con.execute(
                f"""
                WITH source AS (
                    SELECT *, time_bucket(INTERVAL '{bucket}', trade_time) AS bucket
                    FROM bars_5m
                    WHERE code = ? AND trade_time BETWEEN ? AND ?
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
