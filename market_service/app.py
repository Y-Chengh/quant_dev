from __future__ import annotations

import io
import os
from datetime import date, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

try:
    from .client import MarketDataClient
    from .models import KlinePeriod, KlineQuery, RawBarQuery
except ImportError:  # 支持直接从源码目录运行
    from client import MarketDataClient
    from models import KlinePeriod, KlineQuery, RawBarQuery


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DB_PATH = Path(os.getenv("MARKET_DB_PATH", r"D:\量化\market.duckdb"))
client = MarketDataClient(DB_PATH)

app = FastAPI(title="本地5分钟行情服务", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def records(frame):
    """把行情表转换为可安全序列化的REST记录列表。

    参数：
        frame: 待输出的行情数据表；时间列会格式化到秒，所有缺失值转换为
            JSON ``null``。

    返回：
        保持原行序和列名的字典列表。
    """
    frame = frame.copy()
    for column in frame.columns:
        if str(frame[column].dtype).startswith("datetime"):
            frame[column] = frame[column].dt.strftime("%Y-%m-%d %H:%M:%S")
    numeric_columns = frame.select_dtypes(include="number").columns
    frame[numeric_columns] = frame[numeric_columns].where(
        frame[numeric_columns].abs().lt(float("inf"))
    )
    return frame.astype(object).where(frame.notna(), None).to_dict(orient="records")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "database": str(DB_PATH)}


@app.get("/api/meta")
def meta():
    try:
        return client.get_metadata()
    except Exception as exc:
        raise HTTPException(503, f"数据库尚未就绪：{exc}") from exc


@app.get("/api/symbols")
def symbols(q: str = "", limit: int = Query(20, ge=1, le=100)):
    return {"items": client.search_symbols(q, limit)}


@app.get("/api/kline")
def kline(code: str, start: datetime, end: datetime, period: str = "5m"):
    if start > end:
        raise HTTPException(400, "开始时间不能晚于结束时间")
    try:
        query = KlineQuery(code, start, end, KlinePeriod(period))
        frame = client.get_kline(query)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"code": code.upper(), "period": period, "count": len(frame), "items": records(frame)}


@app.get("/api/snapshot")
def snapshot(at: datetime, codes: str | None = None):
    selected = [item.strip() for item in codes.split(",") if item.strip()] if codes else None
    frame = client.get_snapshot(at, selected)
    return {"time": at.isoformat(sep=" "), "count": len(frame), "items": records(frame)}


def raw_query(
    code: str,
    trade_date: date,
    start_time: str | None,
    end_time: str | None,
    min_close: float | None,
    max_close: float | None,
    min_volume: int | None,
    max_volume: int | None,
    min_amount: float | None,
    max_amount: float | None,
    page: int,
    page_size: int,
):
    result = client.get_raw_bars(RawBarQuery(
        code=code, trade_date=trade_date, start_time=start_time, end_time=end_time,
        min_close=min_close, max_close=max_close,
        min_volume=min_volume, max_volume=max_volume,
        min_amount=min_amount, max_amount=max_amount,
        page=page, page_size=page_size,
    ))
    return result.data, result.total


@app.get("/api/raw")
def raw(
    code: str,
    trade_date: date,
    start_time: str | None = None,
    end_time: str | None = None,
    min_close: float | None = None,
    max_close: float | None = None,
    min_volume: int | None = None,
    max_volume: int | None = None,
    min_amount: float | None = None,
    max_amount: float | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=1000),
):
    frame, total = raw_query(code, trade_date, start_time, end_time, min_close, max_close,
                             min_volume, max_volume, min_amount, max_amount, page, page_size)
    return {"total": total, "page": page, "page_size": page_size, "items": records(frame)}


@app.get("/api/raw.csv")
def raw_csv(
    code: str,
    trade_date: date,
    start_time: str | None = None,
    end_time: str | None = None,
    min_close: float | None = None,
    max_close: float | None = None,
    min_volume: int | None = None,
    max_volume: int | None = None,
    min_amount: float | None = None,
    max_amount: float | None = None,
):
    frame, _ = raw_query(code, trade_date, start_time, end_time, min_close, max_close,
                         min_volume, max_volume, min_amount, max_amount, 1, 1000)
    output = io.StringIO()
    frame.to_csv(output, index=False)
    filename = f"{code.upper().replace('.', '_')}_{trade_date}.csv"
    return Response(
        "\ufeff" + output.getvalue(), media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
