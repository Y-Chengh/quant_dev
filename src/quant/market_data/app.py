from __future__ import annotations

import io
from datetime import date, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from quant.config import default_market_database

from .client import MarketDataClient
from .codes import normalize_security_code
from .models import KlinePeriod, KlineQuery, RawBarQuery

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DB_PATH = default_market_database()
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
def kline(code: str, start: datetime, end: datetime, period: str = "5m") -> dict:
    """响应单只证券K线查询，并返回规范化后的证券代码。

    参数：
        code: 证券代码；六位沪深裸代码会自动补全交易所后缀。
        start: 查询起始时间，包含该时刻。
        end: 查询结束时间，包含该时刻。
        period: K线周期，缺省为5分钟，可选值由 ``KlinePeriod`` 定义。

    返回：
        包含规范证券代码、周期、K线数量和可序列化行情记录的字典。
    """
    if start > end:
        raise HTTPException(400, "开始时间不能晚于结束时间")
    try:
        query = KlineQuery(code, start, end, KlinePeriod(period))
        frame = client.get_kline(query)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {
        "code": normalize_security_code(code),
        "period": period,
        "count": len(frame),
        "items": records(frame),
    }


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
) -> Response:
    """导出单只证券在指定交易日内最多1000条原始行情记录。

    参数：
        code: 证券代码；六位沪深裸代码会自动补全交易所后缀，并用于下载文件名。
        trade_date: 目标交易日，不包含其他日期的行情。
        start_time: 可选日内起始时间，包含该时刻。
        end_time: 可选日内结束时间，包含该时刻。
        min_close: 可选最低收盘价，价格单位与行情源一致。
        max_close: 可选最高收盘价，价格单位与行情源一致。
        min_volume: 可选最低成交量，单位与行情源一致。
        max_volume: 可选最高成交量，单位与行情源一致。
        min_amount: 可选最低成交额，金额单位与行情源一致。
        max_amount: 可选最高成交额，金额单位与行情源一致。

    返回：
        带UTF-8 BOM的CSV响应，下载文件名使用补全后的规范证券代码。
    """
    frame, _ = raw_query(code, trade_date, start_time, end_time, min_close, max_close,
                         min_volume, max_volume, min_amount, max_amount, 1, 1000)
    output = io.StringIO()
    frame.to_csv(output, index=False)
    filename = f"{normalize_security_code(code).replace('.', '_')}_{trade_date}.csv"
    return Response(
        "\ufeff" + output.getvalue(), media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
