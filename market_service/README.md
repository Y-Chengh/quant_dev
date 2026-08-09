# 本地行情服务

## 启动网页

```powershell
cd C:\Users\win10\Documents\quant\market_service
C:\Users\win10\AppData\Local\Programs\Python\Python311\python.exe run_server.py
```

浏览器打开 <http://127.0.0.1:9000>，接口文档位于 <http://127.0.0.1:9000/docs>。

服务只监听 `127.0.0.1`，不会暴露到局域网或公网。

## Python公共接口

外部代码只从包入口导入，不应直接使用 `database.py`：

```python
from datetime import date, datetime
from market_service import (
    KlinePeriod,
    KlineQuery,
    MarketDataClient,
    RawBarQuery,
)

client = MarketDataClient(r"D:\量化\market.duckdb")

# 单只证券K线
kline = client.get_kline(KlineQuery(
    code="600000.SH",
    start=datetime(2025, 1, 1),
    end=datetime(2026, 4, 17, 23, 59, 59),
    period=KlinePeriod.DAY,
))

# 某一时刻全市场截面
snapshot = client.get_snapshot(datetime(2026, 4, 17, 10, 30))

# 某交易日5分钟原始数据
page = client.get_raw_bars(RawBarQuery(
    code="600000.SH",
    trade_date=date(2026, 4, 17),
    min_volume=100_000,
))
print(page.data, page.total)
```

公共接口仅包括：

- `get_metadata()`
- `get_kline(query)`
- `get_snapshot(at, codes=None)`
- `get_raw_bars(query)`
- `search_symbols(text, limit)`

## REST接口

- `GET /api/meta`
- `GET /api/kline`
- `GET /api/snapshot`
- `GET /api/raw`
- `GET /api/raw.csv`
- `GET /api/symbols`

`GET /api/kline` 的每根K线除开、高、低、收、成交量和成交额外，还返回
`pre_close`、`change`、`pct_change`、`intraday_pct_change`。其中涨跌幅口径为
`(本根收盘价 / 上一根有效K线收盘价 - 1) × 100`，日内涨跌幅口径为
`(本根收盘价 / 本根开盘价 - 1) × 100`；查询区间首根K线会向前
查找最近的同周期有效收盘价；没有更早行情时，`pre_close`、`change` 和
`pct_change` 为 `null`，日内涨跌幅仍独立按本根开盘价和收盘价计算。当聚合
周期的查询起点落在一个周期内部时，首根K线会包含该周期起点至查询起点之间的
历史行情，以保证同一根K线的OHLCV不随查询起点改变；结束时间之后的数据不会被
读取。

REST层不提供任意SQL执行能力。
