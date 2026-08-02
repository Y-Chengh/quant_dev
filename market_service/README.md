# 本地行情服务

## 启动网页

```powershell
cd C:\Users\win10\Documents\quant\market_service
C:\Users\win10\AppData\Local\Programs\Python\Python311\python.exe run_server.py
```

浏览器打开 <http://127.0.0.1:8000>，接口文档位于 <http://127.0.0.1:8000/docs>。

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

REST层不提供任意SQL执行能力。
