# 本地行情服务

## 启动网页

```powershell
quant-market-server
```

等价写法：`python -m quant.cli.market_server`；也可直接双击
`scripts\start_market_service.bat`。可用 `--host`、`--port` 覆盖默认值。

浏览器打开 <http://127.0.0.1:8000>，接口文档位于 <http://127.0.0.1:8000/docs>。

服务只监听 `127.0.0.1`，不会暴露到局域网或公网。

## Python公共接口

外部代码只从包入口导入，不应直接使用 `database.py`：

```python
from datetime import date, datetime
from quant.market_data import (
    KlinePeriod,
    KlineQuery,
    MarketDataClient,
    RawBarQuery,
)

client = MarketDataClient(r"D:\量化\market.duckdb")

# 单只证券K线
kline = client.get_kline(KlineQuery(
    code="600000",
    start=datetime(2025, 1, 1),
    end=datetime(2026, 4, 17, 23, 59, 59),
    period=KlinePeriod.DAY,
))

# 某一时刻全市场截面
snapshot = client.get_snapshot(datetime(2026, 4, 17, 10, 30))

# 某交易日5分钟原始数据
page = client.get_raw_bars(RawBarQuery(
    code="600000",
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

K线、批量5分钟行情、快照和原始行情查询均支持省略沪深交易所后缀。例如，
`600000` 会按 `600000.SH` 查询，`000001` 会按 `000001.SZ` 查询；已带后缀的
代码仍可照常使用。

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

## QMT日线库

日线数据来自大 QMT，落在**独立**的 `qmt_daily.duckdb` 与 `bars_1d` 月度 Parquet 上，
与上面的 5 分钟库互不影响。构建、增量同步、复权口径与并发约束见
[qmt_daily_store.md](qmt_daily_store.md)，合法性审计见 [market_check.md](market_check.md)。

```python
from datetime import date
from quant.market_data.daily import AdjustMode, DailyMarketClient

client = DailyMarketClient()
client.get_klines_1d(["000001.SZ"], date(2024, 1, 1), date(2024, 12, 31),
                     adjust=AdjustMode.HFQ)
client.list_universe(date(2024, 6, 30))
```

公共接口仅包括：

- `get_metadata()`
- `get_daily_bars(query)` / `get_klines_1d(codes, start, end, ...)`
- `get_corporate_actions(codes, end)`
- `get_trading_calendar(start, end)`
- `get_instruments(codes)`
- `list_universe(as_of, ...)` / `list_universe_over_window(start, end, ...)`
- `search_symbols(text, limit)`

注意 `search_symbols` 与 5 分钟库同义，只反映「库里有没有行情」；做研究选股请用
`list_universe`，它按上市退市日筛选，包含当时还在、如今已退市的证券，不引入
幸存者偏差。
