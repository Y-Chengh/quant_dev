# QMT 日线数据上手指南

从大 QMT 下载日线，到入库、体检、跑因子实验的完整流程。每一步都给出实测耗时与
常见故障的处理办法。

分工速查：

| 想干什么 | 看哪儿 |
| --- | --- |
| 一步步照着做 | **本文** |
| 库的结构、增量机制、复权口径 | [qmt_daily_store.md](qmt_daily_store.md) |
| 每条校验规则的口径与阈值 | [market_check.md](market_check.md) |
| 参数逐个查 | [cli_arguments.md](cli_arguments.md) |
| 5 分钟库与网页服务 | [market_data.md](market_data.md) |
| 下载器本身怎么用 | [qmt_downloader.md](qmt_downloader.md) |

---

## 0. 一次性准备

```powershell
.venv\Scripts\python.exe -m pip install -e ".[research,service,dev]"
```

设两个环境变量（否则用内置回退值 `D:\量化\qmt_daily\qmt_daily.duckdb` 和
`D:\qmt_kline_test1`）：

```powershell
# 当前会话生效
$env:QMT_OUTPUT_ROOT  = "D:\qmt_kline_test1"          # 大 QMT 的落盘目录
$env:QMT_DAILY_DB_PATH = "D:\量化\qmt_daily\qmt_daily.duckdb"   # 日线库

# 永久生效
[Environment]::SetEnvironmentVariable("QMT_OUTPUT_ROOT", "D:\qmt_kline_test1", "User")
[Environment]::SetEnvironmentVariable("QMT_DAILY_DB_PATH", "D:\量化\qmt_daily\qmt_daily.duckdb", "User")
```

路径解析优先级统一为：**命令行显式传参 > 环境变量 > 下载器 JSONC 配置里的
`output_root` > 内置回退值**。所以不设环境变量、只靠 `configs/qmt_downloader/*.json`
也能跑通。

---

## 1. 用大 QMT 下载日线

在大 QMT 编辑器里跑 `scripts/qmt_run_downloader.py`，详见
[qmt_downloader.md](qmt_downloader.md)。日线库需要它产出这四个数据集：

| 数据集 | 必需 | 用途 |
| --- | :---: | --- |
| `kline_1d` | 必需 | 日线行情本体 |
| `instrument_info` | **必需** | 上市退市日；缺了它入库会直接失败退出（见下） |
| `trading_calendar` | 强烈建议 | 没有它就查不出「整天缺失」 |
| `corporate_actions` | 复权必需 | 除权送转记录 |

确认下载器配置里 `datasets` 至少包含这几项：

```jsonc
"datasets": ["kline_1d", "corporate_actions", "trading_calendar"]
```

`instrument_info` 不在 `datasets` 里，它随 `kline_1d` 自动产出。

---

## 2. 首次全量入库

```powershell
quant-build-daily-store --rebuild-all
```

实测参考（6450 个交易日 × 约 5200 只证券）：

```
扫描 12902 个分区 → 重写 320 个月度分片 → 1633 万行，用时 71.6s
```

只有约一半源行会进库——大 QMT 会给**还没上市**的证券也返回一行（`suspend_flag=1`、
`volume=0`），入库时按上市日过滤掉了。2000-01-04 那天源文件 5209 行里只有 750 行是
真实行情。这是刻意的：不过滤的话每只股票都会在 IPO 当天出现一个假跳变。

产出：

```
D:\量化\qmt_daily\
├─ qmt_daily.duckdb                          几 MB 的目录库
├─ bars_1d\year=YYYY\month=MM\bars.parquet   约 294 MB（源 CSV 2.6 GB）
└─ .sync.lock
```

---

## 3. 日常增量

**平时什么都不用做。** `quant-factor-demo --data-source qmt_daily` 启动时会自动
检查并入库。想手动跑：

```powershell
quant-build-daily-store
```

实测耗时：

| 场景 | 耗时 |
| --- | --- |
| 无增量（命中水位短路） | **0.2 秒** |
| 无增量（`--sync-mode full` 完整扫描 12902 个分区） | **2.7 秒** |
| 新增一个交易日 | 秒级，只重写它所在的那个月 |

之所以这么快，是因为检查严格停在 `stat()` 层，无增量时**一个 CSV 都不会打开**。
细节见 [qmt_daily_store.md](qmt_daily_store.md#增量检查为什么这么设计)。

常用开关：

```powershell
quant-build-daily-store --dry-run              # 只看有没有增量，不写
quant-build-daily-store --sync-mode full       # 跳过水位短路，完整扫描
quant-build-daily-store --sync-verify-hash     # 额外重算源文件 SHA-256
quant-build-daily-store --rebuild-all          # 整库重建
```

**什么时候需要 `--rebuild-all`**：`instrument_info` 快照变了（比如终于有退市股了），
因为生命周期过滤的依据变了，历史分片需要按新口径重切。

---

## 4. 数据体检

不需要每次都跑，怀疑数据有问题时手动触发：

```powershell
quant-market-check                                    # 全库
quant-market-check --start-date 20240101 --end-date 20241231
quant-market-check --codes 000001.SZ 600000.SH
quant-market-check --cross-check-5m                   # 额外与 5 分钟库对账
```

退出码：**0 通过 / 1 有 ERROR / 2 执行失败**，可以直接用在定时任务里。

报告写到 `<库目录>\reports\market_check\<时间戳>\`，先看 `summary.md`，再按需要
查明细 CSV。2024 全年的真实结果长这样：

```
- 结论：未通过
- 交易日：242        证券数：5208
- 理论应有记录：1218186   实际记录：1218186   缺失记录：0
- 连续缺失区间：0
- 涨跌停越界：26
- 成交量单位倍数：100.0
- ERROR：22   WARNING：42
```

**怎么读这份报告**：

- `缺失记录：0` 是最重要的一行——上市到退市之间每个交易日都有数据。
- `成交量单位倍数：100.0` 是自动标定出来的（手，不是股），校验成交额时会用到。
- `DAILY_ADJUST_FACTOR_MISMATCH` 实测 2024 年有 153 处（占 3.4%）。多数只差两三
  分钱，是 QMT 自己的 `pre_close` 与 `adjustment_factor` 各自四舍五入造成的；
  但有约 30 条是大额分歧（比如声明因子 2.39、行情却毫无跳空），**那些会让后复权
  算错**，值得逐条查 `adjust_factor_audit.csv`。
- `DAILY_NO_DELISTED_SYMBOLS` 是提醒你：当前证券池里一只退市股都没有，长周期回测
  的结果被幸存者偏差抬高了。这是数据源的问题，入库侧修不了。

每条规则的完整口径见 [market_check.md](market_check.md)。

---

## 5. 跑因子实验

```powershell
# 用日线库
quant-factor-demo --data-source qmt_daily

# 现成的对照配置
quant-factor-demo --config configs\factor_research\qmt_daily.example.yaml

# 不写 --data-source 就还是原来的 5 分钟库，行为逐字节不变
quant-factor-demo --config configs\factor_research\example.yaml
```

### 复权怎么选

| 口径 | 用途 |
| --- | --- |
| `--adjust hfq`（默认） | **研究用这个。** 历史取值不随之后的分红送转改变，不引入未来信息，因子缓存也不会因为一次除权全量失效 |
| `--adjust qfq` | 看图用。会随新分红重算全部历史价；用于研究时必须配 `--adjust-anchor` 固定基准日，否则同一条命令过几天就算出不同结果 |
| `--adjust none` | 原始不复权。除权日会有假跳空，`return_1d`、`overnight_gap` 这类因子在那天是错的 |

### 八个分钟因子在日线源下不可用

`close_to_vwap`、`downside_semivol`、`intraday_path_efficiency`、`last_30m_return`、
`last_30m_volume_ratio`、`positive_bar_ratio`、`realized_vol`、`signed_volume_imbalance`
需要分钟行情。

- **不写 `--factors`**：自动跳过它们并告警，用剩下 22 个日频因子。
- **显式点名**：直接报错，不会静默给你一列 NaN。

```
ValueError: 因子 ['realized_vol'] 需要分钟行情，数据源 'qmt_daily' 只提供日频；
请改用 --data-source market_service，或从 --factors 中移除这些因子
```

所以 `configs\factor_research\example.yaml` 不能直接配日线源（它选了两个分钟因子），
用 `qmt_daily.example.yaml`。

### 因子缓存是隔离的

缓存路径带命名空间，形如 `.factor_cache\qmt_daily\hfq\-\no_susp\rawvol\`。数据源、
复权口径、前复权基准日、是否含停牌、是否调整成交量——任何一项不同都落在不同目录，
不可能互相命中。5 分钟库仍用 `.factor_cache\` 根目录，路径逐字节不变。

---

## 6. 直接用 Python 接口

```python
from datetime import date
from quant.market_data.daily import AdjustMode, DailyMarketClient

client = DailyMarketClient()                      # 按 QMT_DAILY_DB_PATH 解析

client.get_metadata()
# {'first_date': date(2000,1,4), 'last_date': date(2026,8,11),
#  'rows': 16338977, 'symbols': 5208, 'adjust_anchor_date': '2026-08-11', ...}

# 后复权日线，默认剔除停牌日
bars = client.get_klines_1d(
    ["000001.SZ", "600000.SH"], date(2024, 1, 1), date(2024, 12, 31),
    adjust=AdjustMode.HFQ,
)

# 无幸存者偏差的证券池
client.list_universe(date(2020, 6, 30))                       # 某一天的截面
client.list_universe_over_window(date(2020, 1, 1), date(2024, 12, 31))  # 滚动回测用这个

client.get_trading_calendar(date(2024, 1, 1), date(2024, 1, 31))
client.get_instruments(["000001.SZ"])
client.get_corporate_actions(["000001.SZ"])
```

**选股请用 `list_universe*`，不要用 `search_symbols`。** 后者只反映「库里有没有
这只证券的行情」，等于只保留活到今天的证券，会引入幸存者偏差；前者按 `instruments`
的上市退市日筛选，包含当时还在、如今已退市的证券。

---

## 7. 常见故障

**`日线库不存在: ...；请先运行 quant-build-daily-store`**
还没建库，或 `QMT_DAILY_DB_PATH` 指错了地方。

**`证券生命周期快照缺失: ...\instrument_info\snapshot=latest\data.csv`**
入库**故意**在这里硬失败：没有上市日就没法过滤未上市填充行，宁可不入库，也不能
把 40% 的伪造数据当真行情写进去。回到下载器补出 `instrument_info` 再来。

**`日线库同步skipped_locked` / 退出码 3**
另一个进程正在写库（DuckDB 单写多读）。等它跑完即可。注意这**不会**让实验挂掉——
自动增量同步遇锁会降级为警告并继续用现有数据跑。

**`日线库同步锁被占用: ...\.sync.lock`**
上一次同步异常退出留下的残留锁。超过 1 小时会自动夺回；急的话直接删掉该文件。

**`使用 --adjust qfq 时必须显式指定 --adjust-anchor`**
库里还没记录基准日（没同步过）。要么先同步一次，要么自己传
`--adjust-anchor 2026-08-11`。

**审计报了一堆 `DAILY_PRICE_LIMIT_VIOLATION`**
默认按板块基础幅度判定（主板 10%、创业板/科创板 20%、北交所 30%），**不**按当前
ST 状态收紧到 5%——证券简称只有当前快照，历史某天是不是 ST 无从得知。实测按当前
状态收紧会产生 3860 条误报，不收紧只剩 26 条。真要严格：`--apply-st-limit`。

**想确认某天到底为什么缺数据**
先跑 `quant-qmt-self-check` 看源侧 CSV 是不是也缺——那一侧查的是分区完整性
（哈希、行数），和 `quant-market-check` 查的业务合法性是互补的。源侧就缺的话，
回下载器补下载；源侧有而库里没有，就 `--rebuild-all`。

---

## 8. 两条数据栈的关系

```
iFinD 5 分钟 zip ──quant-build-market-db──→ market.duckdb (bars_5m)  ──┐
                                                                       ├─→ quant-factor-demo
大 QMT 日线 CSV ──quant-build-daily-store─→ qmt_daily.duckdb (bars_1d)─┘      --data-source
                                                    ↑
                                          quant-market-check
```

两个库**完全独立**，互不影响。`quant-market-check --cross-check-5m` 会把 5 分钟
行情聚合成日频跟日线对账——两者来自完全不同的数据源，对得上是很强的正确性证据。
