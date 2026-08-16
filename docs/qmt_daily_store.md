# QMT 日线库

> 想按步骤照着做，先看 [上手指南](qmt_daily_guide.md)；本文是口径与参数的参考手册。

把大 QMT 落盘的日线 CSV 增量转换成 market service 自己的存储格式，供因子研究与
合法性审计使用。日线库与现有 5 分钟库**完全独立**，互不影响。

## 存储布局

```
<daily_root>/                                默认 D:\量化\qmt_daily
├─ qmt_daily.duckdb                          目录库：视图 + 小表，只有几 MB
├─ bars_1d/year=YYYY/month=MM/bars.parquet   唯一的大数据集，月粒度原子重写
├─ reports/market_check/<时间戳>/            python -m quant.cli.market_check 报告
└─ .sync.lock                                进程间同步锁
```

`qmt_daily.duckdb` 中：

| 对象 | 内容 |
| --- | --- |
| `VIEW bars_1d` | `code, trade_date, open, high, low, close, pre_close, volume, amount, suspend_flag`，外加 hive 分区列 `year`、`month` |
| `TABLE instruments` | 代码、简称、上市日、退市日、交易状态、板块、风险警示标记 |
| `TABLE trading_calendar` | 交易日与日历基准证券 |
| `TABLE corporate_actions` | 除权送转原始记录，含 `adjustment_factor` |
| `TABLE ingest_state` | 每个源分区的文件指纹与入库结果，增量检查的全部依据 |
| `TABLE sync_runs` | 每次同步的执行记录 |
| `TABLE monthly_inventory` / `symbols` | 月度与逐证券库存统计 |
| `TABLE dataset_metadata` | 结构版本、源目录、复权策略、前复权基准日等 |

**库里一律保存原始不复权价**，复权在读取层按需计算，原始值始终可审计、可对账。

## 命令

```powershell
# 首次全量入库
python -m quant.cli.build_daily_store --qmt-output-root D:\qmt_kline_test1 --rebuild-all
# 日常增量（无增量时约 0.2 秒返回）
python -m quant.cli.build_daily_store
# 只检查不写入
python -m quant.cli.build_daily_store --dry-run
# 强制完整扫描，并重算源文件摘要
python -m quant.cli.build_daily_store --sync-mode full --sync-verify-hash
```

路径解析优先级统一为「显式传参 > 环境变量 > 下载器配置 > 内置回退」：

- 日线库：`--database` > `QMT_DAILY_DB_PATH` > `D:\量化\qmt_daily\qmt_daily.duckdb`
- 源目录：`--qmt-output-root` > `QMT_OUTPUT_ROOT` > `--qmt-config` 指定的 JSONC 里的
  `output_root` > `D:\qmt_kline_test1`

退出码：0 正常，2 配置或执行失败，3 日线库被其它进程占用而跳过。

## 增量检查为什么这么设计

源目录实测有 6450 个日线分区、6450 个除权分区，**每个 `_SUCCESS.json` 约 106 KB**
（`partition_scope` 内嵌了整个约 5000 只的证券池）。全部解析一遍等于读 1.3 GB JSON，
放在每次启动的路径上不可接受。因此检查分四级，只有字节真的变了才会读文件：

| 级别 | 做什么 | 实测耗时 |
| --- | --- | --- |
| Tier −1 | 比 `run_complete` 最新日期 + 四个数据集目录的 mtime | 0.2 秒 |
| Tier 0 | 每数据集一次 `scandir`，与 `ingest_state` 的键做差 | — |
| Tier 1 | 每分区两次 `stat`，比 mtime 与 size | 全量 12902 个分区共 2.7 秒 |
| Tier 2 | 只对指纹变化的分区解析 `_SUCCESS.json`，比 `sha256` | 按需 |
| Tier 3 | `--sync-verify-hash` 时重算 `data.csv` 摘要 | 按需 |

Tier −1 有一个已知局限：NTFS 上目录的修改时间只在增删条目时变化，**孙文件被原地
重写不会冒泡**。因此它只用于缺省的 `auto` 模式；`--sync-mode full` 与
`python -m quant.cli.market_check` 一律从 Tier 0 开始。

全量重建实测 71.6 秒（12902 个分区 → 320 个月度分片 → 1633 万行）。其中约
八成时间花在分片重写上；重活跑在内存连接并按 CPU 核数放开 DuckDB 线程。

## 生命周期过滤：必须做，否则数据废掉一半

大 QMT 的 `fill_data=True` 会给**还没上市**的证券也返回一行，`suspend_flag=1`、
`volume=0`、开高低收全等。实测每个交易日源文件都恰好 5209 行：

| 交易日 | 源行数 | 其中已上市 | 已上市里真正停牌 |
| --- | ---: | ---: | ---: |
| 2000-01-04 | 5209 | 750 | 7 |
| 2015-01-05 | 5209 | 2365 | 214 |
| 2024-01-02 | 5209 | 4996 | 4 |

约 40% 是填充行。入库时按 `instruments.list_date` 过滤掉它们，否则：库体积膨胀到
三倍；`symbols.first_date` 完全失真；**上市前那段平价零量数据会让波动率、收益率类
因子在 IPO 当天炸出一个假跳变**。过滤之后 `suspend_flag=1` 才真正等价于「停牌」。

因此 `instruments` 必须先于 `bars_1d` 入库；`instrument_info` 快照缺失时入库直接
失败退出，不会退化成不过滤地全量写入。

另注：`expire_date` 的 QMT 哨兵不止 `99999999`，实测快照里还有 `19700427` 与
`19700428`（共 725 行），入库时一律归一化为空。

`open_date` 本身也可能缺失（QMT 未返回或字段损坏）。这种情况不按上市日过滤，
放行该证券全部历史行情，交给 `python -m quant.cli.market_check` 的 `DAILY_OPEN_DATE_MISSING`
提示核对，口径与 `quant.qmt_downloader.self_check` 保持一致。这条判据只在源 CSV
内容变化触发重写的月份分片上生效；已建好的存量库不会因为规则改了就自动重写，
升级后需要对已有库执行一次 `python -m quant.cli.build_daily_store --rebuild-all` 才能统一口径。

## 过滤原因与样例

每行源数据都会被打上一个过滤原因。判定写成一条 `CASE`，**一行只命中第一个成立的
分支**，因此下表的顺序就是优先级，各原因的行数互不重叠：

| 原因 | 判定条件 | 典型来源 |
| --- | --- | --- |
| `trade_date_null` | `trade_date` 为 NULL | 该行是空字段，或同月某个日分区文件缺这一列（该月全部文件都缺则读 CSV 直接报错，走不到这里） |
| `trade_date_bad_length` | 去空格后长度不是 8 | 写成了 `2024-01-02` 或截断成 7 位 |
| `trade_date_unparsable` | 长度对但 `strptime('%Y%m%d')` 解析不出来 | `20241332` 这类非法日期、乱码 |
| `unknown_code` | 代码不在 `instrument_info` 快照里 | 快照落后于日线，或代码写错 |
| `before_listing` | `open_date` 非空且 `trade_date < open_date` | `fill_data=True` 的上市前占位行 |
| `after_delisting` | `expire_date` 非空且 `trade_date > expire_date` | 退市后仍被填充的占位行 |

边界是闭区间：`trade_date` 正好等于 `open_date` 或 `expire_date` 的当天保留。
`open_date` / `expire_date` 为空就不查对应那一侧。这一层**不做**价格合法性校验，也
不去重，那些由 `python -m quant.cli.market_check` 负责。

每个原因除了行数，还会带回最多 10 条**随机样例**，日志与命令行摘要都直接打印：

```
本次同步累计过滤（按原因）:
  before_listing: 17252623 行
    随机样例 10 条:
      000338.SZ 20000113 open=2007-04-30 expire=- suspend=1 volume=0 close=0.0000
```

`suspend=1 volume=0 close=0` 正是 `fill_data` 占位行的特征，一眼就能确认滤对了。
样例分两级抽取：每个月度分片先在本月该原因的全部丢弃行里随机抽最多 10 条，同步
结束时再从这些月度样例里随机抽最终 10 条。两级都不按行数加权，所以它**不是全库
均匀抽样**，只用于人工核对具体 case，不能拿来做统计推断。

样例展示的是**勘误前**的源值：勘误在写 Parquet 那一步才 `COALESCE`，而勘误既不改
`code`/`trade_date`，也不参与过滤判定，所以原始值才是判断「该不该滤」的正确依据。

抽样用 `arg_min(struct, random(), 10)` 的定长堆而不是 `ORDER BY random()` 开窗，
避免对上千万丢弃行做全排序。在真实源数据上取 2000-01、2005-06、2015-01、2024-01
四个月各跑 5 轮取最小值，分片重写整体从 0.965s 变成 1.000s，慢约 4%。

## 并发与锁

DuckDB 是**单写多读**，一个活跃的写连接会同时挡住其它进程的读。为此：

- 重活（读 CSV、写 Parquet、统计库存）全部跑在**内存连接**上，目录库不上锁；
- 只有最后更新辅助表与指纹的那个短事务才持有写锁，实测约 5 秒；
- 另有 `.sync.lock` 文件锁，给同一套工具链的其它进程一个明确的中文提示；
- **自动增量同步拿不到锁时降级为警告并跳过**，继续用现有数据跑，绝不让实验挂掉。

## 复权

系数口径已在真实数据上标定（2024 年全年 4524 个除权事件）：

```
adjustment_factor(t) == close(t-1) / pre_close(t)      # t 为除权日
```

比值中位数 1.0，10%~90% 分位落在 `[1.0, 1.000001]`。于是：

- **后复权**（`hfq`，研究推荐）：`adj(t) = price(t) × Π{ f(s) : ex_date s ≤ t }`。
  历史取值不随之后新增的分红送转改变，既不引入未来信息，也不会让因子缓存整体失效。
- **前复权**（`qfq`）：`adj(t) = hfq(t) / hfq(anchor)`。每次出现新的除权事件都会
  重算全部历史价格，因此**必须显式固定基准日**，否则同样的查询会随每次同步给出
  不同的特征值。基准日会进入因子缓存命名空间，不同基准的缓存不可能互相命中。
- 成交额永不调整；成交量只在 `--adjust-volume` 时反向调整。

系数一律由 `corporate_actions` 全历史累乘得到，**不**依赖查询窗口内的 `pre_close`，
否则窗口起点不同就会算出不同的复权价。

`python -m quant.cli.market_check` 会把 `adjustment_factor` 与行情反推的比例逐条对账，实测 2024 年
有约 3.4% 的事件两者对不上（多数只差两三分钱，约 30 条是大额分歧），报告里可以逐条查。

## Python 接口

```python
from datetime import date
from quant.market_data.daily import AdjustMode, DailyMarketClient

client = DailyMarketClient()          # 按 QMT_DAILY_DB_PATH 解析

client.get_metadata()
client.get_klines_1d(["000001.SZ"], date(2024, 1, 1), date(2024, 12, 31),
                     adjust=AdjustMode.HFQ)
client.list_universe(date(2024, 6, 30))          # 无幸存者偏差的证券池
client.list_universe_over_window(date(2020, 1, 1), date(2024, 12, 31))
client.get_trading_calendar(date(2024, 1, 1), date(2024, 12, 31))
client.get_instruments(["000001.SZ"])
client.get_corporate_actions(["000001.SZ"])
```

`list_universe` 读 `instruments` 而不是 `bars_1d`，因此包含在该日之后才退市的证券，
这是消除幸存者偏差的关键。`search_symbols` 保留了与 5 分钟库一致的行为，
只反映「库里有没有行情」，**不是**无偏池。

**当前数据源的局限**：实测 `instrument_info` 快照里一只退市股都没有，
`python -m quant.cli.market_check` 会以 `DAILY_NO_DELISTED_SYMBOLS` 如实报告。做长周期回测时
需要知道结果被幸存者偏差抬高了。
