# QMT 原始数据勘误记录

用途：记录人工核实确认为**大 QMT 源数据本身有误**（而非本仓库落盘/自检逻辑的 bug）
的具体股票、具体交易日问题，供后续做数据清洗、自检白名单或结果解读时查阅。

数据文件：[qmt_source_data_errata.csv](qmt_source_data_errata.csv)，长表格式，每行只
覆盖一只证券、一个交易日的一个字段，列含义：

| 列 | 含义 |
| --- | --- |
| `symbol` | 证券代码，格式与 QMT 一致（如 `002062.SZ`） |
| `trade_date` | 出问题的交易日，`YYYYMMDD` |
| `field` | 需要覆盖的日线字段名，取值只能是 `open`/`high`/`low`/`close`/`pre_close`/`volume`/`amount`/`suspend_flag` 之一 |
| `value` | 覆盖后的取值；只改这一个字段，同一行的其它字段保持源数据原值不变 |
| `issue_type` | 问题类型简短标签，如 `missing_suspension`（停牌未在源数据中体现） |
| `description` | 人工核实结论，说明实际情况与 QMT 数据的差异 |
| `found_date` | 发现/记录日期，`YYYYMMDD` |
| `source_report` | 发现该问题时对应的自检报告路径，便于追溯上下文 |

同一只证券同一交易日如需覆盖多个字段，写多行即可，每行一个 `field`/`value`。

新增记录前须人工核实（如查交易所公告、其它行情源比对），不得仅凭自检报告的一条
issue 就直接归入本文件——自检报告本身可能是本仓库逻辑的 bug，而非源数据问题。

## 谁会读取并应用这份勘误表

`src/quant/qmt_downloader/errata.py` 提供装载与应用函数（`load_errata_overrides`、
`apply_errata_overrides`、`errata_pivot_frame`），目前有两处消费方，都在**读完源
数据之后、产出各自结果之前**应用同一份勘误：

- `quant-qmt-self-check`：逐日扫描读入某个交易日的日线分区后，先应用勘误覆盖，
  再做缺失/异常等校验，避免已知的源数据问题被重复上报。可用
  `--errata-csv` 覆盖默认路径（`docs/qmt_source_data_errata.csv`）。
- `quant.market_data.daily` 的增量入库（`ingest/sync.py`）：按月重建
  `bars_1d` Parquet 分片时，把勘误表转成宽表后与源 CSV 一起参与那条 `COPY`
  语句，因此落盘（dump）到日线库里的数据已经是勘误后的结果，读库的下游
  （因子研究、回测）不需要再各自处理。

两处都只覆盖 `errata.py` 声明的字段集合，不修改其余字段，也不会删除或新增行。

`quant.market_data.daily` 的增量同步只会重建本次检测到变化的月度分片：新增一条
勘误记录后，如果对应月份的源 CSV 没有其它变化，该月分片不会被自动重写。要让
已入库月份追溯应用新增的勘误，需要用 `--sync-mode rebuild`（或等价的
`DailySyncConfig(mode="rebuild")`）触发一次整库重建。
