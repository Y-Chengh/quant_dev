# 大 QMT 日级数据保存器

本目录只负责通过大 QMT 内置 Python 获取并保存数据，不提供 QMT 内的数据读取、选股或回测功能，也不依赖 `xtquant`。

## 文件职责

- `scripts/qmt_run_downloader.py`：在大 QMT 编辑器中运行的唯一入口。
- `config.py`：JSONC 配置读取和校验。
- `gateway.py`：大 QMT 行情、上市退市、原始财务和除权接口适配。
- `runner/`：批量回溯、日增量、分批和断点续传编排。按职责拆为 mixin：
  `instruments`（证券池）、`issues`（问题降级）、`progress`（进度与并行落盘）、
  `kline`（日线与缺口修复）、`finance_steps`（财务）、
  `corporate_actions`（除权）、`trading_calendar`（交易日历落表）、
  `partitions`（分区写入与水位），
  `downloader` 只做编排，共享状态声明在 `base`。
- `storage.py`：按日分区、临时文件原子替换和完成标记。
- `errata.py`：人工核实的源数据勘误表装载与应用，供 `self_check` 与
  `quant.market_data.daily` 增量入库共用，详见
  [qmt_source_data_errata.md](qmt_source_data_errata.md)。
- `finance.py`：根据公告日生成日级财务快照。
- `validation.py`：缺失、重复和行情价格关系检查。
- `state.py`：SQLite 批次状态；不保存业务数据。
- `logging_setup.py`：QMT 输出窗口和外部滚动日志。

## 第一次小范围测试

1. 确认项目位于 `C:\Users\win10\Documents\quant`。如果移动了目录，只需修改 `scripts/qmt_run_downloader.py` 的 `PROJECT_ROOT`，`SOURCE_ROOT` 与 `CONFIG_PATH` 会随之推导。
2. 在大 QMT 的“数据管理”中先下载财务数据。内置 Python 的财务读取接口只读取客户端已有财务缓存，不能在本工具内自动补齐财务缓存。
3. 打开 `scripts/qmt_run_downloader.py`，使用大 QMT 内置 Python 运行，**不要勾选“启动本地 Python”**。
4. 测试配置放在 `configs/qmt_downloader/` 下：两只股票、`20260810` 至 `20260812`、每批一只，结果写入 `D:\qmt_data_test`。
5. 查看 QMT 输出窗口，同时检查 `D:\qmt_data_test\logs\downloader.*.log`（每次运行一个文件）和 `D:\qmt_data_test\reports\issues_*.csv`。

测试日期位于未来或服务器尚无该交易日数据时，日线会为空并在问题报告中明确提示。此时应把配置日期改为客户端已有的最近三个交易日。

## 配置文件格式

配置文件扩展名保持 `.json`，但按 JSONC 解析：支持 `//` 行注释、`/* */` 块注释，以及对象和数组闭合括号前的多余逗号（注释掉最后一项时不必再回头删逗号）。字符串内部的 `//`、`/*` 和逗号原样保留，Windows 路径的 `\\` 不会被误判为字符串结束。注释在解析前替换为等长空白并保留换行，因此报错的行列号仍指向原文件位置。未知键同样会被忽略，`quant-qmt-self-check` 读取同一份配置时使用相同的解析规则。

VS Code 会把带注释的 `.json` 标红，`.vscode/settings.json` 已把 `configs/qmt_downloader/*.json` 关联为 `jsonc` 语言模式。

## 退市标的与幸存者偏差

`sector` 指定的板块（如 `沪深A股`）只包含**当前存续**的证券，用它回溯出来的数据集不含任何已退市标的。用这样的数据回测会系统性高估收益、低估回撤，尤其是低价股、小市值和困境反转类策略：退市前的连续跌停完全不在样本里。

`expired_sectors` 用于消除这一偏差，取值为大 QMT 的过期板块名列表，例如 `["过期沪深A股"]`。它是**附加项**：无论存续证券来自 `symbols` 还是 `sector`，过期板块的代码都会并入同一个证券池。留空（默认）时行为与该功能加入之前完全一致。

启用前需要在客户端做一次性准备：

1. 大 QMT 界面端「数据管理 → 过期合约数据 → 过期合约列表」勾选下载。
2. **重启客户端**，否则内置 Python 读不到过期板块。
3. 在内置 Python 里执行 `print([s for s in ContextInfo.get_sector_list() if "过期" in s])`，以本机实际返回的板块名为准填写配置。常见名称有 `过期沪深A股`、`过期上证A股`、`过期深证A股`、`过期科创板`。

证券池解析整体是 fail-closed 的：宁可报错，也不静默缩池——静默缩池等于偏差重新出现且毫无痕迹。具体判定如下。

| 情况 | 处理 |
| --- | --- |
| 过期板块没有成分证券，且板块名不在 `get_sector_list()` 返回的板块列表中（写错名字，或过期合约列表根本没下载） | 报错，提示上述准备步骤 |
| 过期板块没有成分证券，但板块名确实在板块列表中（例如该市场尚无退市标的） | 记 `WARNING` 继续，不因多填一个正确板块名而阻断其余板块 |
| 配置的过期板块**合计**没带回任何证券 | 报错，此时证券池已经退回只含存续标的。因此只配一个板块且它为空时，上一行的放行不生效 |
| `sector` 指定的存续板块没有成分证券 | 报错。这一条单独判定，否则过期板块会让整体证券池非空，存续板块名写错就会静默跑出一份纯退市数据集 |

`get_sector_list()` 不可用时退回更严格的判定：过期板块为空一律报错。“不可用”包括接口缺失、调用抛错和返回空列表三种情况。

`symbols` 与 `sector` 同时留空、只填 `expired_sectors` 是合法配置，用于单独回补退市标的；此时不会拿空板块名去查接口。

两点使用注意：

- **证券池规模明显变大**，`batch_size`、单次回溯耗时和分区行数都会上升。
- **不要在已有输出目录上直接打开该开关**。证券池变化会同时改变 `job_key`、`partition_scope` 和 `watermark_scope`：断点全部失效，已完成分区的范围核验也会失败并明确报错（见「保存布局」末段）。正确做法是换一个新的 `output_root` 做全量回溯，或按 `repair` 模式重写。反过来，`expired_sectors` 留空时 `watermark_scope` 不会新增该键，既有输出目录的自动增量不受这次改动影响。

过期板块给出的是**代码和合约信息**，不等于行情数据。退市标的的日线能否补齐取决于客户端「数据管理 → 补充行情数据」是否覆盖过期合约，首次启用后应检查问题报告中这批代码的日线缺口。

## 保存布局

```text
D:\qmt_data\
├─ kline_1d\date=20260812\data.csv
├─ instrument_info\snapshot=latest\data.csv
├─ instrument_info\observed_date=20260812\data.csv
├─ trading_calendar\snapshot=latest\data.csv
├─ finance_raw\table=income\announce_date=20260812\data.csv
├─ finance_daily\date=20260812\data.csv
├─ corporate_actions\ex_date=20260812\data.csv
├─ run_complete\date=20260812\_SUCCESS.json
├─ staging\qmt_xxx\...\batch_00000.csv
├─ state\downloader_state.sqlite
├─ logs\downloader.20260812.09.inc.log
├─ reports\issues_20260812_xxx.csv
└─ reports\line_correct\corrections_20260812_xxx.csv
```

根目录即配置中的 `output.root`，其下分两类：`kline_1d` 至 `corporate_actions` 为业务数据，`run_complete` 及之后为运行时资产。各目录含义如下。

| 目录 | 分区键 | 含义 |
| --- | --- | --- |
| `kline_1d\` | `date=YYYYMMDD` | 日 K 线，一个交易日一个分区，列为 `code,trade_date,open,high,low,close,pre_close,volume,amount,suspend_flag`。非交易日不建目录，因此日期序列本身带有节假日跳跃。 |
| `instrument_info\` | `snapshot=latest` | 标的基础信息，不按日期分区而是整体覆盖为最新状态，列为 `code,instrument_name,open_date,expire_date,is_trading,instrument_status`。 |
| `instrument_info\` | `observed_date=YYYYMMDD` | 与快照同列同内容的**观测日**副本，累积证券简称等状态的历史，见下文“证券名称历史”。由 `save_instrument_history` 控制，缺省开启。分区名不是 `date`，因为它是观测日而非交易日。 |
| `trading_calendar\` | `snapshot=latest` | 大 QMT 交易日历，同样整体覆盖为最新状态，列为 `trade_date,calendar_symbol`。仅当 `datasets` 含 `trading_calendar` 时生成。 |
| `finance_raw\` | `table=<来源表>\announce_date=YYYYMMDD` | 原始财务，先按来源表再按实际公告日两级分区；公告日缺失的记录落到 `announce_date=unknown`。 |
| `finance_daily\` | `date=YYYYMMDD` | 日级财务快照，每天每只股票一行。 |
| `corporate_actions\` | `ex_date=YYYYMMDD` | 除权送转事件，按除权日分区；当日无事件时保留只有表头的空 CSV。 |
| `run_complete\` | `date=YYYYMMDD` | 整日水位标记，目录内只有 `_SUCCESS.json`，没有数据文件。 |
| `staging\` | `<任务键>\<staging 数据集>` | 断点续跑的中间批次，见下文。 |
| `state\` | 无 | `downloader_state.sqlite` 批次状态库，不保存业务数据。 |
| `logs\` | 无 | 每次运行一个 `downloader.<日期>.<小时>.<模式>.log`，见“运行日志”。 |
| `reports\` | 无 | `issues_<run_id>.csv` 问题报告，以及 `line_correct\corrections_<run_id>.csv` 行级修正明细。 |

业务分区目录固定包含 `data.csv` 和 `_SUCCESS.json` 两个文件，缺任何一个都视为未完成。`staging` 下第一层是任务键 `qmt_<哈希>`，第二层是 staging 数据集名：日线为 `kline_1d`（批量读取阶段）和 `kline_daily_<日期>`（按日切分），原始财务为 `finance_raw_<表名>_<公告日>`；第三层是 `batch_00000.csv` 与配套的 `batch_00000.meta.json`（只存 `rows` 和 `sha256`）。staging 不会在任务成功后自动删除，它同时是下次同配置运行的断点来源。

据此可以判断一份输出目录的完成状态：`run_complete` 缺失或日期数少于 `kline_1d`，说明所选数据集未全部成功，自动增量不会从这些日期前进；`staging` 子目录数明显多于最终分区数，说明上一次运行中断在合并落盘之前。两者都不需要手工清理，以相同配置重跑即可续上。

每个业务分区的 `_SUCCESS.json` 包含完成时间、行数、`job_key`、`mode`、CSV 的 SHA-256，以及记录当次证券池、数据集列表和财务字段的 `partition_scope`。写入顺序为 `data.csv.tmp` → 原子替换 `data.csv` → `_SUCCESS.json.tmp` → 原子替换 `_SUCCESS.json`。增量模式默认跳过同一证券池的已有完成分区，因此不会重写历史文件；若同一输出目录和日期改用了不同证券池，任务会明确报错，需使用 `repair` 或新输出目录，避免静默丢股票。

选择 `kline_1d` 时，会先逐只保存 `instrument_info/snapshot=latest`，包含上市日期、退市日期、当前交易状态和停牌状态，然后才开始收集日线。逐日缺口检查直接用上市/退市日期跳过未上市或已退市的证券日，只有上市期间填充后仍无日线的记录才进入问题报告。生命周期信息必须先取得：全区间回溯下未上市和已退市的证券日占三成以上，若等日线收集结束后再过滤，中间会堆积上千万条随即被丢弃的提示。也正因如此，上市退市信息获取失败时会直接跳过本次日线收集并报 `ERROR`，避免耗时数小时后才发现问题报告不可用；修复后重新运行会从断点继续。

日线源数据偶见最高价低于开收低、最低价高于开收高的脏行（多为九十年代历史缓存，重新下载结果不变）。这类行不再判为 `ERROR` 阻断整跑，而是行级修正：最高价抬到开、收、低三者最大值，最低价压到开、收、高三者最小值。每次修正都会写一条 `WARNING` 进问题报告，并把原值与修正值逐行落入 `reports/line_correct/corrections_<run_id>.csv`（每次运行都会生成该报告，无修正时只有表头）；运行摘要中的 `corrections` 为修正行数。主键重复、成交量或金额为负等其他质量错误仍按 `ERROR` 阻断批次。

每次运行结束都会在日志写一行 `任务结束 用时=H:MM:SS summary=...`，摘要里的 `elapsed` 为同一文本、`elapsed_seconds` 为浮点秒。计时从证券池解析开始，覆盖下载、校验和分区落盘的全过程；交易日历失败、区间无交易日和增量已是最新这三条提前结束的路径同样带耗时字段，便于对比历次全市场回溯的实际开销。

跳过分区前会重新核验数据文件、行数和 SHA-256。一次任务的全部所选数据集成功后，才会在 `run_complete/date=YYYYMMDD/_SUCCESS.json` 写整日水位；自动增量只根据这个全局水位前进，因此日 K 成功但财务或除权失败时不会越过失败日期。

`corporate_actions` 的现金、送股、转增和配股数量均保留大 QMT 的“每股”口径。没有事件的实际交易日也会生成只有表头的空 CSV，表示已检查但当日无事件，而不是下载失败。

## 运行日志

每次运行单独写一个日志文件，命名为 `logs\downloader.<YYYYMMDD>.<HH>.<模式>.log`。模式片段为 `back_fill`、`inc` 或 `repair`，分别对应配置中的 `backfill`、`incremental` 和 `repair`；例如 2026-08-12 上午九点启动的日增量写入 `downloader.20260812.09.inc.log`。文件名只精确到小时，同一小时内重复运行会追加从 `2` 起的序号（`downloader.20260812.09.inc.2.log`），因此两次运行不会混进同一份日志。历史日志一律保留，不再被后续运行覆盖或续写，何时清理由使用者决定。

`log_max_bytes` 和 `log_backup_count` 仍然生效，但只在单次运行内部轮转：一次全市场回溯写满 `log_max_bytes` 后生成 `....log.1`，最多保留 `log_backup_count` 个，不会波及其它运行的日志。

日志起始固定记录本次运行的完整配置，事后复现不必再回头比对配置文件：

```text
2026-08-12 09:00:01 INFO 本次运行日志文件=D:\qmt_data\logs\downloader.20260812.09.inc.log
2026-08-12 09:00:01 INFO 生效配置开始
2026-08-12 09:00:01 INFO 配置 config_path=C:\Users\win10\Documents\quant\configs\qmt_downloader\incremental.example.json
2026-08-12 09:00:01 INFO 配置 mode=incremental
2026-08-12 09:00:01 INFO 配置 start_date=20260810
2026-08-12 09:00:01 INFO 配置 end_date=20260811
...
2026-08-12 09:00:01 INFO 生效配置结束
```

记录的是生效值而不是配置文件原文：`start_date` 与 `end_date` 的 `auto` 已解析为实际日期，`incremental_lag_days` 的 `auto` 已解析为 `0` 或 `1` 并附带判定时刻 `incremental_lag_decision_time`，`datasets` 也已派生出 `business_datasets` 和 `save_trading_calendar`。每项单独一行，`配置 batch_size=` 这类前缀可直接检索，也避免超长单行在 QMT 输出窗口被截断。新增配置项必须同步写入 `DownloaderConfig.describe()`，该约束由 `test_describe_covers_every_configuration_key` 守护。

## 自检进度日志

`quant-qmt-self-check` 会把阶段起止、耗时和长循环进度写到标准输出，全量运行不再是一段完全静默的等待：

```text
2026-08-15 13:02:17 INFO [自检] 开始 output_root=D:\qmt_kline_test1 start_date=20260810 end_date=20260815
2026-08-15 13:02:17 INFO [自检] 阶段开始 发现日线分区
2026-08-15 13:02:17 INFO [自检] 待校验日线分区 6450 个 root=D:\qmt_kline_test1\kline_1d
2026-08-15 13:02:18 INFO [自检] 分区校验 322/6450 (5.0%) 已用 0.7s 预计剩余 14.0s date=20010517
...
2026-08-15 13:02:36 INFO [自检] 阶段完成 发现日线分区 耗时 18.7s 来源=partitions 分区=6450 个
2026-08-15 13:02:52 INFO [自检] 逐日扫描 1/3 (33.3%) 已用 0.5s 预计剩余 1.1s date=20260810 行数=5209 覆盖率=100.00% 累计问题=19
2026-08-15 13:02:53 INFO [自检] 结束 状态=failed 总耗时 36.0s errors=28 warnings=2 report=...
```

阶段依次为发现日线分区、装载证券信息与证券池、装载交易日历、解析证券生命周期与除权事件、审计日历外分区、逐日扫描行情、汇总并写出报告。其中分区校验（逐个分区读完成标记并对 `data.csv` 做全文件 SHA-256）、除权事件读取和逐日扫描三个长循环带百分比、已用时间和预计剩余时间；剩余时间按已完成元素的平均耗时线性外推，首条进度还没有可测速率，显示为“未知”。

三个开关控制输出量：

- `--log-level`：缺省 `INFO`；`DEBUG` 逐个分区、逐个交易日输出，用于定位卡在哪一个日期；`WARNING` 及以上只保留告警。
- `--progress-every N`：每前进 N 个元素输出一条 INFO 进度。缺省 `0` 表示按总量的 5% 自动选择，即每个长循环约 20 条。
- `--log-file`：把同样的日志追加写入指定 UTF-8 文件，终端输出不受影响。

自检是库层代码，只向 `quant.qmt_downloader.self_check` 这个日志器写入而不安装任何处理器，是否输出、输出到哪里由命令行入口决定；被其它程序作为库调用时不会污染宿主的日志配置。

## 自检的区间裁剪

`--start-date`/`--end-date`（或配置里的日期范围）不只筛选参与统计的交易日，也决定读哪些分区：分区校验和除权事件读取都先按目录名把 `date=YYYYMMDD`、`kline_daily_YYYYMMDD`、`ex_date=YYYYMMDD` 裁到审计区间内，再进入耗时循环。区间外的分区既不参与覆盖率统计，也不会被行级校验读取，逐个做全文件 SHA-256 纯属浪费——在 6450 个分区、5209 只证券的库上只查三天，裁剪前后是 39 秒对 4 秒，errors/warnings 完全一致。

代价是区间外分区的哈希、行数和口径一致性在本次运行中不再被检查。要覆盖全部历史分区就不要传日期范围：两端都缺省时不做任何裁剪，行为与以前完全相同。目录名不是合法八位日期的分区无法判断归属，一律保留并照常报 `INVALID_PARTITION_DATE`，不会因为设了区间被静默跳过。

## 交易日历落表

`trading_calendar` 是与 `kline_1d` 平级的数据集开关，写进 `datasets` 即落表，去掉即不落表；`datasets` 整个缺省时（配置未写该键）默认全选，因此也包含它。程序本来每次运行都要按 `calendar_symbol` 请求一次交易日历用于校验预期交易日，落表只是把这份已经拿到的结果保存下来，不产生任何额外接口调用。

保存位置是 `trading_calendar/snapshot=latest/`，与 `instrument_info` 一样按快照整体覆盖，同样带 `data.csv` 和 `_SUCCESS.json`，因此哈希和行数可被校验。列为 `trade_date` 和取得该日历所用的 `calendar_symbol`。

落表是**累积**的：每次运行读回已有快照，与本次区间求并集后重写，本次区间内的日期以本次结果为准，区间外的历史日期原样保留。日增量每天只请求一两天，若直接覆盖，历史日历会被截成一天。代价是合并只增不删——某个日期一旦写入就不会因为后续运行不再返回它而消失；需要彻底重建时删除该快照目录，再以覆盖完整区间的配置跑一次。

该开关不参与任务键、分区范围和整日水位范围的计算：它不产生 staging 批次，也不改变任何业务分区的内容。因此在已有输出目录上随时开关它都不会让断点失效、不会触发已完成分区的范围冲突，也不会让自动增量的水位重新对不上。只把 `datasets` 设为 `["trading_calendar"]` 是合法配置，表示只导日历、不下载业务数据，此时不写整日水位。

落表失败按所选数据集失败处理：记 `ERROR`、不推进整日水位，重跑时已完成的业务分区会因范围一致而快速跳过。

`quant-qmt-self-check` 会自动读取这份快照作为权威交易日历，无需再传 `--calendar-csv`；没有权威日历时自检只能用已有分区日期推导，会直接报 `CALENDAR_INFERRED_FROM_PARTITIONS` 错误，因为那种退化模式发现不了整个交易日分区完全缺失。手工导出到 `trading_calendar/data.csv` 的扁平单列文件仍然被识别。

## 证券名称历史

`instrument_info/snapshot=latest` 每次运行整体覆盖，只保留**当前**的证券简称，因此还原不了「某只票 2019 年叫什么名字」，也就还原不了历史 ST 状态——而大 QMT 没有任何接口能补回过去的名称，这份信息一旦没存就永久丢失。为此下载器在写快照的同时，会把同一份内容按观测日再存一份到 `instrument_info/observed_date=YYYYMMDD/`，列与快照完全一致，从此累积出可审计的名称历史。

分区键是**观测日**而不是交易日或 `end_date`：回补 2020 年行情时，大 QMT 返回的仍是今天的简称，按请求区间归档等于凭空造出一段假历史。按观测日归档只声明「这一天我们看到的是这些名称」，回补、重跑和跨年长任务都不会污染其它日期。日期取配置读取时刻的 Asia/Shanghai 日历日，日志和 `describe()` 中的 `observation_date` 就是它。分区名刻意不叫 `date`：同一根目录下 `kline_1d/date=` 是交易日，两者语义不同，同名迟早会被下游当成一回事。

**不是每次运行都会产生观测日分区**：它随 `instrument_info` 一起写，而后者只在 `datasets` 含 `kline_1d` 时收集，且排在 `no_work`（增量已是最新）、区间无交易日、交易日历失败这三条提前返回之后。因此只跑财务或日历的任务不会留下当天的记录；周末和节假日则取决于水位——自动增量已是最新时不写，而固定区间的 `backfill`/`repair` 任务在周末跑照样会写下当天的观测记录。名称历史序列本身带空洞是正常的，按前一个有记录的观测日向前填充即可，不要当成缺陷。

同一观测日重复运行时与已有分区**求并集**而不是直接覆盖，本次结果优先，本次没有返回的代码保留已有行。正常情况下窄证券池的任务也不会让内容缩水——查询池已经并入了上一份快照中未退市的代码——但以下三种情况会让当日快照真的变小：快照被删除或损坏时它退化成只有本次证券池；因详情读取失败被移出快照的代码随之消失；`end_date` 靠前的回补任务会按 `end_date` 前一日裁掉更早退市的代码。求并集是这些情况下唯一的兜底。已有分区存在、非空却读不出来时**放弃写入**并记 `WARNING`，绝不覆盖：那份文件里可能还有当天唯一的一份名称。

用 `save_instrument_history` 关闭，缺省开启。全市场一天的 `data.csv` 约几百 KB，但每个分区还有一份内嵌完整证券池的 `_SUCCESS.json`（实测约 106 KB），合计一年约 130 MB。该开关不参与任务键、分区范围和整日水位范围的计算，随时开关都不会让断点失效或触发范围冲突。落表失败只记 `WARNING` 不中断任务：它是纯增量的旁路数据，没有下游流程依赖，此时快照本身已经写成功。

这份历史目前只负责积累，`quant-qmt-self-check` 和日线入库都仍然只读 `snapshot=latest`，历史涨跌停校验的 ST 口径限制见 [market_check.md](market_check.md)。

## 财务日期口径

- `finance_raw` 保存报告期 `report_date` 和实际公告日 `announce_date`，按公告日分区。
- `finance_daily` 每天每只股票一行，并保留每个来源表的报告期和公告日。
- 日快照只使用 `announce_date < trade_date` 的记录。这样当天盘后发布的财报不会进入当天收盘特征；它从下一个实际交易日开始可见。
- 公告日缺失的原始记录保存到 `announce_date=unknown`，同时写问题报告，但不会进入日快照。
- `allow_partial_finance` 默认为 `false`：任一证券或财务表缺失时批次保持可重试，不推进整日水位。确认缺失属于正常业务事实且愿意接受不完整数据时，才改为 `true`。

全市场长区间不会把全部日线或“股票 × 交易日”财务宽表一次放入内存：程序按 `batch_size` 逐批生成按日 staging，再一次只合并一个交易日。原始财务也只为未完成的公告日生成片段，已有同范围历史公告分区不会重复展开。`save_workers` 控制本地 CSV staging、日线最终分区和除权最终分区的写入线程数，默认 `1`；它只并行互不冲突的文件写入，不并行调用 QMT 接口，建议磁盘较快时设置为 `4`～`8`，最大为 `16`。

`batch_size` 除了决定单批证券数，还直接决定 staging 小文件的数量：按日 staging 会产生 `交易日数 × 批次数` 个 CSV，每个再配一个 `.meta.json`。全市场跨三十余年回溯时，`batch_size=100` 约产生七十万个文件，`batch_size=300` 降到约二十五万个，实测每交易日的写入和读回耗时都缩短一半以上。长区间回溯建议用 `300`，并把输出目录加入杀毒软件的实时扫描排除列表——小文件数量级下，实时扫描往往比代码本身更影响耗时。

## 批量回溯与日增量

批量回溯把 `mode` 设为 `backfill` 并填写较长起止区间。正式运行前建议先用两只股票、三个已结束交易日验证数据权限和字段。

日增量复制 `config.incremental.example.json`，保持 `start_date` 和 `end_date` 为 `auto`。程序会从最近一个带 `_SUCCESS.json` 的日线分区下一自然日开始，一直下载到自动目标日；周末会自然落空，不会创建伪交易日。首次运行尚无日线分区时，从 `incremental_initial_start_date` 开始。

`incremental_lag_days` 默认为 `1`，适合每天早晨定时运行并保存截至昨日的完整收盘数据；如果确定任务在当日收盘后运行，可以设为 `0`。设为 `"auto"` 时以上海时间 `incremental_lag_auto_cutoff`（默认 `16:00`，格式 `HH:MM`）为界：边界前按滞后 1 天、边界及以后按 0 天。该判定发生在读取配置的时刻，请确保定时任务在大 QMT 完成当日数据同步之后再启动；若 16:00 时本地缓存尚未就绪，应把边界调晚，否则全市场缺少当日日线会被记为错误并要求重跑。已是最新时任务返回 `status=up_to_date`，不会再次请求数据。运行结束后每个新交易日形成独立目录，不修改历史分区。若需主动替换一个已完成分区，将 `mode` 改为 `repair` 并明确设置 `overwrite_completed_partition: true`。

程序使用 `calendar_symbol` 的大 QMT 交易日历（默认 `000001.SH`）检查预期交易日。整个市场某个预期交易日都没有返回日线时，批次不会标为完成，也不会推进水位。

每只证券首个和最后一个已有交易日之间如果存在日线缺口，程序会按证券合并连续缺失日期区间，再调用大 QMT 执行定向补下载。`kline_gap_retry_count` 控制补下载轮数，默认 `2`，允许范围为 `1`～`10`。达到上限后仍缺失会升级为 `ERROR`，批次保持可重试且不会生成最终日线分区；停牌补齐行不会被误判为缺口，上市前和退市后的日期也不属于“中间缺口”。定向补下载始终会先调用大 QMT 历史下载刷新对应区间的本地缓存；`download_kline` 只控制每批首轮批量读取前是否全量下载。补齐后的数据在写最终分区时按业务主键差异原地重写同范围旧分区，不再需要预先删除完成标记。

补充本地缓存时优先使用大 QMT 的批量历史下载接口 `download_history_data2`，一次请求整批证券。逐只下载的开销几乎全在接口往返上：实测 600 只证券的一批耗时约 30 秒、占单批总耗时的一半以上，批量化后这部分基本消失。入口脚本会依次尝试大 QMT 注入的全局 `download_history_data2` 和 `xtquant.xtdata.download_history_data2`，都取不到时自动改为逐只下载并在日志中说明。批量调用整体失败也会立即回退逐只下载，保留把失败归因到具体证券的能力；批量成功但个别证券实际未补齐时，后续的缺口检查和定向补下载仍会兜住。若批量接口在你的大 QMT 版本上行为异常，把配置项 `download_kline_batch` 设为 `false` 即可一键关闭。下载阶段的开始和完成都会写日志，可用 `stage=download_history` 检索并核对 `mode=batch` 还是 `mode=per_symbol`。

任务键由日期范围、模式、数据集和证券池等关键配置生成。异常退出后，以完全相同配置再次运行会跳过已完成批次，从未完成批次继续。每个 staging CSV 也有独立的行数和 SHA-256 元数据，损坏后会自动使断点失效并重新下载。不要手动删除 `staging` 或 `state/downloader_state.sqlite`，否则相应断点会失效。

## 常见提示

- 财务表全空：先在大 QMT 数据管理中下载相应日期范围的财务数据，再以相同配置重跑。
- 财务记录存在但业务字段全空：默认视为未完成；个别字段缺失会写入问题报告并保留原始空值，不会用其他数值静默填充。
- 日线全空：确认目标日期已经收盘、行情权限可用，或改成最近已有的交易日。
- 某只股票某日缺失：报告会提示；常见原因包括停牌、未上市或本地缓存不完整。
- 批次失败定位：终端和当次运行日志会输出 `dataset`、批次序号、证券列表、日期区间、接口阶段、异常类型和原始错误；问题 CSV 同时保留逐条 `code/date/message`。
- 已完成分区被跳过：这是默认安全行为；确需覆盖时使用 `repair`。
- 勾选“启动本地 Python”后没有输出：该模式不会为本入口注入大 QMT 的 `ContextInfo` 和内置全局下载函数。
