# 大 QMT 日级数据保存器

本目录只负责通过大 QMT 内置 Python 获取并保存数据，不提供 QMT 内的数据读取、选股或回测功能，也不依赖 `xtquant`。

## 文件职责

- `scripts/qmt_run_downloader.py`：在大 QMT 编辑器中运行的唯一入口。
- `config.py`：JSONC 配置读取和校验。
- `gateway.py`：大 QMT 行情、上市退市、原始财务和除权接口适配。
- `runner.py`：批量回溯、日增量、分批和断点续传编排。
- `storage.py`：按日分区、临时文件原子替换和完成标记。
- `finance.py`：根据公告日生成日级财务快照。
- `validation.py`：缺失、重复和行情价格关系检查。
- `state.py`：SQLite 批次状态；不保存业务数据。
- `logging_setup.py`：QMT 输出窗口和外部滚动日志。

## 第一次小范围测试

1. 确认项目位于 `C:\Users\win10\Documents\quant`。如果移动了目录，只需修改 `scripts/qmt_run_downloader.py` 的 `PROJECT_ROOT`，`SOURCE_ROOT` 与 `CONFIG_PATH` 会随之推导。
2. 在大 QMT 的“数据管理”中先下载财务数据。内置 Python 的财务读取接口只读取客户端已有财务缓存，不能在本工具内自动补齐财务缓存。
3. 打开 `scripts/qmt_run_downloader.py`，使用大 QMT 内置 Python 运行，**不要勾选“启动本地 Python”**。
4. 测试配置放在 `configs/qmt_downloader/` 下：两只股票、`20260810` 至 `20260812`、每批一只，结果写入 `D:\qmt_data_test`。
5. 查看 QMT 输出窗口，同时检查 `D:\qmt_data_test\logs\downloader.log` 和 `D:\qmt_data_test\reports\issues_*.csv`。

测试日期位于未来或服务器尚无该交易日数据时，日线会为空并在问题报告中明确提示。此时应把配置日期改为客户端已有的最近三个交易日。

## 配置文件格式

配置文件扩展名保持 `.json`，但按 JSONC 解析：支持 `//` 行注释、`/* */` 块注释，以及对象和数组闭合括号前的多余逗号（注释掉最后一项时不必再回头删逗号）。字符串内部的 `//`、`/*` 和逗号原样保留，Windows 路径的 `\\` 不会被误判为字符串结束。注释在解析前替换为等长空白并保留换行，因此报错的行列号仍指向原文件位置。未知键同样会被忽略，`quant-qmt-self-check` 读取同一份配置时使用相同的解析规则。

VS Code 会把带注释的 `.json` 标红，`.vscode/settings.json` 已把 `configs/qmt_downloader/*.json` 关联为 `jsonc` 语言模式。

## 保存布局

```text
D:\qmt_data\
├─ kline_1d\date=20260812\data.csv
├─ instrument_info\snapshot=latest\data.csv
├─ finance_raw\table=income\announce_date=20260812\data.csv
├─ finance_daily\date=20260812\data.csv
├─ corporate_actions\ex_date=20260812\data.csv
├─ run_complete\date=20260812\_SUCCESS.json
├─ staging\qmt_xxx\...\batch_00000.csv
├─ state\downloader_state.sqlite
├─ logs\downloader.log
├─ reports\issues_20260812_xxx.csv
└─ reports\line_correct\corrections_20260812_xxx.csv
```

每个业务分区都有 `_SUCCESS.json`，其中包含行数、证券池范围和 CSV 的 SHA-256。写入顺序为 `data.csv.tmp` → 原子替换 `data.csv` → `_SUCCESS.json.tmp` → 原子替换 `_SUCCESS.json`。增量模式默认跳过同一证券池的已有完成分区，因此不会重写历史文件；若同一输出目录和日期改用了不同证券池，任务会明确报错，需使用 `repair` 或新输出目录，避免静默丢股票。

选择 `kline_1d` 时，会先逐只保存 `instrument_info/snapshot=latest`，包含上市日期、退市日期、当前交易状态和停牌状态，然后才开始收集日线。逐日缺口检查直接用上市/退市日期跳过未上市或已退市的证券日，只有上市期间填充后仍无日线的记录才进入问题报告。生命周期信息必须先取得：全区间回溯下未上市和已退市的证券日占三成以上，若等日线收集结束后再过滤，中间会堆积上千万条随即被丢弃的提示。也正因如此，上市退市信息获取失败时会直接跳过本次日线收集并报 `ERROR`，避免耗时数小时后才发现问题报告不可用；修复后重新运行会从断点继续。

日线源数据偶见最高价低于开收低、最低价高于开收高的脏行（多为九十年代历史缓存，重新下载结果不变）。这类行不再判为 `ERROR` 阻断整跑，而是行级修正：最高价抬到开、收、低三者最大值，最低价压到开、收、高三者最小值。每次修正都会写一条 `WARNING` 进问题报告，并把原值与修正值逐行落入 `reports/line_correct/corrections_<run_id>.csv`（每次运行都会生成该报告，无修正时只有表头）；运行摘要中的 `corrections` 为修正行数。主键重复、成交量或金额为负等其他质量错误仍按 `ERROR` 阻断批次。

每次运行结束都会在日志写一行 `任务结束 用时=H:MM:SS summary=...`，摘要里的 `elapsed` 为同一文本、`elapsed_seconds` 为浮点秒。计时从证券池解析开始，覆盖下载、校验和分区落盘的全过程；交易日历失败、区间无交易日和增量已是最新这三条提前结束的路径同样带耗时字段，便于对比历次全市场回溯的实际开销。

跳过分区前会重新核验数据文件、行数和 SHA-256。一次任务的全部所选数据集成功后，才会在 `run_complete/date=YYYYMMDD/_SUCCESS.json` 写整日水位；自动增量只根据这个全局水位前进，因此日 K 成功但财务或除权失败时不会越过失败日期。

`corporate_actions` 的现金、送股、转增和配股数量均保留大 QMT 的“每股”口径。没有事件的实际交易日也会生成只有表头的空 CSV，表示已检查但当日无事件，而不是下载失败。

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

任务键由日期范围、模式、数据集和证券池等关键配置生成。异常退出后，以完全相同配置再次运行会跳过已完成批次，从未完成批次继续。每个 staging CSV 也有独立的行数和 SHA-256 元数据，损坏后会自动使断点失效并重新下载。不要手动删除 `staging` 或 `state/downloader_state.sqlite`，否则相应断点会失效。

## 常见提示

- 财务表全空：先在大 QMT 数据管理中下载相应日期范围的财务数据，再以相同配置重跑。
- 财务记录存在但业务字段全空：默认视为未完成；个别字段缺失会写入问题报告并保留原始空值，不会用其他数值静默填充。
- 日线全空：确认目标日期已经收盘、行情权限可用，或改成最近已有的交易日。
- 某只股票某日缺失：报告会提示；常见原因包括停牌、未上市或本地缓存不完整。
- 批次失败定位：终端和 `downloader.log` 会输出 `dataset`、批次序号、证券列表、日期区间、接口阶段、异常类型和原始错误；问题 CSV 同时保留逐条 `code/date/message`。
- 已完成分区被跳过：这是默认安全行为；确需覆盖时使用 `repair`。
- 勾选“启动本地 Python”后没有输出：该模式不会为本入口注入大 QMT 的 `ContextInfo` 和内置全局下载函数。
