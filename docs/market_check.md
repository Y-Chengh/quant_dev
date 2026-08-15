# 日线库合法性审计

> 想按步骤照着做，先看 [上手指南](qmt_daily_guide.md)；本文是口径与参数的参考手册。

`quant-market-check` 对**入库之后**的日线库做全样本业务校验，只读取并报告，
不会修改、填充或删除任何数据。

与 `quant-qmt-self-check` 的分工：那一侧审计大 QMT 落盘的 CSV 分区本身是否完整
（哈希、行数、分区范围），这一侧审计转换入库后的数据在业务上是否合法，因此能发现
转换过程引入的问题，也能与 5 分钟库交叉对账。两者共用同一套问题记录结构与报告列，
输出可以直接拼在一起看。

## 用法

```powershell
quant-market-check                                   # 审计全库
quant-market-check --start-date 20240101 --end-date 20241231
quant-market-check --codes 000001.SZ 600000.SH
quant-market-check --cross-check-5m                  # 额外与 5 分钟库对账
quant-market-check --no-auto-sync                    # 跳过启动时的自动增量检查
```

退出码与 `quant-qmt-self-check` 一致：0 通过，1 存在 ERROR，2 配置或执行失败。

## 报告

写入 `<库所在目录>/reports/market_check/<时间戳>/`：

| 文件 | 内容 |
| --- | --- |
| `summary.md` / `summary.json` | 结论、区间、覆盖率、各类问题计数 |
| `issues.csv` | 全部问题明细，列与 `quant-qmt-self-check` 完全一致 |
| `missing_spans.csv` | 逐证券的连续缺失区间 |
| `coverage_by_date.csv` / `coverage_by_symbol.csv` | 逐日与逐证券覆盖率 |
| `adjust_factor_audit.csv` | 每个除权事件的观测比例、声明因子与偏差 |
| `price_limit_violations.csv` | 涨跌停越界明细 |
| `cross_5m_diff.csv` | 与 5 分钟库的差异明细 |

## 校验规则

### 行级完整性

| 编号 | 级别 | 规则 |
| --- | --- | --- |
| `DAILY_DUPLICATE_KEY` | ERROR | `(code, trade_date)` 重复 |
| `DAILY_NULL_FIELD` | ERROR | `code`/`trade_date`/`close` 为空 |
| `DAILY_NONPOSITIVE_PRICE` | ERROR | 开高低收出现非正值 |
| `DAILY_OHLC_VIOLATION` | ERROR | 最高价低于开收低，或最低价高于开收高 |
| `DAILY_NEGATIVE_QUANTITY` | ERROR | 成交量或成交额为负 |
| `DAILY_CODE_FORMAT_INVALID` | ERROR | 代码缺少交易所后缀 |
| `DAILY_INVALID_SUSPEND_FLAG` | ERROR | 停牌标记不是 0 或 1 |

### 覆盖与日历

| 编号 | 级别 | 规则 |
| --- | --- | --- |
| `DAILY_MISSING_SPAN` | ERROR | 上市至退市之间、日历上有而库里没有的连续区间 |
| `DAILY_COVERAGE_BELOW_THRESHOLD` | ERROR | 单日覆盖率低于 `--coverage-error-threshold`（缺省 0.95） |
| `DAILY_DATE_NOT_IN_CALENDAR` | ERROR | 库里出现交易日历之外的日期 |
| `DAILY_CALENDAR_DATE_EMPTY` | ERROR | 日历上的交易日在库中一根 K 线都没有 |
| `DAILY_CALENDAR_UNAVAILABLE` | ERROR | 没有交易日历，只能按库中日期反推，发现不了整天缺失 |
| `DAILY_DATA_BEFORE_LISTING` / `DAILY_DATA_AFTER_DELISTING` | ERROR | 行情越过存续期 |
| `DAILY_INVALID_LIFECYCLE_RANGE` | ERROR | 退市日早于上市日 |
| `DAILY_LIFECYCLE_UNKNOWN` | WARNING | 有行情但 `instruments` 里没有该证券 |
| `DAILY_OPEN_DATE_MISSING` | WARNING | 证券缺少上市日期 |
| `DAILY_DELISTED_SYMBOL_MISSING` | WARNING | 区间内退市的证券却零行情 |
| `DAILY_NO_DELISTED_SYMBOLS` | WARNING | 整个证券池没有任何退市股，存在幸存者偏差 |

**停牌不豁免缺失判定**：大 QMT 会给停牌日也写一行（`suspend_flag=1`），入库时又
已经切掉了未上市的填充行，所以存续期内某个交易日一行都没有，就是真的缺数据。

### 成交状态

| 编号 | 级别 | 规则 |
| --- | --- | --- |
| `DAILY_ACTIVE_ZERO_VOLUME` | ERROR | 非停牌日成交量为零 |
| `DAILY_ACTIVE_ZERO_AMOUNT` | ERROR | 非停牌日有量无额 |
| `DAILY_SUSPENDED_WITH_TURNOVER` | ERROR | 停牌日却有成交 |
| `DAILY_VWAP_OUT_OF_RANGE` | WARNING | 由成交额与成交量推出的均价落在当日价格区间之外 |
| `DAILY_VOLUME_UNIT_AMBIGUOUS` | WARNING | 成交量单位既不像股也不像手 |

成交量单位由 `median(amount / (volume × close))` 标定一次：接近 1 是股，接近 100
是手。实测该库为 **100（手）**。这个倍数会写进报告，交叉校验成交量时也依赖它。

### 价格连续性与除权

| 编号 | 级别 | 规则 |
| --- | --- | --- |
| `DAILY_PRE_CLOSE_MISMATCH` | ERROR | 前收与上一交易日收盘不一致，且当日无除权记录 |
| `DAILY_ADJUST_FACTOR_MISMATCH` | ERROR | 有除权记录，但 `close(t-1)/adjustment_factor` 与 `pre_close` 对不上 |
| `DAILY_ADJUST_THEORY_MISMATCH` | WARNING | 行情推出的比例与分红送转字段推出的理论值不一致 |
| `DAILY_CORPORATE_ACTION_WITHOUT_BAR` | WARNING | 有除权记录但当日无行情 |

价格比较在**价格维度**而不是比值维度进行，容差为一个完整报价单位（0.01 元）加上
相对容差。原因是 `pre_close` 与 `adjustment_factor` 在源数据里是各自独立四舍五入的，
两次舍入最多相差一整分钱；按纯相对容差判定，2024 年会多出四百余条纯舍入造成的假阳性。

### 涨跌停

| 编号 | 级别 | 规则 |
| --- | --- | --- |
| `DAILY_PRICE_LIMIT_VIOLATION` | WARNING | 涨跌幅超过所属板块的最大可能幅度 |

口径：科创板 20%（2019-07-22 起）、创业板 20%（2020-08-24 起，之前 10%）、
北交所 30%、主板 10%；上市首日与注册制新股前 5 个**交易日**不设限制；除权日排除。

两个刻意的设计选择：

1. **默认不按 ST 收紧到 5%。** `is_st` 来自证券简称的**当前**快照，无法还原历史
   某一天的风险警示状态——今天是 ST 的股票 2019 年未必是。实测按当前 ST 状态收紧，
   2024 一年产生 3860 条误报；按板块基础幅度只剩 26 条。因此这条校验是「有没有超过
   该板块可能的最大幅度」这一**必要条件**，需要更严格时加 `--apply-st-limit`。
2. **新股豁免按交易日而非自然日计数。** 2024 年国庆长假让 5 个交易日跨越了 13 个
   自然日，按自然日算会把仍在豁免期内的新股误判为越界。

`--price-limit-level error` 可以把这条提升为硬错误。

### 与 5 分钟库交叉校验（`--cross-check-5m`）

把 `market.duckdb` 以只读方式 `ATTACH` 进来，将 5 分钟 bar 聚合成日频后逐行比对。
两库来自完全不同的数据源（iFinD 与大 QMT），对得上是很强的正确性证据。

| 编号 | 级别 | 规则 |
| --- | --- | --- |
| `CROSS_5M_MISSING_DAILY` / `CROSS_5M_MISSING_INTRADAY` | WARNING | 一侧有、另一侧没有 |
| `CROSS_5M_CLOSE_MISMATCH` / `CROSS_5M_VOLUME_MISMATCH` | WARNING | 收盘价或成交量超出相对误差阈值 |
| `CROSS_5M_UNAVAILABLE` | INFO | 5 分钟库不存在或被占用，本项跳过 |

只比两库共同覆盖的证券，避免宇宙差异刷屏；停牌日没有 5 分钟 bar，不计入缺失。

## 已知局限

- 历史 ST 状态、历史板块归属都只有当前快照口径。
- `instrument_info` 目前不含退市股，缺失的历史证券无法被发现，只能报告偏差存在。
- 审计只报告不修复；需要补数据请回到下载器侧重新下载后再同步。
