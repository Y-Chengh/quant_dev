# 命令行参数说明

本文档以主实验入口 `python -m quant.cli.factor_demo` 为主，说明当前 `argparse` 参数、默认行为和常用命令。文末附有数据构建、行情下载和单因子检查脚本的参数速查。

## 1. 基本用法

```powershell
python -m quant.cli.factor_demo [通用参数] [所选模型的专属参数]
```

本文所有命令都写成 `python -m <模块>`，在仓库根目录、已激活 `.venv` 的会话中
执行；`pip install -e .` 之后还可用的等价短命令别名见
[README.md](../README.md#常用命令)。

不传参数时，程序会：

- 使用环境变量 `MARKET_DB_PATH` 指向的数据库；未设置时使用 `D:\量化\market.duckdb`。
- 以数据库最后时间为研究终点，向前取 3 年数据。
- 从代码表中取前 20 只证券。
- 使用全部已注册因子和因子缓存。
- 使用 `simple_decision_tree` 模型。
- 从研究终点向前 1 年开始逐日扩展窗口验证。
- 每个验证日按模型分数选择前 10 只证券，开盘等权买入、收盘卖出，并计算收益曲线和年化夏普比率。
- 以 `INFO` 等级同时写终端日志和 `logs/run/YYYY-MM-DD/HH/` 下的运行日志、评估报告、准确率趋势图、IC/Rank IC 双周期趋势图及收益曲线。

模型参数采用两阶段解析：程序先读取 `--model`，然后只注册所选模型的参数。因此，不同模型可以有同名参数；某个模型的专属参数不能用于另一个模型。

查看实际可用参数：

```powershell
python -m quant.cli.factor_demo --help
python -m quant.cli.factor_demo --model gradient_boosting_tree --help
python -m quant.cli.factor_demo --model lightgbm --help
```

任务类型由 `--task` 控制：`classification`（默认）执行涨跌二分类，
`regression` 预测下一交易日开盘至收盘的连续涨跌幅。回归任务当前需选择
`--model lightgbm`。LightGBM 的 `--objective` 默认随任务选择 `binary` 或
`regression`，也可显式指定与任务兼容的目标函数。

## 2. 主实验通用参数

| 参数 | 类型/取值 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `--database` | 路径 | `MARKET_DB_PATH`，否则 `D:\量化\market.duckdb` | `market.duckdb` 文件路径。命令行值优先于环境变量。 |
| `--start` | 日期或时间 | 研究终点向前 3 年 | 研究窗口开始时间；若早于数据库首条数据，会自动截到数据起点。建议使用 `YYYY-MM-DD`。 |
| `--end` | 日期或时间 | 数据库最后时间 | 研究窗口结束时间；若晚于数据库末条数据，会自动截到数据终点。建议使用 `YYYY-MM-DD`。 |
| `--codes` | 一个或多个证券代码 | 未指定 | 明确选择证券，例如 `000001.SZ 600000.SH`。未指定时按 `--symbol-limit` 自动选择。 |
| `--symbol-limit` | 整数，1～100 | `20` | 未指定 `--codes` 时，从代码表中选取的证券数。程序始终校验该值在 1～100 内。 |
| `--backtest-top-n` | 正整数 | `10` | 每个验证交易日按模型分数降序选择并等权买入的最多证券数；当日有效证券不足时全部买入。 |
| `--slippage-bps` | `[0, 10000)` | `0` | 单边滑点，单位为基点；买入价上浮、卖出价下调，买卖两边分别应用一次。 |
| `--commission-bps` | `[0, 10000)` | `0` | 单边手续费率，单位为基点；买卖两边分别收取一次。 |
| `--validation-start` | 日期 | 研究终点向前 1 年 | 滚动验证开始日。该日之前的数据作为初始训练历史，此后按目标交易日逐日扩展训练。必须满足 `start < validation-start <= end`。 |
| `--task` | `classification`、`regression` | `classification` | 选择涨跌二分类或连续涨跌幅预测。回归任务可搭配 `lightgbm` 或用于因子口径核对的 `factor_passthrough`。 |
| `--model` | `simple_decision_tree`、`gradient_boosting_tree`、`lightgbm`、`factor_passthrough` | `simple_decision_tree` | 方向预测模型；其取值决定后续可使用的模型专属参数。 |
| `--factors` | 零个或多个已注册因子名 | 全部已注册因子 | 指定本次训练使用的正式因子。显式写出空的 `--factors` 可只使用 `--factor-expressions`；两者不能同时为空。 |
| `--factor-expressions` | 一个或多个 DSL 字符串 | 空 | 直接加载搜索报告中的 `canonical`/`expression_str`，并按稳定 `fg_...` ID 加入模型。PowerShell 应使用外层单引号；内部单引号写成两个，或直接复制搜索报告生成的命令。 |
| `--factor-cache-dir` | 路径 | `.factor_cache` | 因子 Parquet 缓存目录。 |
| `--no-factor-cache` | 开关 | 关闭 | 出现该参数时完全禁用因子缓存，`--factor-cache-dir` 不再生效。适合核对最新因子实现。 |
| `--log-level` | `DEBUG`、`INFO`、`WARNING`、`ERROR` | `INFO` | 控制终端和文件日志等级。 |
| `--debug` | 开关 | 关闭 | 开启调试模式，并强制把日志等级设为 `DEBUG`；其优先级高于 `--log-level`。 |
| `--log-dir` | 路径 | `logs` | 日志、Markdown 评估报告和 SVG 趋势图的归档根目录。 |

回测使用验证集的样本外预测：分类任务按 `up_probability` 排序，回归任务按
`predicted_return` 排序。每天只持有开盘至收盘，不跨日；收益曲线按扣除双边
成本后的日收益复利，夏普比率采用零无风险利率、日收益样本标准差和 252 日年化。

当前可用于 `--factors` 的名称如下（默认全部使用）：

```text
amplitude
breakout_strength_20d
close_position
close_to_ma_5d
intraday_return
last_30m_return
last_30m_volume_ratio
ma_5d_slope
ma_distance_change_5d
ma_spread_5d_20d
ma_spread_change_5d_20d
momentum_acceleration_5d_20d
positive_bar_ratio
realized_vol
return_1d
return_5d
return_10d
return_20d
up_days_ratio_5d
volatility_5d
volume_ratio_5d
```

## 3. 模型专属参数

### 3.1 `simple_decision_tree`

项目内置的轻量级 CART 二分类树，适合作为快速基线。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--max-depth` | `3` | 树的最大深度。更大时模型表达能力更强，但更容易过拟合且运行更慢。 |
| `--min-samples-leaf` | `20` | 每个叶节点允许的最少训练样本数。调大可减少过小叶节点。 |
| `--max-thresholds` | `32` | 每个因子最多尝试的候选切分阈值数。调大可搜索得更细，但会增加训练耗时。 |

### 3.2 `gradient_boosting_tree`

基于 scikit-learn 的梯度提升树。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--n-estimators` | `100` | 提升迭代次数，即树的数量。 |
| `--learning-rate` | `0.1` | 每棵树的贡献缩减系数；通常与树数量配合调整。 |
| `--max-depth` | `3` | 每棵基学习树的最大深度。 |
| `--min-samples-leaf` | `20` | 每棵基学习树叶节点的最少样本数。 |
| `--subsample` | `1.0` | 每轮训练使用的样本比例，合法范围为 `(0, 1]`；小于 1 时引入随机采样。 |
| `--random-state` | `42` | 随机种子，用于复现实验结果。 |

### 3.3 `lightgbm`

基于 LightGBM 的二分类与连续涨跌幅回归模型。使用前需安装 `pyproject.toml` 中的 `research` 可选依赖。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--n-estimators` | `300` | 提升迭代次数，即树的数量。 |
| `--learning-rate` | `0.03` | 每棵树的贡献缩减系数；较小值通常需要更多树。 |
| `--num-leaves` | `15` | 单棵树的最大叶节点数，控制模型复杂度。 |
| `--max-depth` | `5` | 单棵树最大深度。 |
| `--min-child-samples` | `50` | 一个叶节点所需的最少样本数。 |
| `--subsample` | `0.8` | 每轮训练的行采样比例；小于 1 时程序会启用每轮行采样。 |
| `--colsample-bytree` | `0.8` | 每棵树使用的特征比例。 |
| `--reg-alpha` | `0.1` | L1 正则化系数。 |
| `--reg-lambda` | `1.0` | L2 正则化系数。 |
| `--n-jobs` | `-1` | 训练线程数；`-1` 表示使用全部可用 CPU。共享机器上可设为固定正整数。 |
| `--random-state` | `42` | 随机种子，用于复现实验结果。 |
| `--objective` | 随 `--task` 选择 | LightGBM 目标函数。分类默认为 `binary`，还支持 `cross_entropy`、`cross_entropy_lambda`；回归默认为 `regression`，还支持 `regression_l1`、`huber`、`fair`、`quantile`。目标函数必须与任务类型兼容。 |
| `--objective-alpha` | `0.9` | `huber` 的残差截断阈值，或 `quantile` 的目标分位点；其他目标函数忽略该参数。收益率以小数表示时，`huber` 阈值也使用相同单位，例如 `0.02` 表示 2%。 |

### 3.4 `factor_passthrough`

仅用于核对搜索因子与主实验的指标口径。该模型不学习参数，固定把模型特征矩阵
最后一列原样作为连续预测值，因此必须搭配 `--task regression`。它没有专属参数；
当同时提供正式因子和搜索表达式时，搜索表达式位于特征矩阵末列。直出列中的
NaN/无穷值样本会在日期切分前排除，与搜索 IC 的有效样本口径一致。

## 4. 常用设置与命令

### 4.1 按默认配置运行

```powershell
python -m quant.cli.factor_demo
```

### 4.2 设置 Top N、滑点和手续费

以下示例每天选择前 10 只证券，假设单边滑点 3 bps、单边手续费 1 bps：

```powershell
python -m quant.cli.factor_demo --backtest-top-n 10 --slippage-bps 3 --commission-bps 1
```

### 4.3 指定数据库和研究区间

```powershell
python -m quant.cli.factor_demo `
  --database C:\data\market.duckdb `
  --start 2022-01-01 `
  --end 2025-01-01 `
  --validation-start 2024-01-01
```

也可为当前 PowerShell 会话设置默认数据库：

```powershell
$env:MARKET_DB_PATH = "C:\data\market.duckdb"
python -m quant.cli.factor_demo
```

### 4.4 指定证券

```powershell
python -m quant.cli.factor_demo --codes 000001.SZ 600000.SH 600519.SH
```

让程序自动选择前 50 只证券：

```powershell
python -m quant.cli.factor_demo --symbol-limit 50
```

### 4.5 只研究部分因子

```powershell
python -m quant.cli.factor_demo `
  --factors return_1d return_5d realized_vol volume_ratio_5d
```

只研究一个搜索表达式：

```powershell
python -m quant.cli.factor_demo --factors `
  --factor-expressions 'cs_rank(delta(column(close),periods=5))'
```

因子缓存异常或需要强制重新计算时：

```powershell
python -m quant.cli.factor_demo --no-factor-cache
```

把缓存和运行产物放到指定目录：

```powershell
python -m quant.cli.factor_demo `
  --factor-cache-dir D:\factor-cache `
  --log-dir D:\factor-logs
```

### 4.6 快速决策树基线

```powershell
python -m quant.cli.factor_demo `
  --model simple_decision_tree `
  --max-depth 3 `
  --min-samples-leaf 20 `
  --max-thresholds 32
```

如果只想快速检查流程，可同时缩小证券数、日期范围和因子集合：

```powershell
python -m quant.cli.factor_demo `
  --start 2024-01-01 `
  --validation-start 2024-10-01 `
  --symbol-limit 5 `
  --factors return_1d return_5d realized_vol `
  --model simple_decision_tree
```

### 4.7 梯度提升树

```powershell
python -m quant.cli.factor_demo `
  --model gradient_boosting_tree `
  --n-estimators 100 `
  --learning-rate 0.1 `
  --max-depth 3 `
  --min-samples-leaf 20 `
  --subsample 0.8 `
  --random-state 42
```

### 4.8 LightGBM

偏稳健的常用起点：

```powershell
python -m quant.cli.factor_demo `
  --model lightgbm `
  --n-estimators 300 `
  --learning-rate 0.03 `
  --num-leaves 15 `
  --max-depth 5 `
  --min-child-samples 50 `
  --subsample 0.8 `
  --colsample-bytree 0.8 `
  --reg-alpha 0.1 `
  --reg-lambda 1.0 `
  --n-jobs 4 `
  --random-state 42
```

### 4.9 调试和详细日志

```powershell
python -m quant.cli.factor_demo --debug
```

仅希望减少输出时：

```powershell
python -m quant.cli.factor_demo --log-level WARNING
```

## 5. 使用注意事项

- `--codes` 和 `--factors` 都接收多个值；后面的另一个参数必须带 `--`，以便 `argparse` 判断列表结束。
- `--validation-start` 控制验证区间，不会改变特征只能使用当日及以前数据、训练样本必须满足 `target_date < T` 的防泄漏规则。
- 比较模型或参数时，应固定研究窗口、证券、因子、验证起点和随机种子，否则结果不可直接归因于模型设置。
- `--debug` 会覆盖 `--log-level` 并使用 `DEBUG`，即使命令中同时指定了其他日志等级。
- 路径包含空格时需加双引号，例如 `--database "D:\quant data\market.duckdb"`。
- 完整运行会逐日重新训练模型。增加证券数、验证天数、因子数、树数量或树复杂度都会提高耗时。

## 6. 其他命令行脚本速查

### 6.1 `python -m quant.cli.single_factor_test`

使用最近一个月行情检查 `return_1d` 因子。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--database` | `MARKET_DB_PATH`，否则 `D:\量化\market.duckdb` | 数据库路径。 |
| `--codes` | 未指定 | 证券代码列表；未指定时自动选择。 |
| `--symbol-limit` | `20` | 自动选择的证券数，范围 1～100。 |
| `--output` | 未指定 | 可选的 CSV 输出路径。 |
| `--debug` | 关闭 | 开启调试标记。 |

```powershell
python -m quant.cli.single_factor_test `
  --codes 000001.SZ 600000.SH `
  --output .\output\return_1d.csv
```

### 6.2 `python -m quant.market_data.build_database`

从年度压缩包构建标准化 Parquet 数据和 DuckDB 目录。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--root` | `D:\量化` | 原始年度压缩包所在目录，也是生成 `extracted_daily/`、`bars_5m/` 和 `market.duckdb` 的根目录。 |

```powershell
python -m quant.market_data.build_database --root D:\量化
```

### 6.3 `python -m quant.cli.ifind_download`

通过 iFinD 下载单只证券的分钟行情。账号密码优先从 `IFIND_USERNAME`、`IFIND_PASSWORD` 读取，缺失时交互输入。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--code` | `600000.SH` | 证券代码。 |
| `--start` | `2026-03-01` | 开始日期，格式 `YYYY-MM-DD`。 |
| `--end` | `2026-03-30` | 结束日期，格式 `YYYY-MM-DD`。 |
| `--start-time` | `09:15:00` | 每日请求开始时间。 |
| `--end-time` | `15:15:00` | 每日请求结束时间。 |
| `--indicators` | 脚本内置指标串 | 传给 `THS_HF` 的分号分隔指标。 |
| `--params` | `Fill:Original` | 传给 `THS_HF` 的请求参数。 |
| `--output` | `data/600000_SH_202603` | 每日 CSV 和合并 CSV 的输出目录。 |
| `--retry` | `3` | 单日请求失败时的最大尝试次数。 |

```powershell
python -m quant.cli.ifind_download `
  --code 600000.SH `
  --start 2026-03-01 `
  --end 2026-03-31 `
  --output .\data\600000_SH_202603 `
  --retry 3
```

## 7. 数据源选择

> 日线数据从下载到跑实验的完整流程见 [qmt_daily_guide.md](qmt_daily_guide.md)。

`python -m quant.cli.factor_demo` 通过 `--data-source` 选择行情来源，缺省 `market_service`
（本地 5 分钟库，与改动前行为完全一致）。数据源参数与模型参数一样采用两阶段解析：
程序先读取 `--data-source`，再只注册所选数据源的专属参数，因此不同数据源可以有同名
参数，某个数据源的专属参数不能用于另一个。

```powershell
python -m quant.cli.factor_demo --help
python -m quant.cli.factor_demo --data-source qmt_daily --help
```

### 7.1 `market_service`（默认，5 分钟）

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--database` | `MARKET_DB_PATH` 或 `D:\量化\market.duckdb` | 5 分钟库路径 |

### 7.2 `qmt_daily`（QMT 日线）

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--daily-database` | `QMT_DAILY_DB_PATH` 或 `D:\量化\qmt_daily\qmt_daily.duckdb` | 日线库路径 |
| `--adjust` | `hfq` | 复权口径：`hfq` 后复权、`qfq` 前复权、`none` 不复权 |
| `--adjust-anchor` | 库中记录的基准日 | 前复权基准日，格式 `YYYY-MM-DD` |
| `--include-suspended` | 关闭 | 保留停牌日行情 |
| `--adjust-volume` | 关闭 | 复权时同步反向调整成交量 |
| `--qmt-output-root` | `QMT_OUTPUT_ROOT` 或下载器配置 | QMT 落盘根目录 |
| `--sync-mode` | `auto` | 增量检查力度：`auto`/`full`/`rebuild` |
| `--sync-verify-hash` | 关闭 | 对变化的分区重算 SHA-256 |
| `--no-auto-sync` | 关闭 | 跳过启动时的自动增量检查 |

日线源只有日频行情，因此八个分钟因子（`close_to_vwap`、`downside_semivol`、
`intraday_path_efficiency`、`last_30m_return`、`last_30m_volume_ratio`、
`positive_bar_ratio`、`realized_vol`、`signed_volume_imbalance`）不可用：
不指定 `--factors` 时自动跳过并告警，显式点名则直接报错。

**`--factors` 的默认值变化**：不写该参数表示「使用该数据源支持的全部因子」，
写了但不给值仍表示显式空集合。行为与之前一致，只是默认集合现在随数据源而定。

样例配置见 `configs/factor_research/qmt_daily.example.yaml`。

## 8. 日线库相关命令

### 8.1 `python -m quant.cli.build_daily_store`

把大 QMT 落盘的日线 CSV 增量转成日线库。详见
[qmt_daily_store.md](qmt_daily_store.md)。

| 参数 | 含义 |
| --- | --- |
| `--database` | 日线库路径 |
| `--qmt-output-root` / `--qmt-config` | QMT 落盘根目录，或从下载器 JSONC 里读 `output_root` |
| `--rebuild-all` | 整库重建 |
| `--dry-run` | 只检查不写入 |
| `--sync-mode` / `--sync-verify-hash` | 增量检查力度 |

退出码：0 正常，2 失败，3 日线库被占用而跳过。

### 8.2 `python -m quant.cli.market_check`

审计入库后的日线库，详见 [market_check.md](market_check.md)。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--database` | `QMT_DAILY_DB_PATH` | 日线库路径 |
| `--start-date` / `--end-date` | 全库 | 审计区间，格式 `YYYYMMDD` |
| `--codes` | 全市场 | 只审计指定证券 |
| `--report-dir` | `<库目录>/reports/market_check/<时间戳>` | 报告目录 |
| `--coverage-error-threshold` | 0.95 | 单日覆盖率下限 |
| `--pre-close-tolerance` / `--adjust-factor-tolerance` | 1e-4 | 价格比较的相对容差 |
| `--price-limit-level` | `warning` | 涨跌停越界的级别 |
| `--apply-st-limit` | 关闭 | 按当前 ST 状态收紧到 5%，会大量误报 |
| `--cross-check-5m` / `--market-database` | 关闭 | 与 5 分钟库交叉对账 |

退出码：0 通过，1 存在 ERROR，2 失败。
