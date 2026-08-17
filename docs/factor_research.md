# 日频方向预测研究框架

该模块使用某交易日收盘及之前可见的数据，预测下一有效交易日从开盘到收盘上涨或下跌，或直接预测连续涨跌幅。目标收益定义为下一有效交易日的 `close / open - 1`。框架用于离线研究，并提供验证集 Top N 日内等权回测及双边滑点、手续费估算；它不模拟盘口成交量、涨跌停、停牌或部分成交等实盘约束。

使用 `--task classification`（默认）执行涨跌二分类；使用 `--task regression --model lightgbm` 预测连续涨跌幅。LightGBM 可通过 `--objective` 选择与任务兼容的目标函数，例如分类使用 `binary`，回归使用 `regression`、`regression_l1` 或 `huber`。

## 快速运行

程序直接通过 `quant.market_data.client.MarketDataClient` 读取 `bars_5m` 数据。默认以数据库最后一个行情时点为终点，向前取3年，并研究代码表中的前20只股票。

```powershell
python -m quant.cli.factor_demo
```

主实验会对验证集预测同步执行 Top 10 日内等权回测：按模型分数选股，在目标日
开盘买入、收盘卖出，并在报告中输出收益曲线和 252 日年化夏普比率。可通过
`--backtest-top-n`、`--slippage-bps` 和 `--commission-bps` 调整选股数量、
单边滑点及单边手续费。

常态化配置可以放入 YAML，并在命令行按需覆盖其中的值：

```powershell
python -m quant.cli.factor_demo --config experiment.yaml --log-level DEBUG
```

训练方式由 `--training-mode` 控制。默认 `rolling` 会在每个验证日使用此前全部
可用标签重新训练；`single` 仅使用 `validation-start` 之前的训练集拟合一次，
再对整个验证集测试：

```powershell
python -m quant.cli.factor_demo --training-mode single --validation-start 2024-01-01
```

YAML 使用与命令行参数对应的扁平 `snake_case` 键；列表和布尔参数分别使用
YAML 列表和 `true`/`false`。未知参数或无效值会直接报错。完整配置示例见
`configs/factor_research/example.yaml`。如果命令行切换了 YAML 中配置的模型，原模型独有的
配置项会被忽略，共用配置项仍会应用到新模型。

运行时可以只选择部分因子，并指定缓存目录：

```powershell
python -m quant.cli.factor_demo --factors return_1d return_5d realized_vol --factor-cache-dir .factor_cache
```

搜索报告中的 `canonical`/`expression_str` 可以直接作为临时因子加入实验；程序会
安全解析表达式，并用稳定的 `fg_...` 因子 ID 作为模型特征列名：

```powershell
python -m quant.cli.factor_demo --factors --factor-expressions 'cs_rank(delta(column(close),periods=5))'
```

这里显式传入空的 `--factors`，表示只测试搜索因子；省略它则会在默认正式因子
之外追加搜索因子。也可以在 YAML 中配置 `factors: []` 和
`factor_expressions` 字符串列表。表达式引用的注册因子会
自动作为计算依赖加载，但只有 `factors` 和 `factor_expressions` 明确选择的特征
会进入模型。

模型通过 `--model` 选择。当前默认模型为 `simple_decision_tree`，其参数由模型
模块自行注册：

```powershell
python -m quant.cli.factor_demo --model simple_decision_tree --max-depth 3 --min-samples-leaf 20
```

使用 scikit-learn 梯度提升树：

```powershell
python -m quant.cli.factor_demo --model gradient_boosting_tree --n-estimators 100 --learning-rate 0.1 --max-depth 3
```

使用支持多线程的 LightGBM（`--n-jobs -1` 表示使用全部可用 CPU）：

```powershell
python -m quant.cli.factor_demo --model lightgbm --n-estimators 300 --learning-rate 0.03 --num-leaves 15 --n-jobs -1
```

原样输出最后一个搜索因子，以复核网格搜索与主实验的 IC 口径：

```powershell
python -m quant.cli.factor_demo --model factor_passthrough --task regression --factors --factor-expressions 'cs_rank(delta(column(close),periods=5))'
```

`factor_passthrough` 不拟合目标，只能用于指标核对，不是可部署的预测模型。复核
搜索报告时应直接复制报告生成的完整命令，确保数据库、日期范围、验证起点和证券池
均与搜索一致。

新增模型时，在 `src/quant/factor_research/models/` 中增加具体模型和工厂，并使用
`@register_model_factory` 注册。工厂通过 `add_arguments()` 声明自己的命令行
参数，通过 `from_args()` 从 `args` 构建实例；无需修改主实验入口 `quant.cli.factor_demo`。

使用 `--log-level` 控制日志详细程度，默认是 `INFO`：

```powershell
python -m quant.cli.factor_demo --log-level DEBUG
```

`INFO` 显示因子和验证进度，`DEBUG` 额外显示缓存路径、数据指纹和逐日训练明细，`WARNING` 显示缓存损坏或校验失败，`ERROR` 显示计算异常。

日志默认同时输出到终端和 `logs/` 目录。程序在 `run/` 下按运行开始日期和 24 小时制小时建立 `YYYY-MM-DD/HH` 归档子目录，每次运行在其中生成独立的“时间戳 + 随机ID”文件，例如 `logs/run/2026-08-02/20/factor_demo_20260802_203015_a1b2c3d4.log`。日志首行记录本次全部运行参数。文件达到 10 MB 后自动轮转，最多保留 5 个历史文件；可通过 `--log-dir` 修改归档根目录：

```powershell
python -m quant.cli.factor_demo --log-dir D:\factor-logs
```

运行完成后，同一小时归档目录还会生成同名 `.md` 评估报告、`_accuracy.svg`
准确率趋势图、`_ic_trend.svg` IC/Rank IC 20 日与 60 日动态纵轴趋势图、
`_equity.svg` 收益曲线，以及 `_drawdown.svg` 历史回撤与修复图。报告汇总
ROC AUC 等模型指标、Top N 回测指标（含夏普比率）、回撤诊断（最大回撤、
修复时长、Calmar 比率与历次回撤区间）、每日预测结果、因子重要性
（模型提供时）和运行参数。

回撤按扣除双边成本后的 Top N 净值曲线计算，期初净值 1.0 也参与历史峰值统计，
因此首日就亏损时同样计入回撤；取值不大于 0，0 表示当日创出新高。回撤图上面板
叠加累计净值与历史峰值并填充水下区间，下面板画逐日回撤深度、标注最大回撤谷底
与其修复位置，并叠加全市场等权对照回撤。验证区间末尾仍未回到峰值的区间记为
未修复，其修复日期和修复交易日数显示为缺失。

运行结束时，`INFO` 还会输出行情加载、基础聚合、因子耗时排行、数据集构建，以及验证预处理/训练/预测的分项耗时，可用于定位性能瓶颈。

每个因子由 `src/quant/factor_research/factor_factories/` 下独立的工厂文件计算。缓存按因子实现指纹和输入行情指纹保存为 Parquet；工厂或公共计算逻辑变化时旧缓存会自动失效，缓存缺失或校验不通过时会自动重新计算。使用 `--no-factor-cache` 可以临时禁用缓存。

新增因子时不需要修改注册表，只需在该目录新增模块并使用装饰器：

```python
import pandas as pd

from .base import FactorFactory
from .registry import register_factor


@register_factor
class MyFactorFactory(FactorFactory):
    name = "my_factor"

    def compute(self, bars: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
        return daily["close"] / daily["open"] - 1
```

### 因子必须对价格尺度零次齐次

上面这个因子是收盘价与开盘价的比值，把当天全部价格乘以同一个正常数后取值不变，
这就是零次齐次。日线库只存原始不复权价、在读取层按 `--adjust` 复权，而后复权系数
`hfq(code, t)` 是**逐证券**的：持有 20 年高分红股的系数可能是 50，次新股是 1.0。
于是任何带价格量纲的因子（收盘价水平、`MA20`、`ATR`）在横截面上会主要由
「上市时长 × 分红历史」决定，而不是由信号决定，而且因子值看上去完全正常，
不会触发任何异常检测。

这里的「不变」针对更换**前复权基准日**：`qfq(s | anchor) = hfq(s) / hfq(anchor)`，
分母对该证券是常数。它不意味着 `--adjust none` 与 `--adjust hfq` 会算出同一个值——
`hfq(code, t)` 跨除权日会跳档，两者本就不同，而且只有复权后的取值才是对的。

确需偏离时在因子自己的模块里声明 `price_homogeneity`：整数 `k` 表示
`g(c·P) = c**k · g(P)`，`None` 表示不满足任何齐次度。未声明按 `0` 处理。该属性
不能上提到 `FactorFactory` 基类，否则会作废全部因子的历史缓存。

```python
from quant.factor_research.factor_factories.capabilities import (
    dimensional_factor_names,     # k 为非零整数：带价格量纲，可按 c**k 解析换算
    non_homogeneous_factor_names, # k 为 None：无解析形式，需个案评估
)

dimensional_factor_names()      # frozenset()
non_homogeneous_factor_names()  # frozenset({'alpha_001'})
```

`alpha_001` 忠实复刻 WorldQuant 原式，在 `SignedPower` 里混用了一次齐次的 `close`
和零次齐次的 `stddev(returns, 20)`，因此没有齐次度可言。翻转门槛约为「日收益率
波动率 / 价格水平」，真实后复权系数恒为正且通常不小于 1，`close` 项始终压倒波动率
项，所以它在实践中取值稳定——但这是个案结论，不能推广成 `k = 0`。

`tests/factor_research/test_scale_invariance.py` 用逐证券缩放校验这些声明。注意扰动
必须逐证券取不同系数：全截面统一乘一个常数时，横截面排名与 z-score 恒等不变，反而
会给带价格量纲的因子发出假的合格证。

另外，复权只作用于 `open`/`high`/`low`/`close`/`pre_close`，`amount` 恒不复权、
`volume` 仅在 `--adjust-volume` 时反向调整。因子层拿得到的基础列是
`quant.factor_research.factors.BASE_DAILY_COLUMNS`，即
`code`/`trade_date`/`open`/`high`/`low`/`close`/`volume`/`adjust_factor`
（`_prepare_daily_bars` 会丢掉 `amount`、`pre_close`、`suspend_flag`），
所以同一表达式里 `close` 与成交量的相对尺度会随该开关变化，`close * volume` 这类
近似成交额的写法在两种配置下口径不同。

### 基础列 `adjust_factor`

`adjust_factor` 是当前复权口径乘到价格上的那个正系数，因此

```
close / adjust_factor
```

在 `none`/`hfq`/`qfq` 三种口径下都还原为**同一个原始不复权价**。分钟数据源没有复权
概念，那条路径上该列恒为 1.0，两条路径的基础列宇宙一致，读它的因子换数据源不会
`KeyError`。

它的用途和陷阱：

- **零次齐次**：`close / adjust_factor` 对「更换前复权基准日」这类逐证券常数缩放
  严格不变，可以直接进横截面。这是它唯一无争议的用法。
- **一次齐次**：`adjust_factor` 单独作为特征带价格量纲，横截面上排出来的是
  「上市时长 × 分红送转历史」而不是信号。要用必须声明 `price_homogeneity = 1`。
- **只能同日横截面用**：`adjust_factor(t)` 在时间上是阶梯函数，还原出的原始价跨
  除权日会跳档。`close / adjust_factor` 上再叠任何时序算子（`delta`、`returns`、
  `ts_mean` …）都会算出假的除权跳空——那正是复权要修掉的东西。
- **`qfq` 下有未来数据泄漏**：`hfq` 与 `none` 下 `adjust_factor(t)` 只由不晚于 `t`
  的除权事件累乘而来，是因果的；但 `qfq` 下它等于 `hfq(t) / hfq(anchor)`，`anchor`
  晚于 `t` 时分母含有 `t` 之后才发生的分红送转。所以在 `qfq` 口径下**只能**把它
  用作还原原始价的分母（比值里 `hfq(anchor)` 自动约掉），不得把它本身或它的时序
  变化直接当特征。

搜索的列来源都是显式配置的，因此加这一列不会自动扩大搜索空间：`genetic/config.py`
的 `sources` 缺省为 `("close", "volume", "return_1d")`，`cli/grid_search.py` 的
`build_search_space()` 与 `build_genetic_search_config()` 各自硬编码
`close/volume/return_1d/high/low/open`，三处都不含 `adjust_factor`。若手工把它放进
`sources`，务必按上面四条先确认候选表达式的齐次度；另外分钟数据源上该列恒为 1.0，
放进去只会得到一个常数终端。

指定窗口和股票：

```powershell
python -m quant.cli.factor_demo --start 2022-01-01 --end 2025-01-01 --codes 000001.SZ 600000.SH
```

如数据库不在默认的 `D:\量化\market.duckdb`：

```powershell
python -m quant.cli.factor_demo --database C:\data\market.duckdb
```

框架默认生成日收益、5日动量、波动率、成交量、日内波动、上涨K线比例和尾盘量价等简单因子。`validation-start` 默认是研究结束日期往前1年；该日期之前是训练集，该日期起是验证集。`rolling` 模式从该日期开始逐日扩展训练，`single` 模式则固定使用这份训练集。

核心API：

```python
from factor_research import DirectionExperiment, build_daily_features, build_direction_dataset

daily = build_daily_features(bars_5m)
dataset = build_direction_dataset(daily)
result = DirectionExperiment(
    validation_start="2024-01-01",
    training_mode="single",
    max_depth=3,
    min_samples_leaf=20,
).run(dataset)

print(result.metrics)
print(result.daily_accuracy_trend)
print(result.daily_ic_trend)
print(result.feature_importance)
print(result.predictions.head())
```

`result.metrics["auc"]` 是全部验证样本的 ROC AUC。`result.metrics["ic"]` 和
`result.metrics["rank_ic"]` 分别是逐交易日横截面 Pearson IC、Spearman Rank IC
的有效日均值；`pooled_ic` 保留全部验证样本混合计算的 Pearson 相关作为辅助诊断。
`result.daily_ic_trend` 保存每日横截面明细；`result.daily_accuracy_trend` 按
`target_date` 给出每日样本数、预估准度以及较前一交易日的准度变化。

`feature_date` 是特征截止日，`target_date` 是被预测日。框架按 `target_date` 整日切分，确保同一天的不同股票不会同时出现在训练集与测试集中；缺失值填充中位数也只使用训练集拟合。

## 因子 DSL 与并行网格搜索

项目提供独立的链式因子表达式、固定/模板网格、selection/holdout IC 初筛、进程
并行和 Top K 模型验证。搜索候选不会写入正式因子注册表，固定因子在搜索前只
计算一次。完整设计、算子语义和调用示例见 [FACTOR_GRID_SEARCH.md](FACTOR_GRID_SEARCH.md)。
