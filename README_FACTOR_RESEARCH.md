# 日频方向预测研究框架

该模块使用某交易日收盘及之前可见的数据，预测下一有效交易日从开盘到收盘上涨或下跌，或直接预测连续涨跌幅。目标收益定义为下一有效交易日的 `close / open - 1`。框架用于离线研究，并提供验证集 Top N 日内等权回测及双边滑点、手续费估算；它不模拟盘口成交量、涨跌停、停牌或部分成交等实盘约束。

使用 `--task classification`（默认）执行涨跌二分类；使用 `--task regression --model lightgbm` 预测连续涨跌幅。LightGBM 可通过 `--objective` 选择与任务兼容的目标函数，例如分类使用 `binary`，回归使用 `regression`、`regression_l1` 或 `huber`。

## 快速运行

程序直接通过 `market_service.client.MarketDataClient` 读取 `bars_5m` 数据。默认以数据库最后一个行情时点为终点，向前取3年，并研究代码表中的前20只股票。

```powershell
python run_factor_demo.py
```

主实验会对验证集预测同步执行 Top 10 日内等权回测：按模型分数选股，在目标日
开盘买入、收盘卖出，并在报告中输出收益曲线和 252 日年化夏普比率。可通过
`--backtest-top-n`、`--slippage-bps` 和 `--commission-bps` 调整选股数量、
单边滑点及单边手续费。

常态化配置可以放入 YAML，并在命令行按需覆盖其中的值：

```powershell
python run_factor_demo.py --config experiment.yaml --log-level DEBUG
```

训练方式由 `--training-mode` 控制。默认 `rolling` 会在每个验证日使用此前全部
可用标签重新训练；`single` 仅使用 `validation-start` 之前的训练集拟合一次，
再对整个验证集测试：

```powershell
python run_factor_demo.py --training-mode single --validation-start 2024-01-01
```

YAML 使用与命令行参数对应的扁平 `snake_case` 键；列表和布尔参数分别使用
YAML 列表和 `true`/`false`。未知参数或无效值会直接报错。完整配置示例见
`factor_config.example.yaml`。如果命令行切换了 YAML 中配置的模型，原模型独有的
配置项会被忽略，共用配置项仍会应用到新模型。

运行时可以只选择部分因子，并指定缓存目录：

```powershell
python run_factor_demo.py --factors return_1d return_5d realized_vol --factor-cache-dir .factor_cache
```

搜索报告中的 `canonical`/`expression_str` 可以直接作为临时因子加入实验；程序会
安全解析表达式，并用稳定的 `fg_...` 因子 ID 作为模型特征列名：

```powershell
python run_factor_demo.py --factors --factor-expressions 'cs_rank(delta(column(close),periods=5))'
```

这里显式传入空的 `--factors`，表示只测试搜索因子；省略它则会在默认正式因子
之外追加搜索因子。也可以在 YAML 中配置 `factors: []` 和
`factor_expressions` 字符串列表。表达式引用的注册因子会
自动作为计算依赖加载，但只有 `factors` 和 `factor_expressions` 明确选择的特征
会进入模型。

模型通过 `--model` 选择。当前默认模型为 `simple_decision_tree`，其参数由模型
模块自行注册：

```powershell
python run_factor_demo.py --model simple_decision_tree --max-depth 3 --min-samples-leaf 20
```

使用 scikit-learn 梯度提升树：

```powershell
python run_factor_demo.py --model gradient_boosting_tree --n-estimators 100 --learning-rate 0.1 --max-depth 3
```

使用支持多线程的 LightGBM（`--n-jobs -1` 表示使用全部可用 CPU）：

```powershell
python run_factor_demo.py --model lightgbm --n-estimators 300 --learning-rate 0.03 --num-leaves 15 --n-jobs -1
```

原样输出最后一个搜索因子，以复核网格搜索与主实验的 IC 口径：

```powershell
python run_factor_demo.py --model factor_passthrough --task regression --factors --factor-expressions 'cs_rank(delta(column(close),periods=5))'
```

`factor_passthrough` 不拟合目标，只能用于指标核对，不是可部署的预测模型。复核
搜索报告时应直接复制报告生成的完整命令，确保数据库、日期范围、验证起点和证券池
均与搜索一致。

新增模型时，在 `factor_research/models/` 中增加具体模型和工厂，并使用
`@register_model_factory` 注册。工厂通过 `add_arguments()` 声明自己的命令行
参数，通过 `from_args()` 从 `args` 构建实例；无需修改 `run_factor_demo.py`。

使用 `--log-level` 控制日志详细程度，默认是 `INFO`：

```powershell
python run_factor_demo.py --log-level DEBUG
```

`INFO` 显示因子和验证进度，`DEBUG` 额外显示缓存路径、数据指纹和逐日训练明细，`WARNING` 显示缓存损坏或校验失败，`ERROR` 显示计算异常。

日志默认同时输出到终端和 `logs/` 目录。程序在 `run/` 下按运行开始日期和 24 小时制小时建立 `YYYY-MM-DD/HH` 归档子目录，每次运行在其中生成独立的“时间戳 + 随机ID”文件，例如 `logs/run/2026-08-02/20/factor_demo_20260802_203015_a1b2c3d4.log`。日志首行记录本次全部运行参数。文件达到 10 MB 后自动轮转，最多保留 5 个历史文件；可通过 `--log-dir` 修改归档根目录：

```powershell
python run_factor_demo.py --log-dir D:\factor-logs
```

运行完成后，同一小时归档目录还会生成同名 `.md` 评估报告、`_accuracy.svg`
准确率趋势图、`_ic_trend.svg` IC/Rank IC 20 日与 60 日动态纵轴趋势图，以及
`_equity.svg` 收益曲线。报告汇总 ROC AUC 等模型指标、Top N
回测指标（含夏普比率）、每日预测结果、因子重要性（模型提供时）和运行参数。

运行结束时，`INFO` 还会输出行情加载、基础聚合、因子耗时排行、数据集构建，以及验证预处理/训练/预测的分项耗时，可用于定位性能瓶颈。

每个因子由 `factor_research/factor_factories/` 下独立的工厂文件计算。缓存按因子实现指纹和输入行情指纹保存为 Parquet；工厂或公共计算逻辑变化时旧缓存会自动失效，缓存缺失或校验不通过时会自动重新计算。使用 `--no-factor-cache` 可以临时禁用缓存。

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

指定窗口和股票：

```powershell
python run_factor_demo.py --start 2022-01-01 --end 2025-01-01 --codes 000001.SZ 600000.SH
```

如数据库不在默认的 `D:\量化\market.duckdb`：

```powershell
python run_factor_demo.py --database C:\data\market.duckdb
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
