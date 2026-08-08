# 日频方向预测研究框架

该模块使用某交易日收盘及之前可见的数据，预测下一有效交易日收盘相对前收盘上涨或下跌。它只做因子和模型的离线研究，不模拟订单、持仓或交易费用。

## 快速运行

程序直接通过 `market_service.client.MarketDataClient` 读取 `bars_5m` 数据。默认以数据库最后一个行情时点为终点，向前取3年，并研究代码表中的前20只股票。

```powershell
python run_factor_demo.py
```

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

新增模型时，在 `factor_research/models/` 中增加具体模型和工厂，并使用
`@register_model_factory` 注册。工厂通过 `add_arguments()` 声明自己的命令行
参数，通过 `from_args()` 从 `args` 构建实例；无需修改 `run_factor_demo.py`。

使用 `--log-level` 控制日志详细程度，默认是 `INFO`：

```powershell
python run_factor_demo.py --log-level DEBUG
```

`INFO` 显示因子和验证进度，`DEBUG` 额外显示缓存路径、数据指纹和逐日训练明细，`WARNING` 显示缓存损坏或校验失败，`ERROR` 显示计算异常。

日志默认同时输出到终端和 `logs/` 目录。程序按运行开始日期建立 `YYYY-MM-DD` 归档子目录，每次运行在其中生成独立的“时间戳 + 随机ID”文件，例如 `logs/2026-08-02/factor_demo_20260802_203015_a1b2c3d4.log`。日志首行记录本次全部运行参数。文件达到 10 MB 后自动轮转，最多保留 5 个历史文件；可通过 `--log-dir` 修改归档根目录：

```powershell
python run_factor_demo.py --log-dir D:\factor-logs
```

运行完成后，同一日期归档目录还会生成同名 `.md` 评估报告和 `_accuracy.svg` 趋势图。报告汇总
ROC AUC 等整体指标、每日预测结果、准确率趋势、因子重要性和运行参数。

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
print(result.feature_importance)
print(result.predictions.head())
```

`result.metrics["auc"]` 是全部验证样本的 ROC AUC，`result.metrics["ic"]` 是
样本外上涨概率与下一交易日收益率的 Pearson 相关系数；`result.daily_accuracy_trend` 按
`target_date` 给出每日样本数、预估准度以及较前一交易日的准度变化。

`feature_date` 是特征截止日，`target_date` 是被预测日。框架按 `target_date` 整日切分，确保同一天的不同股票不会同时出现在训练集与测试集中；缺失值填充中位数也只使用训练集拟合。
