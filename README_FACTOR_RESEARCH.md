# 日频方向预测研究框架

该模块使用某交易日收盘及之前可见的数据，预测下一有效交易日收盘相对前收盘上涨或下跌。它只做因子和模型的离线研究，不模拟订单、持仓或交易费用。

## 快速运行

程序直接通过 `market_service.client.MarketDataClient` 读取 `bars_5m` 数据。默认以数据库最后一个行情时点为终点，向前取3年，并研究代码表中的前20只股票。

```powershell
python run_factor_demo.py
```

运行时可以只选择部分因子，并指定缓存目录：

```powershell
python run_factor_demo.py --factors return_1d return_5d realized_vol --factor-cache-dir .factor_cache
```

使用 `--log-level` 控制日志详细程度，默认是 `INFO`：

```powershell
python run_factor_demo.py --log-level DEBUG
```

`INFO` 显示因子和滚动验证进度，`DEBUG` 额外显示缓存路径、数据指纹和逐日训练明细，`WARNING` 显示缓存损坏或校验失败，`ERROR` 显示计算异常。

日志默认同时输出到终端和 `logs/` 目录。每次运行生成独立的“时间戳 + 随机ID”文件，例如 `factor_demo_20260802_203015_a1b2c3d4.log`。日志首行记录本次全部运行参数。文件达到 10 MB 后自动轮转，最多保留 5 个历史文件；可通过 `--log-dir` 修改目录：

```powershell
python run_factor_demo.py --log-dir D:\factor-logs
```

运行结束时，`INFO` 还会输出行情加载、基础聚合、因子耗时排行、数据集构建，以及滚动验证预处理/训练/预测的分项耗时和平均单日耗时，可用于定位性能瓶颈。

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

框架默认生成日收益、5日动量、波动率、成交量、日内波动、上涨K线比例和尾盘量价等简单因子。`validation-start` 默认是研究结束日期往前1年；该日期之前是初始训练历史，从该日期开始逐日扩展训练并滚动验证。

核心API：

```python
from factor_research import DirectionExperiment, build_daily_features, build_direction_dataset

daily = build_daily_features(bars_5m)
dataset = build_direction_dataset(daily)
result = DirectionExperiment(validation_start="2024-01-01", max_depth=3, min_samples_leaf=20).run(dataset)

print(result.metrics)
print(result.feature_importance)
print(result.predictions.head())
```

`feature_date` 是特征截止日，`target_date` 是被预测日。框架按 `target_date` 整日切分，确保同一天的不同股票不会同时出现在训练集与测试集中；缺失值填充中位数也只使用训练集拟合。
