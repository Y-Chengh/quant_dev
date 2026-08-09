# 因子 DSL、网格搜索与遗传搜索逻辑说明

## 1. 设计目标

本工具把“因子如何表达”和“候选如何搜索”拆成两个独立模块：

- `factor_research.factor_dsl` 只负责构建、校验和计算因子表达式。
- `factor_research.factor_search` 只负责生成候选、准备一次性上下文、并行调度和
  评价候选。

现有 `FactorFactory`、`build_daily_features`、`build_direction_dataset` 和
`DirectionExperiment` 的调用方式保持不变。搜索期间产生的临时候选不会注册到
`FACTOR_FACTORIES`，因此不会改变 `DEFAULT_FEATURES` 或现有 demo 的默认行为。

## 2. 整体数据流

```text
分钟行情
  └─ prepare_search_context
       ├─ 固定因子非空：build_daily_features 只调用一次
       ├─ 固定因子为空：aggregate_daily_bars 只聚合 OHLCV
       └─ build_forward_targets 只构建一次目标和日期切分
             ↓
        只读 SearchContext
             ↓
        SearchSpace 生成并去重表达式
             ↓
        SequentialBackend / ProcessBackend 分批执行
             ↓
        IcEvaluator 仅用 selection 全量初筛
             ↓
        HoldoutIcEvaluator 只评价 selection Top K
             ↓
        可选 ModelCandidateEvaluator 验证 Top K
             ↓
        FactorSearchResult 排行榜与表达式物化
```

候选 worker 只接收已经准备好的 `SearchContext`，不会持有分钟行情，也不会调用
因子工厂。进程后端在每个 worker 初始化时传入一次上下文，每个任务只传一批很小
的表达式对象。

## 3. 链式因子表达式

### 3.1 基本用法

```python
from factor_research.factor_dsl import DailyFactorFrame

daily = DailyFactorFrame(daily_frame)
factor = (
    daily.close()
    .rank()
    .stddev(window=20)
    .argmax(window=5)
)
values = factor.compute(name="close_rank_std_argmax")
```

`DailyFactorFrame` 会在内部按 `code, trade_date` 排序，计算完成后恢复调用方原来的
行顺序和索引。调用方的 DataFrame 不会被修改。

`rank()` 是按 `trade_date` 的横截面百分位排名。`stddev()`、`argmax()` 等滚动
算子始终按 `code` 独立计算。为了减少歧义，`rank()` 也可以写成 `cs_rank()`，
时间序列排名使用 `ts_rank(window)`。

### 3.2 数据源

```python
daily.open()
daily.high()
daily.low()
daily.close()
daily.volume()
daily.feature("return_1d")  # 引用已经计算好的固定因子
daily.constant(1.0)
```

### 3.3 数学与条件表达式

```python
intraday_return = (daily.close() - daily.open()) / daily.open()
logged_volume = daily.volume().log()
signed_square = intraday_return.signed_power(2)
conditional = daily.where(
    intraday_return < 0,
    intraday_return.stddev(20),
    daily.close(),
)
```

支持加、减、乘、除、负号、绝对值、`log`、`sign`、`power`、`signed_power` 和
大小比较。除数为零、非正数取对数或运算产生无穷值时返回 `NaN`。比较运算沿用
pandas 语义：任一输入缺失时条件为 False，`where` 会选择 false 分支；被选中
分支自身的缺失值不会被填充。

### 3.4 时间序列算子

```python
source.delay(1)
source.delta(5)
source.returns(1)
source.sum(10)
source.mean(10)
source.stddev(20, ddof=1)
source.min(10)
source.max(10)
source.ts_rank(10)
source.argmax(10)
source.argmin(10)
source.correlation(other, 20)
source.covariance(other, 20)
```

- 窗口包含特征当日。
- `min_periods` 默认等于 `window`，即要求完整窗口。
- `argmax/argmin` 返回从窗口最旧观测开始的 1 基位置；并列时取第一个。
- `delay` 只接受非负整数，其他回看周期只接受正整数。
- 不提供 `lead`、负数位移或中心滚动窗口。

### 3.5 横截面算子

```python
source.rank()                 # 等价于 cs_rank()
source.demean()
source.zscore()
source.scale()                # 同日绝对值之和归一为 1
source.winsorize(0.01, 0.99)
```

这些算子只在同一个 `trade_date` 内比较证券，不会混用其他日期数据。

### 3.6 表达式身份和序列化

链式调用不会立即计算，而是构造不可变表达式树。例如：

```text
cs_rank(ts_stddev(delta(column(close),periods=2),ddof=1,min_periods=5,window=5))
```

```python
payload = factor.to_dict()
factor_id = factor.node.factor_id      # fg_ 开头的稳定短哈希
canonical = factor.node.canonical
lookback = factor.node.lookback
```

可使用 `ExpressionNode.from_dict(payload)` 安全恢复表达式。恢复过程不会执行任意
Python 代码，并会重新校验算子名称、输入数量和全部参数。

## 4. 在现有正式因子中复用 DSL

现有工厂接口无需变化：

```python
from factor_research.factor_dsl import DailyFactorFrame
from .base import FactorFactory
from .registry import register_factor


@register_factor
class MyDslFactorFactory(FactorFactory):
    name = "my_dsl_factor"

    def compute(self, bars, daily):
        frame = DailyFactorFrame(daily)
        return frame.close().returns(1).stddev(20).rank().compute(self.name)
```

正式因子仍然放在自己的独立文件中并按原方式注册。DSL 注册表和正式因子注册表
相互独立。

## 5. 准备只计算一次的数据

```python
from factor_research.factor_search import prepare_search_context

context = prepare_search_context(
    bars_5m,
    fixed_features=[
        "return_1d",
        "volatility_5d",
        "volume_ratio_5d",
    ],
    cache_dir=".factor_cache",
    selection_start="2022-01-01",
    holdout_start="2025-01-01",
    holdout_end="2025-12-31",
)
```

固定因子只在该步骤调用一次 `build_daily_features`。现有 `FactorCache` 仍负责固定
因子的磁盘缓存；命中缓存时工厂不执行。目标收益是每只证券特征日期之后实际存在
行情的下一有效交易日开盘至收盘收益。

如果调用方已经有包含 OHLCV 和固定因子的日频表，可以跳过分钟聚合：

```python
from factor_research.factor_search import SearchContext

context = SearchContext.from_daily(
    daily_frame,
    fixed_features=["return_1d"],
    selection_start="2022-01-01",
    holdout_start="2025-01-01",
)
```

`selection` 用于候选排名和确定方向；`holdout` 只报告结果，不参与筛选。

## 6. 定义搜索空间

### 6.1 单输入流水线网格

```python
from factor_research.factor_search import PipelineGrid, identity, op

space = PipelineGrid(
    sources=["close", "volume", "return_1d"],
    stages=[
        [
            op("delta", periods=[1, 2, 5]),
            identity(),
        ],
        [
            op("ts_stddev", window=[5, 10, 20]),
            op("ts_argmax", window=[5, 10, 20]),
        ],
        [op("cs_rank"), identity()],
    ],
)
```

注意：一个 `op()` 内的参数执行完整笛卡尔积。如需让 `window` 与
`min_periods` 一一配对，应为每组配对值分别写一个 `op()`，不要把两组列表放在
同一个 `op()` 中。

`PipelineGrid` 只接受单输入算子。相关性、协方差和条件表达式使用模板网格。

### 6.2 多输入模板网格

```python
from factor_research.factor_search import ExpressionGrid

space = ExpressionGrid(
    builder=lambda daily, parameters: -(
        daily.volume().log().delta(int(parameters["delta"])).rank()
        .correlation(
            ((daily.close() - daily.open()) / daily.open()).rank(),
            int(parameters["window"]),
        )
    ),
    parameters={
        "delta": [1, 2, 5],
        "window": [5, 10, 20],
    },
)
```

模板只在主进程生成表达式，lambda 不会传入 worker。传给进程后端的是已经规范化
的 `ExpressionNode`。

多个空间可以用 `CombinedGrid([space_a, space_b])` 合并，重复表达式会再次去重。

## 7. 执行搜索

### 7.1 单进程调试

```python
from factor_research.factor_search import FactorGridSearch

result = FactorGridSearch(
    backend="sequential",
    max_candidates=10_000,
    max_depth=5,
    max_lookback=120,
    min_coverage=0.6,
).run(context, space, holdout_top_k=1)
```

### 7.2 多进程搜索

```python
def main():
    # 分钟行情加载、固定因子准备和搜索空间构建也应放在 main 内，避免 Windows
    # 子进程导入主模块时重复执行。
    context = prepare_search_context(load_bars(), fixed_features=["return_1d"])
    space = PipelineGrid(["close"], [[op("cs_rank")]])
    result = FactorGridSearch(
        backend="process",
        n_jobs=8,        # -1 表示使用可用 CPU
        batch_size=32,
        max_candidates=10_000,
    ).run(context, space, holdout_top_k=1)
    print(result.leaderboard.head())


if __name__ == "__main__":
    # Windows 使用 spawn 创建进程，进程入口必须放在该保护块内。
    main()
```

每个 worker 初始化一次日频上下文。每批候选共享同一个表达式节点缓存，因此公共
前缀在批次内只计算一次；批次结束后清理缓存，避免大型搜索持续占用内存。每个
候选的异常会记录到 `result.errors`，不会终止其他候选。

Windows 普通脚本必须使用上例的 `if __name__ == "__main__"` 保护。交互式环境
或无法安全创建子进程的运行器应使用 `backend="sequential"`。

进程并行时应避免模型内部再次占满全部 CPU。例如外层使用 8 个候选进程时，建议
把 LightGBM 的内部线程数设为 1。

### 7.3 遗传编程搜索

当多输入算子使网格空间过大时，可以直接搜索表达式树。算子的输入会递归生成，
因此相关性的左右输入既可以是源列，也可以是搜索得到的子表达式：

```python
from factor_research.factor_search import FactorGeneticSearch, GeneticSearchConfig

config = GeneticSearchConfig(
    sources=("close", "volume", "return_1d"),
    operator_parameters={
        "delta": {"periods": (1, 2, 5, 10, 20)},
        "ts_stddev": {"window": (5, 10, 20, 60)},
        "ts_correlation": {"window": (5, 10, 20, 60)},
        "cs_rank": {},
        "cs_zscore": {},
    },
    population_size=300,
    max_generations=20,
    max_evaluations=5_000,
    max_depth=5,
    max_nodes=15,
    max_lookback=120,
    free_node_count=3,
    length_penalty=0.0005,
    random_seed=20260809,
)

result = FactorGeneticSearch(
    config,
    backend="process",
    n_jobs=8,
    batch_size=16,
).run(context, holdout_top_k=20)
```

表达式长度是终端节点和算子节点的总数，不是字符串字符数。长度惩罚为：

```text
length_penalty_value =
    length_penalty × max(0, node_count - free_node_count)
```

适应度使用 selection 指标并扣除 Rank IC 标准误、长度、深度和覆盖率惩罚。
`max_nodes`、`max_depth` 和 `max_lookback` 是不可突破的硬限制。相同表达式按稳定
`factor_id` 跨代缓存，失败结果也不会重复执行。

遗传选择、交叉和变异只在主进程中使用指定随机种子执行；候选计算由持久化进程池
并行完成。worker 返回后会恢复候选原始顺序，因此相同配置下串行和多进程搜索的
种群轨迹、适应度和最终排序一致。`result.history` 记录每代种群大小、新增评价数、
累计评价数和最优适应度。

交换律算子会规范化输入顺序；默认禁止 `ts_correlation(x, x)`。所有进化和停止
判断都只使用 selection，搜索完全结束后才计算预先指定数量候选的 holdout 指标。

仓库中的 `grid_search_smoke.py` 已使用该遗传搜索入口：主进程按固定种子进化，
由 `SEARCH_N_JOBS` 个 worker 并行评价，最多搜索 8 代和 500 个唯一表达式。报告目录会额外生成
`evolution_history.csv`，记录每代新增/累计评价数、合格候选数和最优适应度；Top K
模型复验与 holdout 一样只在进化结束后运行，不参与适应度或父代选择。报告还会
生成 `rolling_ic_stability.svg`，展示 selection 排名最高且已完成 holdout 评价候选
的 IC、Rank IC 20 日及 60 日滚动趋势，四个小面板分别动态缩放纵轴、以红色虚线标记
holdout 起点；滚动图沿用 selection 锁定的因子方向，只用于观察近期拐点以及长期
衰减、漂移和符号翻转，不参与重排。

`FactorGeneticSearch.run(progress_callback=...)` 支持批次级进度回调；内置串行和
多进程会话每完成一个 `batch_size` 候选批次，就在主进程报告阶段、代次、完成数、
失败数、selection 总预算、耗时和 ETA。`grid_search_smoke.py` 默认启用控制台输出，
并将 `SEARCH_BATCH_SIZE` 设为 8；减小该值会提高刷新频率，但也会增加任务调度开销。

## 8. 指标和排行榜

```python
print(result.leaderboard.head(20))
print(result.errors)
print(result.best_candidate.canonical)
```

`canonical` 是可由 `ExpressionNode.from_string()` 安全恢复的规范字符串。搜索报告
的 `candidates.json` 同时写出同值的 `expression_str` 字段；可直接复制到主实验：

```powershell
python run_factor_demo.py --factors --factor-expressions 'cs_rank(delta(column(close),periods=5))'
```

默认排序目标是 `selection_oriented_rank_ic`，并且 `objective` 强制要求以
`selection_` 开头，不能配置 holdout 指标。方向只根据 selection 的 Rank IC
决定：负 Rank IC 的因子方向为 -1，holdout 使用同一个已锁定方向，避免从
holdout 反向选择。

全量候选初筛不会计算 holdout。`holdout_top_k` 默认是 1，表示只对 selection
第一名补充 holdout 指标；设为更大的数只应当用于事先确定的最终候选数量，不能
在看到 holdout 后再次改变入选结果。

排行榜包含：

- 表达式、因子 ID、深度、最大回看长度和计算耗时；
- selection 的覆盖率、IC、Rank IC、标准差、ICIR；只有预先入选的 Top K 行
  才包含 holdout 对应指标；
- 有效 IC 日期数、正 Rank IC 比例、平均横截面样本数；
- `eligible`，表示覆盖率和排序目标是否满足要求。

这里的 ICIR 是“日均 IC / 日 IC 样本标准差”，没有年化。

## 9. Top K 复用现有模型实验

```python
from factor_research.factor_search import ModelCandidateEvaluator
from factor_research.models.simple_decision_tree import (
    SimpleDecisionTreeModelFactory,
)

model_evaluator = ModelCandidateEvaluator(
    model_factory=SimpleDecisionTreeModelFactory(
        max_depth=3,
        min_samples_leaf=20,
    ),
    training_mode="rolling",
    task="classification",
)

result = search.run(
    context,
    space,
    model_evaluator=model_evaluator,
    model_top_k=10,
)
```

模型评价把“固定因子 + 当前候选”组装成普通日频特征，调用现有
`build_direction_dataset` 和 `DirectionExperiment`。固定因子值不会重新计算，
但每个候选改变了模型输入，因此模型训练必须相互独立。

## 10. 把最优因子交给现有流程

```python
best_daily = result.materialize(context, oriented=True)
best_name = result.best_candidate.factor_id

dataset = build_direction_dataset(
    best_daily,
    feature_columns=[*context.fixed_features, best_name],
)
experiment_result = DirectionExperiment(
    validation_start="2025-01-01",
    feature_columns=[*context.fixed_features, best_name],
).run(dataset)
```

`materialize` 只是新增一个普通 DataFrame 列，不会自动注册因子。如果需要长期生产
使用，应把最终表达式写入独立 `FactorFactory` 文件并增加精确数值测试。

## 11. 防止未来数据泄漏

- 表达式注册表中的内置算子全部标记为因果算子。
- `delay` 禁止负周期，其余时序周期必须为正整数。
- 所有滚动窗口都以当前行为右端点，不支持中心窗口。
- 目标构建集中在 `build_forward_targets`，与现有数据集使用相同实现。
- selection 和 holdout 按 `target_date` 切分且互不重叠。
- 排序目标只能来自 selection；全量排行榜不计算所有候选的 holdout。
- 因子方向只在 selection 确定。
- Top K 模型验证继续使用现有 `DirectionExperiment` 的
  `training target_date < prediction target_date` 规则。

新增算子或搜索能力后，应同时增加“修改未来数据不影响历史输出”的前缀不变性
测试，以及多证券分组隔离测试。
