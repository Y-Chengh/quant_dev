# AGENTS.md

## 项目概述

本项目是一个基于日频和分钟行情的轻量级量化因子研究框架。主要流程包括：

1. 将分钟行情聚合为日频行情。
2. 通过独立的因子工厂生成特征。
3. 使用当日及以前可见的数据预测下一有效交易日开盘至收盘的涨跌方向或涨跌幅。
4. 支持扩展窗口逐日训练验证，以及固定训练集单次训练后在验证集测试。

## 目录说明

- `factor_research/factor_factories/`：因子工厂，每个具体因子使用独立文件。
- `factor_research/factor_dsl/`：不可变因子表达式、算子注册和日频执行上下文。
- `factor_research/factor_search/`：网格/遗传搜索、一次性上下文、候选评价及并行后端。
- `factor_research/models/`：预测模型抽象、具体模型及模型工厂。
- `factor_research/factors.py`：日频聚合、因子计算和缓存流程。
- `factor_research/dataset.py`：特征、标签及训练数据集构建。
- `factor_research/experiment.py`：滚动训练、预测和实验结果汇总。
- `factor_research/backtesting.py`：验证集 Top N 日内等权回测、交易成本、随机/等权
  基准、横截面分组及收益价差诊断。
- `factor_research/metrics.py`：分类、回归及每日横截面 IC/Rank IC 评估指标。
- `factor_research/reporting.py`：Markdown/HTML 双格式评估报告和图表输出。
- `tests/`：单元测试和端到端测试。

## 因子开发规范

- 每个具体因子放在 `factor_research/factor_factories/` 下的独立 Python 文件中。
- 因子类必须继承 `FactorFactory`，使用 `@register_factor` 注册，并定义唯一的
  `snake_case` 名称。
- `compute(bars, daily)` 必须为 `daily` 的每一行返回一个值，并保留
  `daily.index`。
- 日频滚动计算必须按 `code` 独立执行。调用滚动函数前，确认数据已经按
  `code` 和 `trade_date` 排序。
- 分钟因子使用 `intraday_values`，按 `code`、`trade_date` 分组计算并对齐回
  日频索引。
- 注释必须说明计算公式、窗口是否包含当日、最小观测数、缺失值规则，以及
  是否进行了年化或归一化。
- 优先使用连续因子表达方向和强度，避免同时创建互为正反且信息重复的二元
  因子。
- 新增或修改因子后，更新自动注册集合测试，并增加具体数值和边界条件测试。
- 正式因子可以调用 `factor_dsl` 的基础算子，但搜索产生的临时候选不得注册到
  `FACTOR_FACTORIES`，避免改变 `DEFAULT_FEATURES`。

## 因子搜索开发规范

- 算子层不得依赖搜索、模型、指标或正式因子注册表；搜索空间不得读取行情数据。
- 时间序列算子必须按 `code` 独立计算，横截面算子必须按 `trade_date` 独立计算，
  并在计算前由 `DailyFactorFrame` 完成稳定排序。
- 新算子必须声明输入数量、参数校验、作用域、额外回看长度和因果性；不得提供
  负数位移、未来引用或中心滚动窗口。
- `SearchContext` 负责一次性保存日频数据、固定因子、目标和日期切分；候选 worker
  不得重新调用固定因子工厂。
- 单进程和多进程后端必须返回相同的候选顺序、数值指标和错误隔离结果。
- 遗传搜索的随机选择只允许在主进程按固定种子执行；worker 只计算候选。表达式
  必须限制节点数、深度和回看长度，相同 `factor_id` 应跨代缓存，复杂度惩罚只能
  使用表达式结构及 selection 指标计算。
- 候选方向和排名只能使用 selection 区间确定，holdout 只用于最终报告。
- 遗传搜索的等价候选去重只能使用 selection 上方向调整后的候选值和逐日横截面
  Rank；每个等价类只保留节点数最少者参与父代选择、排行榜和后续 holdout，全部
  已评价表达式仍应保留在候选附件中供审计。
- 选中的表达式只有在转为独立 `FactorFactory` 并补齐测试后，才能进入正式默认
  因子集合。
- 搜索输出的 `canonical`/`expression_str` 可以通过主实验的
  `factor_expressions` 配置作为临时候选复用；解析必须走 DSL 白名单，不得使用
  `eval`，且不得把临时候选注册到 `FACTOR_FACTORIES`。

## 防止未来数据泄漏

- 特征只能使用特征日期收盘时已经可见的数据。
- 预测目标日期为 `T` 时，训练样本必须满足 `target_date < T`。
- 表示“突破前高”等概念时，历史基准窗口必须先 `shift(1)`，不能把当日价格
  放入突破基准。
- 缺失值填充值只能根据当前训练窗口计算，不能使用完整数据集或验证集统计量。
- 不得使用下一交易日价格、收益率、标签或未来滚动窗口构造特征。

## 模型开发规范

- 模型相关代码统一放在 `factor_research/models/`。
- 通用抽象定义在 `models/base.py`；一个具体模型及其工厂放在独立模块中。
- 具体模型继承 `DirectionModel`，并实现：
  - `fit(X, y)`
  - `predict_proba(X)`
- 回归模型还需实现 `predict(X)`，返回与输入样本数相同的一维有限涨跌幅数组。
- `feature_importances_` 是可选能力。模型可以不定义该属性或将其设为 `None`；
  如果提供，必须是与输入特征数相同的一维有限数值数组。
- 模型工厂可通过 `required_finite_feature_indices` 声明不得填充缺失值的特征列
  下标；实验会在日期切分前删除这些列含 NaN 或无穷值的样本。缺省为空元组，
  仍沿用通用缺失值填充流程。
- 模型工厂继承 `DirectionModelFactory`，定义唯一的 `name` 并实现 `create()`。
- 模型工厂使用 `@register_model_factory` 注册，并负责通过 `add_arguments()` 声明
  自身 CLI 参数、通过 `from_args()` 从命令行参数构建工厂。主程序不得为具体
  模型增加选择分支。
- CLI 必须先解析 `--model`，再只注册所选模型的专属参数；不得把所有模型参数
  同时添加到一个解析器，以免不同模型的同名参数发生冲突。
- 主实验 CLI 支持通过 `--config` 加载扁平 YAML 配置；键名使用 `snake_case`，
  命令行参数优先于 YAML，并且 YAML 值必须经过与命令行一致的类型和选项校验。
- `create()` 每次必须返回一个无历史训练状态的新模型，以保证逐日滚动验证相互
  独立。
- `DirectionExperiment` 只能依赖模型抽象和模型工厂，不应直接实例化具体算法。
- 替换模型时优先注入新的 `model_factory`，不要复制或分叉滚动验证流程。

## 代码修改原则

- 保持修改范围聚焦，不顺带重写无关模块。
- 按指令完成代码修改后，检查项目结构、开发约束、公共接口、验证命令或协作
  流程是否发生变化；如有必要，必须同步更新 `AGENTS.md`，确保说明与代码现状
  一致。若修改不影响这些内容，则不要仅为留痕而改动 `AGENTS.md`。
- 保留工作区中已有且与当前任务无关的用户改动。
- 公共接口发生变化时尽量保持向后兼容，并同步更新导出入口和引用位置。
- 使用清晰的类型注解、中文文档字符串和必要的行内注释。
- 不要为了消除缺失值而静默改变因子的金融含义或窗口口径。
- 不提交生成的缓存、日志、报告、`__pycache__` 或虚拟环境文件。

## 函数开发规范

- 每个函数和方法都必须有中文文档字符串，说明职责、关键口径及返回结果。
- 文档字符串的“参数”部分必须逐项列出每个形参（隐式实例或类参数
  `self`/`cls` 除外），并说明该变量的具体业务含义、单位、可选值或缺省行为；
  不得只重复参数名或类型注解。
- 可变位置参数和可变关键字参数必须分别说明每个元素或键值的语义与约束。

## 本地 Python 环境

- 本机已安装 Python 3.11.9，解释器路径为
  `C:\Users\win10\AppData\Local\Programs\Python\Python311\python.exe`。
- 对话内的受限终端可能因沙箱权限无法访问用户目录，从而误报找不到 `python`；
  这不代表本机未安装 Python。遇到此情况时，应使用上述解释器路径，或在确有
  必要时申请沙箱外执行权限后再次验证，不要建议用户重复安装 Python。

## 验证要求

修改完成后按风险执行以下检查：

```powershell
python -m compileall -q factor_research tests
python -m unittest discover -s tests -v
git diff --check
```

- 后续运行 `run_factor_demo.py` 的回测（包括修改后的自动验证回测）必须统一加载
  `C:\Users\win10\Documents\quant\factor_config.example.diff.yaml`：
  `python run_factor_demo.py --config C:\Users\win10\Documents\quant\factor_config.example.diff.yaml`。
- 不涉及模型或数据代码的修改无需运行 `run_factor_demo.py` 模型验证；涉及模型或
  数据代码时，应按改动风险运行上述回测并核对验证结果。

如果本地 Python、依赖或虚拟环境不可用，应至少执行 `git diff --check` 和静态
引用检查，并在交付说明中明确指出未运行的验证及原因。

每次代码修改和必要验证完成后，必须新创建一个 sub-agent 对本次修改进行独立
code review。review 范围只包含当前任务的改动，并重点检查正确性、未来数据
泄漏、接口兼容性、边界条件和测试覆盖：

- 如果 code review 没有发现问题，自动生成本次修改摘要，并提交仅属于当前
  任务的文件或变更块；不得将工作区中原有的无关修改一并提交。
- 如果 code review 发现需要修改的问题，必须自动修复、重新执行必要验证，并
  再次创建独立 sub-agent 复审；在 review 无问题前不得提交。只有遇到无法通过
  代码、测试或项目约束自行判定的外部业务决策时，才请求人工介入。
- code review 结论、已执行的验证、提交摘要和 commit hash 必须在最终交付说明
  中列出。

测试至少应覆盖：

- 正常输入下的精确计算结果。
- 最小窗口和缺失历史数据。
- 多证券分组之间不存在状态串联。
- 滚动训练不使用预测日期及未来标签。
- 新模型可以通过工厂注入，而无需修改实验核心流程。
