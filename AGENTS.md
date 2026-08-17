# AGENTS.md

## 项目概述

本项目是一个基于日频和分钟行情的轻量级量化因子研究框架。主要流程包括：

1. 将分钟行情聚合为日频行情。
2. 通过独立的因子工厂生成特征。
3. 使用当日及以前可见的数据预测下一有效交易日开盘至收盘的涨跌方向或涨跌幅。
4. 支持扩展窗口逐日训练验证，以及固定训练集单次训练后在验证集测试。

## 目录说明

项目采用 src layout，全部库代码位于 `src/quant/` 下的统一命名空间包中，
通过 `pip install -e .` 安装后以 `quant.*` 导入。

顶层目录：

- `src/quant/`：全部库代码。
- `scripts/`：不经 pip 安装、需按文件路径直接运行的入口（大 QMT 编辑器等）。
- `configs/`：配置样例，按子系统分目录。
- `docs/`：详细文档。
- `tests/`：测试，目录结构与 `src/quant/` 一一对应。
- `pyproject.toml`：打包、依赖分组、命令行入口与 ruff/mypy/pytest 配置。
- `.claude/`：Claude Code 项目级配置。`settings.json` 注册 `UserPromptSubmit`
  hook，`hooks/review-reminder.md` 是该 hook 每轮注入的正文，用途见下文验证要求；
  二者入库共享。`settings.local.json` 是本机权限授予，已在 `.gitignore` 中排除。

因子研究 `src/quant/factor_research/`：

- `factor_factories/`：因子工厂，每个具体因子使用独立文件。
- `factor_dsl/`：不可变因子表达式、算子注册和日频执行上下文。
- `factor_search/`：网格/遗传搜索、一次性上下文、候选评价及并行后端；其中
  `genetic/` 按 `config`/`events`/`session`/`trees`/`generator`/`search` 分层。
- `models/`：预测模型抽象、具体模型及模型工厂。
- `reporting/`：Markdown/HTML 双格式评估报告和图表输出，按
  `labels`/`formatting`/`markdown`/`charts`/`report` 分层。
- `search_report/`：因子搜索结果的可审计报告生成，按
  `constants`/`formatting`/`reproduction`/`summaries`/`charts`/`report` 分层。
- `data_sources/`：可切换的行情数据源注册表（5 分钟库与 QMT 日线库），
  结构与 `models/` 的工厂注册表一致。
- `factors.py`：日频聚合、因子计算和缓存流程。
- `dataset.py`：特征、标签及训练数据集构建。
- `experiment.py`：滚动训练、预测和实验结果汇总。
- `backtesting.py`：验证集 Top N 日内等权回测、交易成本、随机/等权基准、
  横截面分组及收益价差诊断。
- `metrics.py`：分类、回归及每日横截面 IC/Rank IC 评估指标。

其余子包：

- `src/quant/config/`：项目根目录、配置目录和行情库路径的唯一解析入口；
  优先级为显式传参 > 环境变量 > 内置回退值。任何模块都不得再内联书写绝对路径。
- `src/quant/market_data/`：本地 DuckDB 行情库、查询客户端与 FastAPI 网页服务。
  - `daily/`：与 5 分钟库相互独立的 QMT 日线库，按 `models`/`schema`/`database`/
    `adjust`/`universe`/`client` 分层；库中只存**原始不复权**价，复权在读取层计算。
    其中 `daily/ingest/` 负责把大 QMT 落盘的日线 CSV 增量转成月度 Parquet 与目录表。
  - `daily_check/`：`quant-market-check` 的实现，对入库后的日线库做全样本业务校验，
    按 `config`/`loading`/`rules_*`/`limits`/`cross_5m`/`summary`/`checker` 分层。
- `src/quant/qmt_downloader/`：仅使用大 QMT 内置 Python 的日线、财务和除权数据
  按日分区保存工具；`scripts/qmt_run_downloader.py` 是大 QMT 策略入口。
  其中 `runner/` 与 `self_check/` 都按职责拆为 mixin 包：各 mixin 只承担一类
  职责，共享状态集中声明在 `base.py`，`downloader.py`/`checker.py` 只做编排。
- `src/quant/cli/`：命令行入口，只做参数解析、配置装载、日志初始化和调用库层，
  不得承载可复用的业务逻辑。每个模块对应 `pyproject.toml` 中的一个
  `console_scripts` 入口：

  | 命令 | 模块 |
  | --- | --- |
  | `quant-build-daily-store` | `quant.cli.build_daily_store` |
  | `quant-factor-demo` | `quant.cli.factor_demo` |
  | `quant-market-check` | `quant.cli.market_check` |
  | `quant-grid-search` | `quant.cli.grid_search` |
  | `quant-market-server` | `quant.cli.market_server` |
  | `quant-qmt-self-check` | `quant.cli.qmt_self_check` |
  | `quant-single-factor-test` | `quant.cli.single_factor_test` |
  | `quant-ifind-download` | `quant.cli.ifind_download` |
  | `quant-build-market-db` | `quant.market_data.build_database` |

  上表是 `pyproject.toml` 的入口注册名，只用于说明「哪个模块对应哪个入口」；
  面向用户输出命令时的写法见下文「命令行输出约定」。

  其中 `quant-qmt-self-check` 在外部 Python 中全量审计 QMT 日线分区、证券生命
  周期、缺失区间、停牌成交量和统计异常，输出不修改原始数据的详细报告。
  `quant-market-check` 与它分工互补：前者查 CSV 源本身是否完整，后者查**入库之后**
  的日线库在业务上是否合法，并可与 5 分钟库交叉对账。

## 分层约束

- `quant.qmt_downloader` 必须能在大 QMT 内置 Python 中直接导入，因此只允许依赖
  标准库和 pandas，不得导入 `quant` 的其它子包，也不得使用较新的语法糖。
- 大 QMT 内置 Python 早于 3.7：导入链上的 `src/quant/__init__.py` 和
  `src/quant/qmt_downloader/*.py`（仅在外部 Python 运行的 `self_check.py` 除外）
  都不得写 `from __future__ import annotations`，否则大 QMT 直接抛
  `SyntaxError: future feature annotations is not defined`。入口脚本 `init()`
  会打印 `sys.version`，便于确认实际解释器版本。该约束由
  `test_qmt_import_chain_avoids_future_annotations` 守护。
- `scripts/qmt_run_downloader.py` 必须保持纯 ASCII 源码：该入口需整份复制进大 QMT
  编辑器，而编辑器按 GBK 保存源码，文件头的 UTF-8 编码声明会把任何非 ASCII 字节
  解成 `SyntaxError: (unicode error) 'utf-8' codec can't decode byte`。因此其文档
  字符串和注释一律使用英文，中文说明放在 `docs/qmt_downloader.md`。该约束由
  `test_qmt_entry_source_is_ascii_safe` 守护。
- `src/quant/__init__.py` 与 `src/quant/cli/__init__.py` 不得导入任何子模块，
  避免轻量场景被迫加载全部三方依赖。
- 命令行层可以依赖库层，库层不得反向依赖 `quant.cli`。
- `quant.market_data` 可以单向依赖 `quant.qmt_downloader` 的公开列常量、分区校验
  函数与报告写入工具（`gateway.*_COLUMNS`、`storage.validate_partition_directory`、
  `self_check.writers.write_*_atomic`、`self_check.AuditIssue`、`config.strip_jsonc`
  这个纯文本工具，以及 `errata.load_errata_overrides`/`errata.errata_pivot_frame`
  这两个源数据勘误函数）。这样 CSV 列契约与报告格式只有一份定义。反向依赖仍然
  禁止：`quant.qmt_downloader` 必须能在大 QMT 内置 Python 中独立导入。

## 模块规模与拆分约定

- 单个模块超过约 700 行时应拆为包。拆分只做机械搬运：函数体逐行不改，
  公开接口保持不变，调用方不需要改导入语句。
- 拆分后包内依赖必须单向无环，并在包 `__init__.py` 的文档字符串中列出各子模块
  的职责，同时只重导出对外公开的名字；包内私有实现由使用方从对应子模块直接
  导入，不在包入口再导出。
- 拆分共享大量可变状态的大类时使用 mixin：每个 mixin 只承担一类职责，
  主类只保留 `__init__` 与编排方法，基类顺序与原方法定义顺序一致以保证行为不变。
  跨 mixin 共享的实例属性必须集中声明在包内 `base.py` 的状态契约类中，
  逐项说明业务含义；不得依赖隐式约定。
  - `quant.factor_research` 与 `quant.market_data` 下的状态契约用变量注解声明。
  - `quant.qmt_downloader` 下的状态契约只能用文档字符串说明，因为大 QMT 内置
    Python 早于 3.7，不支持变量注解。

## 因子开发规范

- 每个具体因子放在 `src/quant/factor_research/factor_factories/` 下的独立 Python 文件中。
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
- 使用 `intraday_values` 的因子必须在**自身模块**声明 `requires_intraday = True`，
  由 `factor_factories/capabilities.py` 统一读取，日频数据源据此自动排除它们。
  **不得**把该属性上提到 `FactorFactory` 基类：`_implementation_fingerprint` 会哈希
  `factor_factories/base.py` 的文件字节，改动基类会作废全部已有因子缓存。
  同理，`_build_daily_bars`、`_input_fingerprint` 和 `factor_dsl/` 下的文件也不得
  为了无关目的改动。
- 因子默认必须对价格尺度零次齐次，即把窗口内全部价格乘以同一个正常数后取值不
  变。后复权系数逐证券不同，带价格量纲的因子在横截面上会主要由「上市时长 ×
  分红历史」决定，而不是由信号决定。确需偏离时在**自身模块**声明
  `price_homogeneity`：整数 `k` 表示 `g(c·P) = c**k · g(P)`，`None` 表示不满足
  任何齐次度（例如同一表达式里混用一次齐次的价格与零次齐次的收益率）。该属性
  同样由 `factor_factories/capabilities.py` 用 `getattr` 读取，未声明按 `0` 处理，
  且与 `requires_intraday` 一样**不得**上提到 `FactorFactory` 基类。汇总入口是
  `dimensional_factor_names()`（`k` 为非零整数，可按 `c**k` 解析换算）与
  `non_homogeneous_factor_names()`（`k` 为 `None`，无解析形式，只能个案评估）。
  这里说的「不变」针对更换前复权基准日，因为 `qfq(s|anchor) = hfq(s)/hfq(anchor)`
  的分母对该证券是常数；它**不**意味着 `--adjust` 选 `none` 与 `hfq` 结果相同，
  `hfq(code, t)` 跨除权日会跳档，且只有复权后的取值才是对的。
- 上一条由 `tests/factor_research/test_scale_invariance.py` 用**逐证券**缩放校验。
  绝不能改成全截面统一缩放：那样横截面排名与 z-score 恒等不变，会给带价格量纲的
  因子发出假的合格证。缩放系数还必须跨越多个数量级且不与样本价位同向单调，否则
  横截面顺序不被打乱，用例会空转；该文件用一条独立用例守数量级跨度，并在横截面
  排名用例里以前提断言守单调性。
- 因子不得混用复权价与未复权量。日线库只复权 `open`/`high`/`low`/`close`/
  `pre_close`，`amount` 恒不复权，`volume` 仅在 `adjust_volume=True` 时反向调整。
  因此同一个表达式里 `close` 与成交量类字段的齐次度会随该开关变化，`close * volume`
  这类近似成交额的写法在两种配置下口径不同。因子可见的基础列**契约**由
  `quant.factor_research.factors.BASE_DAILY_COLUMNS` 唯一定义（`_prepare_daily_bars`、
  `_daily_input_fingerprint` 与两处表达式列守卫都读它），日频因子拿不到 `amount`；
  若将来把它加入基础列，`amount / volume` 推出的 VWAP 与已复权的 `close` 不同尺度，
  直接相比会在每个除权日产生虚假跳变。注意分钟路径**实际产出**哪些列仍写在
  `_build_daily_bars` 里，往常量加列时必须手工同步——不能改 `_build_daily_bars` 来
  加列，那会作废全部分钟缓存；两条路径由 `tests/factor_research/test_adjust_factor_column.py`
  的列集合用例对齐，漏同步会直接测试失败而不是静默分叉。
- 基础列 `adjust_factor` 是当前复权口径乘到价格上的那个正系数，于是
  `close / adjust_factor` 在三种口径下都还原为同一个原始不复权价；分钟路径没有复权
  概念，`build_daily_features` 与 `aggregate_daily_bars` 在那里补 1.0，使两条路径的
  列宇宙一致。使用它必须守住三条：**只能同日横截面用**（该系数在时间上是阶梯函数，
  还原出的原始价跨除权日会跳档，再叠任何时序算子都会算出假跳空）；**单独当特征时
  是一次齐次**，必须声明 `price_homogeneity = 1`，否则横截面上排的是「上市时长 ×
  分红送转历史」；**`qfq` 口径下有未来数据泄漏**，因为 `hfq(t) / hfq(anchor)` 的分母
  含有 `t` 之后的分红，只有作为还原原始价的分母（`hfq(anchor)` 自动约掉）才安全。
  `_rescale_prices` 必须与价格同步缩放该列，否则「还原原始价」会被误判成一次齐次。
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
  因子集合；转正前必须按上文的逐证券缩放校验确认它对价格尺度零次齐次。DSL 允许
  写出 `close` 这类带价格量纲的表达式，`rank`/`zscore` 也**不能**把它救回来——
  横截面归一化只对全截面统一缩放不变，对逐证券的后复权系数并不免疫。
- 搜索输出的 `canonical`/`expression_str` 可以通过主实验的
  `factor_expressions` 配置作为临时候选复用；解析必须走 DSL 白名单，不得使用
  `eval`，且不得把临时候选注册到 `FACTOR_FACTORIES`。

## 数据源开发规范

- 行情数据源相关代码统一放在 `src/quant/factor_research/data_sources/`。
- 通用抽象定义在 `data_sources/base.py`；一个具体数据源放在独立模块中。
- 具体数据源继承 `MarketDataSource`，使用 `@register_data_source` 注册，定义唯一
  的 `name`、`frequency` 和 `provides_intraday`，并实现 `metadata()`、
  `list_symbols()`、`load_bars()` 与 `build_features()`。
- 数据源负责通过 `add_arguments()` 声明自身 CLI 参数、通过 `from_args()` 构建实例。
  主程序不得为具体数据源增加选择分支。
- CLI 必须先解析 `--data-source`，再只注册所选数据源的专属参数；不同数据源允许有
  同名参数。切换数据源时，其它数据源的 YAML 键应通过 `data_source_argument_names()`
  列入忽略集合，而不是报未知参数。
- 数据源必须实现 `cache_namespace()`，把数据源名与一切影响取值的口径（如复权方式、
  前复权基准日）放进因子缓存路径。缺省返回空串，以保证既有 5 分钟缓存路径不变。

## 防止未来数据泄漏

- 特征只能使用特征日期收盘时已经可见的数据。
- 前复权会在每次出现新的分红送转时重算全部历史价格，等于把未来信息带进历史序列。
  研究默认使用后复权；确需前复权时必须显式固定基准日，并让基准日进入因子缓存
  命名空间，避免同一份缓存混入两个基准。
- 预测目标日期为 `T` 时，训练样本必须满足 `target_date < T`。
- 表示“突破前高”等概念时，历史基准窗口必须先 `shift(1)`，不能把当日价格
  放入突破基准。
- 缺失值填充值只能根据当前训练窗口计算，不能使用完整数据集或验证集统计量。
- 不得使用下一交易日价格、收益率、标签或未来滚动窗口构造特征。

## 模型开发规范

- 模型相关代码统一放在 `src/quant/factor_research/models/`。
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

- 每个函数和方法都必须有中文文档字符串，说明职责、关键口径及返回结果。唯一例外
  是 `scripts/qmt_run_downloader.py`，原因见上文分层约束。
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
- 项目依赖只装在仓库内的 `.venv` 里，因此跑验证命令时优先激活 `.venv` 或直接用
  `.venv\Scripts\python.exe`；上面那个系统解释器路径只用于确认本机确实装了
  Python，用它跑 `pytest` 会因为缺少依赖而 `ImportError`。

## 命令行输出约定

- 凡是输出给用户的命令行（对话回复、交付说明、`README.md`、`docs/`、代码里的
  报错提示与修复建议）都尽可能写成 `python xxx` 的形式，模块入口统一用
  `python -m <模块>`，例如
  `python -m quant.cli.factor_demo --config configs\factor_research\example.yaml`。
  命令默认在仓库根目录、已激活项目虚拟环境（`.venv`）的会话中执行，因此不要在
  命令里写死 `.venv\Scripts\python.exe` 这类解释器绝对路径，让用户可以直接复制。
  唯一需要额外交代的是创建虚拟环境本身：`python -m venv .venv` 之后必须先给出
  激活命令（见 README 安装一节），再用 `python -m pip install`，否则 `python`
  仍指向系统解释器。
- 上述约束只针对可直接复制执行的命令行。正文里泛指某个工具（例如“`pip install -e .`
  之后可用短命令”）不受影响。助手自己在未激活环境的终端里执行时可以用
  `.venv\Scripts\python.exe`，但写给用户看的命令仍按上面的写法给出。
- 不要直接给出 `quant-factor-demo` 这类 `console_scripts` 短命令，它们只在
  `pip install -e .` 之后可用。短命令仅作为等价别名出现在 README 的对照表中，
  与上文 CLI 入口表的说明保持一致。
- Python 生态的工具一律走 `-m`：用 `python -m pip`、`python -m pytest`、
  `python -m unittest`、`python -m ruff`、`python -m mypy`，而不是 `pip`、
  `pytest`、`ruff` 等裸命令。只有本身不属于 Python 的可执行文件（如 `git`）
  按原样给出。

## 验证要求

首次准备环境（或依赖变化后）：

```powershell
python -m pip install -e ".[research,service,dev]"
```

修改完成后按风险执行以下检查：

```powershell
python -m compileall -q src tests scripts
python -m unittest discover -s tests -v
python -m ruff check .
git diff --check
```

`python -m pytest` 与 `unittest discover` 等价，两者都能跑通全量用例。
上面的写法遵循「命令行输出约定」，假定会话已激活 `.venv`；助手自己所在的终端
未激活时，把 `python` 换成 `.venv\Scripts\python.exe` 再执行。

- 涉及模型或数据代码的改动，应按风险运行回测验证并核对结果：
  `python -m quant.cli.factor_demo --config configs\factor_research\example.yaml`。
  同一次对比实验必须始终使用同一份配置，避免前后结果不可比。
- 不涉及模型或数据代码的修改无需运行回测。

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

上述要求还由 `.claude/settings.json` 注册的 `UserPromptSubmit` hook 在每一轮对话
开头注入一次，正文见 `.claude/hooks/review-reminder.md`。原因是 Claude Code 的内置
提示默认不主动调用 sub-agent，仅靠本文件的书面约定容易被忽略。修改本节时必须同步
更新该正文，避免两处口径不一致。

测试至少应覆盖：

- 正常输入下的精确计算结果。
- 最小窗口和缺失历史数据。
- 多证券分组之间不存在状态串联。
- 滚动训练不使用预测日期及未来标签。
- 新模型可以通过工厂注入，而无需修改实验核心流程。
