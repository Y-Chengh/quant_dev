"""支持多输入表达式、长度惩罚和并行评价的遗传编程因子搜索。"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import combinations, product
from time import perf_counter
from types import TracebackType

import numpy as np
import pandas as pd

from quant.factor_research.factor_dsl import ExpressionNode, get_operator, operation_node

from .backends import (
    BatchProgressCallback,
    CandidateTaskResult,
    ExecutionBackend,
    ExecutionSession,
    ProcessBackend,
    SequentialBackend,
)
from .context import SearchContext
from .evaluators import CandidateEvaluator, HoldoutIcEvaluator, IcEvaluator
from .result import FactorSearchResult
from .space import FactorCandidate


def _default_operator_parameters() -> dict[str, dict[str, tuple[object, ...]]]:
    """返回第一版遗传搜索使用的因果算子和离散参数白名单。"""

    windows: tuple[object, ...] = (5, 10, 20, 60)
    periods: tuple[object, ...] = (1, 2, 5, 10, 20)
    return {
        "add": {},
        "subtract": {},
        "multiply": {},
        "divide": {},
        "absolute": {},
        "delta": {"periods": periods},
        "returns": {"periods": periods},
        "ts_mean": {"window": windows},
        "ts_stddev": {"window": windows},
        "ts_rank": {"window": windows},
        "ts_correlation": {"window": windows},
        "cs_rank": {},
        "cs_zscore": {},
    }


@dataclass(frozen=True)
class GeneticSearchConfig:
    """声明遗传表达式空间、种群规模、复杂度限制和适应度惩罚。"""

    sources: tuple[str, ...] = ("close", "volume", "return_1d")
    operator_parameters: Mapping[str, Mapping[str, Sequence[object]]] = field(
        default_factory=_default_operator_parameters
    )
    commutative_operators: frozenset[str] = frozenset(
        {"add", "multiply", "ts_correlation"}
    )
    disallow_same_input_operators: frozenset[str] = frozenset(
        {"ts_correlation"}
    )
    population_size: int = 300
    max_generations: int = 20
    max_evaluations: int = 5_000
    initial_max_depth: int = 3
    max_depth: int = 5
    max_nodes: int = 15
    max_lookback: int = 120
    min_coverage: float = 0.70
    target_coverage: float = 0.90
    objective: str = "selection_oriented_rank_ic"
    stability_weight: float = 0.5
    free_node_count: int = 3
    length_penalty: float = 0.0005
    free_depth: int = 2
    depth_penalty: float = 0.0005
    coverage_penalty: float = 0.02
    elite_ratio: float = 0.10
    tournament_size: int = 5
    crossover_probability: float = 0.35
    subtree_mutation_probability: float = 0.30
    parameter_mutation_probability: float = 0.20
    reproduction_probability: float = 0.15
    terminal_probability: float = 0.25
    patience: int = 5
    min_improvement: float = 0.0001
    random_seed: int = 20260809

    def __post_init__(self) -> None:
        """校验搜索规模、概率、复杂度和算子白名单。"""

        if not self.sources or any(
            not isinstance(source, str) or not source for source in self.sources
        ):
            raise ValueError("sources 必须包含至少一个非空列名")
        if len(self.sources) != len(set(self.sources)):
            raise ValueError("sources 不能包含重复列名")
        positive_integers = {
            "population_size": self.population_size,
            "max_generations": self.max_generations,
            "max_evaluations": self.max_evaluations,
            "max_depth": self.max_depth,
            "max_nodes": self.max_nodes,
            "patience": self.patience,
            "tournament_size": self.tournament_size,
        }
        for name, value in positive_integers.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} 必须是正整数")
        nonnegative_integers = {
            "initial_max_depth": self.initial_max_depth,
            "max_lookback": self.max_lookback,
            "free_node_count": self.free_node_count,
            "free_depth": self.free_depth,
            "random_seed": self.random_seed,
        }
        for name, value in nonnegative_integers.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} 必须是非负整数")
        if not 0 <= self.initial_max_depth <= self.max_depth:
            raise ValueError("initial_max_depth 必须在 0 到 max_depth 之间")
        for name, value in {
            "min_coverage": self.min_coverage,
            "target_coverage": self.target_coverage,
            "elite_ratio": self.elite_ratio,
            "terminal_probability": self.terminal_probability,
        }.items():
            if not 0 <= value <= 1:
                raise ValueError(f"{name} 必须在 0 到 1 之间")
        if self.target_coverage < self.min_coverage:
            raise ValueError("target_coverage 不能小于 min_coverage")
        probabilities = (
            self.crossover_probability,
            self.subtree_mutation_probability,
            self.parameter_mutation_probability,
            self.reproduction_probability,
        )
        if any(value < 0 for value in probabilities) or not math.isclose(
            sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("交叉、两类变异和复制概率必须非负且合计为 1")
        for name, value in {
            "stability_weight": self.stability_weight,
            "length_penalty": self.length_penalty,
            "depth_penalty": self.depth_penalty,
            "coverage_penalty": self.coverage_penalty,
            "min_improvement": self.min_improvement,
        }.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} 必须是有限非负数")
        if not self.objective.startswith("selection_"):
            raise ValueError("objective 必须是 selection 指标，禁止使用 holdout 进化")
        if not self.operator_parameters:
            raise ValueError("operator_parameters 不能为空")
        for operator, parameters in self.operator_parameters.items():
            definition = get_operator(operator)
            if not definition.causal:
                raise ValueError(f"遗传搜索拒绝非因果算子 {operator!r}")
            for parameter, values in parameters.items():
                if (
                    not isinstance(values, Sequence)
                    or isinstance(values, (str, bytes))
                    or not values
                ):
                    raise ValueError(
                        f"算子 {operator!r} 的参数 {parameter!r} 必须是非空取值序列"
                    )


@dataclass(frozen=True)
class GeneticSearchResult(FactorSearchResult):
    """在通用因子搜索结果之外保存逐代进化摘要。"""

    history: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass(frozen=True)
class GeneticProgressEvent:
    """描述一个候选批次完成后的遗传搜索进度快照。"""

    stage: str
    generation: int | None
    max_generations: int
    completed: int
    total: int
    successful: int
    failed: int
    selection_evaluations: int
    max_evaluations: int
    elapsed_seconds: float
    eta_seconds: float


GeneticProgressCallback = Callable[[GeneticProgressEvent], None]
"""遗传搜索进度回调，在主进程完成一个候选批次后触发。"""


@dataclass(frozen=True)
class _Provenance:
    """记录候选首次出现时的代次、生成方式和父代。"""

    generation: int
    operation: str
    parent_ids: tuple[str, ...] = ()


class _BackendSessionAdapter:
    """把只实现一次性 run 的旧执行后端适配为遗传搜索会话。"""

    def __init__(
        self,
        backend: ExecutionBackend,
        context: SearchContext,
        evaluator: CandidateEvaluator,
        batch_size: int,
    ) -> None:
        """保存每代调用旧执行后端所需的固定参数。

        参数：
            backend: 仅提供批量 ``run`` 接口的自定义执行后端。
            context: 全部代次共享的只读搜索上下文。
            evaluator: 把候选值转换为 selection 指标的评价器。
            batch_size: 每次调用后端时使用的候选批次大小。
        """

        self._backend = backend
        self._context = context
        self._evaluator = evaluator
        self._batch_size = batch_size

    def __enter__(self) -> _BackendSessionAdapter:
        """进入无额外资源的兼容会话并返回自身。"""

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """退出兼容会话；旧后端没有需要持续持有的资源。

        参数：
            exc_type: 会话内异常的类型；正常退出时为空。
            exc_value: 会话内异常对象；正常退出时为空。
            traceback: 会话内异常的回溯；正常退出时为空。
        """

    def run(
        self, candidates: Sequence[FactorCandidate]
    ) -> list[CandidateTaskResult]:
        """通过旧后端的一次性接口评价当前代新增候选。

        参数：
            candidates: 当前代尚未出现在跨代缓存中的候选。
        """

        return self._backend.run(
            candidates,
            self._context,
            self._evaluator,
            self._batch_size,
        )

    def run_with_progress(
        self,
        candidates: Sequence[FactorCandidate],
        progress_callback: BatchProgressCallback | None,
    ) -> list[CandidateTaskResult]:
        """通过旧后端评价候选，并在整批完成后至少报告一次进度。

        参数：
            candidates: 当前代尚未出现在跨代缓存中的候选。
            progress_callback: 可选批次回调；旧后端只能在全部完成后调用一次。
        """

        results = self.run(candidates)
        if progress_callback is not None and candidates:
            progress_callback(
                len(results),
                len(candidates),
                sum(result.error is not None for result in results),
            )
        return results


class _ProgressReporter:
    """把执行会话的批次计数转换为带阶段和 ETA 的公开进度事件。"""

    def __init__(
        self,
        callback: GeneticProgressCallback,
        *,
        stage: str,
        generation: int | None,
        config: GeneticSearchConfig,
        selection_base: int,
    ) -> None:
        """保存当前评价阶段生成进度事件所需的固定上下文。

        参数：
            callback: 接收公开进度事件的调用方回调。
            stage: 当前阶段名称，只使用 selection、holdout 或 model。
            generation: 当前 selection 的一基代次；后置阶段为空。
            config: 提供总代数与 selection 候选预算的遗传配置。
            selection_base: 本批开始前已经完成的 selection 候选数量。
        """

        self._callback = callback
        self._stage = stage
        self._generation = generation
        self._config = config
        self._selection_base = selection_base
        self._started = perf_counter()

    def __call__(self, completed: int, total: int, failed: int) -> None:
        """根据当前批次累计数计算速率和阶段剩余时间并触发回调。

        参数：
            completed: 当前阶段累计完成的候选数量。
            total: 当前阶段计划评价的候选总数。
            failed: 当前阶段累计失败的候选数量。
        """

        elapsed = perf_counter() - self._started
        eta = (
            elapsed * max(0, total - completed) / completed
            if completed > 0
            else float("nan")
        )
        selection_evaluations = self._selection_base
        if self._stage == "selection":
            selection_evaluations += completed
        self._callback(
            GeneticProgressEvent(
                stage=self._stage,
                generation=self._generation,
                max_generations=self._config.max_generations,
                completed=completed,
                total=total,
                successful=completed - failed,
                failed=failed,
                selection_evaluations=selection_evaluations,
                max_evaluations=self._config.max_evaluations,
                elapsed_seconds=elapsed,
                eta_seconds=eta,
            )
        )


def _run_session_with_progress(
    session: ExecutionSession,
    candidates: Sequence[FactorCandidate],
    progress_callback: BatchProgressCallback | None,
) -> list[CandidateTaskResult]:
    """优先使用细粒度会话接口，并兼容只实现旧 ``run`` 的自定义会话。

    参数：
        session: 已进入且负责实际候选评价的执行会话。
        candidates: 当前阶段按稳定顺序排列的候选。
        progress_callback: 可选批次回调；旧会话在全部完成后补报一次。

    返回：
        与候选输入顺序一致的评价结果。
    """

    run_with_progress = getattr(session, "run_with_progress", None)
    if callable(run_with_progress):
        return run_with_progress(candidates, progress_callback)
    results = session.run(candidates)
    if progress_callback is not None and candidates:
        progress_callback(
            len(results),
            len(candidates),
            sum(result.error is not None for result in results),
        )
    return results


def _parameter_combinations(
    parameters: Mapping[str, Sequence[object]],
) -> list[dict[str, object]]:
    """按参数名稳定顺序展开一个算子的离散参数笛卡尔积。

    参数：
        parameters: 参数名到允许离散取值序列的映射；空映射表示无参数算子。
    """

    names = sorted(parameters)
    if not names:
        return [{}]
    return [
        dict(zip(names, values))
        for values in product(*(tuple(parameters[name]) for name in names))
    ]


def _tree_paths(node: ExpressionNode, prefix: tuple[int, ...] = ()) -> list[tuple[int, ...]]:
    """按先序遍历返回表达式树中全部节点路径。

    参数：
        node: 要遍历的表达式根节点。
        prefix: 当前节点相对最初根节点的输入下标路径；缺省表示根节点。
    """

    paths = [prefix]
    for index, child in enumerate(node.inputs):
        paths.extend(_tree_paths(child, (*prefix, index)))
    return paths


def _tree_operators(node: ExpressionNode) -> frozenset[str]:
    """返回表达式树实际使用的全部非终端算子名。

    参数：
        node: 要统计算子覆盖情况的表达式根节点。

    返回：
        不含 ``column`` 和 ``constant`` 终端的算子名集合。
    """

    operators = set().union(*(_tree_operators(child) for child in node.inputs))
    if node.operator not in {"column", "constant"}:
        operators.add(node.operator)
    return frozenset(operators)


def _node_at(node: ExpressionNode, path: tuple[int, ...]) -> ExpressionNode:
    """按输入下标路径返回表达式子树。

    参数：
        node: 路径起点的表达式根节点。
        path: 从根节点逐层进入 inputs 的下标序列。
    """

    current = node
    for index in path:
        current = current.inputs[index]
    return current


def _replace_subtree(
    node: ExpressionNode,
    path: tuple[int, ...],
    replacement: ExpressionNode,
) -> ExpressionNode:
    """把指定路径的子树替换为新节点并返回不可变新树。

    参数：
        node: 待修改的原表达式根节点。
        path: 要替换节点相对根节点的输入下标路径。
        replacement: 写入指定位置的合法表达式子树。
    """

    if not path:
        return replacement
    index = path[0]
    children = list(node.inputs)
    children[index] = _replace_subtree(children[index], path[1:], replacement)
    return operation_node(node.operator, tuple(children), node.parameter_map)


class _ExpressionGenerator:
    """根据算子白名单生成、变异、交叉并规范化表达式树。"""

    def __init__(self, config: GeneticSearchConfig) -> None:
        """预展开算子参数并保存复杂度约束。

        参数：
            config: 数据源、算子参数空间和表达式复杂度限制。
        """

        self._config = config
        self._sources = tuple(ExpressionNode.column(name) for name in config.sources)
        self._calls = {
            operator: tuple(_parameter_combinations(parameters))
            for operator, parameters in sorted(config.operator_parameters.items())
        }
        # 用虚拟输入提前触发所有参数标准化，配置错误应在搜索开始前暴露。
        dummy = ExpressionNode.column(config.sources[0])
        for operator, calls in self._calls.items():
            arity = get_operator(operator).arity
            for parameters in calls:
                operation_node(operator, (dummy,) * arity, parameters)

    def _build(
        self,
        operator: str,
        inputs: tuple[ExpressionNode, ...],
        parameters: Mapping[str, object],
    ) -> ExpressionNode:
        """应用交换律规范和同输入限制后构造算子节点。

        参数：
            operator: 已启用且经过注册表校验的算子名。
            inputs: 按算子输入数量生成的子表达式。
            parameters: 从离散白名单选择的算子参数。
        """

        children = inputs
        if operator in self._config.commutative_operators:
            children = tuple(sorted(children, key=lambda child: child.canonical))
        if (
            operator in self._config.disallow_same_input_operators
            and len({child.canonical for child in children}) != len(children)
        ):
            raise ValueError(f"算子 {operator!r} 不允许使用相同输入")
        return operation_node(operator, children, parameters)

    def normalize(self, node: ExpressionNode) -> ExpressionNode:
        """递归恢复交换律算子的输入顺序并重新执行节点校验。

        参数：
            node: 可能由子树替换产生且需要重新规范化的表达式。
        """

        if not node.inputs:
            return node
        children = tuple(self.normalize(child) for child in node.inputs)
        return self._build(node.operator, children, node.parameter_map)

    def valid(self, node: ExpressionNode) -> bool:
        """判断表达式是否满足因果性、深度、节点数和回看长度硬限制。

        参数：
            node: 待检查的候选表达式根节点。
        """

        return (
            node.causal
            and node.depth <= self._config.max_depth
            and node.node_count <= self._config.max_nodes
            and node.lookback <= self._config.max_lookback
        )

    def random_tree(
        self,
        rng: np.random.Generator,
        max_depth: int,
        *,
        force_operator: bool = False,
    ) -> ExpressionNode:
        """递归随机生成不超过给定深度的合法表达式树。

        参数：
            rng: 只由主进程持有的确定性 NumPy 随机数生成器。
            max_depth: 当前子树允许包含的最大算子嵌套深度。
            force_operator: 是否强制根节点使用算子；缺省允许直接返回数据源。
        """

        if max_depth <= 0 or (
            not force_operator and rng.random() < self._config.terminal_probability
        ):
            return self._sources[int(rng.integers(len(self._sources)))]
        for _ in range(50):
            operator = tuple(self._calls)[int(rng.integers(len(self._calls)))]
            definition = get_operator(operator)
            children = tuple(
                self.random_tree(rng, max_depth - 1)
                for _ in range(definition.arity)
            )
            parameters = self._calls[operator][
                int(rng.integers(len(self._calls[operator])))
            ]
            try:
                node = self._build(operator, children, parameters)
            except ValueError:
                continue
            if self.valid(node):
                return node
        return self._sources[int(rng.integers(len(self._sources)))]

    def random_tree_covering_source(
        self,
        source: str,
        rng: np.random.Generator,
    ) -> ExpressionNode:
        """随机生成明确引用指定数据源的合法表达式。

        优先把指定源放入随机根算子的一个输入；如果结构约束或算子空间使包装
        始终失败，则退回源列本身，保证补全候选仍然合法且可执行。

        参数：
            source: 下一代当前缺失且必须被表达式引用的数据源列名。
            rng: 只由主进程持有的确定性 NumPy 随机数生成器。

        返回：
            满足全部复杂度限制并包含指定数据源的表达式。
        """

        if source not in self._config.sources:
            raise ValueError(f"未知遗传搜索数据源 {source!r}")
        terminal = ExpressionNode.column(source)
        for _ in range(50):
            operator = tuple(self._calls)[int(rng.integers(len(self._calls)))]
            definition = get_operator(operator)
            required_index = int(rng.integers(definition.arity))
            children = tuple(
                terminal
                if index == required_index
                else self.random_tree(rng, max(0, self._config.max_depth - 1))
                for index in range(definition.arity)
            )
            parameters = self._calls[operator][
                int(rng.integers(len(self._calls[operator])))
            ]
            try:
                node = self._build(operator, children, parameters)
            except ValueError:
                continue
            if self.valid(node):
                return node
        return terminal

    def simple_child_pool(self) -> tuple[ExpressionNode, ...]:
        """构造终端及其一层算子变换组成的最简单合法子表达式池。

        该池用于单一数据源无法直接满足相关性等“输入不得相同”约束时，提供
        规范字符串不同且复杂度尽可能低的候选输入。每个算子只使用最简单的
        终端输入，避免为覆盖补全递归展开整个遗传搜索空间。

        返回：
            按回看长度、节点数、深度和规范字符串稳定排序的去重表达式元组。
        """

        nodes: dict[str, ExpressionNode] = {
            source.canonical: source for source in self._sources
        }
        for operator, calls in self._calls.items():
            definition = get_operator(operator)
            if (
                operator in self._config.disallow_same_input_operators
                and definition.arity > len(self._sources)
            ):
                continue
            if operator in self._config.disallow_same_input_operators:
                children = self._sources[: definition.arity]
            else:
                children = (self._sources[0],) * definition.arity
            for parameters in calls:
                try:
                    node = self._build(operator, children, parameters)
                except ValueError:
                    continue
                if self.valid(node):
                    nodes.setdefault(node.canonical, node)
        return tuple(
            sorted(
                nodes.values(),
                key=lambda node: (
                    node.lookback,
                    node.node_count,
                    node.depth,
                    node.canonical,
                ),
            )
        )

    def random_tree_with_root(
        self,
        operator: str,
        rng: np.random.Generator,
    ) -> ExpressionNode | None:
        """随机生成以指定缺失算子为根节点的合法表达式。

        参数：
            operator: 下一代当前缺失且必须作为根节点出现的已配置算子名。
            rng: 只由主进程持有的确定性 NumPy 随机数生成器。

        返回：
            找到时返回满足全部复杂度限制的表达式；若当前硬限制使该算子无法
            构成合法候选，则返回 ``None``。
        """

        if operator not in self._calls:
            raise ValueError(f"未知遗传搜索算子 {operator!r}")
        definition = get_operator(operator)

        # 先随机排列全部离散参数，并用终端子树逐一验证。终端结构拥有最小的
        # 深度、节点数和回看长度；只要该算子存在直接可构造的合法形式，就不会
        # 因有限次随机抽样没有命中短窗口而漏掉本轮覆盖。
        parameter_order = rng.permutation(len(self._calls[operator]))
        for parameter_index in parameter_order:
            if (
                operator in self._config.disallow_same_input_operators
                and definition.arity > len(self._sources)
            ):
                break
            if operator in self._config.disallow_same_input_operators:
                source_indices = rng.choice(
                    len(self._sources), size=definition.arity, replace=False
                )
            else:
                source_indices = rng.integers(
                    len(self._sources), size=definition.arity
                )
            children = tuple(
                self._sources[int(index)] for index in source_indices
            )
            try:
                node = self._build(
                    operator,
                    children,
                    self._calls[operator][int(parameter_index)],
                )
            except ValueError:
                continue
            if self.valid(node):
                return node

        if operator in self._config.disallow_same_input_operators:
            child_pool = self.simple_child_pool()
            if len(child_pool) >= definition.arity:
                for parameter_index in parameter_order:
                    for children in combinations(child_pool, definition.arity):
                        try:
                            node = self._build(
                                operator,
                                children,
                                self._calls[operator][int(parameter_index)],
                            )
                        except ValueError:
                            continue
                        if self.valid(node):
                            return node

        # 极少数配置可能只有一个数据源，却允许用不同的嵌套表达式组成二元
        # 算子；终端快速路径无法覆盖这种情况，因此保留有限次递归随机尝试。
        for _ in range(50):
            children = tuple(
                self.random_tree(rng, max(0, self._config.max_depth - 1))
                for _ in range(definition.arity)
            )
            parameters = self._calls[operator][
                int(rng.integers(len(self._calls[operator])))
            ]
            try:
                node = self._build(operator, children, parameters)
            except ValueError:
                continue
            if self.valid(node):
                return node
        return None

    def initial_population(self, rng: np.random.Generator) -> list[ExpressionNode]:
        """生成包含原始数据源和分层随机树的去重初始种群。

        参数：
            rng: 控制全部随机选择且只在主进程使用的生成器。
        """

        nodes: dict[str, ExpressionNode] = {}
        for source in self._sources:
            nodes.setdefault(source.canonical, source)
            if len(nodes) >= self._config.population_size:
                return list(nodes.values())
        attempts = 0
        limit = self._config.population_size * 100
        while len(nodes) < self._config.population_size and attempts < limit:
            depth = int(rng.integers(self._config.initial_max_depth + 1))
            node = self.random_tree(rng, depth, force_operator=depth > 0)
            if self.valid(node):
                nodes.setdefault(node.canonical, node)
            attempts += 1
        return list(nodes.values())

    def crossover(
        self,
        left: ExpressionNode,
        right: ExpressionNode,
        rng: np.random.Generator,
    ) -> ExpressionNode:
        """把右父代的随机子树替换到左父代随机位置。

        参数：
            left: 提供主体结构的左父代表达式。
            right: 提供替换子树的右父代表达式。
            rng: 控制子树路径选择的主进程随机数生成器。
        """

        for _ in range(30):
            left_paths = _tree_paths(left)
            right_paths = _tree_paths(right)
            left_path = left_paths[int(rng.integers(len(left_paths)))]
            right_path = right_paths[int(rng.integers(len(right_paths)))]
            try:
                child = self.normalize(
                    _replace_subtree(left, left_path, _node_at(right, right_path))
                )
            except ValueError:
                continue
            if self.valid(child):
                return child
        return left

    def subtree_mutation(
        self, node: ExpressionNode, rng: np.random.Generator
    ) -> ExpressionNode:
        """用随机生成的新子树替换候选中的随机节点。

        参数：
            node: 提供主体结构的父代表达式。
            rng: 控制替换位置和新子树的主进程随机数生成器。
        """

        paths = _tree_paths(node)
        for _ in range(30):
            path = paths[int(rng.integers(len(paths)))]
            remaining_depth = max(0, self._config.max_depth - len(path))
            replacement_depth = int(rng.integers(remaining_depth + 1))
            replacement = self.random_tree(
                rng, replacement_depth, force_operator=replacement_depth > 0
            )
            try:
                child = self.normalize(_replace_subtree(node, path, replacement))
            except ValueError:
                continue
            if self.valid(child):
                return child
        return node

    def parameter_mutation(
        self, node: ExpressionNode, rng: np.random.Generator
    ) -> ExpressionNode:
        """随机替换表达式中一个算子节点的离散参数组合。

        参数：
            node: 参数将被修改的父代表达式。
            rng: 控制节点和参数选择的主进程随机数生成器。
        """

        paths = [
            path
            for path in _tree_paths(node)
            if _node_at(node, path).operator in self._calls
            and len(self._calls[_node_at(node, path).operator]) > 1
        ]
        if not paths:
            return node
        for _ in range(30):
            path = paths[int(rng.integers(len(paths)))]
            target = _node_at(node, path)
            choices = [
                parameters
                for parameters in self._calls[target.operator]
                if operation_node(
                    target.operator, target.inputs, parameters
                ).parameter_map
                != target.parameter_map
            ]
            if not choices:
                continue
            parameters = choices[int(rng.integers(len(choices)))]
            replacement = self._build(target.operator, target.inputs, parameters)
            try:
                child = self.normalize(_replace_subtree(node, path, replacement))
            except ValueError:
                continue
            if self.valid(child):
                return child
        return node


@dataclass(frozen=True)
class FactorGeneticSearch:
    """以主进程确定性进化、执行后端并行评价候选的遗传因子搜索器。"""

    config: GeneticSearchConfig = field(default_factory=GeneticSearchConfig)
    backend: str | ExecutionBackend = "sequential"
    n_jobs: int = -1
    batch_size: int = 16

    def __post_init__(self) -> None:
        """校验执行后端名称、worker 数和候选批次大小。"""

        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size <= 0
        ):
            raise ValueError("batch_size 必须是正整数")
        if (
            isinstance(self.n_jobs, bool)
            or not isinstance(self.n_jobs, int)
            or self.n_jobs == 0
            or self.n_jobs < -1
        ):
            raise ValueError("n_jobs 必须是 -1 或正整数")
        if isinstance(self.backend, str) and self.backend not in {"sequential", "process"}:
            raise ValueError("backend 必须是 'sequential'、'process' 或 ExecutionBackend")

    def _backend(self) -> SequentialBackend | ProcessBackend | ExecutionBackend:
        """把后端名称解析为可创建持久化执行会话的后端对象。"""

        if not isinstance(self.backend, str):
            return self.backend
        if self.backend == "sequential":
            return SequentialBackend()
        return ProcessBackend(self.n_jobs)

    def _open_session(
        self,
        backend: SequentialBackend | ProcessBackend | ExecutionBackend,
        context: SearchContext,
        evaluator: CandidateEvaluator,
    ) -> ExecutionSession:
        """优先创建持久化会话，并兼容仅实现旧批量接口的自定义后端。

        参数：
            backend: 已解析的内置或调用方注入执行后端。
            context: 全部遗传代次共享的只读搜索上下文。
            evaluator: 把候选值转换为 selection 或 holdout 指标的评价器。
        """

        open_session = getattr(backend, "open_session", None)
        if callable(open_session):
            return open_session(context, evaluator, self.batch_size)
        return _BackendSessionAdapter(backend, context, evaluator, self.batch_size)

    def _fitness_row(
        self,
        candidate: FactorCandidate,
        result: CandidateTaskResult,
        provenance: _Provenance,
    ) -> dict[str, object]:
        """把原始 selection 指标、复杂度惩罚和来源合并为排行榜行。

        参数：
            candidate: 当前已完成规范化的因子表达式。
            result: worker 返回的原始 selection 指标或隔离错误。
            provenance: 候选首次出现时的代次、生成方式和父代。
        """

        identity: dict[str, object] = {
            "factor_id": candidate.factor_id,
            "expression": candidate.canonical,
            "expression_str": candidate.expression_str,
            "node_count": candidate.node_count,
            "depth": candidate.depth,
            "lookback": candidate.lookback,
            "first_generation": provenance.generation,
            "genetic_operation": provenance.operation,
            "parent_ids": ",".join(provenance.parent_ids),
            "elapsed_seconds": result.elapsed_seconds,
        }
        if result.error is not None:
            return {
                **identity,
                "eligible": False,
                "fitness": float("-inf"),
                "error": result.error,
            }
        metrics: dict[str, object] = dict(result.metrics)
        objective = float(metrics.get(self.config.objective, float("nan")))
        rank_std = float(metrics.get("selection_rank_ic_std", float("nan")))
        rank_dates = float(metrics.get("selection_rank_ic_dates", 0.0))
        coverage = float(metrics.get("selection_coverage", float("nan")))
        standard_error = 0.0 if self.config.stability_weight == 0 else (
            rank_std / math.sqrt(rank_dates)
            if math.isfinite(rank_std) and rank_dates > 0
            else float("inf")
        )
        stability_penalty = self.config.stability_weight * standard_error
        length_penalty = self.config.length_penalty * max(
            0, candidate.node_count - self.config.free_node_count
        )
        depth_penalty = self.config.depth_penalty * max(
            0, candidate.depth - self.config.free_depth
        )
        coverage_penalty = (
            self.config.coverage_penalty
            * max(0.0, self.config.target_coverage - coverage)
            if math.isfinite(coverage)
            else float("inf")
        )
        eligible = (
            math.isfinite(objective)
            and math.isfinite(coverage)
            and coverage >= self.config.min_coverage
            and math.isfinite(standard_error)
        )
        fitness = (
            objective
            - stability_penalty
            - length_penalty
            - depth_penalty
            - coverage_penalty
            if eligible
            else float("-inf")
        )
        return {
            **identity,
            **metrics,
            "rank_ic_standard_error": standard_error,
            "stability_penalty": stability_penalty,
            "length_penalty": length_penalty,
            "depth_penalty": depth_penalty,
            "coverage_penalty": coverage_penalty,
            "fitness": fitness,
            "eligible": eligible,
        }

    @staticmethod
    def _sort_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
        """按适应度、长度、深度和稳定 ID 对排行榜执行确定性排序。

        参数：
            rows: 已计算适应度且包含候选身份的排行榜记录。
        """

        return sorted(
            rows,
            key=lambda row: (
                not bool(row["eligible"]),
                -float(row["fitness"]),
                int(row["node_count"]),
                int(row["depth"]),
                str(row["factor_id"]),
            ),
        )

    @staticmethod
    def _deduplicate_equivalent_rows(
        rows: Sequence[dict[str, object]],
    ) -> list[dict[str, object]]:
        """按 selection 方向值或逐日秩指纹保留最简单候选。

        参数：
            rows: 已计算适应度的当前代或全量候选排行榜行；没有等价指纹的
                自定义评价结果保持原样。

        返回：
            每个等价类只保留节点数最少、再按深度和适应度稳定决胜的排行榜行，
            最终仍按正式适应度顺序排列。
        """

        simplest_first = sorted(
            rows,
            key=lambda row: (
                int(row["node_count"]),
                int(row["depth"]),
                not bool(row["eligible"]),
                -float(row["fitness"]),
                str(row["factor_id"]),
            ),
        )
        retained: list[dict[str, object]] = []
        seen_values: set[str] = set()
        seen_ranks: set[str] = set()
        for row in simplest_first:
            value_key = str(
                row.get("selection_oriented_value_fingerprint", "")
            )
            rank_key = str(
                row.get("selection_oriented_rank_fingerprint", "")
            )
            if (value_key and value_key in seen_values) or (
                rank_key and rank_key in seen_ranks
            ):
                continue
            retained.append(row)
            if value_key:
                seen_values.add(value_key)
            if rank_key:
                seen_ranks.add(rank_key)
        return FactorGeneticSearch._sort_rows(retained)

    def _tournament(
        self,
        ranked: Sequence[FactorCandidate],
        rng: np.random.Generator,
    ) -> FactorCandidate:
        """从随机抽取的合格候选中返回排名最靠前的父代。

        参数：
            ranked: 已按适应度稳定排序的合格候选序列。
            rng: 只由主进程使用的确定性随机数生成器。
        """

        size = min(self.config.tournament_size, len(ranked))
        indices = rng.choice(len(ranked), size=size, replace=False)
        return ranked[min(int(index) for index in indices)]

    def _inject_missing_coverage(
        self,
        selected: dict[str, FactorCandidate],
        provenance: dict[str, _Provenance],
        generator: _ExpressionGenerator,
        rng: np.random.Generator,
        generation: int,
    ) -> None:
        """向下一代优先注入缺失数据源和缺失算子的随机候选。

        数据源和算子分别按主进程随机顺序补全，并在每次注入后重新累计覆盖，
        因而一个随机表达式可以同时补足多个缺失项。若种群容量或表达式硬限制
        不足，则保持已获得的最大覆盖，不突破既有规模与复杂度约束。

        参数：
            selected: 已包含本轮精英、并将被原地补充的下一代候选映射。
            provenance: 与 ``selected`` 同步记录首次生成方式的来源映射。
            generator: 负责生成并校验覆盖候选的表达式生成器。
            rng: 控制缺失项顺序和表达式结构的主进程随机数生成器。
            generation: 正在构造的下一代零基代次编号。

        返回：
            无；通过原地修改 ``selected`` 和 ``provenance`` 返回补全结果。
        """

        covered_sources = set().union(
            *(candidate.expression.columns for candidate in selected.values())
        )
        covered_operators = set().union(
            *(_tree_operators(candidate.expression) for candidate in selected.values())
        )

        source_order = list(self.config.sources)
        rng.shuffle(source_order)
        for source in source_order:
            if len(selected) >= self.config.population_size:
                return
            if source in covered_sources:
                continue
            node = generator.random_tree_covering_source(source, rng)
            candidate = FactorCandidate(node)
            if candidate.factor_id in selected:
                continue
            selected[candidate.factor_id] = candidate
            provenance[candidate.factor_id] = _Provenance(
                generation, "coverage_source"
            )
            covered_sources.update(node.columns)
            covered_operators.update(_tree_operators(node))

        operator_order = list(self.config.operator_parameters)
        rng.shuffle(operator_order)
        for operator in operator_order:
            if len(selected) >= self.config.population_size:
                return
            if operator in covered_operators:
                continue
            node = generator.random_tree_with_root(operator, rng)
            if node is None:
                continue
            candidate = FactorCandidate(node)
            if candidate.factor_id in selected:
                continue
            selected[candidate.factor_id] = candidate
            provenance[candidate.factor_id] = _Provenance(
                generation, "coverage_operator"
            )
            covered_sources.update(node.columns)
            covered_operators.update(_tree_operators(node))

    def _next_population(
        self,
        ranked: Sequence[FactorCandidate],
        generator: _ExpressionGenerator,
        rng: np.random.Generator,
        generation: int,
    ) -> tuple[list[FactorCandidate], dict[str, _Provenance]]:
        """执行精英保留、锦标赛选择、交叉、变异和随机移民。

        参数：
            ranked: 当前代按适应度排序的合格候选。
            generator: 负责构造和校验表达式树的生成器。
            rng: 控制全部遗传选择且只在主进程使用的随机数生成器。
            generation: 即将生成的新种群代次编号。
        """

        elite_count = max(1, math.ceil(len(ranked) * self.config.elite_ratio))
        selected: dict[str, FactorCandidate] = {
            candidate.factor_id: candidate for candidate in ranked[:elite_count]
        }
        provenance: dict[str, _Provenance] = {
            candidate.factor_id: _Provenance(
                generation, "elite", (candidate.factor_id,)
            )
            for candidate in ranked[:elite_count]
        }
        self._inject_missing_coverage(
            selected,
            provenance,
            generator,
            rng,
            generation,
        )
        attempts = 0
        limit = self.config.population_size * 100
        boundaries = np.cumsum(
            [
                self.config.crossover_probability,
                self.config.subtree_mutation_probability,
                self.config.parameter_mutation_probability,
                self.config.reproduction_probability,
            ]
        )
        while len(selected) < self.config.population_size and attempts < limit:
            parent = self._tournament(ranked, rng)
            draw = rng.random()
            operation = "reproduction"
            parent_ids = (parent.factor_id,)
            node = parent.expression
            if draw < boundaries[0]:
                other = self._tournament(ranked, rng)
                node = generator.crossover(parent.expression, other.expression, rng)
                operation = "crossover"
                parent_ids = (parent.factor_id, other.factor_id)
            elif draw < boundaries[1]:
                node = generator.subtree_mutation(parent.expression, rng)
                operation = "subtree_mutation"
            elif draw < boundaries[2]:
                node = generator.parameter_mutation(parent.expression, rng)
                operation = "parameter_mutation"
            candidate = FactorCandidate(node)
            if generator.valid(node) and candidate.factor_id not in selected:
                selected[candidate.factor_id] = candidate
                provenance[candidate.factor_id] = _Provenance(
                    generation, operation, parent_ids
                )
            attempts += 1

        # 搜索空间较小时，随机移民比重复个体更能维持有效种群多样性。
        immigrant_attempts = 0
        while (
            len(selected) < self.config.population_size
            and immigrant_attempts < limit
        ):
            depth = int(rng.integers(self.config.max_depth + 1))
            node = generator.random_tree(rng, depth, force_operator=depth > 0)
            candidate = FactorCandidate(node)
            if generator.valid(node) and candidate.factor_id not in selected:
                selected[candidate.factor_id] = candidate
                provenance[candidate.factor_id] = _Provenance(generation, "immigrant")
            immigrant_attempts += 1
        return list(selected.values()), provenance

    def run(
        self,
        context: SearchContext,
        *,
        evaluator: CandidateEvaluator | None = None,
        holdout_evaluator: CandidateEvaluator | None = None,
        holdout_top_k: int = 1,
        model_evaluator: CandidateEvaluator | None = None,
        model_top_k: int = 0,
        progress_callback: GeneticProgressCallback | None = None,
    ) -> GeneticSearchResult:
        """并行进化，并在结束后评价 selection 入选候选的 holdout 和模型指标。

        参数：
            context: 一次性准备的日频行情、目标和严格隔离的日期掩码。
            evaluator: 每代候选的 selection 评价器；缺省使用横截面 IC。
            holdout_evaluator: 最终候选的 holdout 评价器；缺省使用横截面 IC。
            holdout_top_k: 搜索结束后披露 holdout 指标的 selection 前 K 名数量。
            model_evaluator: 可选的现有模型实验评价器；为空时跳过模型复验。
            model_top_k: 搜索结束后进行模型复验的 selection 前 K 名数量。
            progress_callback: 可选主进程进度回调，每完成一个候选批次触发一次。
        """

        for name, value in {
            "holdout_top_k": holdout_top_k,
            "model_top_k": model_top_k,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} 必须是非负整数")
        if progress_callback is not None and not callable(progress_callback):
            raise TypeError("progress_callback 必须可调用或为 None")
        missing_sources = set(self.config.sources).difference(context.daily.columns)
        if missing_sources:
            raise ValueError(f"遗传搜索日频数据缺少源列: {sorted(missing_sources)}")
        generator = _ExpressionGenerator(self.config)
        rng = np.random.default_rng(self.config.random_seed)
        initial_nodes = generator.initial_population(rng)
        population = [FactorCandidate(node) for node in initial_nodes]
        provenance: dict[str, _Provenance] = {
            candidate.factor_id: _Provenance(0, "initial")
            for candidate in population
        }
        candidates: dict[str, FactorCandidate] = {}
        cache: dict[str, CandidateTaskResult] = {}
        errors: dict[tuple[str, str], dict[str, object]] = {}
        history: list[dict[str, object]] = []
        best_fitness = float("-inf")
        stale_generations = 0
        backend = self._backend()
        selection_evaluator = evaluator or IcEvaluator()
        if isinstance(selection_evaluator, IcEvaluator):
            selection_evaluator = replace(
                selection_evaluator,
                include_equivalence_fingerprints=True,
            )
        with self._open_session(
            backend, context, selection_evaluator
        ) as session:
            for generation in range(self.config.max_generations):
                remaining = self.config.max_evaluations - len(cache)
                new_candidates = [
                    candidate
                    for candidate in population
                    if candidate.factor_id not in cache
                ][: max(0, remaining)]
                if new_candidates:
                    for candidate in new_candidates:
                        candidates[candidate.factor_id] = candidate
                    reporter = (
                        _ProgressReporter(
                            progress_callback,
                            stage="selection",
                            generation=generation + 1,
                            config=self.config,
                            selection_base=len(cache),
                        )
                        if progress_callback is not None
                        else None
                    )
                    for result in _run_session_with_progress(
                        session, new_candidates, reporter
                    ):
                        cache[result.factor_id] = result
                population = [
                    candidate
                    for candidate in population
                    if candidate.factor_id in cache
                ]
                rows = [
                    self._fitness_row(
                        candidate,
                        cache[candidate.factor_id],
                        provenance[candidate.factor_id],
                    )
                    for candidate in population
                ]
                ranked_rows = self._deduplicate_equivalent_rows(rows)
                for row in ranked_rows:
                    if "error" in row:
                        factor_id = str(row["factor_id"])
                        errors[(factor_id, "screening")] = {
                            **row,
                            "stage": "screening",
                        }
                eligible_ids = [
                    str(row["factor_id"])
                    for row in ranked_rows
                    if bool(row["eligible"])
                ]
                generation_best = (
                    float(ranked_rows[0]["fitness"])
                    if ranked_rows and bool(ranked_rows[0]["eligible"])
                    else float("-inf")
                )
                history.append(
                    {
                        "generation": generation,
                        "population": len(population),
                        "new_evaluations": len(new_candidates),
                        "total_evaluations": len(cache),
                        "eligible": len(eligible_ids),
                        "best_fitness": generation_best,
                        "best_factor_id": eligible_ids[0] if eligible_ids else None,
                    }
                )
                if generation_best > best_fitness + self.config.min_improvement:
                    best_fitness = generation_best
                    stale_generations = 0
                else:
                    stale_generations += 1
                if (
                    not eligible_ids
                    or len(cache) >= self.config.max_evaluations
                    or stale_generations >= self.config.patience
                    or generation + 1 >= self.config.max_generations
                ):
                    break
                ranked_candidates = [candidates.get(factor_id) for factor_id in eligible_ids]
                # 当前代新产生且合格的候选均已进入 candidates；精英旧候选同样存在。
                parents = [candidate for candidate in ranked_candidates if candidate is not None]
                population, next_provenance = self._next_population(
                    parents, generator, rng, generation + 1
                )
                for factor_id, item in next_provenance.items():
                    provenance.setdefault(factor_id, item)

        all_rows = [
            self._fitness_row(candidate, cache[factor_id], provenance[factor_id])
            for factor_id, candidate in candidates.items()
            if factor_id in cache and cache[factor_id].error is None
        ]
        final_rows = self._deduplicate_equivalent_rows(all_rows)
        leaderboard = pd.DataFrame(final_rows)
        error_frame = pd.DataFrame(list(errors.values()))

        if (
            holdout_top_k > 0
            and context.holdout_mask.any()
            and not leaderboard.empty
        ):
            holdout_ids = leaderboard.loc[
                leaderboard["eligible"], "factor_id"
            ].head(holdout_top_k)
            holdout_candidates = [
                candidates[str(factor_id)] for factor_id in holdout_ids
            ]
            with self._open_session(
                backend, context, holdout_evaluator or HoldoutIcEvaluator()
            ) as holdout_session:
                holdout_reporter = (
                    _ProgressReporter(
                        progress_callback,
                        stage="holdout",
                        generation=None,
                        config=self.config,
                        selection_base=len(cache),
                    )
                    if progress_callback is not None
                    else None
                )
                holdout_results = _run_session_with_progress(
                    holdout_session, holdout_candidates, holdout_reporter
                )
            holdout_rows: list[dict[str, object]] = []
            for result in holdout_results:
                if result.error is None:
                    holdout_rows.append(
                        {
                            "factor_id": result.factor_id,
                            "holdout_elapsed_seconds": result.elapsed_seconds,
                            **result.metrics,
                        }
                    )
                else:
                    errors[(result.factor_id, "holdout")] = {
                        "factor_id": result.factor_id,
                        "stage": "holdout",
                        "error": result.error,
                        "elapsed_seconds": result.elapsed_seconds,
                    }
            if holdout_rows:
                leaderboard = leaderboard.merge(
                    pd.DataFrame(holdout_rows), on="factor_id", how="left"
                )
                if "direction" in leaderboard and "holdout_rank_ic" in leaderboard:
                    leaderboard["holdout_oriented_rank_ic"] = (
                        leaderboard["holdout_rank_ic"] * leaderboard["direction"]
                    )
                if "direction" in leaderboard and "holdout_ic" in leaderboard:
                    leaderboard["holdout_oriented_ic"] = (
                        leaderboard["holdout_ic"] * leaderboard["direction"]
                    )
            error_frame = pd.DataFrame(list(errors.values()))

        if model_evaluator is not None and model_top_k > 0 and not leaderboard.empty:
            model_ids = leaderboard.loc[
                leaderboard["eligible"], "factor_id"
            ].head(model_top_k)
            model_candidates = [
                candidates[str(factor_id)] for factor_id in model_ids
            ]
            with self._open_session(
                backend, context, model_evaluator
            ) as model_session:
                model_reporter = (
                    _ProgressReporter(
                        progress_callback,
                        stage="model",
                        generation=None,
                        config=self.config,
                        selection_base=len(cache),
                    )
                    if progress_callback is not None
                    else None
                )
                model_results = _run_session_with_progress(
                    model_session, model_candidates, model_reporter
                )
            model_rows: list[dict[str, object]] = []
            for result in model_results:
                if result.error is None:
                    model_rows.append(
                        {
                            "factor_id": result.factor_id,
                            "model_elapsed_seconds": result.elapsed_seconds,
                            **result.metrics,
                        }
                    )
                else:
                    errors[(result.factor_id, "model")] = {
                        "factor_id": result.factor_id,
                        "stage": "model",
                        "error": result.error,
                        "elapsed_seconds": result.elapsed_seconds,
                    }
            if model_rows:
                leaderboard = leaderboard.merge(
                    pd.DataFrame(model_rows), on="factor_id", how="left"
                )
            error_frame = pd.DataFrame(list(errors.values()))

        return GeneticSearchResult(
            candidates=tuple(candidates.values()),
            leaderboard=leaderboard.reset_index(drop=True),
            errors=error_frame.reset_index(drop=True),
            objective="fitness",
            history=pd.DataFrame(history),
        )
