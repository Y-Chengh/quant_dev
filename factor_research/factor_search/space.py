"""因子候选及网格搜索空间定义。"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Mapping, Protocol, Sequence

from factor_research.factor_dsl import (
    ExpressionNode,
    FactorExpression,
    SymbolicDailyFrame,
    get_operator,
    operation_node,
)


@dataclass(frozen=True)
class FactorCandidate:
    """一个已经完成规范化和去重的搜索候选。"""

    expression: ExpressionNode

    @property
    def factor_id(self) -> str:
        """返回底层表达式生成的稳定短因子 ID。"""

        return self.expression.factor_id

    @property
    def canonical(self) -> str:
        """返回可用于日志、去重和复现的规范表达式文本。"""

        return self.expression.canonical

    @property
    def depth(self) -> int:
        """返回底层表达式的最大算子嵌套深度。"""

        return self.expression.depth

    @property
    def lookback(self) -> int:
        """返回底层表达式需要的最大额外历史行数。"""

        return self.expression.lookback


class SearchSpace(Protocol):
    """搜索空间只负责生成表达式，不接触任何行情数据。"""

    def generate(self) -> list[FactorCandidate]:
        """按确定顺序返回已经去重的候选。"""

    def estimate_size(self) -> int:
        """返回去重前的候选数量上界，供执行前阻止组合爆炸。"""


@dataclass(frozen=True)
class OperatorGrid:
    """一个流水线阶段中的算子及其参数取值。"""

    name: str | None
    parameters: tuple[tuple[str, tuple[object, ...]], ...] = ()

    def expand(self) -> list[tuple[str | None, dict[str, object]]]:
        """展开参数笛卡尔积，并校验该阶段适用于单输入流水线。"""

        if self.name is None:
            if self.parameters:
                raise ValueError("identity 阶段不能配置参数")
            return [(None, {})]
        definition = get_operator(self.name)
        if definition.arity != 1:
            raise ValueError(
                f"PipelineGrid 只接受单输入算子，{self.name!r} 的输入数为 "
                f"{definition.arity}；多输入公式请使用 ExpressionGrid"
            )
        if not self.parameters:
            # 即使没有参数也走 operation_node，确保不接受参数的算子得到校验。
            return [(self.name, {})]
        names = [name for name, values in self.parameters]
        choices = [values for name, values in self.parameters]
        if any(not values for values in choices):
            raise ValueError(f"算子 {self.name!r} 的参数网格不能包含空取值")
        return [
            (self.name, dict(zip(names, values)))
            for values in product(*choices)
        ]


def op(name: str, **parameters: Sequence[object] | object) -> OperatorGrid:
    """构造算子网格；标量会自动转换为只有一个取值的网格。

    参数：
        name: 每组参数将调用的已注册单输入算子名。
        **parameters: 参数名到候选取值序列或单一取值的映射。
    """

    normalized: list[tuple[str, tuple[object, ...]]] = []
    for parameter_name, values in sorted(parameters.items()):
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            choices = tuple(values)
        else:
            choices = (values,)
        normalized.append((parameter_name, choices))
    return OperatorGrid(name=name, parameters=tuple(normalized))


def identity() -> OperatorGrid:
    """返回不改变当前表达式的可选流水线阶段。"""

    return OperatorGrid(name=None)


def _deduplicate(nodes: Sequence[ExpressionNode]) -> list[FactorCandidate]:
    """按规范表达式去重，并额外防御极低概率的短哈希碰撞。

    参数：
        nodes: 待去重的表达式节点，首次出现顺序会被保留。
    """

    by_canonical: dict[str, FactorCandidate] = {}
    ids: dict[str, str] = {}
    for node in nodes:
        canonical = node.canonical
        candidate = FactorCandidate(node)
        existing = ids.get(candidate.factor_id)
        if existing is not None and existing != canonical:
            raise RuntimeError(
                f"因子短哈希发生碰撞: {candidate.factor_id} 对应多个表达式"
            )
        ids[candidate.factor_id] = canonical
        by_canonical.setdefault(canonical, candidate)
    return list(by_canonical.values())


@dataclass(frozen=True)
class PipelineGrid:
    """按数据源和一系列可选阶段构造笛卡尔积候选。"""

    sources: Sequence[str]
    stages: Sequence[Sequence[OperatorGrid]]

    def estimate_size(self) -> int:
        """估计各数据源与阶段参数组合的去重前候选上界。"""

        if not self.sources:
            return 0
        size = len(self.sources)
        for stage in self.stages:
            size *= sum(len(grid.expand()) for grid in stage)
        return size

    def generate(self) -> list[FactorCandidate]:
        """逐层展开单输入算子组合，并在每一层及时去除等价表达式。"""

        if not self.sources:
            raise ValueError("PipelineGrid 至少需要一个数据源")
        nodes = [ExpressionNode.column(source) for source in self.sources]
        for position, stage in enumerate(self.stages, start=1):
            if not stage:
                raise ValueError(f"PipelineGrid 第 {position} 个阶段不能为空")
            calls = [call for grid in stage for call in grid.expand()]
            next_nodes: list[ExpressionNode] = []
            for node in nodes:
                for operator, parameters in calls:
                    next_nodes.append(
                        node
                        if operator is None
                        else operation_node(operator, (node,), parameters)
                    )
            # 每一层立即去重，防止多个 identity 或等价参数使中间数量膨胀。
            nodes = [candidate.expression for candidate in _deduplicate(next_nodes)]
        return _deduplicate(nodes)


@dataclass(frozen=True)
class ExpressionGrid:
    """使用符号 DSL 构建相关性、条件表达式等多输入公式网格。"""

    builder: Callable[[SymbolicDailyFrame, Mapping[str, object]], FactorExpression]
    parameters: Mapping[str, Sequence[object]]

    def estimate_size(self) -> int:
        """返回模板参数取值笛卡尔积的组合数量。"""

        size = 1
        for values in self.parameters.values():
            size *= len(values)
        return size

    def generate(self) -> list[FactorCandidate]:
        """对每组参数调用符号构建器，并校验、收集和去重多输入表达式。"""

        names = sorted(self.parameters)
        choices = [tuple(self.parameters[name]) for name in names]
        if any(not values for values in choices):
            raise ValueError("ExpressionGrid 参数不能包含空取值")
        combinations = product(*choices) if names else [()]
        namespace = SymbolicDailyFrame()
        nodes: list[ExpressionNode] = []
        for values in combinations:
            parameters = dict(zip(names, values))
            expression = self.builder(namespace, parameters)
            if not isinstance(expression, FactorExpression):
                raise TypeError("ExpressionGrid builder 必须返回 FactorExpression")
            if expression.frame is not None:
                raise ValueError("ExpressionGrid builder 不能绑定真实 DailyFactorFrame")
            nodes.append(expression.node)
        return _deduplicate(nodes)


@dataclass(frozen=True)
class CombinedGrid:
    """合并多个搜索空间，并对跨空间的相同公式再次去重。"""

    spaces: Sequence[SearchSpace]

    def estimate_size(self) -> int:
        """汇总所有子空间的去重前候选数量上界。"""

        return sum(space.estimate_size() for space in self.spaces)

    def generate(self) -> list[FactorCandidate]:
        """按子空间顺序合并候选，并再次消除跨空间重复表达式。"""

        if not self.spaces:
            raise ValueError("CombinedGrid 至少需要一个子空间")
        nodes = [
            candidate.expression
            for space in self.spaces
            for candidate in space.generate()
        ]
        return _deduplicate(nodes)
