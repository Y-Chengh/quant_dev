"""受配置约束的表达式随机生成、交叉与变异。"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations

import numpy as np

from quant.factor_research.factor_dsl import ExpressionNode, get_operator, operation_node

from .config import GeneticSearchConfig
from .trees import _node_at, _parameter_combinations, _replace_subtree, _tree_paths


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
