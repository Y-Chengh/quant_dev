"""表达式树的路径枚举、子树定位与替换等无状态操作。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import product

from quant.factor_research.factor_dsl import ExpressionNode, operation_node


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
