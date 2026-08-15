"""可组合、可序列化且无未来数据操作的日频因子 DSL。"""

# 导入 operators 会完成内置算子注册；注册只影响本 DSL 的私有注册表，不会改变
# 现有 FACTOR_FACTORIES 或 DEFAULT_FEATURES。
from . import operators as _operators  # noqa: F401  # 仅为触发算子注册副作用
from .expression import (
    ExpressionNode,
    FactorExpression,
    SymbolicDailyFrame,
    operation_node,
)
from .frame import DailyFactorFrame
from .registry import OPERATORS, OperatorDefinition, get_operator

__all__ = [
    "DailyFactorFrame",
    "ExpressionNode",
    "FactorExpression",
    "OPERATORS",
    "OperatorDefinition",
    "SymbolicDailyFrame",
    "get_operator",
    "operation_node",
]
