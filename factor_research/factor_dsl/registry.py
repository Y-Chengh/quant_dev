"""因子算子注册表。

注册表只描述一个算子如何校验参数、如何计算以及需要多少历史数据；它不感知
网格搜索、模型或因子工厂，从而让算子层保持独立且容易复用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import pandas as pd


OperatorParameters = Mapping[str, object]
NormalizedParameters = dict[str, object]
OperatorEvaluator = Callable[
    [pd.DataFrame, tuple[pd.Series, ...], OperatorParameters], pd.Series
]
ParameterNormalizer = Callable[[OperatorParameters], NormalizedParameters]
LookbackCalculator = Callable[[OperatorParameters], int]


@dataclass(frozen=True)
class OperatorDefinition:
    """保存一个算子的稳定元数据和执行函数。"""

    name: str
    arity: int
    scope: str
    evaluator: OperatorEvaluator
    normalize_parameters: ParameterNormalizer
    additional_lookback: LookbackCalculator
    causal: bool = True


OPERATORS: dict[str, OperatorDefinition] = {}


def register_operator(
    *,
    name: str,
    arity: int,
    scope: str,
    normalize_parameters: ParameterNormalizer,
    additional_lookback: LookbackCalculator | None = None,
    causal: bool = True,
) -> Callable[[OperatorEvaluator], OperatorEvaluator]:
    """注册算子并返回原执行函数，便于使用装饰器声明内置算子。"""

    if not name or not name.isidentifier():
        raise ValueError(f"算子名称必须是非空标识符，实际为 {name!r}")
    if arity < 1:
        raise ValueError("算子输入数量必须大于等于 1")
    if name in OPERATORS:
        raise ValueError(f"算子名称重复: {name!r}")

    lookback = additional_lookback or (lambda parameters: 0)

    def decorator(evaluator: OperatorEvaluator) -> OperatorEvaluator:
        OPERATORS[name] = OperatorDefinition(
            name=name,
            arity=arity,
            scope=scope,
            evaluator=evaluator,
            normalize_parameters=normalize_parameters,
            additional_lookback=lookback,
            causal=causal,
        )
        return evaluator

    return decorator


def get_operator(name: str) -> OperatorDefinition:
    """按名称返回算子；未知名称给出所有可选项，便于定位表达式错误。"""

    try:
        return OPERATORS[name]
    except KeyError as exc:
        raise ValueError(f"未知因子算子 {name!r}；可选算子: {sorted(OPERATORS)}") from exc


def normalize_no_parameters(parameters: OperatorParameters) -> NormalizedParameters:
    """校验不接受任何参数的算子。"""

    if parameters:
        raise ValueError(f"该算子不接受参数，实际收到 {sorted(parameters)}")
    return {}
