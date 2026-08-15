"""因子算子注册表。

注册表只描述一个算子如何校验参数、如何计算以及需要多少历史数据；它不感知
网格搜索、模型或因子工厂，从而让算子层保持独立且容易复用。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

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
    """注册算子并返回原执行函数，便于使用装饰器声明内置算子。

    参数：
        name: 算子在表达式中使用的唯一标识符。
        arity: 算子要求的输入表达式数量。
        scope: 计算作用域，如逐元素、时序或横截面。
        normalize_parameters: 校验并标准化算子参数的函数。
        additional_lookback: 根据参数计算额外历史行数的函数；为空时回看长度为 0。
        causal: 算子是否仅依赖当日及以前可见数据，缺省为 ``True``。
    """

    if not name or not name.isidentifier():
        raise ValueError(f"算子名称必须是非空标识符，实际为 {name!r}")
    if arity < 1:
        raise ValueError("算子输入数量必须大于等于 1")
    if name in OPERATORS:
        raise ValueError(f"算子名称重复: {name!r}")

    lookback = additional_lookback or (lambda parameters: 0)

    def decorator(evaluator: OperatorEvaluator) -> OperatorEvaluator:
        """把被装饰的执行函数及其元数据写入 DSL 私有注册表。

        参数：
            evaluator: 接收日频表、输入序列和参数并返回因子值的函数。
        """

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
    """按名称返回算子；未知名称给出所有可选项，便于定位表达式错误。

    参数：
        name: 要查找的已注册算子名称。
    """

    try:
        return OPERATORS[name]
    except KeyError as exc:
        raise ValueError(f"未知因子算子 {name!r}；可选算子: {sorted(OPERATORS)}") from exc


def normalize_no_parameters(parameters: OperatorParameters) -> NormalizedParameters:
    """校验不接受任何参数的算子。

    参数：
        parameters: 调用方传入的算子参数映射，此处必须为空。
    """

    if parameters:
        raise ValueError(f"该算子不接受参数，实际收到 {sorted(parameters)}")
    return {}
