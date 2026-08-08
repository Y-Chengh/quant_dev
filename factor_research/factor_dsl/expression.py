"""不可变因子表达式树及链式 API。"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import TYPE_CHECKING, Callable, Mapping

from .registry import get_operator

if TYPE_CHECKING:
    import pandas as pd

    from .frame import DailyFactorFrame


ScalarParameter = str | int | float | bool | None


def _stable_parameter(value: object) -> ScalarParameter:
    """只允许可稳定序列化的标量进入表达式，避免缓存键依赖对象地址。"""

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"表达式参数必须是标量，实际为 {type(value).__name__}")


@dataclass(frozen=True)
class ExpressionNode:
    """一个不携带实际数据的不可变表达式节点。"""

    operator: str
    inputs: tuple["ExpressionNode", ...] = ()
    parameters: tuple[tuple[str, ScalarParameter], ...] = ()

    @classmethod
    def column(cls, name: str) -> "ExpressionNode":
        if not isinstance(name, str) or not name:
            raise ValueError("列名必须是非空字符串")
        return cls(operator="column", parameters=(("name", name),))

    @classmethod
    def constant(cls, value: float) -> "ExpressionNode":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("常数表达式必须使用数值")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("常数表达式必须使用有限数值")
        return cls(operator="constant", parameters=(("value", numeric),))

    @property
    def parameter_map(self) -> dict[str, ScalarParameter]:
        return dict(self.parameters)

    @property
    def canonical(self) -> str:
        """返回适合作为缓存键和日志字段的稳定表达式字符串。"""

        if self.operator == "column":
            return f"column({self.parameter_map['name']})"
        if self.operator == "constant":
            return f"constant({self.parameter_map['value']!r})"
        arguments = [child.canonical for child in self.inputs]
        arguments.extend(f"{key}={value!r}" for key, value in self.parameters)
        return f"{self.operator}({','.join(arguments)})"

    @property
    def factor_id(self) -> str:
        digest = hashlib.sha256(self.canonical.encode("utf-8")).hexdigest()[:16]
        return f"fg_{digest}"

    @property
    def depth(self) -> int:
        if not self.inputs:
            return 0
        return 1 + max(child.depth for child in self.inputs)

    @property
    def lookback(self) -> int:
        if self.operator in {"column", "constant"}:
            return 0
        definition = get_operator(self.operator)
        child_lookback = max(child.lookback for child in self.inputs)
        return child_lookback + definition.additional_lookback(self.parameter_map)

    @property
    def causal(self) -> bool:
        if self.operator in {"column", "constant"}:
            return True
        definition = get_operator(self.operator)
        return definition.causal and all(child.causal for child in self.inputs)

    def to_dict(self) -> dict[str, object]:
        """导出不包含 Python 代码的安全配置结构。"""

        return {
            "operator": self.operator,
            "inputs": [child.to_dict() for child in self.inputs],
            "parameters": self.parameter_map,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ExpressionNode":
        """从安全配置恢复表达式，同时重新执行全部算子参数校验。"""

        operator = payload.get("operator")
        if not isinstance(operator, str):
            raise ValueError("表达式配置缺少字符串 operator")
        parameters = payload.get("parameters", {})
        inputs = payload.get("inputs", [])
        if not isinstance(parameters, Mapping) or not isinstance(inputs, list):
            raise ValueError("表达式 inputs 或 parameters 格式错误")
        if operator == "column":
            if inputs or set(parameters) != {"name"}:
                raise ValueError("column 表达式只接受 name 且不能包含 inputs")
            name = parameters["name"]
            if not isinstance(name, str):
                raise ValueError("column name 必须是字符串")
            return cls.column(name)
        if operator == "constant":
            if inputs or set(parameters) != {"value"}:
                raise ValueError("constant 表达式只接受 value 且不能包含 inputs")
            value = parameters.get("value")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("constant 表达式缺少数值 value")
            return cls.constant(value)
        children = tuple(
            cls.from_dict(child) if isinstance(child, Mapping) else _invalid_child()
            for child in inputs
        )
        return operation_node(operator, children, parameters)

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )


def _invalid_child() -> ExpressionNode:
    raise ValueError("表达式 inputs 必须由对象组成")


def operation_node(
    operator: str,
    inputs: tuple[ExpressionNode, ...],
    parameters: Mapping[str, object] | None = None,
) -> ExpressionNode:
    """构建已完成输入数量和参数校验的算子节点。"""

    definition = get_operator(operator)
    if len(inputs) != definition.arity:
        raise ValueError(
            f"算子 {operator!r} 需要 {definition.arity} 个输入，实际为 {len(inputs)}"
        )
    normalized = definition.normalize_parameters(parameters or {})
    stable = tuple(
        (key, _stable_parameter(value)) for key, value in sorted(normalized.items())
    )
    node = ExpressionNode(operator=operator, inputs=inputs, parameters=stable)
    if not node.causal:
        raise ValueError(f"表达式包含非因果算子: {operator}")
    return node


@dataclass(frozen=True, eq=False)
class FactorExpression:
    """链式表达式包装器；数据绑定只影响执行，不参与表达式身份。"""

    node: ExpressionNode
    frame: "DailyFactorFrame | None" = field(default=None, repr=False)

    def _coerce(self, value: "FactorExpression | float") -> "FactorExpression":
        if isinstance(value, FactorExpression):
            if self.frame is not None and value.frame not in {None, self.frame}:
                raise ValueError("不能组合来自不同 DailyFactorFrame 的表达式")
            return value
        return FactorExpression(ExpressionNode.constant(value), self.frame)

    def apply(
        self,
        operator: str,
        *others: "FactorExpression | float",
        **parameters: object,
    ) -> "FactorExpression":
        expressions = (self, *(self._coerce(value) for value in others))
        node = operation_node(
            operator,
            tuple(expression.node for expression in expressions),
            parameters,
        )
        frame = next(
            (expression.frame for expression in expressions if expression.frame is not None),
            None,
        )
        return FactorExpression(node, frame)

    def compute(self, name: str | None = None) -> "pd.Series":
        if self.frame is None:
            raise RuntimeError("符号表达式没有绑定 DailyFactorFrame，不能直接计算")
        return self.frame.evaluate(self, name=name)

    def to_dict(self) -> dict[str, object]:
        return self.node.to_dict()

    def __add__(self, other: "FactorExpression | float") -> "FactorExpression":
        return self.apply("add", other)

    def __radd__(self, other: "FactorExpression | float") -> "FactorExpression":
        return self._coerce(other).apply("add", self)

    def __sub__(self, other: "FactorExpression | float") -> "FactorExpression":
        return self.apply("subtract", other)

    def __rsub__(self, other: "FactorExpression | float") -> "FactorExpression":
        return self._coerce(other).apply("subtract", self)

    def __mul__(self, other: "FactorExpression | float") -> "FactorExpression":
        return self.apply("multiply", other)

    def __rmul__(self, other: "FactorExpression | float") -> "FactorExpression":
        return self._coerce(other).apply("multiply", self)

    def __truediv__(self, other: "FactorExpression | float") -> "FactorExpression":
        return self.apply("divide", other)

    def __rtruediv__(self, other: "FactorExpression | float") -> "FactorExpression":
        return self._coerce(other).apply("divide", self)

    def __neg__(self) -> "FactorExpression":
        return self.apply("negative")

    def __abs__(self) -> "FactorExpression":
        return self.apply("absolute")

    def __lt__(self, other: "FactorExpression | float") -> "FactorExpression":
        return self.apply("less_than", other)

    def __le__(self, other: "FactorExpression | float") -> "FactorExpression":
        return self.apply("less_equal", other)

    def __gt__(self, other: "FactorExpression | float") -> "FactorExpression":
        return self.apply("greater_than", other)

    def __ge__(self, other: "FactorExpression | float") -> "FactorExpression":
        return self.apply("greater_equal", other)

    def negative(self) -> "FactorExpression":
        return -self

    def absolute(self) -> "FactorExpression":
        return abs(self)

    def log(self) -> "FactorExpression":
        return self.apply("log")

    def sign(self) -> "FactorExpression":
        return self.apply("sign")

    def power(self, exponent: float) -> "FactorExpression":
        return self.apply("power", exponent=exponent)

    def signed_power(self, exponent: float) -> "FactorExpression":
        return self.apply("signed_power", exponent=exponent)

    def equal(self, other: "FactorExpression | float") -> "FactorExpression":
        return self.apply("equal", other)

    def not_equal(self, other: "FactorExpression | float") -> "FactorExpression":
        return self.apply("not_equal", other)

    def delay(self, periods: int = 1) -> "FactorExpression":
        return self.apply("delay", periods=periods)

    def delta(self, periods: int = 1) -> "FactorExpression":
        return self.apply("delta", periods=periods)

    def returns(self, periods: int = 1) -> "FactorExpression":
        return self.apply("returns", periods=periods)

    @staticmethod
    def _rolling_parameters(window: int, min_periods: int | None) -> dict[str, int]:
        return {
            "window": window,
            "min_periods": window if min_periods is None else min_periods,
        }

    def sum(self, window: int, min_periods: int | None = None) -> "FactorExpression":
        return self.apply("ts_sum", **self._rolling_parameters(window, min_periods))

    def mean(self, window: int, min_periods: int | None = None) -> "FactorExpression":
        return self.apply("ts_mean", **self._rolling_parameters(window, min_periods))

    def stddev(
        self,
        window: int,
        min_periods: int | None = None,
        ddof: int = 1,
    ) -> "FactorExpression":
        return self.apply(
            "ts_stddev",
            **self._rolling_parameters(window, min_periods),
            ddof=ddof,
        )

    def min(self, window: int, min_periods: int | None = None) -> "FactorExpression":
        return self.apply("ts_min", **self._rolling_parameters(window, min_periods))

    def max(self, window: int, min_periods: int | None = None) -> "FactorExpression":
        return self.apply("ts_max", **self._rolling_parameters(window, min_periods))

    def ts_rank(
        self, window: int, min_periods: int | None = None
    ) -> "FactorExpression":
        return self.apply("ts_rank", **self._rolling_parameters(window, min_periods))

    def argmax(
        self, window: int, min_periods: int | None = None
    ) -> "FactorExpression":
        return self.apply("ts_argmax", **self._rolling_parameters(window, min_periods))

    def argmin(
        self, window: int, min_periods: int | None = None
    ) -> "FactorExpression":
        return self.apply("ts_argmin", **self._rolling_parameters(window, min_periods))

    def correlation(
        self,
        other: "FactorExpression",
        window: int,
        min_periods: int | None = None,
    ) -> "FactorExpression":
        return self.apply(
            "ts_correlation",
            other,
            **self._rolling_parameters(window, min_periods),
        )

    def covariance(
        self,
        other: "FactorExpression",
        window: int,
        min_periods: int | None = None,
    ) -> "FactorExpression":
        return self.apply(
            "ts_covariance",
            other,
            **self._rolling_parameters(window, min_periods),
        )

    def rank(self) -> "FactorExpression":
        return self.apply("cs_rank")

    def cs_rank(self) -> "FactorExpression":
        return self.rank()

    def demean(self) -> "FactorExpression":
        return self.apply("cs_demean")

    def zscore(self) -> "FactorExpression":
        return self.apply("cs_zscore")

    def scale(self) -> "FactorExpression":
        return self.apply("cs_scale")

    def winsorize(self, lower: float = 0.01, upper: float = 0.99) -> "FactorExpression":
        return self.apply("cs_winsorize", lower=lower, upper=upper)

    def where(
        self,
        condition: "FactorExpression",
        other: "FactorExpression | float",
    ) -> "FactorExpression":
        """条件成立时取当前表达式，否则取 other。"""

        return condition.apply("where", self, other)


class ExpressionNamespace:
    """为真实日频数据和纯符号模板提供一致的数据源入口。"""

    def __init__(self, frame: "DailyFactorFrame | None" = None):
        self._expression_frame = frame

    def column(self, name: str) -> FactorExpression:
        return FactorExpression(ExpressionNode.column(name), self._expression_frame)

    def feature(self, name: str) -> FactorExpression:
        return self.column(name)

    def constant(self, value: float) -> FactorExpression:
        return FactorExpression(ExpressionNode.constant(value), self._expression_frame)

    def open(self) -> FactorExpression:
        return self.column("open")

    def high(self) -> FactorExpression:
        return self.column("high")

    def low(self) -> FactorExpression:
        return self.column("low")

    def close(self) -> FactorExpression:
        return self.column("close")

    def volume(self) -> FactorExpression:
        return self.column("volume")

    def where(
        self,
        condition: FactorExpression,
        when_true: FactorExpression | float,
        when_false: FactorExpression | float,
    ) -> FactorExpression:
        true_expression = condition._coerce(when_true)
        return condition.apply("where", true_expression, when_false)


class SymbolicDailyFrame(ExpressionNamespace):
    """仅构建表达式、不携带 DataFrame 的模板命名空间。"""

    def __init__(self):
        super().__init__(frame=None)
