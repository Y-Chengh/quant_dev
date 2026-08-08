"""不可变因子表达式树及链式 API。"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
import hashlib
import json
import keyword
import math
import unicodedata
from typing import TYPE_CHECKING, Callable, Mapping

from .registry import get_operator

if TYPE_CHECKING:
    import pandas as pd

    from .frame import DailyFactorFrame


ScalarParameter = str | int | float | bool | None


def _stable_parameter(value: object) -> ScalarParameter:
    """只允许可稳定序列化的标量进入表达式，避免缓存键依赖对象地址。

    参数：
        value: 要写入表达式标识的原始算子参数值。
    """

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
        """创建引用日频表指定列的数据源节点。

        参数：
            name: 执行时从日频表读取的非空列名。
        """

        if not isinstance(name, str) or not name:
            raise ValueError("列名必须是非空字符串")
        return cls(operator="column", parameters=(("name", name),))

    @classmethod
    def constant(cls, value: float) -> "ExpressionNode":
        """创建在全部行上取同一有限数值的常量节点。

        参数：
            value: 表达式每行共享的有限数值常量。
        """

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("常数表达式必须使用数值")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("常数表达式必须使用有限数值")
        return cls(operator="constant", parameters=(("value", numeric),))

    @property
    def parameter_map(self) -> dict[str, ScalarParameter]:
        """把稳定排序的参数元组转换为便于算子读取的字典。"""

        return dict(self.parameters)

    @property
    def canonical(self) -> str:
        """返回适合作为缓存键和日志字段的稳定表达式字符串。"""

        if self.operator == "column":
            name = str(self.parameter_map["name"])
            can_use_bare_name = (
                name.isidentifier()
                and not keyword.iskeyword(name)
                and unicodedata.normalize("NFKC", name) == name
            )
            argument = name if can_use_bare_name else repr(name)
            return f"column({argument})"
        if self.operator == "constant":
            return f"constant({self.parameter_map['value']!r})"
        arguments = [child.canonical for child in self.inputs]
        arguments.extend(f"{key}={value!r}" for key, value in self.parameters)
        return f"{self.operator}({','.join(arguments)})"

    @property
    def factor_id(self) -> str:
        """根据规范表达式生成可复现且适合作为列名的短因子 ID。"""

        digest = hashlib.sha256(self.canonical.encode("utf-8")).hexdigest()[:16]
        return f"fg_{digest}"

    @property
    def depth(self) -> int:
        """返回表达式树从当前节点到最深叶节点的算子层数。"""

        if not self.inputs:
            return 0
        return 1 + max(child.depth for child in self.inputs)

    @property
    def lookback(self) -> int:
        """返回计算当前节点所需的最大额外历史行数。"""

        if self.operator in {"column", "constant"}:
            return 0
        definition = get_operator(self.operator)
        child_lookback = max(child.lookback for child in self.inputs)
        return child_lookback + definition.additional_lookback(self.parameter_map)

    @property
    def causal(self) -> bool:
        """判断当前节点及全部子节点是否只依赖当日和历史数据。"""

        if self.operator in {"column", "constant"}:
            return True
        definition = get_operator(self.operator)
        return definition.causal and all(child.causal for child in self.inputs)

    @property
    def columns(self) -> frozenset[str]:
        """返回表达式直接或间接引用的全部日频列名。"""

        if self.operator == "column":
            return frozenset({str(self.parameter_map["name"])})
        return frozenset().union(*(child.columns for child in self.inputs))

    def to_dict(self) -> dict[str, object]:
        """导出不包含 Python 代码的安全配置结构。"""

        return {
            "operator": self.operator,
            "inputs": [child.to_dict() for child in self.inputs],
            "parameters": self.parameter_map,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ExpressionNode":
        """从安全配置恢复表达式，同时重新执行全部算子参数校验。

        参数：
            payload: 含 ``operator``、``inputs`` 和 ``parameters`` 的递归配置映射。
        """

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
        """把安全配置结构编码为字段顺序稳定的 JSON 字符串。"""

        return json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )

    def to_string(self) -> str:
        """返回可由 :meth:`from_string` 安全恢复的规范表达式字符串。"""

        return self.canonical

    @classmethod
    def from_string(cls, expression: str) -> "ExpressionNode":
        """从搜索输出的规范字符串安全恢复表达式节点。

        解析器只接受 DSL 算子调用、标量命名参数以及 ``column(close)`` 形式的
        列引用，不执行任意 Python 代码。恢复过程中会重新校验算子数量、参数和
        因果性。

        参数：
            expression: 搜索结果中的 ``canonical``/``expression_str`` 字符串。
        """

        if not isinstance(expression, str) or not expression.strip():
            raise ValueError("因子表达式字符串不能为空")
        if len(expression) > 20_000:
            raise ValueError("因子表达式字符串过长")
        try:
            parsed = ast.parse(expression, mode="eval")
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"因子表达式字符串语法错误: {exc}") from exc
        return _node_from_ast(parsed.body)


def _scalar_from_ast(node: ast.AST) -> ScalarParameter:
    """从 AST 节点读取 DSL 允许的标量参数，不执行 Python 表达式。

    参数：
        node: 应表示字符串、布尔值、整数、有限浮点数、空值或负数的 AST 节点。
    """

    if isinstance(node, ast.Constant) and (
        node.value is None or isinstance(node.value, (str, bool, int, float))
    ):
        value = node.value
    elif (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.USub, ast.UAdd))
        and isinstance(node.operand, ast.Constant)
        and not isinstance(node.operand.value, bool)
        and isinstance(node.operand.value, (int, float))
    ):
        value = -node.operand.value if isinstance(node.op, ast.USub) else node.operand.value
    else:
        raise ValueError("因子表达式参数只能使用标量字面量")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("因子表达式参数必须是有限数值")
    return value


def _node_from_ast(node: ast.AST) -> ExpressionNode:
    """递归把受限调用语法转换为经过注册表校验的表达式节点。

    参数：
        node: 当前待解析的调用节点；任何非白名单 Python 语法都会被拒绝。
    """

    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise ValueError("因子表达式只能由 DSL 算子调用组成")
    if any(keyword.arg is None for keyword in node.keywords):
        raise ValueError("因子表达式不支持 ** 参数展开")
    if len({keyword.arg for keyword in node.keywords}) != len(node.keywords):
        raise ValueError("因子表达式包含重复命名参数")

    operator = node.func.id
    parameters = {
        str(keyword.arg): _scalar_from_ast(keyword.value) for keyword in node.keywords
    }
    if operator == "column":
        if node.keywords or len(node.args) != 1:
            raise ValueError("column 表达式只接受一个列名")
        column = node.args[0]
        if isinstance(column, ast.Name):
            return ExpressionNode.column(column.id)
        if isinstance(column, ast.Constant) and isinstance(column.value, str):
            return ExpressionNode.column(column.value)
        raise ValueError("column 列名必须是标识符或字符串")
    if operator == "constant":
        if node.keywords or len(node.args) != 1:
            raise ValueError("constant 表达式只接受一个数值")
        value = _scalar_from_ast(node.args[0])
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("constant 表达式只接受有限数值")
        return ExpressionNode.constant(value)

    inputs = tuple(_node_from_ast(child) for child in node.args)
    return operation_node(operator, inputs, parameters)


def _invalid_child() -> ExpressionNode:
    """为反序列化时发现的非对象子节点抛出统一格式错误。"""

    raise ValueError("表达式 inputs 必须由对象组成")


def operation_node(
    operator: str,
    inputs: tuple[ExpressionNode, ...],
    parameters: Mapping[str, object] | None = None,
) -> ExpressionNode:
    """构建已完成输入数量和参数校验的算子节点。

    参数：
        operator: 要调用的已注册 DSL 算子名。
        inputs: 按算子定义顺序排列的子表达式节点。
        parameters: 算子命名参数；为空时按空映射校验。
    """

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
        """把数值转换为同上下文常量表达式，并拒绝跨 Frame 组合。

        参数：
            value: 要保留或转为常量节点的表达式或有限数值。
        """

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
        """把当前表达式、其他输入和参数组合成一个已校验的新算子节点。

        参数：
            operator: 要应用到所有输入的已注册算子名。
            *others: 在当前表达式之后传入的其他表达式或数值输入。
            **parameters: 传给算子标准化器的命名参数。
        """

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
        """在已绑定的日频执行上下文中计算表达式并返回对齐序列。

        参数：
            name: 返回序列名；缺省时使用表达式的稳定因子 ID。
        """

        if self.frame is None:
            raise RuntimeError("符号表达式没有绑定 DailyFactorFrame，不能直接计算")
        return self.frame.evaluate(self, name=name)

    def to_dict(self) -> dict[str, object]:
        """导出底层表达式树的安全可序列化配置。"""

        return self.node.to_dict()

    def __add__(self, other: "FactorExpression | float") -> "FactorExpression":
        """构建当前表达式加另一表达式或常数的节点。

        参数：
            other: 加法右侧的表达式或数值常量。
        """

        return self.apply("add", other)

    def __radd__(self, other: "FactorExpression | float") -> "FactorExpression":
        """构建另一表达式或常数加当前表达式的节点。

        参数：
            other: 反向加法左侧的表达式或数值常量。
        """

        return self._coerce(other).apply("add", self)

    def __sub__(self, other: "FactorExpression | float") -> "FactorExpression":
        """构建当前表达式减另一表达式或常数的节点。

        参数：
            other: 减法右侧的表达式或数值常量。
        """

        return self.apply("subtract", other)

    def __rsub__(self, other: "FactorExpression | float") -> "FactorExpression":
        """构建另一表达式或常数减当前表达式的节点。

        参数：
            other: 反向减法左侧的表达式或数值常量。
        """

        return self._coerce(other).apply("subtract", self)

    def __mul__(self, other: "FactorExpression | float") -> "FactorExpression":
        """构建当前表达式乘另一表达式或常数的节点。

        参数：
            other: 乘法右侧的表达式或数值常量。
        """

        return self.apply("multiply", other)

    def __rmul__(self, other: "FactorExpression | float") -> "FactorExpression":
        """构建另一表达式或常数乘当前表达式的节点。

        参数：
            other: 反向乘法左侧的表达式或数值常量。
        """

        return self._coerce(other).apply("multiply", self)

    def __truediv__(self, other: "FactorExpression | float") -> "FactorExpression":
        """构建当前表达式除以另一表达式或常数的节点。

        参数：
            other: 除法右侧的分母表达式或数值常量。
        """

        return self.apply("divide", other)

    def __rtruediv__(self, other: "FactorExpression | float") -> "FactorExpression":
        """构建另一表达式或常数除以当前表达式的节点。

        参数：
            other: 反向除法左侧的分子表达式或数值常量。
        """

        return self._coerce(other).apply("divide", self)

    def __neg__(self) -> "FactorExpression":
        """构建当前表达式逐元素取负的节点。"""

        return self.apply("negative")

    def __abs__(self) -> "FactorExpression":
        """构建当前表达式逐元素取绝对值的节点。"""

        return self.apply("absolute")

    def __lt__(self, other: "FactorExpression | float") -> "FactorExpression":
        """构建当前表达式小于另一输入的布尔条件节点。

        参数：
            other: 小于比较右侧的表达式或数值常量。
        """

        return self.apply("less_than", other)

    def __le__(self, other: "FactorExpression | float") -> "FactorExpression":
        """构建当前表达式小于等于另一输入的布尔条件节点。

        参数：
            other: 小于等于比较右侧的表达式或数值常量。
        """

        return self.apply("less_equal", other)

    def __gt__(self, other: "FactorExpression | float") -> "FactorExpression":
        """构建当前表达式大于另一输入的布尔条件节点。

        参数：
            other: 大于比较右侧的表达式或数值常量。
        """

        return self.apply("greater_than", other)

    def __ge__(self, other: "FactorExpression | float") -> "FactorExpression":
        """构建当前表达式大于等于另一输入的布尔条件节点。

        参数：
            other: 大于等于比较右侧的表达式或数值常量。
        """

        return self.apply("greater_equal", other)

    def negative(self) -> "FactorExpression":
        """以链式方法形式构建逐元素取负节点。"""

        return -self

    def absolute(self) -> "FactorExpression":
        """以链式方法形式构建逐元素绝对值节点。"""

        return abs(self)

    def log(self) -> "FactorExpression":
        """构建仅对正数有效的逐元素自然对数节点。"""

        return self.apply("log")

    def sign(self) -> "FactorExpression":
        """构建返回每个值正负方向的符号节点。"""

        return self.apply("sign")

    def power(self, exponent: float) -> "FactorExpression":
        """构建逐元素普通幂运算节点。

        参数：
            exponent: 应用到每个输入值的有限浮点指数。
        """

        return self.apply("power", exponent=exponent)

    def signed_power(self, exponent: float) -> "FactorExpression":
        """构建保留原值符号的绝对值幂运算节点。

        参数：
            exponent: 应用到每个输入绝对值的有限浮点指数。
        """

        return self.apply("signed_power", exponent=exponent)

    def equal(self, other: "FactorExpression | float") -> "FactorExpression":
        """构建当前表达式等于另一输入的布尔条件节点。

        参数：
            other: 相等比较右侧的表达式或数值常量。
        """

        return self.apply("equal", other)

    def not_equal(self, other: "FactorExpression | float") -> "FactorExpression":
        """构建当前表达式不等于另一输入的布尔条件节点。

        参数：
            other: 不等比较右侧的表达式或数值常量。
        """

        return self.apply("not_equal", other)

    def delay(self, periods: int = 1) -> "FactorExpression":
        """构建按证券取若干历史期原值的延迟节点。

        参数：
            periods: 向历史回看的非负交易日数，缺省为 1。
        """

        return self.apply("delay", periods=periods)

    def delta(self, periods: int = 1) -> "FactorExpression":
        """构建当前值减同证券若干历史期值的差分节点。

        参数：
            periods: 差分基准向历史回看的正交易日数，缺省为 1。
        """

        return self.apply("delta", periods=periods)

    def returns(self, periods: int = 1) -> "FactorExpression":
        """构建当前值相对同证券若干历史期值的收益率节点。

        参数：
            periods: 收益率基准向历史回看的正交易日数，缺省为 1。
        """

        return self.apply("returns", periods=periods)

    @staticmethod
    def _rolling_parameters(window: int, min_periods: int | None) -> dict[str, int]:
        """统一展开滚动窗口参数，并令缺省最小观测数等于窗口长度。

        参数：
            window: 包含当日的滚动交易日窗口长度。
            min_periods: 输出非缺失值所需最小观测数；为空时等于窗口长度。
        """

        return {
            "window": window,
            "min_periods": window if min_periods is None else min_periods,
        }

    def sum(self, window: int, min_periods: int | None = None) -> "FactorExpression":
        """构建按证券计算包含当日滚动和的节点。

        参数：
            window: 滚动求和包含的交易日数。
            min_periods: 输出有效和所需最小观测数；缺省等于 ``window``。
        """

        return self.apply("ts_sum", **self._rolling_parameters(window, min_periods))

    def mean(self, window: int, min_periods: int | None = None) -> "FactorExpression":
        """构建按证券计算包含当日滚动均值的节点。

        参数：
            window: 滚动均值包含的交易日数。
            min_periods: 输出有效均值所需最小观测数；缺省等于 ``window``。
        """

        return self.apply("ts_mean", **self._rolling_parameters(window, min_periods))

    def stddev(
        self,
        window: int,
        min_periods: int | None = None,
        ddof: int = 1,
    ) -> "FactorExpression":
        """构建按证券计算包含当日滚动标准差的节点。

        参数：
            window: 滚动标准差包含的交易日数。
            min_periods: 输出有效标准差所需最小观测数；缺省等于 ``window``。
            ddof: 标准差的自由度扣减值，缺省为样本标准差所用的 1。
        """

        return self.apply(
            "ts_stddev",
            **self._rolling_parameters(window, min_periods),
            ddof=ddof,
        )

    def min(self, window: int, min_periods: int | None = None) -> "FactorExpression":
        """构建按证券计算包含当日滚动最小值的节点。

        参数：
            window: 滚动最小值包含的交易日数。
            min_periods: 输出有效最小值所需最小观测数；缺省等于 ``window``。
        """

        return self.apply("ts_min", **self._rolling_parameters(window, min_periods))

    def max(self, window: int, min_periods: int | None = None) -> "FactorExpression":
        """构建按证券计算包含当日滚动最大值的节点。

        参数：
            window: 滚动最大值包含的交易日数。
            min_periods: 输出有效最大值所需最小观测数；缺省等于 ``window``。
        """

        return self.apply("ts_max", **self._rolling_parameters(window, min_periods))

    def ts_rank(
        self, window: int, min_periods: int | None = None
    ) -> "FactorExpression":
        """构建当前值在同证券滚动窗口内的百分位排名节点。

        参数：
            window: 时序排名包含的交易日数。
            min_periods: 输出有效排名所需最小观测数；缺省等于 ``window``。
        """

        return self.apply("ts_rank", **self._rolling_parameters(window, min_periods))

    def argmax(
        self, window: int, min_periods: int | None = None
    ) -> "FactorExpression":
        """构建滚动最大值在原窗口中首次出现的 1 基位置节点。

        参数：
            window: 查找最大值位置的交易日窗口长度。
            min_periods: 输出有效位置所需最小观测数；缺省等于 ``window``。
        """

        return self.apply("ts_argmax", **self._rolling_parameters(window, min_periods))

    def argmin(
        self, window: int, min_periods: int | None = None
    ) -> "FactorExpression":
        """构建滚动最小值在原窗口中首次出现的 1 基位置节点。

        参数：
            window: 查找最小值位置的交易日窗口长度。
            min_periods: 输出有效位置所需最小观测数；缺省等于 ``window``。
        """

        return self.apply("ts_argmin", **self._rolling_parameters(window, min_periods))

    def correlation(
        self,
        other: "FactorExpression",
        window: int,
        min_periods: int | None = None,
    ) -> "FactorExpression":
        """构建两个表达式按证券计算滚动 Pearson 相关系数的节点。

        参数：
            other: 相关性计算的右侧表达式。
            window: 滚动相关性包含的交易日数。
            min_periods: 输出有效相关系数所需最小成对观测数；为空时等于 ``window``。
        """

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
        """构建两个表达式按证券计算滚动样本协方差的节点。

        参数：
            other: 协方差计算的右侧表达式。
            window: 滚动协方差包含的交易日数。
            min_periods: 输出有效协方差所需最小成对观测数；为空时等于 ``window``。
        """

        return self.apply(
            "ts_covariance",
            other,
            **self._rolling_parameters(window, min_periods),
        )

    def rank(self) -> "FactorExpression":
        """构建按交易日计算平均并列百分位排名的横截面节点。"""

        return self.apply("cs_rank")

    def cs_rank(self) -> "FactorExpression":
        """提供与 ``rank`` 等价且作用域更明确的横截面排名别名。"""

        return self.rank()

    def demean(self) -> "FactorExpression":
        """构建逐日减去横截面均值的中心化节点。"""

        return self.apply("cs_demean")

    def zscore(self) -> "FactorExpression":
        """构建逐日使用总体标准差归一化的横截面 Z 分数节点。"""

        return self.apply("cs_zscore")

    def scale(self) -> "FactorExpression":
        """构建逐日令横截面绝对值之和为一的缩放节点。"""

        return self.apply("cs_scale")

    def winsorize(self, lower: float = 0.01, upper: float = 0.99) -> "FactorExpression":
        """构建逐日按给定分位数上下界缩尾的横截面节点。

        参数：
            lower: 横截面下界分位点，缺省为 0.01。
            upper: 横截面上界分位点，缺省为 0.99 且必须大于 ``lower``。
        """

        return self.apply("cs_winsorize", lower=lower, upper=upper)

    def where(
        self,
        condition: "FactorExpression",
        other: "FactorExpression | float",
    ) -> "FactorExpression":
        """条件成立时取当前表达式，否则取 other。

        参数：
            condition: 逐行决定选择真分支或假分支的布尔表达式。
            other: 条件为假时使用的表达式或数值常量。
        """

        return condition.apply("where", self, other)


class ExpressionNamespace:
    """为真实日频数据和纯符号模板提供一致的数据源入口。"""

    def __init__(self, frame: "DailyFactorFrame | None" = None):
        """保存可选执行上下文，使同一入口兼容真实计算和符号建模。

        参数：
            frame: 表达式要绑定的日频执行上下文；为空时只构建符号树。
        """

        self._expression_frame = frame

    def column(self, name: str) -> FactorExpression:
        """创建引用指定日频列并继承当前执行上下文的表达式。

        参数：
            name: 执行时要从日频表读取的列名。
        """

        return FactorExpression(ExpressionNode.column(name), self._expression_frame)

    def feature(self, name: str) -> FactorExpression:
        """以语义化别名引用已经计算好的固定因子列。

        参数：
            name: 日频表中已计算的固定因子列名。
        """

        return self.column(name)

    def constant(self, value: float) -> FactorExpression:
        """创建绑定当前上下文的有限常量表达式。

        参数：
            value: 表达式每行共享的有限数值常量。
        """

        return FactorExpression(ExpressionNode.constant(value), self._expression_frame)

    def open(self) -> FactorExpression:
        """创建引用日频开盘价 ``open`` 的表达式。"""

        return self.column("open")

    def high(self) -> FactorExpression:
        """创建引用日频最高价 ``high`` 的表达式。"""

        return self.column("high")

    def low(self) -> FactorExpression:
        """创建引用日频最低价 ``low`` 的表达式。"""

        return self.column("low")

    def close(self) -> FactorExpression:
        """创建引用日频收盘价 ``close`` 的表达式。"""

        return self.column("close")

    def volume(self) -> FactorExpression:
        """创建引用日频成交量 ``volume`` 的表达式。"""

        return self.column("volume")

    def where(
        self,
        condition: FactorExpression,
        when_true: FactorExpression | float,
        when_false: FactorExpression | float,
    ) -> FactorExpression:
        """构建条件为真、假时分别选择两个输入的表达式。

        参数：
            condition: 逐行决定分支的布尔表达式。
            when_true: 条件成立时的表达式或数值常量。
            when_false: 条件不成立时的表达式或数值常量。
        """

        true_expression = condition._coerce(when_true)
        return condition.apply("where", true_expression, when_false)


class SymbolicDailyFrame(ExpressionNamespace):
    """仅构建表达式、不携带 DataFrame 的模板命名空间。"""

    def __init__(self):
        """初始化不绑定真实数据、仅用于搜索模板展开的命名空间。"""

        super().__init__(frame=None)
