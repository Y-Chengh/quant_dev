"""内置因子算子的 pandas 实现。

所有时间序列算子都按 ``code`` 独立计算，所有横截面算子都按
``trade_date`` 独立计算。滚动窗口包含当日且默认要求完整窗口；这里不提供任何
向未来移动数据的算子。
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .registry import normalize_no_parameters, register_operator


def _replace_infinite(values: pd.Series) -> pd.Series:
    """统一把除零等运算产生的无穷值转成缺失值。

    参数：
        values: 要清理正负无穷值的算子结果序列。
    """

    return pd.Series(values, index=values.index).replace([np.inf, -np.inf], np.nan)


def _exact_parameters(
    parameters: Mapping[str, object],
    expected: set[str],
) -> None:
    """校验算子参数名称集合与声明完全一致，拒绝遗漏和多余参数。

    参数：
        parameters: 调用表达式实际传入的参数映射。
        expected: 当前算子要求且只允许的参数名集合。
    """

    actual = set(parameters)
    if actual != expected:
        raise ValueError(
            f"算子参数不匹配: expected={sorted(expected)} actual={sorted(actual)}"
        )


def _normalize_exponent(parameters: Mapping[str, object]) -> dict[str, object]:
    """校验并标准化幂运算使用的有限浮点指数。

    参数：
        parameters: 必须仅含 ``exponent`` 有限数值的算子参数。
    """

    _exact_parameters(parameters, {"exponent"})
    exponent = parameters["exponent"]
    if isinstance(exponent, bool) or not isinstance(exponent, (int, float)):
        raise TypeError("exponent 必须是有限数值")
    exponent = float(exponent)
    if not np.isfinite(exponent):
        raise ValueError("exponent 必须是有限数值")
    return {"exponent": exponent}


def _normalize_periods(parameters: Mapping[str, object]) -> dict[str, object]:
    """校验差分和收益算子使用的正整数历史周期。

    参数：
        parameters: 必须仅含 ``periods`` 正整数的算子参数。
    """

    _exact_parameters(parameters, {"periods"})
    periods = parameters["periods"]
    if isinstance(periods, bool) or not isinstance(periods, int) or periods <= 0:
        raise ValueError("periods 必须是正整数，禁止负数位移造成未来数据泄漏")
    return {"periods": periods}


def _normalize_delay(parameters: Mapping[str, object]) -> dict[str, object]:
    """校验延迟算子的非负整数周期，禁止向未来移动。

    参数：
        parameters: 必须仅含 ``periods`` 非负整数的延迟参数。
    """

    _exact_parameters(parameters, {"periods"})
    periods = parameters["periods"]
    if isinstance(periods, bool) or not isinstance(periods, int) or periods < 0:
        raise ValueError("delay periods 必须是非负整数，禁止向未来移动数据")
    return {"periods": periods}


def _normalize_window(parameters: Mapping[str, object]) -> dict[str, object]:
    """标准化滚动窗口及其最小有效观测数。

    参数：
        parameters: 含正整数 ``window`` 及可选 ``min_periods`` 的滚动参数。
    """

    allowed = {"window", "min_periods"}
    if "window" not in parameters or set(parameters).difference(allowed):
        raise ValueError(
            "滚动算子必须提供 window，且只接受 window、min_periods 参数"
        )
    window = parameters["window"]
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        raise ValueError("window 必须是正整数")
    min_periods = parameters.get("min_periods", window)
    if (
        isinstance(min_periods, bool)
        or not isinstance(min_periods, int)
        or not 1 <= min_periods <= window
    ):
        raise ValueError("min_periods 必须是 1 到 window 之间的整数")
    return {"min_periods": min_periods, "window": window}


def _normalize_stddev(parameters: Mapping[str, object]) -> dict[str, object]:
    """标准化滚动标准差的窗口、最小观测数和自由度参数。

    参数：
        parameters: 含 ``window`` 以及可选 ``min_periods``、``ddof`` 的标准差参数。
    """

    allowed = {"window", "min_periods", "ddof"}
    if "window" not in parameters or set(parameters).difference(allowed):
        raise ValueError(
            "ts_stddev 必须提供 window，且只接受 window、min_periods、ddof 参数"
        )
    window_parameters = _normalize_window(
        {
            "window": parameters["window"],
            **(
                {"min_periods": parameters["min_periods"]}
                if "min_periods" in parameters
                else {}
            ),
        }
    )
    ddof = parameters.get("ddof", 1)
    if isinstance(ddof, bool) or not isinstance(ddof, int) or ddof < 0:
        raise ValueError("ddof 必须是非负整数")
    return {**window_parameters, "ddof": ddof}


def _normalize_winsorize(parameters: Mapping[str, object]) -> dict[str, object]:
    """校验横截面缩尾使用的上下分位点。

    参数：
        parameters: 必须含 ``lower`` 和 ``upper`` 分位点的缩尾参数。
    """

    _exact_parameters(parameters, {"lower", "upper"})
    lower = float(parameters["lower"])
    upper = float(parameters["upper"])
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError("winsorize 要求 0 <= lower < upper <= 1")
    return {"lower": lower, "upper": upper}


def _period_lookback(parameters: Mapping[str, object]) -> int:
    """返回位移类算子需要额外读取的历史期数。

    参数：
        parameters: 已标准化且含 ``periods`` 的位移算子参数。
    """

    return int(parameters["periods"])


def _window_lookback(parameters: Mapping[str, object]) -> int:
    """返回包含当日的滚动窗口所需额外历史行数。

    参数：
        parameters: 已标准化且含 ``window`` 的滚动算子参数。
    """

    return int(parameters["window"]) - 1


@register_operator(
    name="add", arity=2, scope="elementwise", normalize_parameters=normalize_no_parameters
)
def _add(
    frame: pd.DataFrame, inputs: tuple[pd.Series, ...], parameters: Mapping[str, object]
) -> pd.Series:
    """逐元素相加两个输入，并把无穷结果统一转换为缺失值。

    参数：
        frame: 提供行顺序的日频表，逐元素算子不读取其分组列。
        inputs: 两个逐行对齐的加数序列。
        parameters: 已校验为空的算子参数。
    """

    return _replace_infinite(inputs[0] + inputs[1])


@register_operator(
    name="subtract",
    arity=2,
    scope="elementwise",
    normalize_parameters=normalize_no_parameters,
)
def _subtract(
    frame: pd.DataFrame, inputs: tuple[pd.Series, ...], parameters: Mapping[str, object]
) -> pd.Series:
    """逐元素用第一个输入减第二个输入。

    参数：
        frame: 提供行顺序的日频表，本算子不读取分组列。
        inputs: 依次为被减数和减数的对齐序列。
        parameters: 已校验为空的算子参数。
    """

    return _replace_infinite(inputs[0] - inputs[1])


@register_operator(
    name="multiply",
    arity=2,
    scope="elementwise",
    normalize_parameters=normalize_no_parameters,
)
def _multiply(
    frame: pd.DataFrame, inputs: tuple[pd.Series, ...], parameters: Mapping[str, object]
) -> pd.Series:
    """逐元素相乘两个输入，并规范化溢出的无穷值。

    参数：
        frame: 提供行顺序的日频表，本算子不读取分组列。
        inputs: 两个逐行对齐的乘数序列。
        parameters: 已校验为空的算子参数。
    """

    return _replace_infinite(inputs[0] * inputs[1])


@register_operator(
    name="divide",
    arity=2,
    scope="elementwise",
    normalize_parameters=normalize_no_parameters,
)
def _divide(
    frame: pd.DataFrame, inputs: tuple[pd.Series, ...], parameters: Mapping[str, object]
) -> pd.Series:
    """逐元素相除两个输入，分母为零时返回缺失值。

    参数：
        frame: 提供行顺序的日频表，本算子不读取分组列。
        inputs: 依次为分子和分母的对齐序列。
        parameters: 已校验为空的算子参数。
    """

    denominator = inputs[1].where(inputs[1] != 0)
    return _replace_infinite(inputs[0] / denominator)


@register_operator(
    name="negative",
    arity=1,
    scope="elementwise",
    normalize_parameters=normalize_no_parameters,
)
def _negative(
    frame: pd.DataFrame, inputs: tuple[pd.Series, ...], parameters: Mapping[str, object]
) -> pd.Series:
    """逐元素返回输入值的相反数。

    参数：
        frame: 提供行顺序的日频表，本算子不读取分组列。
        inputs: 只含待取反数序列的输入元组。
        parameters: 已校验为空的算子参数。
    """

    return -inputs[0]


@register_operator(
    name="absolute",
    arity=1,
    scope="elementwise",
    normalize_parameters=normalize_no_parameters,
)
def _absolute(
    frame: pd.DataFrame, inputs: tuple[pd.Series, ...], parameters: Mapping[str, object]
) -> pd.Series:
    """逐元素返回输入值的绝对值。

    参数：
        frame: 提供行顺序的日频表，本算子不读取分组列。
        inputs: 只含待取绝对值序列的输入元组。
        parameters: 已校验为空的算子参数。
    """

    return inputs[0].abs()


@register_operator(
    name="log", arity=1, scope="elementwise", normalize_parameters=normalize_no_parameters
)
def _log(
    frame: pd.DataFrame, inputs: tuple[pd.Series, ...], parameters: Mapping[str, object]
) -> pd.Series:
    """逐元素计算自然对数，非正数返回缺失值。

    参数：
        frame: 提供行顺序的日频表，本算子不读取分组列。
        inputs: 只含待取自然对数序列的输入元组。
        parameters: 已校验为空的算子参数。
    """

    # 对数只在正数上有定义；零、负数和缺失值均保留为缺失值。
    return np.log(inputs[0].where(inputs[0] > 0))


@register_operator(
    name="sign", arity=1, scope="elementwise", normalize_parameters=normalize_no_parameters
)
def _sign(
    frame: pd.DataFrame, inputs: tuple[pd.Series, ...], parameters: Mapping[str, object]
) -> pd.Series:
    """逐元素返回输入值的正负符号。

    参数：
        frame: 提供行顺序的日频表，本算子不读取分组列。
        inputs: 只含待判断符号序列的输入元组。
        parameters: 已校验为空的算子参数。
    """

    return np.sign(inputs[0])


@register_operator(
    name="power", arity=1, scope="elementwise", normalize_parameters=_normalize_exponent
)
def _power(
    frame: pd.DataFrame, inputs: tuple[pd.Series, ...], parameters: Mapping[str, object]
) -> pd.Series:
    """逐元素计算普通幂，并把非法或溢出结果规范为缺失值。

    参数：
        frame: 提供行顺序的日频表，本算子不读取分组列。
        inputs: 只含待作幂运算序列的输入元组。
        parameters: 含有限浮点 ``exponent`` 的已标准化参数。
    """

    with np.errstate(invalid="ignore", over="ignore"):
        result = inputs[0].pow(float(parameters["exponent"]))
    return _replace_infinite(result)


@register_operator(
    name="signed_power",
    arity=1,
    scope="elementwise",
    normalize_parameters=_normalize_exponent,
)
def _signed_power(
    frame: pd.DataFrame, inputs: tuple[pd.Series, ...], parameters: Mapping[str, object]
) -> pd.Series:
    """逐元素计算保留原符号的绝对值幂。

    参数：
        frame: 提供行顺序的日频表，本算子不读取分组列。
        inputs: 只含待作保号幂运算序列的输入元组。
        parameters: 含有限浮点 ``exponent`` 的已标准化参数。
    """

    with np.errstate(invalid="ignore", over="ignore"):
        result = np.sign(inputs[0]) * inputs[0].abs().pow(
            float(parameters["exponent"])
        )
    return _replace_infinite(result)


def _comparison(
    left: pd.Series, right: pd.Series, comparator
) -> pd.Series:
    """对两个序列应用比较器，并令任一输入缺失时条件为假。

    参数：
        left: 比较运算左侧的对齐数值序列。
        right: 比较运算右侧的对齐数值序列。
        comparator: 接收左右序列并执行指定关系比较的函数。
    """

    # 与 pandas 比较语义保持一致：任一输入缺失时条件为 False，使 where 走
    # false 分支；真正的分支值仍会保留自身的缺失值。
    valid = left.notna() & right.notna()
    return pd.Series(valid & comparator(left, right), index=left.index, dtype=bool)


for _name, _comparator in {
    "less_than": np.less,
    "less_equal": np.less_equal,
    "greater_than": np.greater,
    "greater_equal": np.greater_equal,
    "equal": np.equal,
    "not_equal": np.not_equal,
}.items():

    def _make_comparison(comparator):
        """为当前循环中的比较器创建并注册二元算子实现。

        参数：
            comparator: 当前算子名所绑定的 NumPy 关系比较函数。
        """

        @register_operator(
            name=_name,
            arity=2,
            scope="elementwise",
            normalize_parameters=normalize_no_parameters,
        )
        def evaluator(
            frame: pd.DataFrame,
            inputs: tuple[pd.Series, ...],
            parameters: Mapping[str, object],
        ) -> pd.Series:
            """调用闭包绑定的比较器计算逐元素布尔结果。

            参数：
                frame: 提供行顺序的日频表，本算子不读取分组列。
                inputs: 依次为比较左值和右值的两个对齐序列。
                parameters: 已校验为空的算子参数。
            """

            return _comparison(inputs[0], inputs[1], comparator)

        return evaluator

    _make_comparison(_comparator)


@register_operator(
    name="where", arity=3, scope="elementwise", normalize_parameters=normalize_no_parameters
)
def _where(
    frame: pd.DataFrame, inputs: tuple[pd.Series, ...], parameters: Mapping[str, object]
) -> pd.Series:
    """根据第一个布尔输入逐行选择真分支或假分支。

    参数：
        frame: 提供行顺序的日频表，本算子不读取分组列。
        inputs: 依次为条件、真分支和假分支的三个对齐序列。
        parameters: 已校验为空的算子参数。
    """

    condition, when_true, when_false = inputs
    return pd.Series(
        np.where(condition.fillna(False).astype(bool), when_true, when_false),
        index=condition.index,
    )


@register_operator(
    name="delay",
    arity=1,
    scope="time_series",
    normalize_parameters=_normalize_delay,
    additional_lookback=_period_lookback,
)
def _delay(
    frame: pd.DataFrame, inputs: tuple[pd.Series, ...], parameters: Mapping[str, object]
) -> pd.Series:
    """按证券独立返回指定历史期的原始值。

    参数：
        frame: 提供证券 ``code`` 分组键的已排序日频表。
        inputs: 只含待延迟时序序列的输入元组。
        parameters: 含非负历史期数 ``periods`` 的参数。
    """

    return inputs[0].groupby(frame["code"], sort=False).shift(
        int(parameters["periods"])
    )


@register_operator(
    name="delta",
    arity=1,
    scope="time_series",
    normalize_parameters=_normalize_periods,
    additional_lookback=_period_lookback,
)
def _delta(
    frame: pd.DataFrame, inputs: tuple[pd.Series, ...], parameters: Mapping[str, object]
) -> pd.Series:
    """按证券独立计算当前值与指定历史期值之差。

    参数：
        frame: 提供证券 ``code`` 分组键的已排序日频表。
        inputs: 只含待计算差分的时序序列。
        parameters: 含正整数历史期数 ``periods`` 的参数。
    """

    return inputs[0].groupby(frame["code"], sort=False).diff(
        int(parameters["periods"])
    )


@register_operator(
    name="returns",
    arity=1,
    scope="time_series",
    normalize_parameters=_normalize_periods,
    additional_lookback=_period_lookback,
)
def _returns(
    frame: pd.DataFrame, inputs: tuple[pd.Series, ...], parameters: Mapping[str, object]
) -> pd.Series:
    """按证券独立计算当前值相对指定历史期值的变化率。

    参数：
        frame: 提供证券 ``code`` 分组键的已排序日频表。
        inputs: 只含待计算历史变化率的时序序列。
        parameters: 含正整数历史期数 ``periods`` 的参数。
    """

    periods = int(parameters["periods"])
    previous = inputs[0].groupby(frame["code"], sort=False).shift(periods)
    return _replace_infinite(inputs[0] / previous.where(previous != 0) - 1)


def _rolling_transform(
    frame: pd.DataFrame,
    values: pd.Series,
    parameters: Mapping[str, object],
    method: str,
) -> pd.Series:
    """按证券调用 pandas 简单滚动聚合，并保持输入行位置。

    参数：
        frame: 提供证券分组键的已排序日频表。
        values: 要按证券执行滚动聚合的对齐序列。
        parameters: 含 ``window`` 和 ``min_periods`` 的滚动参数。
        method: pandas Rolling 对象上要调用的聚合方法名。
    """

    window = int(parameters["window"])
    min_periods = int(parameters["min_periods"])
    return values.groupby(frame["code"], sort=False).transform(
        lambda group: getattr(
            group.rolling(window, min_periods=min_periods), method
        )()
    )


def _register_simple_rolling(name: str, method: str) -> None:
    """注册可直接映射到 pandas Rolling 方法的单输入算子。

    参数：
        name: 要写入注册表的 DSL 滚动算子名。
        method: 与该算子对应的 pandas Rolling 聚合方法名。
    """

    @register_operator(
        name=name,
        arity=1,
        scope="time_series",
        normalize_parameters=_normalize_window,
        additional_lookback=_window_lookback,
    )
    def evaluator(
        frame: pd.DataFrame,
        inputs: tuple[pd.Series, ...],
        parameters: Mapping[str, object],
    ) -> pd.Series:
        """执行闭包指定的 pandas 滚动聚合方法。

        参数：
            frame: 提供证券分组键的已排序日频表。
            inputs: 只含待执行滚动聚合序列的输入元组。
            parameters: 含 ``window`` 和 ``min_periods`` 的滚动参数。
        """

        return _rolling_transform(frame, inputs[0], parameters, method)


for _operator_name, _rolling_method in {
    "ts_sum": "sum",
    "ts_mean": "mean",
    "ts_min": "min",
    "ts_max": "max",
}.items():
    _register_simple_rolling(_operator_name, _rolling_method)


@register_operator(
    name="ts_stddev",
    arity=1,
    scope="time_series",
    normalize_parameters=_normalize_stddev,
    additional_lookback=_window_lookback,
)
def _ts_stddev(
    frame: pd.DataFrame, inputs: tuple[pd.Series, ...], parameters: Mapping[str, object]
) -> pd.Series:
    """按证券计算包含当日的滚动标准差。

    参数：
        frame: 提供证券分组键的已排序日频表。
        inputs: 只含待计算滚动标准差序列的输入元组。
        parameters: 含 ``window``、``min_periods`` 和 ``ddof`` 的参数。
    """

    window = int(parameters["window"])
    min_periods = int(parameters["min_periods"])
    ddof = int(parameters["ddof"])
    return inputs[0].groupby(frame["code"], sort=False).transform(
        lambda group: group.rolling(window, min_periods=min_periods).std(ddof=ddof)
    )


def _first_argmax(values: np.ndarray) -> float:
    """忽略缺失值并返回最大有限值在原窗口中首次出现的 1 基位置。

    参数：
        values: 一个按时间升序排列的原始滚动窗口数组。
    """

    # min_periods 可以小于 window，此时窗口数组仍可能包含 NaN。只在有限值中
    # 比较大小，但返回其在原窗口中的 1 基位置，不能把 NaN 自身当成最大值。
    finite_positions = np.flatnonzero(np.isfinite(values))
    if not len(finite_positions):
        return float("nan")
    position = finite_positions[np.argmax(values[finite_positions])]
    return float(position + 1)


def _first_argmin(values: np.ndarray) -> float:
    """忽略缺失值并返回最小有限值在原窗口中首次出现的 1 基位置。

    参数：
        values: 一个按时间升序排列的原始滚动窗口数组。
    """

    finite_positions = np.flatnonzero(np.isfinite(values))
    if not len(finite_positions):
        return float("nan")
    position = finite_positions[np.argmin(values[finite_positions])]
    return float(position + 1)


def _current_percentile_rank(values: np.ndarray) -> float:
    """返回窗口最后一个值在窗口内的平均并列百分位排名。

    参数：
        values: 一个按时间升序排列且末元素为当日值的窗口数组。
    """

    return float(pd.Series(values).rank(method="average", pct=True).iloc[-1])


def _register_rolling_apply(name: str, calculator) -> None:
    """注册需要对每个滚动窗口调用自定义 NumPy 计算器的算子。

    参数：
        name: 要写入注册表的 DSL 滚动算子名。
        calculator: 接收单个 NumPy 窗口并返回标量的计算函数。
    """

    @register_operator(
        name=name,
        arity=1,
        scope="time_series",
        normalize_parameters=_normalize_window,
        additional_lookback=_window_lookback,
    )
    def evaluator(
        frame: pd.DataFrame,
        inputs: tuple[pd.Series, ...],
        parameters: Mapping[str, object],
    ) -> pd.Series:
        """按证券把闭包计算器应用到每个包含当日的滚动窗口。

        参数：
            frame: 提供证券分组键的已排序日频表。
            inputs: 只含待应用自定义滚动计算器的序列。
            parameters: 含 ``window`` 和 ``min_periods`` 的滚动参数。
        """

        window = int(parameters["window"])
        min_periods = int(parameters["min_periods"])
        return inputs[0].groupby(frame["code"], sort=False).transform(
            lambda group: group.rolling(window, min_periods=min_periods).apply(
                calculator, raw=True
            )
        )


_register_rolling_apply("ts_argmax", _first_argmax)
_register_rolling_apply("ts_argmin", _first_argmin)
_register_rolling_apply("ts_rank", _current_percentile_rank)


def _rolling_pair(
    frame: pd.DataFrame,
    left: pd.Series,
    right: pd.Series,
    parameters: Mapping[str, object],
    method: str,
) -> pd.Series:
    """按证券计算两个序列的滚动相关性或协方差并按位置回填。

    参数：
        frame: 提供证券分组键和回填行位置的已排序日频表。
        left: 成对滚动统计的左侧对齐序列。
        right: 成对滚动统计的右侧对齐序列。
        parameters: 含 ``window`` 和 ``min_periods`` 的滚动参数。
        method: 统计方法，``corr`` 表示相关系数，其他值表示协方差。
    """

    window = int(parameters["window"])
    min_periods = int(parameters["min_periods"])
    result = np.full(len(frame), np.nan, dtype=float)
    # 使用位置数组而不是标签索引，避免调用方提供重复索引时发生错误对齐。
    for positions in frame.groupby("code", sort=False).indices.values():
        left_group = pd.Series(left.iloc[positions].to_numpy(dtype=float))
        right_group = pd.Series(right.iloc[positions].to_numpy(dtype=float))
        rolling = left_group.rolling(window, min_periods=min_periods)
        if method == "corr":
            values = rolling.corr(right_group)
        else:
            values = rolling.cov(right_group)
        result[positions] = values.to_numpy()
    return pd.Series(result, index=frame.index)


def _register_rolling_pair(name: str, method: str) -> None:
    """注册相关性或协方差形式的双输入滚动算子。

    参数：
        name: 要写入注册表的 DSL 双输入算子名。
        method: pandas Rolling 使用的 ``corr`` 或 ``cov`` 方法名。
    """

    @register_operator(
        name=name,
        arity=2,
        scope="time_series",
        normalize_parameters=_normalize_window,
        additional_lookback=_window_lookback,
    )
    def evaluator(
        frame: pd.DataFrame,
        inputs: tuple[pd.Series, ...],
        parameters: Mapping[str, object],
    ) -> pd.Series:
        """调用闭包指定的双输入滚动统计方法。

        参数：
            frame: 提供证券分组键和行位置的已排序日频表。
            inputs: 依次为左右变量的两个对齐序列。
            parameters: 含 ``window`` 和 ``min_periods`` 的滚动参数。
        """

        return _rolling_pair(frame, inputs[0], inputs[1], parameters, method)


_register_rolling_pair("ts_correlation", "corr")
_register_rolling_pair("ts_covariance", "cov")


@register_operator(
    name="cs_rank",
    arity=1,
    scope="cross_sectional",
    normalize_parameters=normalize_no_parameters,
)
def _cs_rank(
    frame: pd.DataFrame, inputs: tuple[pd.Series, ...], parameters: Mapping[str, object]
) -> pd.Series:
    """按交易日计算平均并列百分位横截面排名。

    参数：
        frame: 提供 ``trade_date`` 横截面分组键的日频表。
        inputs: 只含待计算横截面排名序列的输入元组。
        parameters: 已校验为空的算子参数。
    """

    return inputs[0].groupby(frame["trade_date"], sort=False).rank(
        method="average", pct=True
    )


@register_operator(
    name="cs_demean",
    arity=1,
    scope="cross_sectional",
    normalize_parameters=normalize_no_parameters,
)
def _cs_demean(
    frame: pd.DataFrame, inputs: tuple[pd.Series, ...], parameters: Mapping[str, object]
) -> pd.Series:
    """按交易日从每个值中减去当日横截面均值。

    参数：
        frame: 提供 ``trade_date`` 横截面分组键的日频表。
        inputs: 只含待按日去均值序列的输入元组。
        parameters: 已校验为空的算子参数。
    """

    means = inputs[0].groupby(frame["trade_date"], sort=False).transform("mean")
    return inputs[0] - means


@register_operator(
    name="cs_zscore",
    arity=1,
    scope="cross_sectional",
    normalize_parameters=normalize_no_parameters,
)
def _cs_zscore(
    frame: pd.DataFrame, inputs: tuple[pd.Series, ...], parameters: Mapping[str, object]
) -> pd.Series:
    """按交易日使用总体标准差计算横截面 Z 分数。

    参数：
        frame: 提供 ``trade_date`` 横截面分组键的日频表。
        inputs: 只含待按日标准化序列的输入元组。
        parameters: 已校验为空的算子参数。
    """

    grouped = inputs[0].groupby(frame["trade_date"], sort=False)
    means = grouped.transform("mean")
    standard_deviations = grouped.transform(lambda values: values.std(ddof=0))
    return _replace_infinite((inputs[0] - means) / standard_deviations.where(standard_deviations != 0))


@register_operator(
    name="cs_scale",
    arity=1,
    scope="cross_sectional",
    normalize_parameters=normalize_no_parameters,
)
def _cs_scale(
    frame: pd.DataFrame, inputs: tuple[pd.Series, ...], parameters: Mapping[str, object]
) -> pd.Series:
    """按交易日缩放输入，使当日有效值绝对值之和为一。

    参数：
        frame: 提供 ``trade_date`` 横截面分组键的日频表。
        inputs: 只含待按日缩放序列的输入元组。
        parameters: 已校验为空的算子参数。
    """

    absolute_sum = inputs[0].abs().groupby(frame["trade_date"], sort=False).transform("sum")
    return _replace_infinite(inputs[0] / absolute_sum.where(absolute_sum != 0))


@register_operator(
    name="cs_winsorize",
    arity=1,
    scope="cross_sectional",
    normalize_parameters=_normalize_winsorize,
)
def _cs_winsorize(
    frame: pd.DataFrame, inputs: tuple[pd.Series, ...], parameters: Mapping[str, object]
) -> pd.Series:
    """按交易日使用给定上下分位点对横截面输入执行缩尾。

    参数：
        frame: 提供 ``trade_date`` 横截面分组键的日频表。
        inputs: 只含待按日缩尾序列的输入元组。
        parameters: 含 ``lower`` 和 ``upper`` 分位点的已标准化参数。
    """

    lower = float(parameters["lower"])
    upper = float(parameters["upper"])
    grouped = inputs[0].groupby(frame["trade_date"], sort=False)
    lower_bound = grouped.transform(lambda values: values.quantile(lower))
    upper_bound = grouped.transform(lambda values: values.quantile(upper))
    return inputs[0].clip(lower=lower_bound, upper=upper_bound)
