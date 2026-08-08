"""内置因子算子的 pandas 实现。

所有时间序列算子都按 ``code`` 独立计算，所有横截面算子都按
``trade_date`` 独立计算。滚动窗口包含当日且默认要求完整窗口；这里不提供任何
向未来移动数据的算子。
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from .registry import normalize_no_parameters, register_operator


def _replace_infinite(values: pd.Series) -> pd.Series:
    """统一把除零等运算产生的无穷值转成缺失值。"""

    return pd.Series(values, index=values.index).replace([np.inf, -np.inf], np.nan)


def _exact_parameters(
    parameters: Mapping[str, object],
    expected: set[str],
) -> None:
    actual = set(parameters)
    if actual != expected:
        raise ValueError(
            f"算子参数不匹配: expected={sorted(expected)} actual={sorted(actual)}"
        )


def _normalize_exponent(parameters: Mapping[str, object]) -> dict[str, object]:
    _exact_parameters(parameters, {"exponent"})
    exponent = parameters["exponent"]
    if isinstance(exponent, bool) or not isinstance(exponent, (int, float)):
        raise TypeError("exponent 必须是有限数值")
    exponent = float(exponent)
    if not np.isfinite(exponent):
        raise ValueError("exponent 必须是有限数值")
    return {"exponent": exponent}


def _normalize_periods(parameters: Mapping[str, object]) -> dict[str, object]:
    _exact_parameters(parameters, {"periods"})
    periods = parameters["periods"]
    if isinstance(periods, bool) or not isinstance(periods, int) or periods <= 0:
        raise ValueError("periods 必须是正整数，禁止负数位移造成未来数据泄漏")
    return {"periods": periods}


def _normalize_delay(parameters: Mapping[str, object]) -> dict[str, object]:
    _exact_parameters(parameters, {"periods"})
    periods = parameters["periods"]
    if isinstance(periods, bool) or not isinstance(periods, int) or periods < 0:
        raise ValueError("delay periods 必须是非负整数，禁止向未来移动数据")
    return {"periods": periods}


def _normalize_window(parameters: Mapping[str, object]) -> dict[str, object]:
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
    _exact_parameters(parameters, {"lower", "upper"})
    lower = float(parameters["lower"])
    upper = float(parameters["upper"])
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError("winsorize 要求 0 <= lower < upper <= 1")
    return {"lower": lower, "upper": upper}


def _period_lookback(parameters: Mapping[str, object]) -> int:
    return int(parameters["periods"])


def _window_lookback(parameters: Mapping[str, object]) -> int:
    return int(parameters["window"]) - 1


@register_operator(
    name="add", arity=2, scope="elementwise", normalize_parameters=normalize_no_parameters
)
def _add(
    frame: pd.DataFrame, inputs: tuple[pd.Series, ...], parameters: Mapping[str, object]
) -> pd.Series:
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
    return inputs[0].abs()


@register_operator(
    name="log", arity=1, scope="elementwise", normalize_parameters=normalize_no_parameters
)
def _log(
    frame: pd.DataFrame, inputs: tuple[pd.Series, ...], parameters: Mapping[str, object]
) -> pd.Series:
    # 对数只在正数上有定义；零、负数和缺失值均保留为缺失值。
    return np.log(inputs[0].where(inputs[0] > 0))


@register_operator(
    name="sign", arity=1, scope="elementwise", normalize_parameters=normalize_no_parameters
)
def _sign(
    frame: pd.DataFrame, inputs: tuple[pd.Series, ...], parameters: Mapping[str, object]
) -> pd.Series:
    return np.sign(inputs[0])


@register_operator(
    name="power", arity=1, scope="elementwise", normalize_parameters=_normalize_exponent
)
def _power(
    frame: pd.DataFrame, inputs: tuple[pd.Series, ...], parameters: Mapping[str, object]
) -> pd.Series:
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
    with np.errstate(invalid="ignore", over="ignore"):
        result = np.sign(inputs[0]) * inputs[0].abs().pow(
            float(parameters["exponent"])
        )
    return _replace_infinite(result)


def _comparison(
    left: pd.Series, right: pd.Series, comparator
) -> pd.Series:
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
            return _comparison(inputs[0], inputs[1], comparator)

        return evaluator

    _make_comparison(_comparator)


@register_operator(
    name="where", arity=3, scope="elementwise", normalize_parameters=normalize_no_parameters
)
def _where(
    frame: pd.DataFrame, inputs: tuple[pd.Series, ...], parameters: Mapping[str, object]
) -> pd.Series:
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
    periods = int(parameters["periods"])
    previous = inputs[0].groupby(frame["code"], sort=False).shift(periods)
    return _replace_infinite(inputs[0] / previous.where(previous != 0) - 1)


def _rolling_transform(
    frame: pd.DataFrame,
    values: pd.Series,
    parameters: Mapping[str, object],
    method: str,
) -> pd.Series:
    window = int(parameters["window"])
    min_periods = int(parameters["min_periods"])
    return values.groupby(frame["code"], sort=False).transform(
        lambda group: getattr(
            group.rolling(window, min_periods=min_periods), method
        )()
    )


def _register_simple_rolling(name: str, method: str) -> None:
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
    window = int(parameters["window"])
    min_periods = int(parameters["min_periods"])
    ddof = int(parameters["ddof"])
    return inputs[0].groupby(frame["code"], sort=False).transform(
        lambda group: group.rolling(window, min_periods=min_periods).std(ddof=ddof)
    )


def _first_argmax(values: np.ndarray) -> float:
    # min_periods 可以小于 window，此时窗口数组仍可能包含 NaN。只在有限值中
    # 比较大小，但返回其在原窗口中的 1 基位置，不能把 NaN 自身当成最大值。
    finite_positions = np.flatnonzero(np.isfinite(values))
    if not len(finite_positions):
        return float("nan")
    position = finite_positions[np.argmax(values[finite_positions])]
    return float(position + 1)


def _first_argmin(values: np.ndarray) -> float:
    finite_positions = np.flatnonzero(np.isfinite(values))
    if not len(finite_positions):
        return float("nan")
    position = finite_positions[np.argmin(values[finite_positions])]
    return float(position + 1)


def _current_percentile_rank(values: np.ndarray) -> float:
    return float(pd.Series(values).rank(method="average", pct=True).iloc[-1])


def _register_rolling_apply(name: str, calculator) -> None:
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
    lower = float(parameters["lower"])
    upper = float(parameters["upper"])
    grouped = inputs[0].groupby(frame["trade_date"], sort=False)
    lower_bound = grouped.transform(lambda values: values.quantile(lower))
    upper_bound = grouped.transform(lambda values: values.quantile(upper))
    return inputs[0].clip(lower=lower_bound, upper=upper_bound)
