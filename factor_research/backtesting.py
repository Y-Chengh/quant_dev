from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252
DEFAULT_RANDOM_BASELINE_SIMULATIONS = 1_000
DEFAULT_RANDOM_BASELINE_SEED = 42


@dataclass(frozen=True)
class TopNBacktestResult:
    """保存 Top N 日内策略及其横截面对照诊断。"""

    top_n: int
    score_column: str
    slippage_bps: float
    commission_bps: float
    daily_returns: pd.DataFrame
    metrics: dict[str, float]
    benchmark_metrics: dict[str, float] = field(default_factory=dict)
    relative_metrics: dict[str, float] = field(default_factory=dict)
    selection_metrics: dict[str, float] = field(default_factory=dict)
    decile_returns: pd.DataFrame = field(default_factory=pd.DataFrame)
    top_selections: pd.DataFrame = field(default_factory=pd.DataFrame)


def _compound_metrics(returns: np.ndarray) -> dict[str, float]:
    """计算可复利长仓日收益的累计、年化、波动和夏普指标。

    参数：
        returns: 按交易日排序的有限日收益率数组，单位为一。

    返回：
        含交易日数、累计收益率、复利年化收益率、年化波动率和夏普比率的字典。
    """

    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("复利指标要求非空的一维有限日收益率")
    periods = len(values)
    final_equity = float(np.prod(1.0 + values))
    daily_std = float(np.std(values, ddof=1)) if periods >= 2 else float("nan")
    return {
        "trading_days": float(periods),
        "total_return": final_equity - 1.0,
        "annualized_return": (
            float(final_equity ** (TRADING_DAYS_PER_YEAR / periods) - 1.0)
            if final_equity > 0.0
            else float("nan")
        ),
        "annualized_volatility": (
            float(daily_std * np.sqrt(TRADING_DAYS_PER_YEAR))
            if np.isfinite(daily_std)
            else float("nan")
        ),
        "sharpe_ratio": (
            float(np.sqrt(TRADING_DAYS_PER_YEAR) * np.mean(values) / daily_std)
            if np.isfinite(daily_std) and daily_std > 0.0
            else float("nan")
        ),
    }


def _spread_metrics(returns: np.ndarray, prefix: str) -> dict[str, float]:
    """计算横截面收益差的算术年化、波动率和夏普比率。

    参数：
        returns: Top N 减基准或 Bottom N 的逐日收益差，单位为一。
        prefix: 输出指标键的业务前缀，用于区分等权超额和多空价差。

    返回：
        使用算术年化收益的三项价差诊断指标。
    """

    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("价差指标要求非空的一维有限日收益率")
    daily_std = float(np.std(values, ddof=1)) if len(values) >= 2 else float("nan")
    return {
        f"{prefix}_annualized_return": float(np.mean(values) * TRADING_DAYS_PER_YEAR),
        f"{prefix}_annualized_volatility": (
            float(daily_std * np.sqrt(TRADING_DAYS_PER_YEAR))
            if np.isfinite(daily_std)
            else float("nan")
        ),
        f"{prefix}_sharpe_ratio": (
            float(np.sqrt(TRADING_DAYS_PER_YEAR) * np.mean(values) / daily_std)
            if np.isfinite(daily_std) and daily_std > 0.0
            else float("nan")
        ),
    }


def run_top_n_intraday_backtest(
    predictions: pd.DataFrame,
    score_column: str,
    top_n: int = 10,
    slippage_bps: float = 0.0,
    commission_bps: float = 0.0,
    random_simulations: int = DEFAULT_RANDOM_BASELINE_SIMULATIONS,
    random_seed: int = DEFAULT_RANDOM_BASELINE_SEED,
) -> TopNBacktestResult:
    """回测每日买入预测分数最高证券并在收盘卖出的等权策略。

    每个目标交易日在全部有效预测中按分数降序、证券代码升序稳定选取最多
    ``top_n`` 只。买入成交价相对开盘价上浮滑点，卖出成交价相对收盘价下调
    滑点，手续费在买卖两边各收取一次。组合不跨日持仓，收益曲线从 1.0 开始
    按每日净收益复利。夏普比率采用零无风险利率、样本标准差和 252 日年化。

    参数：
        predictions: 验证集逐证券预测，须含目标日期、证券代码、开盘至收盘
            实际收益及指定模型分数。
        score_column: 用于每日横截面选股的模型分数字段；分类通常为
            ``up_probability``，回归通常为 ``predicted_return``。
        top_n: 每日最多买入的证券数量；当日有效证券不足时全部买入。
        slippage_bps: 单边滑点，单位为基点；买卖两边分别应用一次。
        commission_bps: 单边手续费率，单位为基点；买卖两边分别收取一次。
        random_simulations: 随机等权选取最多 ``top_n`` 只证券的蒙特卡洛路径数；
            缺省为 1,000 次。
        random_seed: 随机基准的伪随机种子；缺省为 42，以保证重复运行结果一致。

    返回：
        回测参数、按日组合收益、年化夏普等汇总指标，以及每日 Top N 证券的
        预测值、当日实际收益、前日收盘相对前前日收盘收益，以及前日收盘相对
        前日开盘收益明细。
    """

    if top_n < 1:
        raise ValueError("top_n 必须是正整数")
    if not 0.0 <= slippage_bps < 10_000.0:
        raise ValueError("slippage_bps 必须在 [0, 10000) 范围内")
    if not 0.0 <= commission_bps < 10_000.0:
        raise ValueError("commission_bps 必须在 [0, 10000) 范围内")
    if random_simulations < 1:
        raise ValueError("random_simulations 必须是正整数")
    required = {"target_date", "code", "target_return", score_column}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Top N 回测缺少列: {sorted(missing)}")

    previous_return_columns = [
        column
        for column in (
            "previous_close_to_close_return",
            "previous_open_to_close_return",
        )
        if column in predictions.columns
    ]
    frame = predictions.loc[
        :,
        [
            "target_date",
            "code",
            "target_return",
            score_column,
            *previous_return_columns,
        ],
    ].copy()
    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="coerce")
    if frame["target_date"].isna().any():
        raise ValueError("Top N 回测的 target_date 包含缺失或无效日期")
    frame[score_column] = pd.to_numeric(frame[score_column], errors="coerce")
    frame["target_return"] = pd.to_numeric(
        frame["target_return"], errors="coerce"
    )
    for column in previous_return_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["code"] = frame["code"].astype(str)
    frame["_row_id"] = np.arange(len(frame), dtype=np.int64)
    # 前日收益按证券和日期独立后移；重复键采用最高有限分数记录，避免依赖输入顺序。
    history_candidates = frame.copy()
    history_candidates["_history_score"] = history_candidates[
        score_column
    ].where(np.isfinite(history_candidates[score_column]), np.nan)
    best_history_score = history_candidates.groupby(
        ["code", "target_date"], sort=False
    )["_history_score"].transform("max")
    best_history = history_candidates.loc[
        history_candidates["_history_score"].eq(best_history_score)
        | (
            history_candidates["_history_score"].isna()
            & best_history_score.isna()
        )
    ].copy()
    conflicting_history = best_history.groupby(
        ["code", "target_date"], sort=False
    )["target_return"].nunique(dropna=False)
    conflicting_history = conflicting_history.loc[conflicting_history > 1]
    if not conflicting_history.empty:
        details = ", ".join(
            f"{code}:{pd.Timestamp(target_date).date()}"
            for code, target_date in conflicting_history.index[:5]
        )
        raise ValueError(f"同证券同日期的最高分记录存在冲突实际收益: {details}")
    return_history = (
        best_history.sort_values(
            ["code", "target_date", "_row_id"], kind="mergesort"
        )
        .drop_duplicates(["code", "target_date"], keep="first")
        .loc[:, ["code", "target_date", "target_return"]]
    )
    return_history["_legacy_previous_open_to_close_return"] = return_history.groupby(
        "code", sort=False
    )["target_return"].shift(1)
    frame = frame.merge(
        return_history.loc[
            :, ["code", "target_date", "_legacy_previous_open_to_close_return"]
        ],
        on=["code", "target_date"],
        how="left",
        validate="many_to_one",
    )
    if "previous_open_to_close_return" not in frame.columns:
        frame["previous_open_to_close_return"] = frame[
            "_legacy_previous_open_to_close_return"
        ]
    if "previous_close_to_close_return" not in frame.columns:
        frame["previous_close_to_close_return"] = np.nan
    # 保留旧字段作为兼容别名；新报告只使用两个明确口径的字段。
    frame["previous_actual_return"] = frame["previous_open_to_close_return"]
    valid_score = np.isfinite(frame[score_column].to_numpy(dtype=float))
    frame = frame.loc[valid_score].copy()
    if frame.empty:
        raise ValueError("Top N 回测没有模型分数为有限值的样本")
    frame = frame.sort_values(
        ["target_date", score_column, "code"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    grouped = frame.groupby("target_date", sort=True, group_keys=False)
    selected = grouped.head(top_n).copy()
    finite_return = np.isfinite(selected["target_return"].to_numpy(dtype=float))
    if not finite_return.all():
        invalid = selected.loc[~finite_return, ["target_date", "code"]]
        details = ", ".join(
            f"{pd.Timestamp(row.target_date).date()}:{row.code}"
            for row in invalid.itertuples(index=False)
        )
        raise ValueError(f"Top N 已选证券的实际收益不是有限值: {details}")

    finite_universe_return = np.isfinite(frame["target_return"].to_numpy(dtype=float))
    if not finite_universe_return.all():
        raise ValueError("横截面对照要求全部有限模型分数样本具有有限实际收益")

    slippage = slippage_bps / 10_000.0
    commission = commission_bps / 10_000.0
    execution_multiplier = (
        (1.0 - slippage) * (1.0 - commission)
        / ((1.0 + slippage) * (1.0 + commission))
    )
    selected["net_return"] = (
        (1.0 + selected["target_return"]) * execution_multiplier - 1.0
    )
    frame["net_return"] = (
        (1.0 + frame["target_return"]) * execution_multiplier - 1.0
    )
    bottom = grouped.tail(top_n).copy()
    rank_position = grouped.cumcount()
    frame["top_rank"] = rank_position + 1
    group_size = grouped["code"].transform("size")
    mid_count = group_size.clip(upper=top_n)
    mid_start = (group_size - mid_count) // 2
    mid_mask = rank_position.ge(mid_start) & rank_position.lt(mid_start + mid_count)
    mid = frame.loc[mid_mask].copy()
    daily = (
        selected.groupby("target_date", as_index=False, sort=True)
        .agg(
            selected_count=("code", "size"),
            gross_return=("target_return", "mean"),
            net_return=("net_return", "mean"),
        )
        .sort_values("target_date", kind="mergesort")
        .reset_index(drop=True)
    )
    daily["equity"] = (1.0 + daily["net_return"]).cumprod()
    metrics = _compound_metrics(daily["net_return"].to_numpy(dtype=float))

    universe_daily = frame.groupby("target_date", sort=True).agg(
        universe_count=("code", "size"),
        universe_gross_return=("target_return", "mean"),
        universe_net_return=("net_return", "mean"),
    )
    bottom_daily = bottom.groupby("target_date", sort=True).agg(
        bottom_count=("code", "size"),
        bottom_gross_return=("target_return", "mean"),
        bottom_net_return=("net_return", "mean"),
    )
    mid_daily = mid.groupby("target_date", sort=True).agg(
        mid_count=("code", "size"),
        mid_gross_return=("target_return", "mean"),
        mid_net_return=("net_return", "mean"),
    )
    daily = daily.merge(universe_daily, on="target_date", validate="one_to_one")
    daily = daily.merge(bottom_daily, on="target_date", validate="one_to_one")
    daily = daily.merge(mid_daily, on="target_date", validate="one_to_one")
    daily["universe_equity"] = (1.0 + daily["universe_net_return"]).cumprod()
    daily["bottom_equity"] = (1.0 + daily["bottom_net_return"]).cumprod()
    daily["mid_equity"] = (1.0 + daily["mid_net_return"]).cumprod()
    daily["top_minus_universe_return"] = (
        daily["gross_return"] - daily["universe_gross_return"]
    )
    daily["top_minus_bottom_return"] = (
        daily["gross_return"] - daily["bottom_gross_return"]
    )

    equal_weight = _compound_metrics(
        daily["universe_net_return"].to_numpy(dtype=float)
    )
    rng = np.random.default_rng(random_seed)
    random_equity = np.ones(random_simulations, dtype=float)
    for _, date_group in frame.groupby("target_date", sort=True):
        date_returns = date_group["net_return"].to_numpy(dtype=float)
        sample_size = min(top_n, len(date_returns))
        random_keys = rng.random((random_simulations, len(date_returns)))
        chosen = np.argpartition(random_keys, sample_size - 1, axis=1)[:, :sample_size]
        portfolio_returns = date_returns[chosen].mean(axis=1)
        random_equity *= 1.0 + portfolio_returns
    periods = len(daily)
    random_annualized = np.where(
        random_equity > 0.0,
        random_equity ** (TRADING_DAYS_PER_YEAR / periods) - 1.0,
        np.nan,
    )
    valid_random = random_annualized[np.isfinite(random_annualized)]
    strategy_annualized = metrics["annualized_return"]
    if valid_random.size:
        random_p05 = float(np.quantile(valid_random, 0.05))
        random_median = float(np.median(valid_random))
        random_p95 = float(np.quantile(valid_random, 0.95))
        strategy_percentile = (
            float(np.mean(valid_random <= strategy_annualized))
            if np.isfinite(strategy_annualized)
            else float("nan")
        )
    else:
        random_p05 = float("nan")
        random_median = float("nan")
        random_p95 = float("nan")
        strategy_percentile = float("nan")
    benchmark_metrics = {
        "equal_weight_total_return": equal_weight["total_return"],
        "equal_weight_annualized_return": equal_weight["annualized_return"],
        "equal_weight_sharpe_ratio": equal_weight["sharpe_ratio"],
        "random_simulations": float(random_simulations),
        "random_annualized_p05": random_p05,
        "random_annualized_median": random_median,
        "random_annualized_p95": random_p95,
        "strategy_random_percentile": strategy_percentile,
    }
    relative_metrics = {
        **_spread_metrics(
            daily["top_minus_universe_return"].to_numpy(dtype=float),
            "top_minus_universe",
        ),
        **_spread_metrics(
            daily["top_minus_bottom_return"].to_numpy(dtype=float),
            "top_minus_bottom",
        ),
    }

    ascending_position = grouped.cumcount().rsub(group_size - 1)
    decile = (ascending_position * 10 // group_size + 1).astype(int)
    # 少于十只证券时无法形成十个等频组，用跨越 1 至 10 的有序标签明确两端。
    small_group = group_size < 10
    multi_security = small_group & (group_size > 1)
    decile.loc[multi_security] = np.rint(
        ascending_position.loc[multi_security]
        * 9
        / (group_size.loc[multi_security] - 1)
    ).astype(int) + 1
    decile.loc[group_size == 1] = 10
    frame["predicted_decile"] = decile
    daily_decile_returns = (
        frame.groupby(
            ["target_date", "predicted_decile"], as_index=False, sort=True
        )
        .agg(
            samples=("code", "size"),
            daily_return=("target_return", "mean"),
            daily_hit_rate=(
                "target_return", lambda values: float((values > 0).mean())
            ),
        )
    )
    decile_returns = (
        daily_decile_returns.groupby(
            "predicted_decile", as_index=False, sort=True
        )
        .agg(
            samples=("samples", "sum"),
            trading_days=("target_date", "size"),
            average_return=("daily_return", "mean"),
            hit_rate=("daily_hit_rate", "mean"),
        )
    )
    decile_returns["annualized_arithmetic_return"] = (
        decile_returns["average_return"] * TRADING_DAYS_PER_YEAR
    )

    selected_mask = frame["_row_id"].isin(selected["_row_id"])
    selected_returns = frame.loc[selected_mask, "target_return"]
    other_returns = frame.loc[~selected_mask, "target_return"]
    top_selections = (
        frame.loc[
            selected_mask,
            [
                "target_date",
                "top_rank",
                "code",
                score_column,
                "target_return",
                "previous_close_to_close_return",
                "previous_open_to_close_return",
                "previous_actual_return",
            ],
        ]
        .rename(
            columns={
                score_column: "predicted_value",
                "target_return": "actual_return",
            }
        )
        .sort_values(["target_date", "top_rank"], kind="mergesort")
        .reset_index(drop=True)
    )
    selection_metrics = {
        "top_samples": float(len(selected_returns)),
        "top_hit_rate": float((selected_returns > 0).mean()),
        "top_average_return": float(selected_returns.mean()),
        "other_samples": float(len(other_returns)),
        "other_hit_rate": float((other_returns > 0).mean()),
        "other_average_return": float(other_returns.mean()),
        "top_minus_other_average_return": float(
            selected_returns.mean() - other_returns.mean()
        ),
    }
    return TopNBacktestResult(
        top_n=top_n,
        score_column=score_column,
        slippage_bps=float(slippage_bps),
        commission_bps=float(commission_bps),
        daily_returns=daily,
        metrics=metrics,
        benchmark_metrics=benchmark_metrics,
        relative_metrics=relative_metrics,
        selection_metrics=selection_metrics,
        decile_returns=decile_returns,
        top_selections=top_selections,
    )
